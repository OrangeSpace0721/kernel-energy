"""Capture the kernel configs a pipeline actually executes.

The obvious way to do this is module forward hooks. It works and it is model-specific:
every pipeline names its blocks differently, MMDiT attention concatenates two streams
before projecting, and the shape you want is often not the shape the module was
constructed with.

So instead this patches the handful of ``torch.nn.functional`` entry points that every
implementation ultimately calls, and records the tensor shapes as they pass through. That
is exact, needs no knowledge of how FLUX differs from SD3.5, and keeps working when a
pipeline is restructured upstream.

The one thing it does not catch is arithmetic written as Python operators -- residual
adds, the AdaLN scale/shift/gate multiplies. Those are real kernels and a real share of
the energy, and they are recovered in ``catalogue.py`` from the token and channel counts
this pass observes, rather than left out.
"""

from __future__ import annotations

import contextlib
from collections import Counter
from dataclasses import dataclass, field

from kernelenergy.kernels.base import KernelConfig

__all__ = ["CaptureContext", "capture_pipeline", "CapturedOp"]


@dataclass
class CapturedOp:
    category: str
    dtype: str
    params: dict
    op_name: str
    count: int = 1

    def to_config(self, model: str) -> KernelConfig:
        return KernelConfig(
            category=self.category,
            dtype=self.dtype,
            params=dict(self.params),
            source_model=model,
            op_name=self.op_name,
        )


_DTYPE_NAME = {
    "torch.float32": "fp32",
    "torch.float16": "fp16",
    "torch.bfloat16": "bf16",
    "torch.float8_e4m3fn": "fp8_e4m3",
    "torch.float8_e5m2": "fp8_e5m2",
    "torch.int8": "int8",
}


def _dt(t) -> str:
    return _DTYPE_NAME.get(str(t.dtype), "fp32")


def _arg(args: tuple, kwargs: dict, index: int, *names, default=None):
    """Fetch an argument that may have been passed positionally or by keyword.

    Every wrapper here goes through this, because callers differ and the failure mode is
    total. diffusers 0.40 calls ``scaled_dot_product_attention(query=..., key=...,
    value=...)`` where earlier versions passed them positionally -- so a wrapper declared
    as ``def wrapper(q, k, v, ...)`` raises TypeError on every single call, which the
    capture loop catches per resolution and reports as "failed at 512x512", producing a
    catalogue of zero rows. SD3.5 still calls positionally, which is why it captured
    cleanly in the same run and made the failure look model-specific rather than
    version-specific.

    Keyword wins over position: a caller that passes something by name means it.
    """
    for n in names:
        if n in kwargs:
            return kwargs[n]
    if len(args) > index:
        return args[index]
    return default


class CaptureContext:
    """Context manager that records every linear, attention, conv and norm call.

    Usage::

        with CaptureContext() as cap:
            pipe(prompt, num_inference_steps=1)
        configs = cap.configs("flux1-dev")

    Counts are kept: a config seen 57 times in one forward pass is one row in the
    catalogue with ``count=57``, which is what weights the end-to-end reconstruction.
    """

    def __init__(self, capture_activations: bool = True):
        self.capture_activations = capture_activations
        self._ops: dict[tuple, CapturedOp] = {}
        self._patches: list = []

    # -- lifecycle ---------------------------------------------------------- #

    def __enter__(self) -> "CaptureContext":
        import torch.nn.functional as F

        self._patch(F, "linear", self._wrap_linear)
        self._patch(F, "scaled_dot_product_attention", self._wrap_sdpa)
        self._patch(F, "conv2d", self._wrap_conv2d)
        self._patch(F, "layer_norm", self._wrap_layer_norm)
        self._patch(F, "group_norm", self._wrap_group_norm)
        if self.capture_activations:
            self._patch(F, "gelu", self._wrap_act("gelu"))
            self._patch(F, "silu", self._wrap_act("silu"))
        return self

    def __exit__(self, *exc) -> None:
        for mod, name, orig in reversed(self._patches):
            setattr(mod, name, orig)
        self._patches.clear()

    def _patch(self, mod, name, factory) -> None:
        orig = getattr(mod, name)
        self._patches.append((mod, name, orig))
        setattr(mod, name, factory(orig))

    # -- recording ---------------------------------------------------------- #

    def _record(self, category: str, dtype: str, params: dict, op_name: str) -> None:
        key = (category, dtype, tuple(sorted(params.items())))
        op = self._ops.get(key)
        if op is None:
            self._ops[key] = CapturedOp(category, dtype, params, op_name, 1)
        else:
            op.count += 1

    # -- wrappers ----------------------------------------------------------- #

    def _wrap_linear(self, orig):
        def wrapper(*args, **kwargs):
            try:
                x = _arg(args, kwargs, 0, "input")
                w = _arg(args, kwargs, 1, "weight")
                b = _arg(args, kwargs, 2, "bias")
                m = 1
                for d in x.shape[:-1]:
                    m *= int(d)
                self._record(
                    "gemm", _dt(x),
                    {"m": m, "n": int(w.shape[0]), "k": int(w.shape[1]),
                     "bias": b is not None},
                    "aten::linear",
                )
            except Exception:
                pass
            return orig(*args, **kwargs)

        return wrapper

    def _wrap_sdpa(self, orig):
        def wrapper(*args, **kwargs):
            try:
                q = _arg(args, kwargs, 0, "query")
                k = _arg(args, kwargs, 1, "key")
                causal = bool(_arg(args, kwargs, 5, "is_causal", default=False))
                b, h, s_q, d = (int(x) for x in q.shape)
                h_kv, s_kv = int(k.shape[1]), int(k.shape[2])
                self._record(
                    "attention", _dt(q),
                    {"b": b, "h": h, "s_q": s_q, "s_kv": s_kv, "d": d,
                     "h_kv": h_kv, "causal": causal},
                    "aten::scaled_dot_product_attention",
                )
            except Exception:
                pass
            return orig(*args, **kwargs)

        return wrapper

    def _wrap_conv2d(self, orig):
        def wrapper(*args, **kwargs):
            try:
                x = _arg(args, kwargs, 0, "input")
                w = _arg(args, kwargs, 1, "weight")
                bias = _arg(args, kwargs, 2, "bias")
                stride = _arg(args, kwargs, 3, "stride", default=1)
                padding = _arg(args, kwargs, 4, "padding", default=0)
                groups = _arg(args, kwargs, 6, "groups", default=1)
                n, c_in, h, w_in = (int(d) for d in x.shape)
                c_out, _, kh, kw = (int(d) for d in w.shape)
                st = stride[0] if isinstance(stride, (tuple, list)) else stride
                pd_ = padding[0] if isinstance(padding, (tuple, list)) else padding
                if isinstance(pd_, str):
                    pd_ = kh // 2 if pd_ == "same" else 0
                self._record(
                    "conv", _dt(x),
                    {"n": n, "c_in": c_in, "c_out": c_out, "h": h, "w": w_in,
                     "kh": kh, "kw": kw, "stride": int(st), "pad": int(pd_),
                     "groups": int(groups), "bias": bias is not None},
                    "aten::conv2d",
                )
            except Exception:
                pass
            return orig(*args, **kwargs)

        return wrapper

    def _wrap_layer_norm(self, orig):
        def wrapper(*args, **kwargs):
            try:
                x = _arg(args, kwargs, 0, "input")
                shape = _arg(args, kwargs, 1, "normalized_shape")
                weight = _arg(args, kwargs, 2, "weight")
                dim = int(shape[-1]) if hasattr(shape, "__len__") else int(shape)
                rows = 1
                for d in x.shape[:-1]:
                    rows *= int(d)
                self._record(
                    "norm", _dt(x),
                    {"rows": rows, "dim": dim, "kind": "layer",
                     "affine": weight is not None},
                    "aten::layer_norm",
                )
            except Exception:
                pass
            return orig(*args, **kwargs)

        return wrapper

    def _wrap_group_norm(self, orig):
        def wrapper(*args, **kwargs):
            try:
                x = _arg(args, kwargs, 0, "input")
                num_groups = int(_arg(args, kwargs, 1, "num_groups"))
                weight = _arg(args, kwargs, 2, "weight")
                n = int(x.shape[0])
                c = int(x.shape[1])
                spatial = 1
                for d in x.shape[2:]:
                    spatial *= int(d)
                self._record(
                    "norm", _dt(x),
                    {"rows": n * num_groups, "dim": (c // num_groups) * spatial,
                     "kind": "group", "groups": num_groups,
                     "affine": weight is not None},
                    "aten::group_norm",
                )
            except Exception:
                pass
            return orig(*args, **kwargs)

        return wrapper

    def _wrap_act(self, kind: str):
        def factory(orig):
            def wrapper(*args, **kwargs):
                try:
                    x = _arg(args, kwargs, 0, "input")
                    n = 1
                    for d in x.shape:
                        n *= int(d)
                    self._record("elementwise", _dt(x),
                                 {"n_elem": n, "kind": kind}, f"aten::{kind}")
                except Exception:
                    pass
                return orig(*args, **kwargs)

            return wrapper

        return factory

    # -- results ------------------------------------------------------------ #

    @property
    def ops(self) -> list[CapturedOp]:
        return sorted(self._ops.values(), key=lambda o: (-o.count, o.category))

    def configs(self, model: str) -> list[KernelConfig]:
        return [o.to_config(model) for o in self.ops]

    def counts(self) -> Counter:
        c = Counter()
        for o in self.ops:
            c[o.category] += o.count
        return c


def capture_pipeline(pipe, *, model: str, prompt: str = "a photograph of a city street",
                     steps: int = 1, height: int = 1024, width: int = 1024,
                     capture_activations: bool = True, **pipe_kwargs):
    """Run one short generation under capture and return ``(configs, counts)``.

    One step is enough for the transformer, because every denoising step executes the
    same kernels with the same shapes -- only the VAE decode differs, and it runs once
    per image regardless. Use ``steps=2`` if you want to confirm that for a pipeline you
    have not profiled before.
    """
    import torch

    with CaptureContext(capture_activations) as cap:
        with torch.no_grad():
            pipe(prompt=prompt, num_inference_steps=steps, height=height,
                 width=width, **pipe_kwargs)
    return cap.configs(model), cap.counts()
