"""Emit PipeWeave's feature vector for a kernel in this project's catalogue.

This is the join between the two halves of the transfer. On one side, a
:class:`~kernelenergy.kernels.base.KernelConfig` captured from a diffusion pipeline. On
the other, the exact 11- or 15-vector one of PipeWeave's published checkpoints expects,
in their order, in their units.

Getting this wrong does not raise. It produces a vector the trunk reads as a different
kernel on a different card, and the only symptom is that transfer "doesn't work". So the
whole file is written to be checkable: the calculators are their code unmodified (see
``vendor/NOTICE.md``), the hardware constants are their JSON verbatim where they have
one, and ``tests/test_pipeweave_parity.py`` re-derives their own datasets.

Operator mapping
----------------

============  ==================  ============================================
category      PipeWeave operator  fit
============  ==================  ============================================
``gemm``      ``gemm``            exact -- same operator, same units
``conv``      ``gemm``            implicit GEMM. Their GEMM set contains no
                                  convolutions, so this is extrapolation in
                                  *shape space*, not in operator semantics.
``attention`` ``attn``            structurally right, distributionally not --
                                  see below
``norm``      ``rmsnorm``         LayerNorm and GroupNorm through an RMSNorm
                                  op model. Same traffic, one extra reduction
                                  pass for the mean.
``elementwise`` ``siluandmul``    same task shape (one CTA per row, one
                                  streaming pass), different arithmetic
============  ==================  ============================================

Three honest gaps, in descending order of how much they should worry you:

1. **Attention is non-causal here and was always causal there.**
   ``aggregator.py`` hardcodes ``causal=True`` for every row of their attention
   corpus. Diffusion attention attends over the whole sequence. The *calculator*
   handles the flag correctly -- it changes the CTA workload and the iteration
   distribution -- but the *checkpoint* has never seen the resulting region. Their
   attention model is also FlashInfer-shaped: paged/ragged KV, GQA, binary-search
   chunking. PyTorch SDPA is none of those. Report attention separately and expect
   it to be the worst category.

2. **Elementwise needs a row width their model does not get from ours.**
   Their ``siluandmul`` op model is per-row: ``total_ctas = seq_len`` and
   ``bytes_per_cta = dim * dtype_size``. Our elementwise configs carry ``n_elem``
   only -- the synthesised AdaLN and gated-residual kernels lost the ``(rows, dim)``
   split they were built from. ``dim`` defaults to ``threads * elems_per_thread``,
   the actual per-CTA element count of the launch that was measured, and ``seq_len``
   follows. That is geometrically faithful to what ran, but it puts ``dim`` at 1024,
   below their training range (768 to ~4608). Fix at the source: record ``rows`` and
   ``dim`` on synthesised elementwise configs at the next capture. Doing it now would
   change every kernel signature and orphan 414 already-measured configs.

3. **Conv has no checkpoint of its own.** Sent through the GEMM model as implicit
   GEMM, which is what cuDNN does anyway, or excluded entirely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from kernelenergy.hardware import GPU, get_gpu
from kernelenergy.kernels.base import KernelConfig, dtype_bytes
from kernelenergy.pipeweave.hardware import (
    PW_HARDWARE,
    gemm_calculator_for,
    to_hardware_spec,
)
from kernelenergy.pipeweave.tiles import GemmLaunch, choose_launch
from kernelenergy.pipeweave.vendor import (
    FaProblemConfig,
    GemmProblemConfig,
    RmsNormProblemConfig,
    SiluMulProblemConfig,
    calculate_fa2_params,
    calculate_fa3_params,
    rmsnorm_calculator,
    silu_mul_calculator,
)

__all__ = [
    "PIPEWEAVE_FEATURES",
    "CATEGORY_TO_OPERATOR",
    "Emission",
    "emit",
    "emit_frame",
]

_MEMORY = (
    "global_in_flight",
    "global_cycle",
    "local_cycle",
    "sm_max_in_flight",
    "sm_max_global_cycle",
    "sm_max_shared_cycle",
    "sm_max_local_cycle",
)
_TENSOR = ("tensor_all_ops", "tensor_all_cycle", "tensor_sm_max_ops", "tensor_sm_max_cycle")
_XU = ("xu_all_ops", "xu_all_cycle", "xu_sm_max_ops", "xu_sm_max_cycle")
_FMA = ("fma_all_ops", "fma_all_cycle", "fma_sm_max_ops", "fma_sm_max_cycle")

#: Exactly the ``model_info.features`` list in each checkpoint's ``metadata.json``,
#: in order. The order is load-bearing: these go straight into a Linear layer.
PIPEWEAVE_FEATURES: dict[str, tuple[str, ...]] = {
    "gemm": _TENSOR + _MEMORY,           # 11
    "attn": _TENSOR + _XU + _MEMORY,     # 15
    "rmsnorm": _FMA + _XU + _MEMORY,     # 15
    "siluandmul": _FMA + _XU + _MEMORY,  # 15
}

CATEGORY_TO_OPERATOR = {
    "gemm": "gemm",
    "conv": "gemm",
    "attention": "attn",
    "norm": "rmsnorm",
    "elementwise": "siluandmul",
}


@dataclass
class Emission:
    """One kernel's feature vector, plus everything needed to audit it."""

    operator: str
    features: np.ndarray          # in PIPEWEAVE_FEATURES[operator] order
    feature_names: tuple[str, ...]
    launch: GemmLaunch | None     # GEMM/conv only
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.feature_names, (float(x) for x in self.features)))


def _pipe_values(pipe, names):
    return [pipe.all_ops, pipe.all_cycle, pipe.sm_max_ops, pipe.sm_max_cycle]


def _memory_values(mp):
    return [
        mp.global_in_flight, mp.global_cycle, mp.local_cycle,
        mp.sm_max_in_flight, mp.sm_max_global_cycle,
        mp.sm_max_shared_cycle, mp.sm_max_local_cycle,
    ]


def _gemm(m, n, k, dsize, gpu_key, spec, tile_mode):
    launch = choose_launch(m, n, k, gpu_key, mode=tile_mode)
    problem = GemmProblemConfig(
        m=int(m), n=int(n), k=int(k),
        tile_m=launch.tile_m, tile_n=launch.tile_n, tile_k=launch.tile_k,
        cta_count=launch.cta_count, is_split_k=launch.is_split_k,
        data_size_bytes=int(dsize),
    )
    f = gemm_calculator_for(gpu_key)(problem, spec)
    return np.array(_pipe_values(f.tensor_pipe, _TENSOR) + _memory_values(f.memory_pipe),
                    dtype=float), launch


def emit(
    config: KernelConfig,
    gpu: GPU | str,
    tile_mode: str = "structural",
    elementwise_dim: int | None = None,
) -> Emission:
    """One :class:`KernelConfig` -> the vector its PipeWeave checkpoint expects."""
    gpu_key = (gpu if isinstance(gpu, str) else gpu.gpu_key).upper()
    if gpu_key not in PW_HARDWARE:
        gpu_key = get_gpu(gpu_key).gpu_key
    spec = to_hardware_spec(gpu_key)
    cat = config.category
    p = config.params
    dsize = dtype_bytes(config.dtype)
    notes: list[str] = []

    if cat == "gemm":
        v, launch = _gemm(int(p["m"]), int(p["n"]), int(p["k"]), dsize, gpu_key, spec, tile_mode)
        return Emission("gemm", v, PIPEWEAVE_FEATURES["gemm"], launch)

    if cat == "conv":
        # Implicit GEMM, the same lowering cuDNN performs.
        stride = int(p.get("stride", 1))
        kh, kw = int(p["kh"]), int(p["kw"])
        pad = int(p.get("pad", kh // 2))
        groups = int(p.get("groups", 1))
        h_out = (int(p["h"]) + 2 * pad - kh) // stride + 1
        w_out = (int(p["w"]) + 2 * pad - kw) // stride + 1
        m = int(p["n"]) * h_out * w_out
        n = int(p["c_out"])
        k = int(p["c_in"]) // groups * kh * kw
        v, launch = _gemm(m, n, k, dsize, gpu_key, spec, tile_mode)
        notes.append(
            "conv lowered to implicit GEMM; PipeWeave's GEMM checkpoint contains no "
            "convolutions, so this is out-of-distribution in shape space"
        )
        return Emission("gemm", v, PIPEWEAVE_FEATURES["gemm"], launch, tuple(notes))

    if cat == "attention":
        b = int(p["b"])
        h, d = int(p["h"]), int(p["d"])
        h_kv = int(p.get("h_kv", h))
        s_q, s_kv = int(p["s_q"]), int(p["s_kv"])
        causal = bool(p.get("causal", False))
        problem = FaProblemConfig(
            batch_size=b,
            q_lengths=[s_q] * b,
            kv_lengths=[s_kv] * b,
            num_qo_heads=h, num_kv_heads=h_kv, head_dim=d,
            # 'ragged' is the contiguous layout; 'paged' models a KV cache read in
            # 16-token pages, which a diffusion forward pass has no equivalent of.
            layout="ragged",
            data_size_q=dsize, data_size_kv=dsize, data_size_o=dsize,
            causal=causal,
        )
        calc = (calculate_fa3_params
                if PW_HARDWARE[gpu_key].architecture == "hopper"
                else calculate_fa2_params)
        f = calc(problem, spec)
        if not causal:
            notes.append(
                "non-causal: aggregator.py hardcodes causal=True for every attention "
                "row upstream trained on, so the checkpoint is extrapolating here"
            )
        if h_kv == h:
            notes.append("MHA (h_kv == h); their corpus is overwhelmingly GQA")
        v = np.array(
            _pipe_values(f.tensor_pipe, _TENSOR)
            + _pipe_values(f.xu_pipe, _XU)
            + _memory_values(f.memory_pipe),
            dtype=float,
        )
        return Emission("attn", v, PIPEWEAVE_FEATURES["attn"], None, tuple(notes))

    if cat == "norm":
        kind = str(p.get("kind", "layer"))
        problem = RmsNormProblemConfig(
            batch_size=int(p["rows"]), dim=int(p["dim"]), dtype_size=dsize
        )
        f = rmsnorm_calculator(problem, spec)
        if kind != "rms":
            notes.append(
                f"{kind}norm through an RMSNorm op model: same traffic and task shape, "
                f"one extra reduction pass for the mean that the op counts omit"
            )
        v = np.array(
            _pipe_values(f.fma_pipe, _FMA)
            + _pipe_values(f.xu_pipe, _XU)
            + _memory_values(f.memory_pipe),
            dtype=float,
        )
        return Emission("rmsnorm", v, PIPEWEAVE_FEATURES["rmsnorm"], None, tuple(notes))

    if cat == "elementwise":
        n_elem = int(p["n_elem"])
        dim = elementwise_dim or int(p.get("dim", 0)) or (
            int(p.get("threads", 256)) * int(p.get("elems_per_thread", 4))
        )
        dim = max(8, dim - dim % 8)  # their model assumes a vec_size of 8
        seq = max(1, math.ceil(n_elem / dim))
        problem = SiluMulProblemConfig(seq_len=seq, dim=dim, dtype_size=dsize)
        f = silu_mul_calculator(problem, spec)
        notes.append(
            f"n_elem={n_elem} factored as seq_len={seq} x dim={dim} from the launch "
            f"geometry; their siluandmul corpus has dim in [768, 4608]"
        )
        if str(p.get("kind")) not in ("silu", "gelu", "gelu_tanh", "swiglu"):
            notes.append(
                f"kind={p.get('kind')!r} is not a SiLU-family activation; the memory "
                f"model still holds, the op counts are a stand-in"
            )
        v = np.array(
            _pipe_values(f.fma_pipe, _FMA)
            + _pipe_values(f.xu_pipe, _XU)
            + _memory_values(f.memory_pipe),
            dtype=float,
        )
        return Emission("siluandmul", v, PIPEWEAVE_FEATURES["siluandmul"], None, tuple(notes))

    raise ValueError(
        f"no PipeWeave operator for category {cat!r}; "
        f"known: {sorted(CATEGORY_TO_OPERATOR)}"
    )


def emit_frame(df, tile_mode: str = "structural", on_error: str = "drop"):
    """Add PipeWeave features to a dataset frame, one column block per operator.

    Returns ``(frame, per_operator_columns)``. Rows keep their original index. Because
    each operator has its own feature *names* and its own checkpoint, the columns are
    prefixed ``pw_`` and rows of other operators are left NaN -- the frame is a
    container, not a single design matrix. ``transfer.py`` splits it by operator.
    """
    import pandas as pd

    from kernelenergy.kernels.base import KernelConfig as _KC

    blocks: dict[str, list[str]] = {}
    rows: list[dict] = []
    failures: list[tuple] = []
    for i, r in df.iterrows():
        try:
            cfg = _KC.from_row(r.to_dict())
            em = emit(cfg, str(r["gpu_key"]), tile_mode=tile_mode)
            rec = {f"pw_{k}": v for k, v in em.as_dict().items()}
            rec["pw_operator"] = em.operator
            rec["pw_notes"] = "; ".join(em.notes)
            if em.launch is not None:
                rec["pw_tile_m"] = em.launch.tile_m
                rec["pw_tile_n"] = em.launch.tile_n
                rec["pw_tile_k"] = em.launch.tile_k
                rec["pw_cta_count"] = em.launch.cta_count
                rec["pw_is_split_k"] = int(em.launch.is_split_k)
                rec["pw_launch_source"] = em.launch.source
            rec["__i"] = i
            rows.append(rec)
            blocks.setdefault(em.operator, [f"pw_{n}" for n in em.feature_names])
        except Exception as e:
            failures.append((i, f"{type(e).__name__}: {e}"))
            if on_error == "raise":
                raise
    if failures:
        # Group by message rather than printing the first five rows. When one thing is
        # broken -- a missing reference file, an unknown GPU -- every row fails the same
        # way, and five identical tracebacks hide how many distinct problems there are.
        from collections import Counter

        counts = Counter(msg for _, msg in failures)
        print(f"emit_frame: {len(failures)} of {len(df)} rows failed, "
              f"{len(counts)} distinct cause(s):")
        for msg, n in counts.most_common(5):
            print(f"  [{n} rows] {msg}")
        if len(counts) > 5:
            print(f"  ... and {len(counts) - 5} more")
    fdf = pd.DataFrame(rows).set_index("__i")
    fdf.index.name = None
    return df.join(fdf, how="left"), blocks
