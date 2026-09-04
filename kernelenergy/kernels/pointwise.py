"""Normalisation and elementwise kernels.

These do almost no arithmetic and move a great deal of data, which makes them the most
interesting kernels in the catalogue from an energy point of view and the least
interesting from a latency one. A latency model can afford to treat them as a rounding
error on a well-fused pipeline. An energy model cannot: moving a byte from HBM costs
roughly two orders of magnitude more energy than the flop that consumes it, so a kernel
that is 2% of runtime can be well over 2% of joules.

They are also where the AdaLN modulation lives -- the scale/shift/gate applied around
every attention and MLP sub-block in a DiT. There are six such tensors per block per
step, and they are pure bandwidth.
"""

from __future__ import annotations

from kernelenergy.kernels.base import KernelSpec, Task

__all__ = ["NormKernel", "ElementwiseKernel"]


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


class NormKernel(KernelSpec):
    """LayerNorm, RMSNorm or GroupNorm over ``rows`` vectors of length ``dim``."""

    category = "norm"
    required = ("rows", "dim")

    def __init__(self, config):
        super().__init__(config)
        self.rows = int(self.p["rows"])
        self.dim = int(self.p["dim"])
        self.kind = str(self.p.get("kind", "layer"))  # layer | rms | group
        self.groups = int(self.p.get("groups", 32))
        self.affine = bool(self.p.get("affine", True))
        # Norms accumulate in FP32 even when their I/O is bf16.
        self.compute_dsize = 4

    # -- replay ------------------------------------------------------------- #

    def build(self, device: str = "cuda"):  # pragma: no cover - needs a GPU
        import torch

        from kernelenergy.kernels.gemm import _torch_dtype

        td = _torch_dtype(self.dtype)
        x = torch.randn(self.rows, self.dim, device=device).to(td)
        w = torch.randn(self.dim, device=device).to(td) if self.affine else None
        b = torch.randn(self.dim, device=device).to(td) if self.affine else None
        return {"x": x, "w": w, "b": b}

    def run(self, state) -> None:  # pragma: no cover - needs a GPU
        import torch
        import torch.nn.functional as F

        x = state["x"]
        if self.kind == "rms":
            var = x.float().pow(2).mean(-1, keepdim=True)
            y = x * torch.rsqrt(var + 1e-6).to(x.dtype)
            if state["w"] is not None:
                y = y * state["w"]
        elif self.kind == "group":
            xg = x.view(1, self.rows, self.dim)
            F.group_norm(xg, self.groups)
        else:
            F.layer_norm(x, (self.dim,), state["w"], state["b"], 1e-6)

    def working_set_bytes(self) -> float:
        return 2 * self.rows * self.dim * self.dsize

    # -- analysis ----------------------------------------------------------- #

    def decompose(self, gpu) -> list[Task]:
        # One task per row. A row is read, reduced, and written -- a single pass in the
        # fused implementations these pipelines use, two passes in a naive one.
        n_elem = self.dim
        # LayerNorm needs mean and variance (2 accumulations per element plus the
        # normalise-and-affine pass); RMSNorm skips the mean.
        acc = 2.0 if self.kind != "rms" else 1.0
        n_fma = acc * n_elem + 2.0 * n_elem  # reduce + (x - mu) * rstd * w + b
        n_xu = 1.0  # one rsqrt per row

        bytes_rw = 2 * n_elem * self.dsize
        affine_bytes = (2 * n_elem * self.dsize) if self.affine else 0.0

        return [
            Task(
                n_mma=0.0,
                n_fma=n_fma,
                n_xu=n_xu,
                bytes_global=bytes_rw,
                # The affine weights are the same vector for every row: read once from
                # HBM, then served from L2 for the remaining rows.
                bytes_l2=bytes_rw + affine_bytes,
                bytes_shared=n_elem * self.compute_dsize,
                count=self.rows,
                label=f"{self.kind}_norm_row",
            )
        ]

    def flops(self) -> float:
        acc = 2.0 if self.kind != "rms" else 1.0
        return self.rows * self.dim * (acc + 2.0)

    def bytes_global(self) -> float:
        b = 2 * self.rows * self.dim * self.dsize
        if self.affine:
            b += 2 * self.dim * self.dsize
        return b


# --------------------------------------------------------------------------- #
# Elementwise
# --------------------------------------------------------------------------- #

#: FMA and XU op counts per element, per kind. GELU's tanh approximation and SiLU's
#: sigmoid both go through the XU pipe; a plain add does not touch it at all. Getting
#: this split right is the difference between a memory-bound kernel that idles the
#: math pipes and one that does not.
_ELEMENTWISE_COST = {
    "add": (1.0, 0.0, 2),
    "mul": (1.0, 0.0, 2),
    "gelu": (6.0, 1.0, 1),
    "gelu_tanh": (6.0, 1.0, 1),
    "silu": (2.0, 1.0, 1),
    "swiglu": (3.0, 1.0, 2),
    "geglu": (7.0, 1.0, 2),
    "scale_shift": (2.0, 0.0, 3),  # AdaLN modulation: x * (1 + scale) + shift
    "gate_residual": (2.0, 0.0, 3),  # residual + gate * branch
    "copy": (0.0, 0.0, 1),
    "softmax": (3.0, 1.0, 1),
}


class ElementwiseKernel(KernelSpec):
    """Activations, residual adds, and AdaLN modulation.

    ``n_elem`` is the number of output elements; ``kind`` selects the arithmetic and the
    number of input tensors read.
    """

    category = "elementwise"
    required = ("n_elem", "kind")

    def __init__(self, config):
        super().__init__(config)
        self.n_elem = int(self.p["n_elem"])
        self.kind = str(self.p["kind"])
        if self.kind not in _ELEMENTWISE_COST:
            raise ValueError(
                f"unknown elementwise kind {self.kind!r}; "
                f"known: {sorted(_ELEMENTWISE_COST)}"
            )
        self.fma_per_elem, self.xu_per_elem, self.n_tensors = _ELEMENTWISE_COST[self.kind]
        # Threads per CTA and elements per thread, matching a standard vectorised
        # elementwise launch (float4 loads).
        self.threads = int(self.p.get("threads", 256))
        self.elems_per_thread = int(self.p.get("elems_per_thread", 4))

    # -- replay ------------------------------------------------------------- #

    def build(self, device: str = "cuda"):  # pragma: no cover - needs a GPU
        import torch

        from kernelenergy.kernels.gemm import _torch_dtype

        td = _torch_dtype(self.dtype)
        n_in = max(self.n_tensors - 1, 1)
        xs = [torch.randn(self.n_elem, device=device).to(td) for _ in range(n_in)]
        return {"xs": xs, "out": torch.empty(self.n_elem, device=device, dtype=td)}

    def run(self, state) -> None:  # pragma: no cover - needs a GPU
        import torch
        import torch.nn.functional as F

        xs, out = state["xs"], state["out"]
        k = self.kind
        if k in ("gelu", "gelu_tanh"):
            torch.nn.functional.gelu(xs[0], approximate="tanh", out=None)
        elif k == "silu":
            torch.nn.functional.silu(xs[0])
        elif k == "softmax":
            F.softmax(xs[0], dim=-1)
        elif k == "add":
            torch.add(xs[0], xs[min(1, len(xs) - 1)], out=out)
        elif k == "mul":
            torch.mul(xs[0], xs[min(1, len(xs) - 1)], out=out)
        elif k in ("scale_shift", "gate_residual"):
            a = xs[0]
            b = xs[min(1, len(xs) - 1)]
            torch.addcmul(a, a, b, out=out)
        elif k in ("swiglu", "geglu"):
            a = xs[0]
            b = xs[min(1, len(xs) - 1)]
            torch.mul(F.silu(a) if k == "swiglu" else F.gelu(a), b, out=out)
        else:
            out.copy_(xs[0])

    def working_set_bytes(self) -> float:
        return self.n_tensors * self.n_elem * self.dsize

    # -- analysis ----------------------------------------------------------- #

    def decompose(self, gpu) -> list[Task]:
        per_cta = self.threads * self.elems_per_thread
        n_ctas = max(1, -(-self.n_elem // per_cta))
        elems = min(per_cta, self.n_elem)
        bytes_per_cta = self.n_tensors * elems * self.dsize
        return [
            Task(
                n_mma=0.0,
                n_fma=self.fma_per_elem * elems,
                n_xu=self.xu_per_elem * elems,
                bytes_global=bytes_per_cta,
                bytes_l2=bytes_per_cta,
                bytes_shared=0.0,
                count=n_ctas,
                label=f"elementwise_{self.kind}",
            )
        ]

    def flops(self) -> float:
        return self.n_elem * (self.fma_per_elem + self.xu_per_elem)

    def bytes_global(self) -> float:
        return self.n_tensors * self.n_elem * self.dsize
