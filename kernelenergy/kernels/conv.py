"""Convolution kernels, which in these pipelines means the VAE.

The VAE decoder is easy to forget and expensive to ignore. It runs once per image rather
than once per denoising step, so at 50 steps it is a small share of runtime -- but at
1-4 steps, which is where distilled FLUX variants and most of the cheap end of a sweep
live, it can be a large one. It also has a completely different energy character from the
transformer: high-resolution, low-channel convolutions at the end of the decoder are
bandwidth-bound in a way nothing in the DiT is.

Decomposition goes through implicit GEMM, which is what cuDNN actually does for these
shapes, so the task structure is the GEMM one with the dimensions remapped.
"""

from __future__ import annotations

import math

from kernelenergy.kernels.base import KernelSpec, Task, choose_gemm_tile

__all__ = ["ConvKernel"]


class ConvKernel(KernelSpec):
    """2D convolution, decomposed as implicit GEMM.

    ``M = n * h_out * w_out``, ``N = c_out``, ``K = c_in/groups * kh * kw``.
    """

    category = "conv"
    required = ("n", "c_in", "c_out", "h", "w", "kh", "kw")

    def __init__(self, config):
        super().__init__(config)
        self.n = int(self.p["n"])
        self.c_in = int(self.p["c_in"])
        self.c_out = int(self.p["c_out"])
        self.h = int(self.p["h"])
        self.w = int(self.p["w"])
        self.kh = int(self.p["kh"])
        self.kw = int(self.p["kw"])
        self.stride = int(self.p.get("stride", 1))
        self.pad = int(self.p.get("pad", self.kh // 2))
        self.groups = int(self.p.get("groups", 1))
        self.bias = bool(self.p.get("bias", True))
        self.accum_dsize = 4 if self.dtype in ("bf16", "fp16") else self.dsize

    @property
    def h_out(self) -> int:
        return (self.h + 2 * self.pad - self.kh) // self.stride + 1

    @property
    def w_out(self) -> int:
        return (self.w + 2 * self.pad - self.kw) // self.stride + 1

    @property
    def gemm_m(self) -> int:
        return self.n * self.h_out * self.w_out

    @property
    def gemm_n(self) -> int:
        return self.c_out // self.groups

    @property
    def gemm_k(self) -> int:
        return (self.c_in // self.groups) * self.kh * self.kw

    # -- replay ------------------------------------------------------------- #

    def build(self, device: str = "cuda"):  # pragma: no cover - needs a GPU
        import torch

        from kernelenergy.kernels.gemm import _torch_dtype

        td = _torch_dtype(self.dtype)
        x = torch.randn(self.n, self.c_in, self.h, self.w, device=device).to(td)
        wt = torch.randn(
            self.c_out, self.c_in // self.groups, self.kh, self.kw, device=device
        ).to(td)
        b = torch.randn(self.c_out, device=device).to(td) if self.bias else None
        return {"x": x, "w": wt, "b": b}

    def run(self, state) -> None:  # pragma: no cover - needs a GPU
        import torch.nn.functional as F

        F.conv2d(
            state["x"], state["w"], state["b"],
            stride=self.stride, padding=self.pad, groups=self.groups,
        )

    def working_set_bytes(self) -> float:
        x = self.n * self.c_in * self.h * self.w * self.dsize
        y = self.n * self.c_out * self.h_out * self.w_out * self.dsize
        wt = self.c_out * (self.c_in // self.groups) * self.kh * self.kw * self.dsize
        return x + y + wt

    # -- analysis ----------------------------------------------------------- #

    def decompose(self, gpu) -> list[Task]:
        m, n, k = self.gemm_m, self.gemm_n, self.gemm_k
        tile = choose_gemm_tile(m, n, k, gpu.sms, self.dtype)
        tasks = self._gemm_tasks(
            m, n, k, self.dsize, tile, alpha=2.0, accum_dsize=self.accum_dsize
        )
        t = tasks[0]
        t.label = f"conv_implicit_gemm_{tile.m}x{tile.n}x{tile.k}"

        # im2col gathers overlap: each input element is touched kh*kw/stride^2 times.
        # Those repeats hit L1/L2, not HBM, so they inflate the L2 column only.
        overlap = (self.kh * self.kw) / (self.stride ** 2)
        t.bytes_l2 *= max(overlap, 1.0)
        # HBM traffic is the honest tensor sizes, unaffected by the gather.
        n_tiles = t.count
        t.bytes_global = self.bytes_global() / max(n_tiles, 1)
        if self.bias:
            t.n_fma += tile.m * tile.n
        # Grouped convolution runs `groups` independent GEMMs.
        if self.groups > 1:
            t.count *= self.groups
        return tasks

    def flops(self) -> float:
        f = 2.0 * self.gemm_m * self.gemm_n * self.gemm_k * self.groups
        if self.bias:
            f += self.n * self.c_out * self.h_out * self.w_out
        return f

    def bytes_global(self) -> float:
        x = self.n * self.c_in * self.h * self.w * self.dsize
        y = self.n * self.c_out * self.h_out * self.w_out * self.accum_dsize
        wt = self.c_out * (self.c_in // self.groups) * self.kh * self.kw * self.dsize
        return x + y + wt

    def n_ctas(self, gpu) -> int:
        t = choose_gemm_tile(self.gemm_m, self.gemm_n, self.gemm_k, gpu.sms, self.dtype)
        return math.ceil(self.gemm_m / t.m) * math.ceil(self.gemm_n / t.n) * self.groups
