"""GEMM / linear-projection kernels.

Every DiT and MMDiT block in FLUX.1, SD3.5 and Qwen-Image is dominated by these: QKV
projection, attention output projection, the two MLP matmuls, and the AdaLN modulation
head. Together with attention they account for the large majority of transformer runtime,
which is why they are first in the catalogue.
"""

from __future__ import annotations

import math

from kernelenergy.kernels.base import KernelSpec, Task, TileChoice, choose_gemm_tile

__all__ = ["GemmKernel"]


class GemmKernel(KernelSpec):
    """``C[m,n] = A[m,k] @ B[k,n]``, optionally with a bias add.

    ``m`` is tokens x batch, ``k`` is input features, ``n`` is output features.
    """

    category = "gemm"
    required = ("m", "n", "k")

    def __init__(self, config):
        super().__init__(config)
        self.m = int(self.p["m"])
        self.n = int(self.p["n"])
        self.k = int(self.p["k"])
        self.bias = bool(self.p.get("bias", True))
        # Accumulation is FP32 for every bf16/fp16 GEMM in these pipelines.
        self.accum_dsize = 4 if self.dtype in ("bf16", "fp16", "fp8_e4m3", "fp8_e5m2") else self.dsize

    # -- replay ------------------------------------------------------------- #

    def build(self, device: str = "cuda"):  # pragma: no cover - needs a GPU
        import torch

        td = _torch_dtype(self.dtype)
        a = torch.randn(self.m, self.k, device=device, dtype=torch.float32).to(td)
        b = torch.randn(self.k, self.n, device=device, dtype=torch.float32).to(td)
        bias = (
            torch.randn(self.n, device=device, dtype=torch.float32).to(td)
            if self.bias else None
        )
        out = torch.empty(self.m, self.n, device=device, dtype=td)
        return {"a": a, "b": b, "bias": bias, "out": out}

    def run(self, state) -> None:  # pragma: no cover - needs a GPU
        import torch

        if state["bias"] is None:
            torch.mm(state["a"], state["b"], out=state["out"])
        else:
            torch.addmm(state["bias"], state["a"], state["b"], out=state["out"])

    def working_set_bytes(self) -> float:
        return (self.m * self.k + self.k * self.n) * self.dsize + self.m * self.n * self.dsize

    # -- analysis ----------------------------------------------------------- #

    def tile(self, gpu) -> TileChoice:
        return choose_gemm_tile(self.m, self.n, self.k, gpu.sms, self.dtype)

    def decompose(self, gpu) -> list[Task]:
        tile = self.tile(gpu)
        tasks = self._gemm_tasks(
            self.m, self.n, self.k, self.dsize, tile,
            alpha=2.0, accum_dsize=self.accum_dsize,
        )
        if self.bias:
            # One FMA per output element. The bias vector is small and broadcast, so it
            # is read once per tile from L2 and effectively never from HBM.
            t = tasks[0]
            t.n_fma += tile.m * tile.n
            t.bytes_l2 += tile.n * self.dsize
        return tasks

    def flops(self) -> float:
        f = 2.0 * self.m * self.n * self.k
        if self.bias:
            f += self.m * self.n
        return f

    def bytes_global(self) -> float:
        return (
            (self.m * self.k + self.k * self.n) * self.dsize
            + self.m * self.n * self.accum_dsize
            + (self.n * self.dsize if self.bias else 0)
        )

    def n_ctas(self, gpu) -> int:
        t = self.tile(gpu)
        return math.ceil(self.m / t.m) * math.ceil(self.n / t.n)


def _torch_dtype(name: str):  # pragma: no cover - needs torch
    import torch

    return {
        "fp32": torch.float32,
        "tf32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp8_e4m3": getattr(torch, "float8_e4m3fn", torch.bfloat16),
        "fp8_e5m2": getattr(torch, "float8_e5m2", torch.bfloat16),
        "int8": torch.int8,
    }[name]
