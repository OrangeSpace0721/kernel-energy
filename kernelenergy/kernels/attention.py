"""Scaled dot-product / FlashAttention kernels.

Diffusion attention differs from LLM attention in one way that matters for the
decomposition: it is **not causal**. FLUX, SD3.5 and Qwen-Image all attend
bidirectionally over the full latent sequence (and, in the MMDiT blocks, over the
concatenated text stream). PipeWeave singles out causal masking as the reason its
Attention error (15.5%) is roughly double its GEMM error (8.4%) -- tasks handling early
query tokens attend to fewer keys, so nominally uniform tiles carry very different
workloads and inter-block completion variance is high.

Without the mask that source of variance is gone, so attention here should be closer to
GEMM in predictability than PipeWeave's numbers suggest. That is a testable prediction
of this port and worth checking once data exists.

The other difference: sequence length is fixed by resolution, not by a sampled request
length, so the config space is small and dense rather than large and ragged.
"""

from __future__ import annotations

import math

from kernelenergy.kernels.base import KernelSpec, Task

__all__ = ["AttentionKernel"]

# FlashAttention-2/3 style block sizes. The query block is what defines a task: one
# CTA owns a block of queries and streams every key/value block past it.
_BLOCK_Q = 128
_BLOCK_KV = 128


class AttentionKernel(KernelSpec):
    """Fused attention over ``(b, h, s_q, d)`` queries and ``(b, h_kv, s_kv, d)`` KV."""

    category = "attention"
    required = ("b", "h", "s_q", "s_kv", "d")

    def __init__(self, config):
        super().__init__(config)
        self.b = int(self.p["b"])
        self.h = int(self.p["h"])
        self.s_q = int(self.p["s_q"])
        self.s_kv = int(self.p["s_kv"])
        self.d = int(self.p["d"])
        self.h_kv = int(self.p.get("h_kv", self.h))  # GQA / MQA
        self.causal = bool(self.p.get("causal", False))
        self.block_q = int(self.p.get("block_q", _BLOCK_Q))
        self.block_kv = int(self.p.get("block_kv", _BLOCK_KV))

    # -- replay ------------------------------------------------------------- #

    def build(self, device: str = "cuda"):  # pragma: no cover - needs a GPU
        import torch

        from kernelenergy.kernels.gemm import _torch_dtype

        td = _torch_dtype(self.dtype)
        q = torch.randn(self.b, self.h, self.s_q, self.d, device=device).to(td)
        k = torch.randn(self.b, self.h_kv, self.s_kv, self.d, device=device).to(td)
        v = torch.randn(self.b, self.h_kv, self.s_kv, self.d, device=device).to(td)
        return {"q": q, "k": k, "v": v}

    def run(self, state) -> None:  # pragma: no cover - needs a GPU
        import torch
        import torch.nn.functional as F

        k, v = state["k"], state["v"]
        if self.h_kv != self.h:  # expand GQA groups the way the pipelines do
            rep = self.h // self.h_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        F.scaled_dot_product_attention(state["q"], k, v, is_causal=self.causal)

    def working_set_bytes(self) -> float:
        q = self.b * self.h * self.s_q * self.d * self.dsize
        kv = 2 * self.b * self.h_kv * self.s_kv * self.d * self.dsize
        return q + kv + q  # + output

    # -- analysis ----------------------------------------------------------- #

    def n_tasks(self) -> int:
        """One task per (batch, head, query block)."""
        return self.b * self.h * math.ceil(self.s_q / self.block_q)

    def decompose(self, gpu) -> list[Task]:
        bq = self.block_q
        # Under causal masking a query block at position i only sees keys up to i, so
        # the mean KV extent across blocks is about half the sequence. Non-causal
        # diffusion attention sees all of it -- this is the branch that makes these
        # kernels more uniform than their LLM equivalents.
        n_q_blocks = math.ceil(self.s_q / bq)
        mean_kv = self.s_kv
        if self.causal:
            mean_kv = sum(
                min(self.s_kv, (i + 1) * bq) for i in range(n_q_blocks)
            ) / n_q_blocks

        # Two chained matmuls per task: QK^T then PV. PipeWeave's alpha = 4.
        n_mma = 4.0 * bq * mean_kv * self.d

        # Softmax: one exp per score, plus running max/sum rescaling.
        n_xu = bq * mean_kv
        n_fma = 3.0 * bq * mean_kv + 2.0 * bq * self.d

        # FlashAttention never materialises the s_q x s_kv score matrix in HBM. Global
        # traffic is Q, K, V and O only. With GQA the same KV block serves several query
        # heads, so it is read once from HBM per kv-head and re-read from L2 per head --
        # exactly the reuse that makes GQA cheap, and it belongs in the L2 column.
        q_bytes = bq * self.d * self.dsize
        o_bytes = bq * self.d * self.dsize
        kv_bytes = 2 * mean_kv * self.d * self.dsize

        tasks_total = self.n_tasks()
        kv_unique = 2 * self.b * self.h_kv * self.s_kv * self.d * self.dsize
        per_task_kv_global = kv_unique / max(tasks_total, 1)

        return [
            Task(
                n_mma=n_mma,
                n_fma=n_fma,
                n_xu=n_xu,
                bytes_global=q_bytes + o_bytes + per_task_kv_global,
                bytes_l2=q_bytes + o_bytes + kv_bytes,
                bytes_shared=kv_bytes + q_bytes,
                count=tasks_total,
                label=f"attn_q{bq}_kv{self.block_kv}",
            )
        ]

    def flops(self) -> float:
        eff_kv = self.s_kv / 2.0 if self.causal else float(self.s_kv)
        return 4.0 * self.b * self.h * self.s_q * eff_kv * self.d

    def bytes_global(self) -> float:
        q = self.b * self.h * self.s_q * self.d * self.dsize
        kv = 2 * self.b * self.h_kv * self.s_kv * self.d * self.dsize
        return 2 * q + kv  # Q in, O out, KV in
