"""Profiler pass: which kernels actually cost the runtime.

The capture pass says *what* runs. This says *how much it matters*, which is what decides
where the hand-written decomposers go. PipeWeave writes one decomposer per kernel
category and each costs real analyst effort, so the honest way to bound that effort is to
rank categories by measured share and stop when the tail stops mattering.

Kernel names are matched to categories by pattern. The patterns are the ones cuBLAS,
cuDNN, FlashAttention and PyTorch's own kernels use; anything unmatched lands in
``other``, and a large ``other`` share is the signal that a category is missing rather
than that the tail is small.
"""

from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd

__all__ = ["profile_pipeline", "categorise_kernel", "coverage_report", "KERNEL_PATTERNS"]

#: Ordered: the first match wins, so more specific patterns come first.
KERNEL_PATTERNS: list[tuple[str, str]] = [
    ("attention", r"flash|fmha|attention|mha_|sdpa|cudnn_.*attn"),
    ("conv", r"conv|winograd|cudnn::|implicit_gemm"),
    # Normalisation kernels are named after the *statistic* they compute, not after the
    # layer: PyTorch's LayerNorm/GroupNorm land as RowwiseMoments and Welford kernels,
    # and RMSNorm shows up as a MeanOps/NormTwoOps reduction. Matching only on "norm"
    # sends 3-4% of runtime to `other` and makes it look uncovered.
    ("norm", r"layer_?norm|rms_?norm|group_?norm|batch_norm|rowwisemoments|welford|"
             r"reduce_kernel.*(meanops|normtwoops)"),
    ("gemm", r"gemm|cutlass|sm\d+_.*_tensorop|nvjet|xmma|cublas|ampere_\w*|"
             r"addmm|baddbmm|matmul|gemv"),
    ("elementwise", r"elementwise|vectorized|unrolled|gelu|silu|swish|"
                    r"catarraybatchedcopy|copy_|fill_|mul_|add_|upsample|interpolate"),
    ("softmax", r"softmax"),
    ("reduce", r"reduce|sum_|mean_"),
    ("memory", r"memcpy|memset|transpose|permute|contiguous"),
]

#: CPU-side entries the profiler also reports. Their device time is the *sum* of the
#: kernels they launch, so counting both an `aten::addmm` and the cutlass kernel it
#: launched double-counts that work -- and since the aten name matches no kernel pattern
#: it lands in `other`, inflating the uncovered share by exactly the amount it
#: double-counted. This is why the first FLUX run reported 34% `other` with
#: `aten::addmm` alone at 22%: that GEMM time was already counted under `gemm`.
_CPU_SIDE_PREFIXES = ("aten::", "autograd::", "torch::", "cuda", "Optimizer",
                      "ProfilerStep", "nn.Module", "enumerate(")


def is_device_kernel(ev) -> bool:
    """True for an actual GPU kernel launch, false for a CPU-side operator record."""
    try:
        from torch.autograd import DeviceType

        dt = getattr(ev, "device_type", None)
        if dt is not None:
            return dt == DeviceType.CUDA
    except Exception:
        pass
    return not str(getattr(ev, "key", "")).startswith(_CPU_SIDE_PREFIXES)


def categorise_kernel(name: str) -> str:
    # Case-insensitive via the flag rather than by lowering the name, because several
    # patterns are CamelCase kernel symbols (CatArrayBatchedCopy, RowwiseMoments) and
    # lowering the haystack while leaving the needle mixed-case meant they never matched.
    for cat, pat in KERNEL_PATTERNS:
        if re.search(pat, name, flags=re.IGNORECASE):
            return cat
    return "other"


def profile_pipeline(pipe, *, prompt: str = "a photograph of a city street",
                     steps: int = 2, height: int = 1024, width: int = 1024,
                     warmup: bool = True, **pipe_kwargs) -> pd.DataFrame:
    """Profile one generation and return per-kernel CUDA time.

    A warmup generation runs first and is discarded: the first call compiles autotuning
    caches, allocates, and picks cuBLAS algorithms, none of which is representative.
    """
    import torch
    from torch.profiler import ProfilerActivity, profile

    def _gen():
        with torch.no_grad():
            pipe(prompt=prompt, num_inference_steps=steps, height=height,
                 width=width, **pipe_kwargs)

    if warmup:
        _gen()
        torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        _gen()
        torch.cuda.synchronize()

    rows = []
    for ev in prof.key_averages():
        if not is_device_kernel(ev):
            continue
        cuda_us = float(
            getattr(ev, "self_device_time_total", 0)
            or getattr(ev, "self_cuda_time_total", 0)
            or 0
        )
        if cuda_us <= 0:
            continue
        rows.append(
            {
                "kernel": ev.key,
                "category": categorise_kernel(ev.key),
                "calls": int(ev.count),
                "cuda_time_us": cuda_us,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            "the profiler recorded no CUDA kernel time -- is the pipeline on the GPU, "
            "and is CPU offload disabled?"
        )
    df = df.sort_values("cuda_time_us", ascending=False).reset_index(drop=True)
    df["share"] = df["cuda_time_us"] / df["cuda_time_us"].sum()
    df["cumshare"] = df["share"].cumsum()
    return df


def coverage_report(prof_df: pd.DataFrame,
                    implemented: tuple[str, ...] = ("gemm", "attention", "conv",
                                                    "norm", "elementwise")) -> pd.DataFrame:
    """Per-category runtime share, and how much of it the catalogue covers.

    The number to look at is the ``covered`` row's share. Below about 0.90 the model is
    being asked to reconstruct an end-to-end figure from less than nine tenths of the
    work, and the missing tenth will not distribute itself evenly.
    """
    by_cat = (
        prof_df.groupby("category")
        .agg(cuda_time_us=("cuda_time_us", "sum"), calls=("calls", "sum"),
             n_kernels=("kernel", "nunique"))
        .sort_values("cuda_time_us", ascending=False)
    )
    total = by_cat["cuda_time_us"].sum()
    by_cat["share"] = by_cat["cuda_time_us"] / total
    by_cat["implemented"] = [c in implemented for c in by_cat.index]
    covered = by_cat.loc[by_cat["implemented"], "share"].sum()
    by_cat.loc["__covered__"] = {
        "cuda_time_us": by_cat.loc[by_cat["implemented"], "cuda_time_us"].sum(),
        "calls": by_cat.loc[by_cat["implemented"], "calls"].sum(),
        "n_kernels": by_cat.loc[by_cat["implemented"], "n_kernels"].sum(),
        "share": covered,
        "implemented": True,
    }
    return by_cat


def top_uncovered(prof_df: pd.DataFrame, implemented=("gemm", "attention", "conv",
                                                      "norm", "elementwise"),
                  n: int = 15) -> pd.DataFrame:
    """The most expensive kernels no decomposer handles. Read this before adding one."""
    m = ~prof_df["category"].isin(implemented)
    return prof_df[m].head(n)[["kernel", "category", "calls", "cuda_time_us", "share"]]
