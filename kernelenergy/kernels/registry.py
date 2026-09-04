"""Category -> KernelSpec lookup."""

from __future__ import annotations

from kernelenergy.kernels.attention import AttentionKernel
from kernelenergy.kernels.base import KernelConfig, KernelSpec, Task
from kernelenergy.kernels.conv import ConvKernel
from kernelenergy.kernels.gemm import GemmKernel
from kernelenergy.kernels.pointwise import ElementwiseKernel, NormKernel

__all__ = ["REGISTRY", "make_kernel", "CATEGORIES"]

REGISTRY: dict[str, type[KernelSpec]] = {
    GemmKernel.category: GemmKernel,
    AttentionKernel.category: AttentionKernel,
    ConvKernel.category: ConvKernel,
    NormKernel.category: NormKernel,
    ElementwiseKernel.category: ElementwiseKernel,
}

CATEGORIES = tuple(REGISTRY)


def make_kernel(config: KernelConfig) -> KernelSpec:
    try:
        cls = REGISTRY[config.category]
    except KeyError:
        raise ValueError(
            f"no KernelSpec for category {config.category!r}; known: {CATEGORIES}"
        ) from None
    return cls(config)
