"""Loading FLUX.1-dev, SD3.5-Large and Qwen-Image, and the sweep of shapes to capture.

Kept deliberately thin. The point of the capture layer is that it does not need to know
anything about these models; this module only knows how to *load* them and what
resolutions to walk, so adding a fourth pipeline is a few lines rather than a new
decomposer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["PipelineSpec", "PIPELINES", "load_pipeline", "DEFAULT_SHAPE_SWEEP"]


@dataclass
class PipelineSpec:
    key: str
    repo: str
    loader: str  # diffusers class name
    dtype: str = "bf16"
    #: kwargs the pipeline needs that others do not
    extra_call_kwargs: dict = field(default_factory=dict)
    notes: str = ""


PIPELINES: dict[str, PipelineSpec] = {
    "flux1-dev": PipelineSpec(
        key="flux1-dev",
        repo="black-forest-labs/FLUX.1-dev",
        loader="FluxPipeline",
        dtype="bf16",
        extra_call_kwargs={"guidance_scale": 3.5, "max_sequence_length": 512},
        notes="MMDiT: 19 double-stream + 38 single-stream blocks, 3072 hidden, 24 heads.",
    ),
    "sd35-large": PipelineSpec(
        key="sd35-large",
        repo="stabilityai/stable-diffusion-3.5-large",
        loader="StableDiffusion3Pipeline",
        dtype="bf16",
        extra_call_kwargs={"guidance_scale": 4.5, "max_sequence_length": 512},
        notes="MMDiT: 38 joint blocks, 2432 hidden, 38 heads. Lowest utilisation of "
              "the three at run level (0.21-0.34), so worth extra shapes.",
    ),
    "qwen-image": PipelineSpec(
        key="qwen-image",
        repo="Qwen/Qwen-Image",
        loader="QwenImagePipeline",
        dtype="bf16",
        extra_call_kwargs={"true_cfg_scale": 4.0},
        notes="MMDiT with a Qwen2.5-VL text encoder; requires diffusers >= 0.35.",
    ),
}


#: (height, width) pairs to walk. Sequence length in the transformer scales as
#: (h/16)*(w/16) for these VAE + patchify configurations, so this sweep spans roughly a
#: 16x range in tokens, which is what makes the shape features identifiable.
DEFAULT_SHAPE_SWEEP: list[tuple[int, int]] = [
    (512, 512),
    (768, 768),
    (1024, 1024),
    (1024, 1536),
    (1536, 1536),
    (2048, 2048),
]


def load_pipeline(key: str, device: str = "cuda", dtype: str | None = None,
                  enable_cpu_offload: bool = False, **kwargs):
    """Load a pipeline by key.

    ``enable_cpu_offload`` keeps the largest models on a 24 GB card, at the cost of
    making end-to-end timings meaningless -- fine for capture, which only needs shapes,
    and not fine for the end-to-end validation script.
    """
    import diffusers
    import torch

    spec = PIPELINES[key]
    td = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        dtype or spec.dtype
    ]
    cls = getattr(diffusers, spec.loader, None)
    if cls is None:
        raise ImportError(
            f"{spec.loader} is not in this diffusers build ({diffusers.__version__}). "
            f"{spec.key} needs a newer diffusers."
        )
    pipe = cls.from_pretrained(spec.repo, torch_dtype=td, **kwargs)
    if enable_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe, spec
