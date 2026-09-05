"""Loading FLUX.1-dev, SD3.5-Large and Qwen-Image, and the sweep of shapes to capture.

Kept deliberately thin. The point of the capture layer is that it does not need to know
anything about these models; this module only knows how to *load* them and what
resolutions to walk, so adding a fourth pipeline is a few lines rather than a new
decomposer.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "PipelineSpec",
    "PIPELINES",
    "load_pipeline",
    "resolve_source",
    "env_var_for",
    "DEFAULT_SHAPE_SWEEP",
]


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


def env_var_for(key: str) -> str:
    """``flux1-dev`` -> ``KE_MODEL_FLUX1_DEV``."""
    return "KE_MODEL_" + re.sub(r"[^A-Za-z0-9]+", "_", key).upper()


def resolve_source(key: str) -> tuple[str, str]:
    """Where to load this pipeline from, and how that was decided.

    Weights are often already on disk somewhere -- staged for other work, shared by a
    group, or downloaded before this repo existed -- and re-fetching 100+ GB to satisfy a
    hardcoded repo id is pure waste. Two layouts turn up in practice and they are not
    interchangeable:

    * **Hub cache** (``<root>/hub/models--org--name/snapshots/<sha>/``), which is what
      ``snapshot_download`` and any ordinary ``from_pretrained`` produce. Point ``HF_HOME``
      at the root and the repo id resolves offline with no code change at all.
    * **A plain directory** (``<somewhere>/FLUX.1-dev/`` with model_index.json in it),
      which is what ``git clone`` or ``snapshot_download(local_dir=...)`` produce. The hub
      client cannot find this from a repo id; the path has to be passed instead.

    ``KE_MODEL_<KEY>`` handles the second case. It wins over the repo id when set, and
    must name a directory containing ``model_index.json`` -- checked here rather than
    left to fail deep inside ``from_pretrained`` with a message about a URL.
    """
    spec = PIPELINES[key]
    override = os.environ.get(env_var_for(key), "").strip()
    if not override:
        return spec.repo, "hub"

    p = Path(override).expanduser()
    if not p.exists():
        raise FileNotFoundError(
            f"{env_var_for(key)} points at {p}, which does not exist"
        )
    if not (p / "model_index.json").exists():
        hint = ""
        if (p / "hub").is_dir() or any(p.glob("models--*")):
            hint = (
                "  It looks like a hub cache rather than a pipeline directory -- set "
                f"HF_HOME to it instead of {env_var_for(key)}, and unset that."
            )
        elif (p / "config.json").exists() or (p / "transformer").is_dir():
            # The likely shape when weights come from a distillation or fine-tuning
            # project: only the denoiser was kept, because that is all a student needs.
            # This harness runs the whole pipeline -- VAE decode is a real share of the
            # energy at low step counts -- so a component directory is not enough.
            present = sorted(c.name for c in p.iterdir() if c.is_dir())[:8]
            hint = (
                "  It looks like a single component rather than a full pipeline "
                f"(subdirs: {present or 'none'}). A pipeline snapshot needs "
                "model_index.json plus vae/, text_encoder*/ and tokenizer*/ beside the "
                f"transformer. Leave {env_var_for(key)} unset and let 00_setup.sh fetch "
                "the complete pipeline for this model."
            )
        raise FileNotFoundError(
            f"{env_var_for(key)} points at {p}, which has no model_index.json, so it is "
            f"not a diffusers pipeline directory.{hint}"
        )

    # A snapshot can carry model_index.json and still be missing the folders it names --
    # a partial download, or an rsync that dropped a large subtree. Catch it here rather
    # than minutes in, when from_pretrained finally reaches for the VAE.
    try:
        index = json.loads((p / "model_index.json").read_text())
        wanted = {
            k for k, v in index.items()
            if not k.startswith("_") and isinstance(v, (list, tuple)) and len(v) == 2
            and v[0] is not None
        }
        absent = sorted(c for c in wanted if not (p / c).exists())
        if absent:
            raise FileNotFoundError(
                f"{env_var_for(key)} points at {p}, whose model_index.json lists "
                f"components that are not on disk: {absent}. The snapshot is incomplete "
                f"-- re-sync it, or unset {env_var_for(key)} and let 00_setup.sh fetch "
                "this pipeline."
            )
    except json.JSONDecodeError:
        raise FileNotFoundError(
            f"{env_var_for(key)}: {p / 'model_index.json'} is not valid JSON"
        ) from None

    return str(p), "local"


def load_pipeline(key: str, device: str = "cuda", dtype: str | None = None,
                  enable_cpu_offload: bool = False, **kwargs):
    """Load a pipeline by key, from the hub cache or a local directory.

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
    source, how = resolve_source(key)
    print(f"loading {key} from {'local directory' if how == 'local' else 'hub cache'}: "
          f"{source}")
    pipe = cls.from_pretrained(source, torch_dtype=td, **kwargs)
    if enable_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe, spec
