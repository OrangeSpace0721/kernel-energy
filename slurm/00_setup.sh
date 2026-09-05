#!/usr/bin/env bash
# Run once, ON A LOGIN NODE (it needs internet). Creates the conda environment and
# downloads all three pipelines into the shared HF cache.
#
#   bash slurm/00_setup.sh
#
# Expect this to take a while and a lot of disk: FLUX.1-dev, SD3.5-Large and Qwen-Image
# together are well over 100 GB. Check your scratch quota before starting, not after.
#
# FLUX.1-dev and SD3.5-Large are gated on HuggingFace. Accept their licences in a browser
# and run `huggingface-cli login` first, or this will fail with a 401 that looks like a
# network error.

set -euo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$_here/config.sh"

echo "root        : $KE_ROOT"
echo "conda env   : $KE_ENV_NAME"
echo "HF cache    : $HF_HOME"
mkdir -p "$KE_ROOT" "$KE_DATA" "$KE_LOGS" "$HF_HOME"

# --- environment -------------------------------------------------------------
if [[ -n "${KE_PRE_ACTIVATE:-}" ]]; then eval "$KE_PRE_ACTIVATE"; fi
# shellcheck source=/dev/null
source "$KE_CONDA_SH"

if conda env list | awk '{print $1}' | grep -qx "$KE_ENV_NAME"; then
  echo "environment $KE_ENV_NAME exists; updating"
  conda env update -n "$KE_ENV_NAME" -f "$_here/environment.yml" --prune
else
  conda env create -n "$KE_ENV_NAME" -f "$_here/environment.yml"
fi
conda activate "$KE_ENV_NAME"

# The package itself, from the repo this script lives in.
pip install -e "$_here/.." --no-deps
pip install hf_transfer  # much faster on the big weight files

python -c "import torch, diffusers, transformers, pynvml; \
print('torch', torch.__version__, '| diffusers', diffusers.__version__, \
      '| transformers', transformers.__version__)"

# --- weights -----------------------------------------------------------------
export HF_HUB_ENABLE_HF_TRANSFER=1
python - <<'PY'
import os, sys
from huggingface_hub import snapshot_download
from kernelenergy.trace.pipelines import PIPELINES, env_var_for, resolve_source

failed = []
for key, spec in PIPELINES.items():
    # Anything already on disk -- a KE_MODEL_* override, or a hub cache HF_HOME already
    # points at -- is left alone. Re-downloading 40 GB you have is the single most
    # expensive mistake available at this step.
    var = env_var_for(key)
    if os.environ.get(var, "").strip():
        try:
            src, _ = resolve_source(key)
            print(f"\n=== {key}: using {var}={src} (skipping download) ===", flush=True)
            continue
        except Exception as e:
            print(f"\n=== {key}: {var} is set but unusable: {e} ===", flush=True)
            failed.append((key, var, str(e)))
            continue

    print(f"\n=== {key}: {spec.repo} ===", flush=True)
    try:
        # Skip the duplicate .bin copies where safetensors exist; roughly halves the
        # download for repos that ship both.
        path = snapshot_download(
            spec.repo,
            ignore_patterns=["*.bin", "*.pth", "*.onnx", "*.msgpack", "*.h5"],
        )
        print(f"    -> {path}")
    except Exception as e:
        print(f"    FAILED: {type(e).__name__}: {e}")
        failed.append((key, spec.repo, str(e)))

if failed:
    print("\nnot staged:")
    for key, repo, err in failed:
        print(f"  {key} ({repo}): {err.splitlines()[0]}")
    print("\nGated repos need `huggingface-cli login` and the licence accepted in a "
          "browser. Compute nodes cannot do this for you.")
    sys.exit(1)
print("\nall pipelines staged")
PY

echo
echo "du -sh $HF_HOME:"; du -sh "$HF_HOME" 2>/dev/null || true
echo
echo "setup complete. Next:"
echo "  sbatch slurm/01_capture.sbatch          # build the kernel catalogue"
echo "  bash   slurm/submit_measure.sh          # sweep every card"
