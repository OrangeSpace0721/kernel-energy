#!/usr/bin/env bash
#SBATCH --job-name=ke-e2e
#SBATCH --output=%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#
# Do the replayed per-kernel energies reconstruct a real generation? Run this on every
# card, for every model, before presenting any kernel-level number as a claim about
# pipelines.
#
#   KE_GPU_KEY=H100 sbatch --partition=gpu --gres=gpu:h100:1 slurm/04_validate_e2e.sbatch
#
# Exclusive for the same reason the measure stage is: this measures energy.

# SLURM copies the batch script into /var/spool before running it, so BASH_SOURCE
# points there, not at the repo -- `source "$_here/job_common.sh"` then fails and the
# job silently runs with whatever environment the submitting shell happened to export.
# That is why the first capture run worked: the shell had sourced job_common.sh by hand.
# From a clean shell it would have run with no conda env and no CUDA_DEVICE_ORDER.
_ke_locate() {
  local c
  for c in "${KE_SLURM_DIR:-}" \
           "${SLURM_SUBMIT_DIR:-}/slurm" \
           "$(cd "$(dirname "${BASH_SOURCE[1]:-$0}")" 2>/dev/null && pwd)"; do
    [[ -n "$c" && -f "$c/job_common.sh" ]] && { echo "$c"; return 0; }
  done
  echo "cannot locate job_common.sh -- set KE_SLURM_DIR to the repo's slurm/ dir" >&2
  return 1
}
_here="$(_ke_locate)" || exit 1
# shellcheck source=/dev/null
source "$_here/job_common.sh"

MODELS=(flux1-dev sd35-large qwen-image)
MODEL="${MODELS[${SLURM_ARRAY_TASK_ID:-0}]}"

# Three or more step counts. The sweep is the point: one ratio has several causes at
# once, while an intercept and a slope separate the once-per-image cost (text encoder,
# VAE decode) from the per-step transformer cost, and each can be checked against its
# predicted counterpart. Two points fit a line through no residual and tell you nothing
# about whether the line is real.
STEPS="${KE_E2E_STEPS:-4,12,20,28}"

ke_banner
echo "measuring $MODEL end to end on ${KE_GPU_KEY:-?}, steps $STEPS"

kernelenergy preflight --stage all --models "$MODEL" --device 0 \
  --catalogue "$KE_DATA/catalogue.csv" || exit 1

kernelenergy e2e \
  --model "$MODEL" \
  --steps "$STEPS" \
  --height "${KE_E2E_HEIGHT:-1024}" \
  --width  "${KE_E2E_WIDTH:-1024}" \
  --repeats "${KE_E2E_REPEATS:-3}" \
  --device 0 \
  --out "$KE_DATA/e2e"

echo
echo "reconcile once every card has finished -- it needs no GPU:"
echo "  kernelenergy reconcile --e2e $KE_DATA/e2e --dataset $KE_DATA/dataset.csv"
echo "done: $(date -Is)"
