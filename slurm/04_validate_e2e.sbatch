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

_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=/dev/null
source "$_here/job_common.sh"

MODELS=(flux1-dev sd35-large qwen-image)
MODEL="${MODELS[${SLURM_ARRAY_TASK_ID:-0}]}"

ke_banner
echo "validating $MODEL on ${KE_GPU_KEY:-?}"

kernelenergy preflight --stage all --models "$MODEL" --device 0 \
  --catalogue "$KE_DATA/catalogue.csv" || exit 1

for steps in 4 20; do
  echo
  echo "--- $steps steps ---"
  python "$_here/../scripts/validate_e2e.py" \
    --dataset   "$KE_DATA/dataset.csv" \
    --catalogue "$KE_DATA/catalogue.csv" \
    --model "$MODEL" \
    --steps "$steps" \
    --device 0 \
    --out "$KE_DATA/e2e_validation.csv"
done

echo "done: $(date -Is)"
