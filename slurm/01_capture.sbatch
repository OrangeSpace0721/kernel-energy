#!/usr/bin/env bash
#SBATCH --job-name=ke-capture
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#
# Capture the kernel configs each pipeline runs, and profile which of them cost the
# runtime. One array task per model.
#
#   sbatch --partition=gpu --gres=gpu:1 slurm/01_capture.sbatch
#
# This stage does NOT need an exclusive node. It records shapes and relative kernel
# times, not energy, so a co-tenant is a nuisance rather than a correctness problem --
# the opposite of the measure stage. Ask for one GPU and let the scheduler pack you.
#
# It does need a lot of host memory: loading a 12B-parameter transformer plus its text
# encoders is where jobs die if --mem is set for a normal workload.

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

ke_banner
echo "capturing: $MODEL"

# Fail early and cheaply if the weights are not staged, rather than 20 minutes in.
kernelenergy preflight --stage capture --models "$MODEL" --device 0 || exit 1

kernelenergy capture \
  --model "$MODEL" \
  --out "$KE_DATA/catalogue" \
  --steps 1 \
  --profile

echo "done: $(date -Is)"
echo
echo "when all three tasks finish, build the catalogue on a login node:"
echo "  kernelenergy catalogue --in $KE_DATA/catalogue --out $KE_DATA/catalogue.csv"
