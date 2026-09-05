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

_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
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
