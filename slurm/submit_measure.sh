#!/usr/bin/env bash
# Submit the measurement sweep for one card, or for the whole fleet.
#
#   bash slurm/submit_measure.sh H100          # one card
#   bash slurm/submit_measure.sh               # every card in KE_GPU_SPEC
#   KE_MEASURE_ARRAY=0-7 bash slurm/submit_measure.sh H100    # 8-way split
#
# Each card gets its own array job, sharded across tasks. The shards are balanced by
# estimated cost and derived deterministically from the array task id, so a requeued task
# reconstructs exactly its own shard -- change the array width between submissions and
# nothing is measured twice, because resume checks every shard file for the card.

set -euo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$_here/config.sh"

if [[ ! -f "$KE_DATA/catalogue.csv" ]]; then
  echo "no catalogue at $KE_DATA/catalogue.csv" >&2
  echo "run 01_capture.sbatch first, then:" >&2
  echo "  kernelenergy catalogue --in $KE_DATA/catalogue --out $KE_DATA/catalogue.csv" >&2
  exit 1
fi

targets=("$@")
if [[ ${#targets[@]} -eq 0 ]]; then
  targets=("${!KE_GPU_SPEC[@]}")
fi

mkdir -p "$KE_LOGS"

for key in "${targets[@]}"; do
  partition="$(ke_gpu_field "$key" 1)" || continue
  gres="$(ke_gpu_field "$key" 2)"
  exclusive="$(ke_gpu_field "$key" 3)"

  args=(
    --job-name="ke-$key"
    --partition="$partition"
    --gres="$gres"
    --array="$KE_MEASURE_ARRAY"
    --time="$KE_MEASURE_TIME"
    --cpus-per-task="$KE_CPUS"
    --mem="$KE_MEM"
    --output="$KE_LOGS/ke-$key-%A_%a.out"
    --error="$KE_LOGS/ke-$key-%A_%a.out"
    --export="ALL,KE_GPU_KEY=$key"
  )
  # Exclusive buys thermal isolation, not counter correctness -- gres already gives sole
  # use of the GPU. Worth it on small nodes, not worth the queue on 8-GPU ones.
  [[ "$exclusive" == "yes" ]] && args+=(--exclusive)
  # shellcheck disable=SC2046
  args+=($(ke_account_flag))

  echo "submitting $key: partition=$partition gres=$gres exclusive=$exclusive array=$KE_MEASURE_ARRAY"
  sbatch "${args[@]}" "$_here/02_measure.sbatch"
done

echo
echo "watch with:  squeue -u \$USER -o '%.18i %.12j %.8T %.10M %R'"
echo "logs in:     $KE_LOGS"
