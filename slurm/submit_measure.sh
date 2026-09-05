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

DRY_RUN=0
targets=()
for a in "$@"; do
  case "$a" in
    -n|--dry-run) DRY_RUN=1 ;;
    -*) echo "unknown flag: $a" >&2; exit 2 ;;
    *) targets+=("$a") ;;
  esac
done

if [[ ! -f "$KE_DATA/catalogue.csv" ]]; then
  echo "no catalogue at $KE_DATA/catalogue.csv" >&2
  echo "the catalogue is captured ONCE, on any single GPU -- kernel shapes do not" >&2
  echo "depend on the hardware. Run:" >&2
  echo "  bash slurm/submit_capture.sh" >&2
  echo "  kernelenergy catalogue --in $KE_DATA/catalogue --out $KE_DATA/catalogue.csv" >&2
  exit 1
fi

if [[ ${#targets[@]} -eq 0 ]]; then
  targets=("${!KE_GPU_SPEC[@]}")
fi

# Two partitions pointing at the same physical card is the failure mode worth catching
# here, because it is silent downstream: both sweeps would write the same gpu_key, and
# merge_results deduplicates on (kernel_sig, gpu_key) -- so one card's rows would
# overwrite the other's and the fold would quietly train on five cards instead of six.
declare -A _seen_partition=()
for key in "${targets[@]}"; do
  p="$(ke_gpu_field "$key" 1)" || exit 1
  if [[ -n "${_seen_partition[$p]:-}" ]]; then
    echo "config error: $key and ${_seen_partition[$p]} both use partition '$p'." >&2
    echo "One of them is pointing at the wrong hardware. Run identify_gpus.sh." >&2
    exit 1
  fi
  _seen_partition[$p]="$key"
done

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
    --export="ALL,KE_GPU_KEY=$key,KE_SLURM_DIR=$_here"
  )
  # Exclusive buys thermal isolation, not counter correctness -- gres already gives sole
  # use of the GPU. Worth it on small nodes, not worth the queue on 8-GPU ones.
  [[ "$exclusive" == "yes" ]] && args+=(--exclusive)
  # shellcheck disable=SC2046
  args+=($(ke_account_flag))

  if [[ $DRY_RUN -eq 1 ]]; then
    printf '%-12s partition=%-14s gres=%-16s exclusive=%-4s array=%s\n' \
      "$key" "$partition" "$gres" "$exclusive" "$KE_MEASURE_ARRAY"
    continue
  fi

  echo "submitting $key: partition=$partition gres=$gres exclusive=$exclusive array=$KE_MEASURE_ARRAY"
  sbatch "${args[@]}" "$_here/02_measure.sbatch"
done

if [[ $DRY_RUN -eq 1 ]]; then
  echo
  echo "(dry run -- nothing submitted; drop -n to go)"
  exit 0
fi

echo
echo "watch with:  bash slurm/status.sh"
echo "logs in:     $KE_LOGS"
