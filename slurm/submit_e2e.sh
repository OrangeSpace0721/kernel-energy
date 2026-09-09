#!/usr/bin/env bash
# Measure real generations end to end, for one card or the whole fleet.
#
#   bash slurm/submit_e2e.sh H100          # one card, all three models
#   bash slurm/submit_e2e.sh               # every card in KE_GPU_SPEC
#   bash slurm/submit_e2e.sh -n            # dry run
#   KE_E2E_STEPS=4,20 bash slurm/submit_e2e.sh L4    # shorter sweep
#
# One array job per card, one task per model. This is the stage that decides whether any
# per-kernel number in this project means anything about a pipeline, so it wants the same
# card list and the same GPU-key resolution as the measurement sweep -- a mismatch here
# would compare one card's kernels against another card's generation.
#
# Unlike `measure`, this is EXCLUSIVE everywhere. The replay sweep only needs sole use of
# the board, which --gres already gives; a full generation is long enough that a noisy
# neighbour's thermals move the clock, and the clock moves the energy.

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

if [[ ! -f "$KE_DATA/dataset.csv" ]]; then
  echo "no dataset at $KE_DATA/dataset.csv" >&2
  echo "reconcile needs per-kernel measurements to sum. Run the measure sweep and" >&2
  echo "  kernelenergy dataset --raw $KE_DATA/raw --out $KE_DATA/dataset.csv" >&2
  echo "first. (The e2e measurement itself does not need it -- but there would be" >&2
  echo "nothing to compare it against.)" >&2
  exit 1
fi

[[ ${#targets[@]} -eq 0 ]] && targets=("${!KE_GPU_SPEC[@]}")

# Same guard as submit_measure.sh: two partitions on one physical card would write two
# sets of rows under one gpu_key, and nothing downstream would notice.
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

mkdir -p "$KE_LOGS" "$KE_DATA/e2e"

for key in "${targets[@]}"; do
  partition="$(ke_gpu_field "$key" 1)" || continue
  gres="$(ke_gpu_field "$key" 2)"

  args=(
    --job-name="ke-e2e-$key"
    --partition="$partition"
    --gres="$gres"
    --array="${KE_E2E_ARRAY:-0-2}"
    --time="${KE_E2E_TIME:-04:00:00}"
    --cpus-per-task="$KE_CPUS"
    --mem="$KE_MEM"
    --exclusive
    --output="$KE_LOGS/ke-e2e-$key-%A_%a.out"
    --error="$KE_LOGS/ke-e2e-$key-%A_%a.out"
    --export="ALL,KE_GPU_KEY=$key,KE_SLURM_DIR=$_here"
  )
  # shellcheck disable=SC2046
  args+=($(ke_account_flag))

  if [[ $DRY_RUN -eq 1 ]]; then
    printf '%-12s partition=%-14s gres=%-16s array=%s steps=%s\n' \
      "$key" "$partition" "$gres" "${KE_E2E_ARRAY:-0-2}" "${KE_E2E_STEPS:-4,12,20,28}"
    continue
  fi

  echo "submitting $key: partition=$partition gres=$gres array=${KE_E2E_ARRAY:-0-2}"
  sbatch "${args[@]}" "$_here/04_validate_e2e.sbatch"
done

if [[ $DRY_RUN -eq 1 ]]; then
  echo
  echo "(dry run -- nothing submitted; drop -n to go)"
  exit 0
fi

cat <<'EOF'

watch with:  bash slurm/status.sh

when every card has finished, reconcile on the login node -- it needs no GPU:

  kernelenergy reconcile \
    --e2e data/e2e --dataset data/dataset.csv \
    --predictions data/predictions/predictions__hardware.csv

The predictions file is optional and gives level B. It comes from
`kernelenergy evaluate --fold hardware --predictions data/predictions`, and because
those folds are leave-one-GPU-out, the predictions for a card were made by a model
that never saw it.
EOF
