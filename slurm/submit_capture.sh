#!/usr/bin/env bash
# Submit the capture stage, with the partition and gres filled in from config.sh.
#
#   bash slurm/submit_capture.sh              # all three pipelines
#   bash slurm/submit_capture.sh flux1-dev    # or named ones
#
# Capture needs a card the models FIT ON, which is the constraint people miss. FLUX.1-dev
# is ~24 GB of transformer plus ~9 GB of T5 in bf16; SD3.5-Large and Qwen-Image are the
# same order. That rules out a 22 GB L4 and points at the 80 GB A100 partition, which is
# why KE_CAPTURE_PARTITION defaults there rather than to whatever is quickest to get.
#
# It does NOT need exclusivity -- capture records tensor shapes and relative kernel
# times, not energy, so a neighbour on another GPU is irrelevant here. That is the one
# stage where sharing is free.

set -euo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$_here/config.sh"

ALL=(flux1-dev sd35-large qwen-image)
targets=("$@")
[[ ${#targets[@]} -eq 0 ]] && targets=("${ALL[@]}")

# Reuse the A100 entry's gres string so the name matches what the partition advertises;
# a bare `gpu:1` is usually accepted but not everywhere.
partition="${KE_CAPTURE_PARTITION:-a100}"
gres="${KE_CAPTURE_GRES:-}"
if [[ -z "$gres" ]]; then
  for k in A100_PCIE A100_SXM4 H100 H200_SXM; do
    if [[ -n "${KE_GPU_SPEC[$k]:-}" ]] && [[ "$(ke_gpu_field "$k" 1)" == "$partition" ]]; then
      gres="$(ke_gpu_field "$k" 2)"
      break
    fi
  done
fi
gres="${gres:-gpu:1}"

mkdir -p "$KE_LOGS"

# Build the array index list for just the requested models, so task ids still line up
# with the MODELS array inside 01_capture.sbatch.
idx=""
for t in "${targets[@]}"; do
  for i in "${!ALL[@]}"; do
    [[ "${ALL[$i]}" == "$t" ]] && idx+="${idx:+,}$i"
  done
done
[[ -z "$idx" ]] && { echo "no known pipelines in: ${targets[*]}" >&2; exit 1; }

args=(
  --job-name="ke-capture"
  --partition="$partition"
  --gres="$gres"
  --array="$idx"
  --time="$KE_CAPTURE_TIME"
  --cpus-per-task="$KE_CPUS"
  --mem="${KE_CAPTURE_MEM:-160G}"
  --output="$KE_LOGS/ke-capture-%A_%a.out"
  --error="$KE_LOGS/ke-capture-%A_%a.out"
  --export="ALL,KE_SLURM_DIR=$_here"
)
# shellcheck disable=SC2046
args+=($(ke_account_flag))

echo "submitting capture: partition=$partition gres=$gres array=$idx (${targets[*]})"
sbatch "${args[@]}" "$_here/01_capture.sbatch"

echo
echo "when all tasks finish:"
echo "  kernelenergy catalogue --in $KE_DATA/catalogue --out $KE_DATA/catalogue.csv"
