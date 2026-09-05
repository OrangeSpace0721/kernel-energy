#!/usr/bin/env bash
# Ask every GPU partition what card it actually has.
#
#   bash slurm/identify_gpus.sh
#
# Two minutes of queue time that saves a whole sweep. The hardware table is keyed on the
# NVML product name, and near-miss names are everywhere: L4 against L40, L40 against
# L40S, H200 against H200 NVL. Attributing rows to the wrong descriptor gives them the
# wrong TDP, bandwidth and peak, and the sweep runs to completion looking fine.
#
# It also prints the enforced power limit next to the card default. Where those differ,
# the site has capped the card and `pi = P_avg / TDP` has an unreachable denominator --
# fit `pi_limit` instead.

set -uo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$_here/config.sh"

QUERY="name,memory.total,power.limit,power.default_limit,power.max_limit,clocks.max.sm,persistence_mode,compute_mode"

printf '%-14s %-12s %s\n' "KEY" "PARTITION" "CARD / limits"
printf '%.0s-' {1..100}; echo

for key in "${!KE_GPU_SPEC[@]}"; do
  partition="$(ke_gpu_field "$key" 1)"
  gres="$(ke_gpu_field "$key" 2)"

  out=$(timeout 300 srun --partition="$partition" --gres="$gres" \
          --time=00:02:00 --cpus-per-task=1 --mem=4G \
          $(ke_account_flag) \
          nvidia-smi --query-gpu="$QUERY" --format=csv,noheader 2>&1 | head -1)

  if [[ -z "$out" ]]; then
    out="(no output -- partition busy, or you lack access)"
  fi
  printf '%-14s %-12s %s\n' "$key" "$partition" "$out"
done

echo
echo "Check, in order:"
echo "  1. Does each CARD name match the KEY you expected? A100 PCIe and A100 SXM4 in"
echo "     particular must come back as different names, or one of the two config.sh"
echo "     lines is pointing at the same hardware twice."
echo "  2. power.limit vs power.default_limit -- if they differ, the card is capped."
echo "  3. compute_mode should be Default or Exclusive_Process, not Prohibited."
echo
echo "Then confirm the table agrees with what came back:"
echo "  kernelenergy info | head -12"
