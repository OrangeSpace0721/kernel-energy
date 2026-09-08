#!/usr/bin/env bash
# Where has the fleet sweep got to?
#
#   bash slurm/status.sh
#
# Two dozen array tasks across six cards is more than `squeue` reads well, and the
# question you actually want answered is not "what is running" but "how many rows do I
# have per card, and is anything wrong with them". So this reports both: the queue, and
# the data on disk.

set -uo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$_here/config.sh"

echo "=== queue ==="
squeue -u "$USER" -o '%.14i %.14j %.10T %.10M %.10L %.20R' 2>/dev/null | head -40
n_run=$(squeue -u "$USER" -h -t RUNNING 2>/dev/null | wc -l)
n_pend=$(squeue -u "$USER" -h -t PENDING 2>/dev/null | wc -l)
echo "  $n_run running, $n_pend pending"

echo
echo "=== rows collected ==="
if [[ ! -d "$KE_DATA/raw" ]]; then
  echo "  nothing yet ($KE_DATA/raw does not exist)"
  exit 0
fi

total=0
n_cat=$(wc -l < "$KE_DATA/catalogue.csv" 2>/dev/null || echo 0)
[[ $n_cat -gt 0 ]] && n_cat=$((n_cat - 1))

# Report on the keys that are ACTUALLY on disk, not only the ones config.sh expects.
#
# Those two sets can differ, and the difference is easy to misread as missing data. The
# file name carries the gpu_key that probe_local_gpu() derived from the card's real NVML
# product name -- so if the `l40` partition turns out to hold an L40S, the rows land in
# L40S__*.csv while config.sh still says L40. An earlier version of this script listed
# only the configured keys and printed "-" beside them, which made a complete, successful
# sweep look like a job that never ran.
declare -A _seen=()
shopt -s nullglob
for f in "$KE_DATA/raw/"*__*.csv; do
  _seen["$(basename "$f" | sed 's/__.*//')"]=1
done
shopt -u nullglob

# configured keys first, then anything else found on disk
_order=$( { printf '%s\n' "${!KE_GPU_SPEC[@]}"; printf '%s\n' "${!_seen[@]}"; } | sort -u )

for key in $_order; do
  configured=""; [[ -n "${KE_GPU_SPEC[$key]:-}" ]] || configured="  <-- not in config.sh"
  files=("$KE_DATA/raw/${key}__"*.csv)
  [[ -e "${files[0]}" ]] || { printf '  %-12s %s\n' "$key" "no data yet"; continue; }
  n=$(cat "${files[@]}" 2>/dev/null | grep -v '^kernel_sig' | cut -d, -f1 | sort -u | wc -l)
  total=$((total + n))
  pct=""
  [[ $n_cat -gt 0 ]] && pct=$(awk -v a="$n" -v b="$n_cat" 'BEGIN{printf "%5.1f%%", 100*a/b}')
  printf '  %-12s %6d rows  %s  (%d shard files)%s\n' \
    "$key" "$n" "$pct" "${#files[@]}" "$configured"
done
echo "  ---"
printf '  %-12s %6d rows across the fleet (catalogue has %d configs)\n' "TOTAL" "$total" "$n_cat"

# A key on disk that config.sh does not know about is not an error -- the card simply
# identified as something other than the label we guessed -- but the label in config.sh
# should be corrected so resubmissions and status agree.
for key in "${!_seen[@]}"; do
  if [[ -z "${KE_GPU_SPEC[$key]:-}" ]]; then
    echo
    echo "  NOTE: rows exist for '$key', which is not a key in config.sh. The card"
    echo "        identified itself differently from the label we assumed. The data is"
    echo "        fine; rename the KE_GPU_SPEC entry to '$key' so status and resubmits"
    echo "        line up. Check with:  head -1 \$KE_DATA/raw/${key}__*.csv | head -2"
  fi
done

echo
echo "=== warnings in logs ==="
if compgen -G "$KE_LOGS/ke-*.out" > /dev/null; then
  # Contention means a row carries a neighbour's joules; OOM and FAILED mean configs
  # were skipped and the card's coverage is incomplete.
  for pat in CONTENDED "skip (oom)" FAILED "PREFLIGHT FAILED" "exiting 64"; do
    c=$(grep -l "$pat" "$KE_LOGS"/ke-*.out 2>/dev/null | wc -l)
    [[ $c -gt 0 ]] && printf '  %-20s in %d log(s)\n' "$pat" "$c"
  done
  echo "  (none listed above means none found)"
else
  echo "  no logs in $KE_LOGS"
fi

echo
echo "when every card is near 100%:  sbatch --partition=\$KE_CPU_PARTITION slurm/03_dataset.sbatch"
