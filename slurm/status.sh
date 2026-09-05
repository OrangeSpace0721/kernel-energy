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

# Rows land in per-(card, category, shard) files, so count uniquely by kernel_sig to
# avoid double-counting a config a requeued task re-measured.
for key in $(printf '%s\n' "${!KE_GPU_SPEC[@]}" | sort); do
  files=("$KE_DATA/raw/${key}__"*.csv)
  [[ -e "${files[0]}" ]] || { printf '  %-12s %s\n' "$key" "-"; continue; }
  n=$(cat "${files[@]}" 2>/dev/null | grep -v '^kernel_sig' | cut -d, -f1 | sort -u | wc -l)
  total=$((total + n))
  pct=""
  [[ $n_cat -gt 0 ]] && pct=$(awk -v a="$n" -v b="$n_cat" 'BEGIN{printf "%5.1f%%", 100*a/b}')
  printf '  %-12s %6d rows  %s  (%d shard files)\n' "$key" "$n" "$pct" "${#files[@]}"
done
echo "  ---"
printf '  %-12s %6d rows across the fleet (catalogue has %d configs)\n' "TOTAL" "$total" "$n_cat"

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
