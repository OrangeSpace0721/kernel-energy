#!/usr/bin/env bash
# Ask every GPU partition what card it actually has.
#
#   bash slurm/identify_gpus.sh            # print gres names, submit probes, poll
#   bash slurm/identify_gpus.sh --collect  # just read whatever probes have finished
#   bash slurm/identify_gpus.sh --gres     # print gres names only, submit nothing
#
# Worth doing before a sweep because the hardware table is keyed on the NVML product
# name, and near-miss names are everywhere: L4 against L40, L40 against L40S, H200
# against H200 NVL. Since the L40 turns out to have exactly half the L40S's tensor
# throughput on the same die, mistaking one for the other is a 2x error in peak -- which
# shows up only as "that card looks inefficient", never as a failure.
#
# It does NOT block on srun. An earlier version did, and on a busy cluster every probe
# just sat in the queue until the timeout killed it, capturing srun's "queued and
# waiting" message instead of an answer. Probes are submitted as tiny batch jobs that
# write to a file; this polls for the files and can be re-run with --collect at any
# point, including after you have closed the terminal.

set -uo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$_here/config.sh"

MODE="submit"
case "${1:-}" in
  --collect) MODE="collect" ;;
  --gres)    MODE="gres" ;;
  "")        ;;
  *) echo "usage: $0 [--collect|--gres]" >&2; exit 2 ;;
esac

OUT="$KE_DATA/gpuinfo"
mkdir -p "$OUT"
QUERY="name,memory.total,power.limit,power.default_limit,power.max_limit,clocks.max.sm,persistence_mode,compute_mode"

# --------------------------------------------------------------------------- #
# 1. The gres names, untruncated. No queue needed, and this alone is often the fix.
# --------------------------------------------------------------------------- #
echo "=== gres advertised by each partition (from sinfo, not truncated) ==="
printf '%-14s %-16s %s\n' "KEY" "PARTITION" "GRES"
for key in $(printf '%s\n' "${!KE_GPU_SPEC[@]}" | sort); do
  p="$(ke_gpu_field "$key" 1)"
  # %G with no width specifier does not truncate; sort -u because a partition can span
  # node types.
  real=$(sinfo -h -p "$p" -o '%G' 2>/dev/null | tr ',' '\n' | sed 's/(.*)//' \
         | grep -v '^(null)$' | sort -u | paste -sd, -)
  configured="$(ke_gpu_field "$key" 2)"
  flag=""
  # Compare the resource *name* only -- config carries a ":1" count that sinfo's
  # per-node total does not.
  cfg_name=$(echo "$configured" | cut -d: -f1-2)
  if [[ -n "$real" ]] && ! echo "$real" | grep -q "${cfg_name#gpu:}"; then
    flag="   <-- config says '$configured', which does not match"
  fi
  printf '%-14s %-16s %s%s\n' "$key" "$p" "${real:-<partition not found>}" "$flag"
done

if [[ "$MODE" == "gres" ]]; then
  echo
  echo "Fix any mismatched line in slurm/config.sh, then re-run without --gres."
  exit 0
fi

# --------------------------------------------------------------------------- #
# 2. Submit one tiny probe per partition.
# --------------------------------------------------------------------------- #
if [[ "$MODE" == "submit" ]]; then
  echo
  echo "=== submitting probes ==="
  for key in $(printf '%s\n' "${!KE_GPU_SPEC[@]}" | sort); do
    p="$(ke_gpu_field "$key" 1)"
    g="$(ke_gpu_field "$key" 2)"
    f="$OUT/$key.txt"
    rm -f "$f" "$OUT/$key.err"

    submit() {
      sbatch --parsable --job-name="ke-id-$key" --partition="$p" --gres="$1" \
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:02:00 \
        --output="$OUT/$key.slurmlog" $(ke_account_flag) \
        --wrap="nvidia-smi --query-gpu=$QUERY --format=csv,noheader > '$f'" 2>"$OUT/$key.err"
    }

    id=$(submit "$g")
    if [[ -z "$id" ]]; then
      # A wrong gres name is the usual cause. Plain gpu:1 is accepted almost everywhere
      # and still lands on the right partition, so it is a useful fallback.
      id=$(submit "gpu:1")
      [[ -n "$id" ]] && echo "  $key: '$g' rejected, fell back to gpu:1 (fix config.sh)"
    fi
    if [[ -z "$id" ]]; then
      printf '  %-14s FAILED to submit: %s\n' "$key" "$(head -1 "$OUT/$key.err")"
    else
      printf '  %-14s job %s\n' "$key" "$id"
    fi
  done
fi

# --------------------------------------------------------------------------- #
# 3. Poll. Safe to interrupt -- re-run with --collect.
# --------------------------------------------------------------------------- #
echo
echo "=== waiting for probes (Ctrl-C is safe; re-run with --collect) ==="
deadline=$(( SECONDS + ${KE_ID_WAIT:-900} ))
while (( SECONDS < deadline )); do
  pending=0
  for key in "${!KE_GPU_SPEC[@]}"; do
    [[ -s "$OUT/$key.txt" ]] || pending=$((pending + 1))
  done
  (( pending == 0 )) && break
  n=$(squeue -u "$USER" -h -n "$(printf 'ke-id-%s,' "${!KE_GPU_SPEC[@]}" | sed 's/,$//')" 2>/dev/null | wc -l)
  printf '\r  %d of %d still pending (%s in queue, %ds elapsed)   ' \
    "$pending" "${#KE_GPU_SPEC[@]}" "$n" "$SECONDS"
  sleep 15
done
echo

# --------------------------------------------------------------------------- #
# 4. Report.
# --------------------------------------------------------------------------- #
echo "=== results ==="
printf '%-14s %-16s %s\n' "KEY" "PARTITION" "CARD / limits"
printf '%.0s-' {1..110}; echo
missing=0
for key in $(printf '%s\n' "${!KE_GPU_SPEC[@]}" | sort); do
  p="$(ke_gpu_field "$key" 1)"
  f="$OUT/$key.txt"
  if [[ -s "$f" ]]; then
    printf '%-14s %-16s %s\n' "$key" "$p" "$(head -1 "$f")"
  else
    missing=$((missing + 1))
    reason="still queued"
    [[ -s "$OUT/$key.err" ]] && reason="$(head -1 "$OUT/$key.err")"
    printf '%-14s %-16s (%s)\n' "$key" "$p" "$reason"
  fi
done

echo
echo "columns: name, memory, power.limit, power.default, power.max, max SM clock,"
echo "         persistence, compute mode"
echo
echo "Check:"
echo "  1. Every CARD name distinct? Two keys reporting the same name means one"
echo "     config.sh line points at the wrong hardware -- submit_measure.sh refuses"
echo "     to launch in that case, since both would write the same gpu_key."
echo "  2. power.limit vs power.default -- if they differ the card is capped, and"
echo "     add_targets should fit pi_limit rather than pi."
echo "  3. compute_mode must not be Prohibited."
(( missing > 0 )) && echo -e "\n  $missing probe(s) outstanding -- re-run: bash slurm/identify_gpus.sh --collect"
exit 0
