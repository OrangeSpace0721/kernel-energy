#!/usr/bin/env bash
#SBATCH --job-name=ke-measure
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:USR1@420
#SBATCH --requeue
#SBATCH --open-mode=append
#
# The measurement sweep. Submit through slurm/submit_measure.sh, which fills in the
# partition, gres and exclusivity for each card -- those differ per site and per GPU, so
# they are not baked in here.
#
# On exclusivity, precisely
# -------------------------
# NVML's energy counter is per *board*, not per node: nvmlDeviceGetTotalEnergyConsumption
# reports joules for one GPU. So a job on a different GPU of the same node does NOT
# contaminate your counter, and `--gres=gpu:1` -- which already gives you sole use of that
# GPU -- is sufficient for the measurement to be attributable.
#
# What node sharing does affect is thermal coupling. Neighbours in the same chassis raise
# inlet temperature and can push your card into thermal throttling, which changes its
# energy at fixed work. That is a real confound, but a second-order one, and every row
# records `temperature_mean_c` and `frac_hw_thermal` so it is visible rather than assumed
# away.
#
# So `--exclusive` is worth having on small nodes and not worth the queue time on large
# ones: demanding all eight GPUs of an H100 node to use one is a good way to never run.
# submit_measure.sh sets it per card from KE_GPU_SPEC. If you do run shared, check
# `frac_hw_thermal` in the quality report afterwards -- if it is materially non-zero on a
# card, re-measure that card exclusively before trusting it.
#
# Two directives above are load-bearing and should not be removed:
#
#   --signal=B:USR1@420
#                      Sends SIGUSR1 seven minutes before walltime. The sweep catches it,
#                      finishes the kernel it is measuring, flushes, and exits cleanly.
#                      Killed instead at the wall, it loses the kernel in flight and may
#                      leave a partial row.
#
#   --requeue          Lets a preempted task come back. The sweep is resumable, so a
#                      requeued task skips everything already recorded and continues.
#
# Exit 64 means "stopped cleanly, work remains" -- see the trailer below.

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

GPU_KEY="${KE_GPU_KEY:?set KE_GPU_KEY, e.g. export KE_GPU_KEY=H100}"

ke_banner
echo "target card: $GPU_KEY"

# Preflight is cheap and catches the failures that would otherwise produce a full run of
# plausible, wrong numbers: the wrong NVML device, a co-tenant, MIG, a capped card.
kernelenergy preflight \
  --stage measure \
  --device 0 \
  --catalogue "$KE_DATA/catalogue.csv" \
  --out "$KE_DATA/raw" || exit 1

# Run through ke_run_forwarding_signals, NOT srun. See the comment on that function in
# job_common.sh: --signal=B:USR1 goes to this bash process, so the sweep only hears about
# an imminent walltime kill if bash passes it down.
#
# --time-budget 0 derives the budget from the job's own remaining walltime, so the sweep
# stops before starting a measurement it cannot finish.
ke_run_forwarding_signals kernelenergy measure \
  --catalogue "$KE_DATA/catalogue.csv" \
  --out "$KE_DATA/raw" \
  --device 0 \
  --window   "${KE_WINDOW:-3.0}" \
  --warmup   "${KE_WARMUP:-2.0}" \
  --repeats  "${KE_REPEATS:-3}" \
  --buffers  "${KE_BUFFERS:-4}" \
  --time-budget 0 \
  --reserve 420 \
  --notes "$GPU_KEY $(hostname) job=${SLURM_JOB_ID}"
status=$?

echo "finished with status $status at $(date -Is)"

if [[ $status -eq 64 ]]; then
  # Clean stop with work remaining -- walltime or a preemption signal. Requeue this
  # exact array task; it will resume from the rows already on disk.
  echo "work remains; requeueing ${SLURM_JOB_ID}"
  scontrol requeue "${SLURM_JOB_ID}" || \
    echo "requeue failed -- resubmit manually, the sweep is resumable"
  exit 0
fi

exit $status
