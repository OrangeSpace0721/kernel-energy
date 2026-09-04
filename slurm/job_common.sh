#!/usr/bin/env bash
# Sourced at the top of every job. Sets up the environment and the invariants that make
# a measurement trustworthy on a shared machine.

set -euo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$_here/config.sh"

# --- CUDA device ordering ---------------------------------------------------
# This one line prevents a whole class of silent corruption. CUDA_DEVICE_ORDER defaults
# to FASTEST_FIRST, which orders CUDA devices by capability rather than by PCI slot. On a
# heterogeneous node that reorders them relative to NVML, so "CUDA device 0 is the first
# entry of CUDA_VISIBLE_DEVICES" stops being true and the energy counter gets read from
# the wrong card. The Python side resolves by UUID and does not rely on this, but it
# refuses to fall back to index arithmetic unless this is set -- so set it.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# --- offline -----------------------------------------------------------------
# Compute nodes have no internet. Without these, the hub client spends minutes on
# connection timeouts before falling back to the cache; with them, a missing weight
# fails in a second with a message that names the file.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

# --- threads -----------------------------------------------------------------
# Uncapped OpenMP on a big node spawns a thread per core and the CPU-side contention
# shows up as jitter in the replay loop's launch cadence.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

# --- conda -------------------------------------------------------------------
if [[ -n "${KE_PRE_ACTIVATE:-}" ]]; then eval "$KE_PRE_ACTIVATE"; fi
if [[ ! -f "$KE_CONDA_SH" ]]; then
  echo "conda profile not found at $KE_CONDA_SH -- set KE_CONDA_SH in slurm/config.sh" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$KE_CONDA_SH"
conda activate "$KE_ENV_NAME"

mkdir -p "$KE_DATA" "$KE_LOGS"

ke_banner() {
  echo "=============================================================="
  echo "job      : ${SLURM_JOB_ID:-local} ${SLURM_ARRAY_TASK_ID:+task $SLURM_ARRAY_TASK_ID}"
  echo "node     : $(hostname)"
  echo "started  : $(date -Is)"
  echo "python   : $(which python)"
  echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "CUDA_DEVICE_ORDER    = ${CUDA_DEVICE_ORDER:-<unset>}"
  echo "data     : $KE_DATA"
  nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,power.limit,persistence_mode \
             --format=csv 2>/dev/null || true
  echo "=============================================================="
}
