#!/usr/bin/env bash
# Site configuration. THIS IS THE ONLY FILE YOU SHOULD NEED TO EDIT.
#
# Everything below is a guess at your site's names. Partitions, constraints and module
# names differ everywhere; `sinfo -o "%P %G %f"` lists yours.

# --- paths (must be on shared scratch, visible from every compute node) ------
# Home quotas are usually far too small for model weights, so nothing here should
# default into $HOME.
export KE_ROOT="${KE_ROOT:-$SCRATCH/kernel-energy}"
export KE_DATA="${KE_DATA:-$KE_ROOT/data}"
export KE_LOGS="${KE_LOGS:-$KE_ROOT/logs}"
export KE_ENV_NAME="${KE_ENV_NAME:-kernelenergy}"

# HuggingFace cache. Weights total well over 100 GB across the three pipelines, so this
# must be shared: six nodes each pulling their own copy is both slow and a quota
# incident waiting to happen.
export HF_HOME="${HF_HOME:-$KE_ROOT/hf}"
export HF_HUB_ENABLE_HF_TRANSFER=1

# --- conda ------------------------------------------------------------------
# Point this at whatever provides `conda` on your site. Some sites want
# `module load anaconda3` first; if so, add it to KE_PRE_ACTIVATE below.
export KE_CONDA_SH="${KE_CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
export KE_PRE_ACTIVATE="${KE_PRE_ACTIVATE:-}"   # e.g. "module load cuda/12.4"

# --- SLURM: which partition and constraint gets you which GPU ---------------
# One entry per card in the fleet, as "PARTITION|CONSTRAINT|GRES".
# Leave CONSTRAINT empty where the partition already implies the GPU type.
declare -gA KE_GPU_SPEC=(
  [L4]="gpu|l4|gpu:l4:1"
  [L40S]="gpu|l40s|gpu:l40s:1"
  [A100_PCIE]="gpu|a100_pcie|gpu:a100:1"
  [A100_SXM4]="gpu|a100_sxm4|gpu:a100:1"
  [H100]="gpu|h100|gpu:h100:1"
  [H200_NVL]="gpu|h200|gpu:h200:1"
)

# --- job sizing -------------------------------------------------------------
export KE_MEASURE_TIME="${KE_MEASURE_TIME:-04:00:00}"
export KE_MEASURE_ARRAY="${KE_MEASURE_ARRAY:-0-3}"   # 4-way split per card
export KE_CAPTURE_TIME="${KE_CAPTURE_TIME:-02:00:00}"
export KE_CPUS="${KE_CPUS:-8}"
export KE_MEM="${KE_MEM:-64G}"
export KE_ACCOUNT="${KE_ACCOUNT:-}"                   # --account, if your site needs one

ke_account_flag() { [[ -n "$KE_ACCOUNT" ]] && echo "--account=$KE_ACCOUNT"; }

ke_gpu_field() {  # $1 = gpu key, $2 = 1|2|3 for partition|constraint|gres
  local spec="${KE_GPU_SPEC[$1]:-}"
  [[ -z "$spec" ]] && { echo "unknown GPU key: $1" >&2; return 1; }
  echo "$spec" | cut -d'|' -f"$2"
}
