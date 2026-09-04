#!/usr/bin/env bash
# Site configuration. THIS IS THE ONLY FILE YOU SHOULD NEED TO EDIT.
#
# Filled in for Iridis X (University of Southampton) from `sinfo` on 2026-09-04.
# For any other cluster, replace KE_GPU_SPEC and the conda block with your own.

# --- paths (shared scratch, visible from every compute node) -----------------
# Iridis puts scratch at /scratch/<username> and does not export $SCRATCH.
export KE_ROOT="${KE_ROOT:-/scratch/$USER/kernel-energy}"
export KE_DATA="${KE_DATA:-$KE_ROOT/data}"
export KE_LOGS="${KE_LOGS:-$KE_ROOT/logs}"
export KE_ENV_NAME="${KE_ENV_NAME:-kernelenergy}"

# HuggingFace cache. Over 100 GB across the three pipelines, so it must be on scratch,
# not home, and shared so six cards do not each pull their own copy.
export HF_HOME="${HF_HOME:-$KE_ROOT/hf}"
export HF_HUB_ENABLE_HF_TRANSFER=1

# --- conda ------------------------------------------------------------------
# Set this to the output of `conda info --base` plus /etc/profile.d/conda.sh.
# If your conda comes from a module, put the module load in KE_PRE_ACTIVATE and this
# will usually resolve afterwards.
export KE_CONDA_SH="${KE_CONDA_SH:-$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh}"
export KE_PRE_ACTIVATE="${KE_PRE_ACTIVATE:-}"   # e.g. "module load conda"

# --- the fleet --------------------------------------------------------------
# "PARTITION|GRES|EXCLUSIVE", one per card.
#
# Iridis has two A100 populations and they are almost certainly different SKUs: `a100`
# (rose[02-13], 2 GPUs and 48 cores per node) reads as PCIe, `swarm_a100`
# (swarma[1001-1005], 4 GPUs and 96 cores) as SXM4. That distinction matters -- they
# differ by 150 W of TDP and 100 GB/s of bandwidth -- so CONFIRM IT before a full sweep:
#
#   srun -p a100       --gres=gpu:1 -t 00:02:00 nvidia-smi --query-gpu=name --format=csv
#   srun -p swarm_a100 --gres=gpu:1 -t 00:02:00 nvidia-smi --query-gpu=name --format=csv
#
# probe_local_gpu() reads the NVML product name, so it will pick the right descriptor on
# its own -- but if both report the same name, one of these two lines is wrong and the
# rows will be attributed to a single card.
#
# Exclusivity: `--gres=gpu:1` already gives sole use of the GPU, and the energy counter
# is per board, so exclusive is about thermal isolation only. Set on the small nodes
# where it is cheap; left off on the 8-GPU nodes where it would mean queueing for seven
# idle cards.
declare -gA KE_GPU_SPEC=(
  #                 partition       gres          exclusive
  [A100_PCIE]="a100|gpu:a100:1|yes"            # rose[02-13], 2/node, 2d12h
  [A100_SXM4]="swarm_a100|gpu:a100sw:1|no"     # swarma, 4/node, 5d
  [H100]="swarm_h100|gpu:h100sw:1|no"          # swarmh, 8/node, 5d
  [H200_SXM]="quad_h200|gpu:h200:1|no"         # blossom[01-04], 4/node, 2d12h
  [L4]="l4|gpu:l4:1|no"                        # cotton[01-02], 8/node, 2d12h
  [L40]="l40|gpu:l40:1|yes"                    # coral01, 8/node, single node
)

# Scavenger equivalents: same hardware, 12 h limit, preemptible. Free-er to get, and the
# sweep is built to survive preemption -- it catches the signal, flushes, and requeues.
# Use these when the main partitions are busy; widen KE_MEASURE_ARRAY so each task holds
# a node for less time and loses less when evicted.
declare -gA KE_GPU_SPEC_SCAVENGER=(
  [A100_SXM4]="scavenger_4a100|gpu:a100sw:1|no"
  [H100]="scavenger_8h100|gpu:h100sw:1|no"
  [L4]="scavenger_l4|gpu:l4swar:1|no"
)
if [[ "${KE_USE_SCAVENGER:-0}" == "1" ]]; then
  for k in "${!KE_GPU_SPEC_SCAVENGER[@]}"; do
    KE_GPU_SPEC[$k]="${KE_GPU_SPEC_SCAVENGER[$k]}"
  done
  export KE_MEASURE_TIME="${KE_MEASURE_TIME:-11:30:00}"
fi

# --- job sizing -------------------------------------------------------------
export KE_MEASURE_TIME="${KE_MEASURE_TIME:-04:00:00}"
export KE_MEASURE_ARRAY="${KE_MEASURE_ARRAY:-0-3}"
export KE_CAPTURE_TIME="${KE_CAPTURE_TIME:-02:00:00}"
export KE_CAPTURE_PARTITION="${KE_CAPTURE_PARTITION:-a100}"
export KE_CPU_PARTITION="${KE_CPU_PARTITION:-amd_serial}"   # ruby nodes, no GPU
export KE_CPUS="${KE_CPUS:-8}"
export KE_MEM="${KE_MEM:-64G}"
export KE_ACCOUNT="${KE_ACCOUNT:-}"

ke_account_flag() { [[ -n "$KE_ACCOUNT" ]] && echo "--account=$KE_ACCOUNT"; }

ke_gpu_field() {  # $1 = gpu key, $2 = 1|2|3 for partition|gres|exclusive
  local spec="${KE_GPU_SPEC[$1]:-}"
  [[ -z "$spec" ]] && { echo "unknown GPU key: $1" >&2; return 1; }
  echo "$spec" | cut -d'|' -f"$2"
}
