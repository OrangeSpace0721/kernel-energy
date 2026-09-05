#!/usr/bin/env bash
# Site configuration. THIS IS THE ONLY FILE YOU SHOULD NEED TO EDIT.
#
# Filled in for Iridis X (University of Southampton) from `sinfo` on 2026-09-04.
# For any other cluster, replace KE_GPU_SPEC and the conda block with your own.

# --- paths (shared scratch, visible from every compute node) -----------------
# Iridis puts scratch at /scratch/<username> and does not export $SCRATCH.
# Some batch environments do not export USER either, and every path below depends on
# it, so derive it rather than trusting it.
export USER="${USER:-$(id -un)}"
export KE_ROOT="${KE_ROOT:-/scratch/$USER/kernel-energy}"
export KE_DATA="${KE_DATA:-$KE_ROOT/data}"
export KE_LOGS="${KE_LOGS:-$KE_ROOT/logs}"
export KE_ENV_NAME="${KE_ENV_NAME:-kernelenergy}"

# --- model weights ----------------------------------------------------------
# Over 100 GB across the three pipelines, so never download twice. Two ways to reuse
# what you already have; check which layout yours is with `ls`:
#
#   1. A HUB CACHE -- contains a `hub/` dir, or `models--org--name/` entries directly.
#      Point HF_HOME at it and nothing else changes; repo ids resolve offline.
#
#        export HF_HOME=/scratch/$USER/existing/huggingface
#
#   2. A PLAIN PIPELINE DIRECTORY -- contains model_index.json, model files in
#      subfolders (transformer/, vae/, text_encoder/...). This is what `git clone` or
#      `snapshot_download(local_dir=...)` gives you, and the hub client cannot find it
#      from a repo id. Name the path per pipeline instead:
#
#        export KE_MODEL_FLUX1_DEV=/scratch/$USER/models/FLUX.1-dev
#        export KE_MODEL_SD35_LARGE=/scratch/$USER/models/stable-diffusion-3.5-large
#        export KE_MODEL_QWEN_IMAGE=/scratch/$USER/models/Qwen-Image
#
# The two can be mixed -- override the ones you already have, let the rest come from
# the cache. `kernelenergy preflight --stage capture` prints exactly which path each
# pipeline resolved to, so run it before committing an allocation.
export HF_HOME="${HF_HOME:-$KE_ROOT/hf}"
export HF_HUB_ENABLE_HF_TRANSFER=1

# Reusing the teacher snapshots from the distillation project. These are the stock
# pretrained pipelines, which is exactly what this measures -- no download needed.
#
# One caveat worth checking once (`kernelenergy preflight --stage capture` does it):
# a distillation project sometimes keeps only the denoiser, since that is all the
# student needs. This harness runs the whole pipeline, so each directory must be a
# complete snapshot -- model_index.json at the top, plus vae/, text_encoder*/ and
# tokenizer*/ beside the transformer. If one turns out to be transformer-only, leave
# that KE_MODEL_* unset and 00_setup.sh will fetch the full pipeline for it alone.
KE_TEACHERS="${KE_TEACHERS:-/iridisfs/scratch/$USER/model_distillation/Model-Distillation}"
export KE_MODEL_FLUX1_DEV="${KE_MODEL_FLUX1_DEV:-$KE_TEACHERS/flux1_teacher}"
export KE_MODEL_SD35_LARGE="${KE_MODEL_SD35_LARGE:-$KE_TEACHERS/sd3.5_large_teacher}"
export KE_MODEL_QWEN_IMAGE="${KE_MODEL_QWEN_IMAGE:-$KE_TEACHERS/qwen_teacher}"

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
# gres names confirmed against `sinfo -h -p <part> -o '%G'` on 2026-09-05. Note the
# swarm partitions use "a100swarm" / "h100swarm", not the "a100sw" / "h100sw" that
# sinfo's default %10G column shows -- that column truncates, and a truncated gres name
# fails as "Requested node configuration is not available", which reads like the
# partition being full rather than a typo. `identify_gpus.sh --gres` prints them
# untruncated and flags mismatches.
declare -gA KE_GPU_SPEC=(
  #                 partition       gres              exclusive
  [A100_PCIE]="a100|gpu:a100:1|yes"                 # rose[02-13], 2/node, 2d12h
  [A100_SXM4]="swarm_a100|gpu:a100swarm:1|no"       # swarma, 4/node, 5d
  [H100]="swarm_h100|gpu:h100swarm:1|no"            # swarmh, 8/node, 5d
  [H200_SXM]="quad_h200|gpu:h200:1|no"              # blossom[01-04], 4/node, 2d12h
  [L4]="l4|gpu:l4:1|no"                             # cotton[01-02], 8/node, 2d12h
  [L40]="l40|gpu:l40:1|yes"                         # coral01, 8/node, single node
)

# Scavenger equivalents: same hardware, 12 h limit, preemptible. Much easier to get, and
# the sweep is built to survive preemption -- it catches the signal, finishes the kernel
# in flight, flushes and requeues, so an eviction costs one kernel rather than the run.
# On a cluster where the main GPU partitions queue for days, this is the faster route to
# a complete dataset, not a compromise.
#
# Widen KE_MEASURE_ARRAY when using these so each task holds a node for less time.
# scavenger_l4 spans two node types (ecsai and swarml), so its gres is left as plain
# gpu:1 rather than assuming both advertise the same name.
declare -gA KE_GPU_SPEC_SCAVENGER=(
  [A100_SXM4]="scavenger_4a100|gpu:a100swarm:1|no"
  [H100]="scavenger_8h100|gpu:h100swarm:1|no"
  [L4]="scavenger_l4|gpu:1|no"
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

ke_account_flag() {
  # The explicit `return 0` is load-bearing. Without it the function's exit status is
  # that of the `[[ -n ... ]]` test, which is 1 whenever KE_ACCOUNT is empty -- and the
  # submit scripts run under `set -e`, so they would die at the first card having
  # submitted nothing and printed no error. A silent no-op is the worst possible failure
  # for a script whose whole job is to launch work.
  [[ -n "$KE_ACCOUNT" ]] && echo "--account=$KE_ACCOUNT"
  return 0
}

ke_gpu_field() {  # $1 = gpu key, $2 = 1|2|3 for partition|gres|exclusive
  local spec="${KE_GPU_SPEC[$1]:-}"
  [[ -z "$spec" ]] && { echo "unknown GPU key: $1" >&2; return 1; }
  echo "$spec" | cut -d'|' -f"$2"
}
