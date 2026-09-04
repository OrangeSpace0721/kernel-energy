#!/usr/bin/env bash
#SBATCH --job-name=ke-dataset
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#
# Merge every shard, attach analytical features, derive targets, and run the folds.
# No GPU: the modelling half is pure NumPy on purpose, so this can run on a CPU
# partition or on a login node.
#
#   sbatch --partition=$KE_CPU_PARTITION slurm/03_dataset.sbatch

_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=/dev/null
source "$_here/job_common.sh"

echo "merging shards from $KE_DATA/raw"
kernelenergy dataset \
  --raw "$KE_DATA/raw" \
  --out "$KE_DATA/dataset.csv" \
  --clock tensor \
  | tee "$KE_DATA/dataset_report.txt"

kernelenergy evaluate \
  --dataset "$KE_DATA/dataset.csv" \
  --fold all \
  --predictions "$KE_DATA/predictions" \
  --out "$KE_DATA/results.txt" \
  | tee -a "$KE_DATA/dataset_report.txt"

# The boost-clock convention, for the comparison the run-level work left open: the tensor
# clock won on latency at every shrinkage level while the energy fold moved the other
# way. Whether that reproduces at kernel level is one flag.
echo
echo "=== same data, boost clock ==="
kernelenergy dataset --raw "$KE_DATA/raw" --out "$KE_DATA/dataset_boost.csv" \
  --clock boost > /dev/null
kernelenergy evaluate --dataset "$KE_DATA/dataset_boost.csv" --fold all \
  | tee "$KE_DATA/results_boost.txt"

echo "done: $(date -Is)"
