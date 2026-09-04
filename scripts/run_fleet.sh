#!/usr/bin/env bash
# Per-GPU collection. Run this once on each card in the fleet, then merge.
#
#   ./scripts/run_fleet.sh capture    # only needed on one box with all three models
#   ./scripts/run_fleet.sh measure    # on every card
#
# The measure stage is resumable: rerun it after a preemption and it picks up where it
# stopped, skipping (kernel_sig, gpu_key) pairs already in the output.
#
# Run measure on an otherwise idle card. Another process sharing the GPU does not merely
# add noise -- it changes the power budget the card is working against, so every row
# collected alongside it is measuring a different machine.

set -euo pipefail

STAGE="${1:-measure}"
DATA="${DATA_DIR:-data}"
DEVICE="${CUDA_DEVICE:-0}"

case "$STAGE" in
  capture)
    for m in flux1-dev sd35-large qwen-image; do
      echo "=== capturing $m ==="
      kernelenergy capture --model "$m" --out "$DATA/catalogue" --profile \
        || echo "  $m failed; continuing"
    done
    kernelenergy catalogue --in "$DATA/catalogue" --out "$DATA/catalogue.csv"
    ;;

  measure)
    echo "=== idle baseline ==="
    kernelenergy idle --device "$DEVICE" | tee "$DATA/idle_$(hostname).txt"

    echo "=== sweep ==="
    kernelenergy measure \
      --catalogue "$DATA/catalogue.csv" \
      --out "$DATA/raw" \
      --device "$DEVICE" \
      --window "${WINDOW:-3.0}" \
      --warmup "${WARMUP:-2.0}" \
      --repeats "${REPEATS:-3}" \
      --buffers "${BUFFERS:-4}" \
      --notes "$(hostname) $(date -Is)"
    ;;

  dataset)
    kernelenergy dataset --raw "$DATA/raw" --out "$DATA/dataset.csv"
    ;;

  evaluate)
    kernelenergy evaluate --dataset "$DATA/dataset.csv" --fold all \
      --predictions "$DATA/predictions" --out "$DATA/results.txt"
    ;;

  *)
    echo "usage: $0 {capture|measure|dataset|evaluate}" >&2
    exit 2
    ;;
esac
