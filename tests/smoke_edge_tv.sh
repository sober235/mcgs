#!/usr/bin/env bash
# 2-iter dry-run of the edge-TV path through arc/cycle.py. Verifies wiring end-to-end.
# Usage: bash tests/smoke_edge_tv.sh <gpu>
set -euo pipefail
GPU="${1:-5}"
cd "$(dirname "$0")/.."
LOG=$(mktemp)
CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONNOUSERSITE=1 PYTHONPATH= \
  /home/congcongliu/anaconda3/envs/GSTMI/bin/python arc/cycle.py \
  --exp smoke_edge_tv --gpu "$GPU" --iters 2 --warmup 0 \
  --lambda_geo 0.06 --kspace_high 2.2 --max_radius 20 \
  --base_cfg configs/winner_v2_edgeTV.yaml 2>&1 | tee "$LOG"
grep -q "\[edge_tv\] kappa_q" "$LOG" || { echo "FAIL: no [edge_tv] kappa log line"; exit 1; }
echo "PASS smoke_edge_tv"
