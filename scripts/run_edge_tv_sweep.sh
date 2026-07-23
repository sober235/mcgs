#!/usr/bin/env bash
# Edge-TV sweep driver (P0 noise floor + P1 kappa_q x lambda cells @1500 iters).
# Splittable across GPUs: pass a gpu and the cells this batch should run.
# Each cell is "KQ:LAM" (edge-TV) or the literal "p0" (isotropic-TV baseline re-run).
# Per-GPU TSV avoids concurrent-append races; merge before selection.
#
# Usage: run_edge_tv_sweep.sh <gpu> <cell> [cell...]
#   e.g. run_edge_tv_sweep.sh 5 p0 0.60:0.06 0.75:0.06 0.85:0.06 0.60:0.10 0.75:0.10
#        run_edge_tv_sweep.sh 7 0.85:0.10 0.60:0.15 0.75:0.15 0.85:0.15
set -euo pipefail
GPU="$1"; shift
cd "$(dirname "$0")/.."
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
TSV="results_arc/edge_scores_g${GPU}.tsv"
run(){ CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONNOUSERSITE=1 PYTHONPATH= "$PY" arc/cycle.py "$@"; }
for cell in "$@"; do
  if [ "$cell" = p0 ]; then
    run --exp p0_noise --gpu "$GPU" --iters 1500 --lambda_geo 0.06 --kspace_high 2.2 \
        --warmup 500 --max_radius 20 --base_cfg configs/winner_v2_aligned.yaml --save_volumes
    PYTHONNOUSERSITE=1 "$PY" scripts/score_edge.py --npz results_arc/p0_noise_volumes.npz --tag p0_noise --tsv "$TSV"
  else
    KQ="${cell%%:*}"; LAM="${cell##*:}"; exp="etv_kq${KQ}_l${LAM}"
    run --exp "$exp" --gpu "$GPU" --iters 1500 --kspace_high 2.2 --warmup 500 --max_radius 20 \
        --edge_tv --edge_tv_kappa_q "$KQ" --edge_tv_wfloor 0.1 --lambda_geo "$LAM" \
        --base_cfg configs/winner_v2_edgeTV.yaml --save_volumes
    PYTHONNOUSERSITE=1 "$PY" scripts/score_edge.py --npz "results_arc/${exp}_volumes.npz" --tag "$exp" --tsv "$TSV"
  fi
done
echo "BATCH DONE gpu=$GPU cells: $*"
