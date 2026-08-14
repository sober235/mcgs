#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 && $# -ne 9 ]]; then
  echo "usage: $0 EXP GPU ITERS LAMBDA_GEO TV_EPS KSPACE_HIGH EDGE_TV_KAPPA_Q EDGE_TV_WFLOOR [KSPACE_HIGH_PLAIN]" >&2
  exit 2
fi

EXP=$1
GPU=$2
ITERS=$3
LAMBDA_GEO=$4
TV_EPS=$5
KSPACE_HIGH=$6
EDGE_TV_KAPPA_Q=$7
EDGE_TV_WFLOOR=$8
KSPACE_HIGH_PLAIN=${9:--1}

PLAIN_HIGH_ARGS=()
if [[ "$KSPACE_HIGH_PLAIN" != "-1" ]]; then
  PLAIN_HIGH_ARGS=(--kspace_high_plain "$KSPACE_HIGH_PLAIN")
fi

ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
CONFIG=configs/winner_v2_edgeTV.yaml
LOG="$ROOT/results_arc/${EXP}.log"
SCORE="$ROOT/results_arc/${EXP}_score.json"
VOLUMES="$ROOT/results_arc/${EXP}_volumes.npz"
RUN_DIR="$ROOT/results_arc/${EXP}"
ARTIFACT="$ROOT/results_arc/autoresearch_af12/${EXP}_artifact.json"

for path in "$LOG" "$SCORE" "$VOLUMES" "$RUN_DIR" "$ARTIFACT"; do
  if [[ -e "$path" ]]; then
    echo "refusing to overwrite existing output: $path" >&2
    exit 1
  fi
done

mkdir -p "$ROOT/results_arc/autoresearch_af12"
cd "$ROOT"

env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID \
  "$PY" -u arc/cycle.py \
  --exp "$EXP" --gpu "$GPU" --iters "$ITERS" \
  --lambda_geo "$LAMBDA_GEO" --warmup 500 --max_radius 20 \
  --kspace_high "$KSPACE_HIGH" "${PLAIN_HIGH_ARGS[@]}" \
  --edge_tv --edge_tv_kappa_q "$EDGE_TV_KAPPA_Q" \
  --edge_tv_wfloor "$EDGE_TV_WFLOOR" --tv_eps "$TV_EPS" \
  --base_cfg "$CONFIG" --save_volumes 2>&1 | tee "$LOG"

env PYTHONNOUSERSITE=1 PYTHONPATH= \
  "$PY" scripts/autoresearch_artifact_score.py \
  --npz "$VOLUMES" --score "$SCORE" --out "$ARTIFACT"
