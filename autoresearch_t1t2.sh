#!/usr/bin/env bash
# One autoresearch experiment on the hyw T1+T2 AF12 pair (see autoresearch_t1t2.md).
#   autoresearch_t1t2.sh EXP GPU ITERS [extra arc/cycle.py args...]
# Frozen working point (edge-TV, Charbonnier eps 0.04, lambda_geo 0.10, kspace_high 2.2) is
# baked in; every experiment differs from the baseline only by the extra args it passes
# (cycle.py flags or --set key.path=value overrides). Never overwrites an existing experiment.
# SAVE_VOLUMES=1 dumps the recon npz (promotion runs only; screening skips it to save minutes).
set -uo pipefail
if [[ $# -lt 3 ]]; then
  echo "usage: $0 EXP GPU ITERS [extra cycle.py args...]" >&2; exit 2
fi
EXP=$1; GPU=$2; ITERS=$3; shift 3
WARMUP=${WARMUP:-500}   # dead knob (no use-site) but cycle.py asserts warmup < iters/2; smoke runs set WARMUP=20
ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
CONFIG=configs/t1t2_af12_hyw.yaml
LOG="$ROOT/results_arc/${EXP}.log"
for path in "$LOG" "$ROOT/results_arc/${EXP}_score.json" "$ROOT/results_arc/${EXP}"; do
  if [[ -e "$path" ]]; then echo "refusing to overwrite existing output: $path" >&2; exit 1; fi
done
VOL=()
if [[ "${SAVE_VOLUMES:-0}" == "1" ]]; then VOL=(--save_volumes); fi
cd "$ROOT"
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID \
  "$PY" -u arc/cycle.py --exp "$EXP" --gpu "$GPU" --iters "$ITERS" \
  --lambda_geo 0.10 --warmup "$WARMUP" --max_radius 20 --kspace_high 2.2 \
  --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 \
  --base_cfg "$CONFIG" "${VOL[@]}" "$@" > "$LOG" 2>&1
RC=$?
STATUS=pending
if [[ $RC -ne 0 ]]; then STATUS=crash; fi
env PYTHONNOUSERSITE=1 PYTHONPATH= "$PY" scripts/autoresearch_t1t2_row.py --exp "$EXP" --status "$STATUS" --note "rc=$RC"
exit $RC
