#!/usr/bin/env bash
# One autoresearch experiment on the jiangyizhou T1+FLAIR AF16 pair (see autoresearch_jyz_af16.md).
#   autoresearch_jyz_af16.sh EXP GPU ITERS [extra arc/cycle.py args...]
# The PARENT is the v2 recipe of the hyw T1+T2 loop (frozen working point + ZF-head-support init,
# 3-NN scales, kappa from ZF, 1.2M points), baked in below; every experiment differs from it only
# by the extra args it passes (cycle.py flags repeat -> last wins; --set key=value overrides apply
# in argv order -> last wins). Never overwrites an existing experiment.
# SAVE_VOLUMES=1 dumps the recon npz (promotion runs only). WARMUP=20 for smoke runs.
set -uo pipefail
if [[ $# -lt 3 ]]; then
  echo "usage: $0 EXP GPU ITERS [extra cycle.py args...]" >&2; exit 2
fi
EXP=$1; GPU=$2; ITERS=$3; shift 3
WARMUP=${WARMUP:-500}
ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
CONFIG=${CONFIG:-configs/t1flair_v3_af16_jyz.yaml}   # single-arm checks point this at the _singleT1/_singleFLAIR configs
LOG="$ROOT/results_arc/${EXP}.log"
for path in "$LOG" "$ROOT/results_arc/${EXP}_score.json" "$ROOT/results_arc/${EXP}"; do
  if [[ -e "$path" ]]; then echo "refusing to overwrite existing output: $path" >&2; exit 1; fi
done
VOL=()
if [[ "${SAVE_VOLUMES:-0}" == "1" ]]; then VOL=(--save_volumes); fi
cd "$ROOT"
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID TORCH_EXTENSIONS_DIR=/data0/congcong/code/GS/mcgs_af16_wt.torch_ext \
  "$PY" -u arc/cycle.py --exp "$EXP" --gpu "$GPU" --iters "$ITERS" \
  --lambda_geo 0.10 --warmup "$WARMUP" --max_radius 20 --kspace_high 2.2 \
  --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 \
  --set gaussian.init_mode=head --set gaussian.init_scale_mode=3nn \
  --set loss.edge_tv_kappa_source=zf --set gaussian.initial_num_points=1200000 \
  --base_cfg "$CONFIG" "${VOL[@]}" "$@" > "$LOG" 2>&1
RC=$?
STATUS=pending
if [[ $RC -ne 0 ]]; then STATUS=crash; fi
env PYTHONNOUSERSITE=1 PYTHONPATH= "$PY" scripts/autoresearch_jyz_af16_row.py --exp "$EXP" --status "$STATUS" --note "rc=$RC cfg=$CONFIG"
exit $RC
