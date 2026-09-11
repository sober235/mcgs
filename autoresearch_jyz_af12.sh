#!/usr/bin/env bash
# One autoresearch experiment on the jiangyizhou T1+FLAIR AF12 pair (see autoresearch_jyz_af12.md).
#   autoresearch_jyz_af12.sh EXP GPU ITERS [extra arc/cycle.py args...]
# The PARENT is P6, the final recipe of the jyz AF16 loop (autoresearch_jyz_af16.md, promotion 4),
# run unchanged on the AF12 mask as ar17_P6_tf_jyz_af12_full4k: frozen working point + ZF-head-support
# init, 3-NN scales, kappa from ZF, 3.6M points, wavelet T1 0.10 / FLAIR 0.07, edge-TV lambda_geo 0.10,
# kappa_q 0.85. Every experiment differs from it only by the extra args it passes (cycle.py flags
# repeat -> last wins; --set key=value overrides apply in argv order -> last wins).
# Never overwrites an existing experiment. SAVE_VOLUMES=1 dumps the recon npz (promotion runs only).
set -uo pipefail
if [[ $# -lt 3 ]]; then
  echo "usage: $0 EXP GPU ITERS [extra cycle.py args...]" >&2; exit 2
fi
EXP=$1; GPU=$2; ITERS=$3; shift 3
WARMUP=${WARMUP:-500}
ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
CONFIG=${CONFIG:-configs/t1flair_v3_af12_hyw.yaml}   # named after the hyw pairing, points at the jyz AF12 topic data
LOG="$ROOT/results_arc/${EXP}.log"
for path in "$LOG" "$ROOT/results_arc/${EXP}_score.json" "$ROOT/results_arc/${EXP}"; do
  if [[ -e "$path" ]]; then echo "refusing to overwrite existing output: $path" >&2; exit 1; fi
done
VOL=()
if [[ "${SAVE_VOLUMES:-0}" == "1" ]]; then VOL=(--save_volumes); fi
cd "$ROOT"
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID TORCH_EXTENSIONS_DIR=/data0/congcong/code/GS/mcgs_af12_wt.torch_ext \
  "$PY" -u arc/cycle.py --exp "$EXP" --gpu "$GPU" --iters "$ITERS" \
  --lambda_geo 0.10 --warmup "$WARMUP" --max_radius 20 --kspace_high 2.2 \
  --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 \
  --set gaussian.init_mode=head --set gaussian.init_scale_mode=3nn \
  --set loss.edge_tv_kappa_source=zf --set gaussian.initial_num_points=3600000 \
  --set loss.lambda_wav=0.07 --set loss.lambda_wav_plain=0.10 \
  --base_cfg "$CONFIG" "${VOL[@]}" "$@" > "$LOG" 2>&1
RC=$?
STATUS=pending
if [[ $RC -ne 0 ]]; then STATUS=crash; fi
env PYTHONNOUSERSITE=1 PYTHONPATH= "$PY" scripts/autoresearch_jyz_af12_row.py --exp "$EXP" --status "$STATUS" --note "rc=$RC cfg=$CONFIG"
exit $RC
