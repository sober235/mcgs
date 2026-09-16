#!/usr/bin/env bash
# One MC-GS run for the hyper-parameter sensitivity study (spec docs/superpowers/specs/2026-09-16-hparam-sensitivity-design.md):
# jiangyizhou T1+FLAIR at nominal AF 16 (the house mask of T1FLAIR_AF16_v3_jiangyizhou, T1 15.19x / FLAIR 16.49x).
#   run_hp_mcgs.sh EXP GPU ITERS [extra arc/cycle.py args...]
#   SAVE_VOLUMES=1 dumps the recon npz.  WARMUP (default 500) for short smokes.
#
# The recipe block below is byte-identical to run_af24_mcgs.sh (the AF16 phase-3 recipe, frozen): P6 cold start, 3.6M points,
# edge-TV lambda_geo 0.10 / kappa_q 0.85 / wfloor 0.10 / eps 0.04, kspace_high 2.2, warmup 500, wavelet db4 finest level
# (T1 0.12 / FLAIR 0.10), common-edge 0.001, phase TV 0.03. The study varies ONLY what the caller appends:
#   --set gaussian.initial_num_points=N     (shared count; --set applies in argv order, the appended value wins)
#   --set gaussian.num_private_points=N     (FLAIR-private residual set; 0 = the published recipe)
set -uo pipefail
if [[ $# -lt 3 ]]; then echo "usage: $0 EXP GPU ITERS [extra cycle.py args...]" >&2; exit 2; fi
EXP=$1; GPU=$2; ITERS=$3; shift 3
if [[ "$EXP" != hp_* ]]; then
  echo "EXP must start with hp_ (keeps this study apart from the AF12/16/20/24 and diff-mask lines): $EXP" >&2; exit 2
fi
WARMUP=${WARMUP:-500}
ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
CONFIG=configs/t1flair_v3_af16_jyz.yaml
LOG="$ROOT/results_arc/${EXP}.log"
for path in "$LOG" "$ROOT/results_arc/${EXP}_score.json" "$ROOT/results_arc/${EXP}"; do
  if [[ -e "$path" ]]; then echo "refusing to overwrite existing output: $path" >&2; exit 1; fi
done
VOL=()
if [[ "${SAVE_VOLUMES:-0}" == "1" ]]; then VOL=(--save_volumes); fi
mkdir -p "$ROOT/results_arc"
cd "$ROOT"
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID TORCH_EXTENSIONS_DIR=/data0/congcong/code/GS/mcgs_hparam_wt.torch_ext \
  "$PY" -u arc/cycle.py --exp "$EXP" --gpu "$GPU" --iters "$ITERS" \
  --lambda_geo 0.10 --warmup "$WARMUP" --max_radius 20 --kspace_high 2.2 \
  --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 \
  --set gaussian.init_mode=head --set gaussian.init_scale_mode=3nn \
  --set loss.edge_tv_kappa_source=zf \
  --set gaussian.initial_num_points=3600000 \
  --set loss.wav_name=db4 --set loss.wav_levels=1 \
  --set loss.lambda_wav_plain=0.12 --set loss.lambda_wav=0.10 \
  --lambda_edge 0.001 --lambda_ptv 0.03 \
  --base_cfg "$CONFIG" "${VOL[@]}" "$@" > "$LOG" 2>&1
RC=$?
echo "[$(date '+%F %H:%M')] $EXP rc=$RC  $(grep -o 'head_psnr[^,]*' "$ROOT/results_arc/${EXP}_score.json" 2>/dev/null | head -2 | tr '\n' ' ')" >> "$ROOT/results_arc/hp_runs.log"
exit $RC
