#!/usr/bin/env bash
# One MC-GS run on the jiangyizhou T1+FLAIR pair at nominal AF 20 (topic T1FLAIR_AF20_v3_jiangyizhou).
#   run_af20_mcgs.sh EXP GPU ITERS [extra arc/cycle.py args...]
#   SINGLE=T1|FLAIR picks that contrast's single-arm config.   SAVE_VOLUMES=1 dumps the recon npz.
#
# The recipe is the AF16 PHASE-3 recipe, frozen, in one place (user decision 2026-09-13: AF20 runs the
# AF16 parameters unchanged, so that the only difference between the two studies is the mask):
#   P6            repaired cold start (init_mode head, 3-NN scales, edge-TV threshold from ZF),
#                 3.6M points, edge-TV lambda_geo 0.10 / kappa_q 0.85 / wfloor 0.10 / eps 0.04,
#                 kspace_high 2.2, warmup 500
#   + phase 3     wavelet db4 on its finest level only, lambda_wav T1 0.12 / FLAIR 0.10
#                 common-edge cross-gradient prior --lambda_edge 0.001
#                 phase-only TV --lambda_ptv 0.03
# At AF16 this reads head PSNR 35.706 / 34.928 (margins over L1-ESPIRiT +0.497 / +0.425).
# Extra args are appended last, so they override (cycle.py: repeated flags -> last wins;
# --set applies in argv order).
set -uo pipefail
if [[ $# -lt 3 ]]; then echo "usage: $0 EXP GPU ITERS [extra cycle.py args...]" >&2; exit 2; fi
EXP=$1; GPU=$2; ITERS=$3; shift 3
if [[ "$EXP" != af20_* ]]; then
  echo "EXP must start with af20_ (keeps this study apart from the AF12 and AF16 lines): $EXP" >&2; exit 2
fi
WARMUP=${WARMUP:-500}
ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
case "${SINGLE:-joint}" in
  joint) CONFIG=configs/t1flair_v3_af20_jyz.yaml ;;
  T1)    CONFIG=configs/t1flair_v3_af20_jyz_singleT1.yaml ;;
  FLAIR) CONFIG=configs/t1flair_v3_af20_jyz_singleFLAIR.yaml ;;
  *)     echo "SINGLE must be T1, FLAIR or unset: ${SINGLE}" >&2; exit 2 ;;
esac
LOG="$ROOT/results_arc/${EXP}.log"
for path in "$LOG" "$ROOT/results_arc/${EXP}_score.json" "$ROOT/results_arc/${EXP}"; do
  if [[ -e "$path" ]]; then echo "refusing to overwrite existing output: $path" >&2; exit 1; fi
done
VOL=()
if [[ "${SAVE_VOLUMES:-0}" == "1" ]]; then VOL=(--save_volumes); fi
mkdir -p "$ROOT/results_arc"
cd "$ROOT"
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID TORCH_EXTENSIONS_DIR=/data0/congcong/code/GS/mcgs_af16_wt.torch_ext \
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
echo "[$(date +%H:%M)] $EXP rc=$RC  $(grep -o 'head_psnr[^,]*' "$ROOT/results_arc/${EXP}_score.json" 2>/dev/null | head -2 | tr '\n' ' ')" >> "$ROOT/results_arc/af20_runs.log"
exit $RC
