#!/usr/bin/env bash
# One autoresearch experiment of the Gaussian-mask tuning loop on the jiangyizhou T1+FLAIR pair (2026-09-16).
#   autoresearch_gauss.sh EXP GPU ITERS [extra arc/cycle.py args...]
#   EXP must start with g_ (this line's namespace).  SINGLE=T1|FLAIR picks that contrast's single-arm config.
#   SAVE_VOLUMES=1 dumps the recon npz (promotion runs only).  WARMUP=20 for smoke runs.
#
# Arena: topic T1FLAIR_AF16_gauss_v3_jiangyizhou (2-D Gaussian variable-density random mask, 2964 points,
# T1 15.20x / FLAIR 15.43x, labelled AF 16 nominal). Program, bar and rules: autoresearch_gauss.md.
#
# PARENT = the AF16 phase-3 recipe as it ran frozen on this mask (af16g_p3_joint_full4k in mcgs_diffmask_wt:
# head PSNR 34.909 / 34.317 at 4000 it). The recipe block below is byte-identical to run_diffmask_mcgs.sh's
# (itself byte-identical to run_af24_mcgs.sh's). Every candidate differs from the parent only by the extra
# args it passes: cycle.py flags repeat -> last wins; --set applies in argv order.
set -uo pipefail
if [[ $# -lt 3 ]]; then echo "usage: $0 EXP GPU ITERS [extra cycle.py args...]" >&2; exit 2; fi
EXP=$1; GPU=$2; ITERS=$3; shift 3
if [[ "$EXP" != g_* ]]; then echo "EXP must start with g_ (gauss-loop namespace): $EXP" >&2; exit 2; fi
WARMUP=${WARMUP:-500}
ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
case "${SINGLE:-joint}" in
  joint) CONFIG=configs/t1flair_v3_af16g_jyz.yaml ;;
  T1)    CONFIG=configs/t1flair_v3_af16g_jyz_singleT1.yaml ;;
  FLAIR) CONFIG=configs/t1flair_v3_af16g_jyz_singleFLAIR.yaml ;;
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
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID TORCH_EXTENSIONS_DIR=/data0/congcong/code/GS/mcgs_gauss_wt.torch_ext \
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
STATUS=pending
if [[ $RC -ne 0 ]]; then STATUS=crash; fi
env PYTHONNOUSERSITE=1 PYTHONPATH= "$PY" scripts/autoresearch_gauss_row.py --exp "$EXP" --status "$STATUS" --note "rc=$RC"
exit $RC
