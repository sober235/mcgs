#!/usr/bin/env bash
# run_gs.sh EXP GPU ITERS [extra arc/cycle.py args]
# The jyz AF16 phase-3 frozen recipe on the pinned trainer snapshot (e778bf2). SINGLE=pre|post selects the
# single-scan config (both channels = that scan, the GS-single arm); unset = the joint config (MC-GS).
# WARMUP=20 for smoke runs. Outputs: trainer/_trainer_e778bf2/results_arc/EXP{.log,_score.json,_volumes.npz}.
set -uo pipefail
if [[ $# -lt 3 ]]; then echo "usage: $0 EXP GPU ITERS [extra cycle.py args...]" >&2; exit 2; fi
EXP=$1; GPU=$2; ITERS=$3; shift 3
T=${RETRO_TOPIC:-/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16}
TR=${RETRO_TRAINER:-$T/trainer/_trainer_e778bf2}   # RETRO_TRAINER: an alternative trainer copy (2026-10-02 autoresearch: _trainer_e778bf2_ar)
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
WARMUP=${WARMUP:-500}
case "${SINGLE:-joint}" in
  joint) CFG=$T/configs/ce_retro_af16.yaml ;;
  pre)   CFG=$T/configs/ce_retro_af16_single_pre.yaml ;;
  post)  CFG=$T/configs/ce_retro_af16_single_post.yaml ;;
  *)     echo "SINGLE must be pre, post or unset: ${SINGLE}" >&2; exit 2 ;;
esac
mkdir -p "$TR/results_arc"
LOG=$TR/results_arc/$EXP.log
for p in "$LOG" "$TR/results_arc/${EXP}_score.json" "$TR/results_arc/$EXP"; do
  if [[ -e "$p" ]]; then echo "refusing to overwrite existing output: $p" >&2; exit 1; fi
done
cd "$TR"
# CUDA_VISIBLE_DEVICES is set by cycle.py from --gpu anyway; exporting the same value here makes the claim on the card
# visible in /proc/<pid>/environ from the first second, so other schedulers (scripts/ar_free_queue.py) skip it (2026-10-10)
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
  "$PY" -u arc/cycle.py --exp "$EXP" --gpu "$GPU" --iters "$ITERS" \
  --lambda_geo 0.10 --warmup "$WARMUP" --max_radius 20 --kspace_high 2.2 \
  --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 \
  --set gaussian.init_mode=head --set gaussian.init_scale_mode=3nn \
  --set loss.edge_tv_kappa_source=zf \
  --set gaussian.initial_num_points=3600000 \
  --set loss.wav_name=db4 --set loss.wav_levels=1 \
  --set loss.lambda_wav_plain=0.12 --set loss.lambda_wav=0.10 \
  --lambda_edge 0.001 --lambda_ptv 0.03 \
  --base_cfg "$CFG" --save_volumes "$@" > "$LOG" 2>&1
RC=$?
echo "rc=$RC log=$LOG"
exit $RC
