#!/usr/bin/env bash
# run_vdpois_classical.sh GPU -- the classical arms of the mask-swap comparison (audit 2026-09-29) on ONE GPU, in series:
#   ZF + CG-SENSE (pre, post), then CS-SENSE ADMM-TV at the registered protocol (--admm_outer 200) over the same
#   seven-point lambda grid the registered line swept, both scans. Every call is the registered recon_classical.py
#   invocation with RETRO_TOPIC pointed at the vdPoisMex topic; summarize.py later keeps the best lambda per scan
#   exactly as it did for the BART-mask line. Log: $RETRO_TOPIC/logs/classical_chain.log (this script appends).
set -uo pipefail
if [[ $# -ne 1 ]]; then echo "usage: $0 GPU" >&2; exit 2; fi
GPU=$1
export RETRO_TOPIC=${RETRO_TOPIC:-/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois}
export RETRO_TRAINER=${RETRO_TRAINER:-$RETRO_TOPIC/trainer/_trainer_e778bf2}
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
S=$(cd "$(dirname "$0")" && pwd)
E=(env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID)
LOG=$RETRO_TOPIC/logs/classical_chain.log
mkdir -p "$RETRO_TOPIC/logs" "$RETRO_TOPIC/recon"
run() { echo "[$(date '+%F %T')] $*" >> "$LOG"; "${E[@]}" "$@" >> "$LOG" 2>&1; echo "[$(date '+%F %T')] rc=$?" >> "$LOG"; }
for s in pre post; do
  run "$PY" "$S/recon_classical.py" --scan "$s" --methods zf,cgsense --gpu "$GPU"
done
for lam in 0.001 0.0007 0.0014 0.0005 0.002 0.00035 0.00025; do
  for s in pre post; do
    run "$PY" "$S/recon_classical.py" --scan "$s" --methods cssense --admm_outer 200 --lam "$lam" --suffix "_o200_lam$lam" --gpu "$GPU"
  done
done
echo "[$(date '+%F %T')] CLASSICAL_CHAIN_DONE" >> "$LOG"
