#!/bin/bash
# run_classical_af24.sh -- every classical arm of the AF24 study, with lambda swept ON THIS SUBJECT
# against the ground truth (the same privilege the MC-GS recipe had at AF16); summarize.py keeps the
# best lambda per contrast. Grids = the AF20 topic's final grids (its edge extensions included) plus one point
# below each, because the one optimum that moved from AF16 to AF20 moved down (L1-ESPIRiT T1 0.0015 -> 0.001):
#     converged TV (cssense, 200 outer ADMM)  AF20 best: T1 0.0007, FLAIR 0.0005
#     L1-wavelet ESPIRiT (BART pics, CPU)     AF20 best: T1 0.001,  FLAIR 0.003
#     joint vectorial TV (200 outer)          AF20 best: T1 0.001 / FLAIR 0.0007 (one run scores both)
# If an optimum lands on an edge of its grid the grid must be extended, exactly as at AF16 and AF20.
# Cards: GPU_TV / GPU_JTV / GPU_L1 below. BART is CPU-only; its --gpu is used for the scoring pass.
set -uo pipefail
J=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF24_v3_jiangyizhou
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
export PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID
GPU_BASE=${GPU_BASE:-7}; GPU_TV=${GPU_TV:-7}; GPU_JTV=${GPU_JTV:-5}; GPU_L1=${GPU_L1:-7}
THREADS=${THREADS:-12}
cd $J
log() { echo "[$(date +%H:%M)] $*" >> logs/classical_af24.log; }

base() { local g=$1 c=$2; [ -f logs/classical_${c}.json ] && return
  $PY -u scripts/recon_classical.py --contrast $c --gpu $g > logs/classical_${c}.log 2>&1
  log "base $c rc=$?"; }
tv()   { local g=$1 c=$2 L=$3; [ -f logs/classical_${c}_o200_lam${L}.json ] && return
  $PY -u scripts/recon_classical.py --contrast $c --gpu $g --methods cssense --admm_outer 200 --lam $L --suffix _o200_lam$L > logs/classical_${c}_o200_lam${L}.log 2>&1
  log "tv $c lam=$L rc=$?"; }
l1()   { local g=$1 c=$2 L=$3; [ -f logs/l1espirit_${c}_lam${L}.json ] && return
  $PY -u scripts/recon_l1espirit.py --topic $J --contrast $c --gpu $g --lam $L --iters 100 --suffix _lam$L --threads $THREADS > logs/l1espirit_${c}_lam${L}.log 2>&1
  log "l1 $c lam=$L rc=$?"; }
jtv()  { local g=$1 L=$2; [ -f logs/jcssense_o200_lam${L}.json ] && return
  $PY -u scripts/recon_joint_cssense.py --topic $J --contrasts T1,FLAIR --gpu $g --admm_outer 200 --lam $L --suffix _o200_lam$L > logs/jcssense_o200_lam${L}.log 2>&1
  log "jtv lam=$L rc=$?"; }

log "START (tv gpu $GPU_TV, jtv gpu $GPU_JTV, l1 gpu $GPU_L1, bart threads $THREADS)"
( base $GPU_BASE T1; base $GPU_BASE FLAIR
  for L in 0.00035 0.0005 0.0007 0.001 0.0014; do tv $GPU_TV T1 $L; done
  for L in 0.00025 0.00035 0.0005 0.0007 0.001; do tv $GPU_TV FLAIR $L; done
  log "CHAIN_TV_DONE" ) &
( for L in 0.00035 0.0005 0.0007 0.001 0.0014; do jtv $GPU_JTV $L; done; log "CHAIN_JTV_DONE" ) &
( for L in 0.00035 0.0005 0.0007;       do l1 $GPU_L1 T1 $L; done;    log "CHAIN_L1_T1a_DONE" ) &
( for L in 0.001 0.0015 0.0025 0.004;   do l1 $GPU_L1 T1 $L; done;    log "CHAIN_L1_T1b_DONE" ) &
( for L in 0.0014 0.002 0.003;          do l1 $GPU_L1 FLAIR $L; done; log "CHAIN_L1_FLa_DONE" ) &
( for L in 0.005 0.007;                 do l1 $GPU_L1 FLAIR $L; done; log "CHAIN_L1_FLb_DONE" ) &
wait
log "ALL_CLASSICAL_AF24_DONE"
