#!/bin/bash
# run_mask_draws_af20.sh -- mask-draw variance of the AF20 margin (2026-09-14).
# Three more draws (b, c, d) of the same AF20 vd-Poisson call; per draw only the two arms that define the margin:
#   MC-GS joint, the frozen AF16 phase-3 recipe (run_af20_mcgs.sh with MASK_DRAW=<d>), 4000 it, no volumes
#   L1-ESPIRiT at the AF20 optimum and both neighbours of the main sweep: T1 0.0007/0.001/0.0015, FLAIR 0.002/0.003/0.005
# The registered results (draw a) are not touched: L1 outputs carry the suffix _d<d>_lam<L> (outside summarize.py's
# glob), MC-GS runs are named af20_p3_joint_d<d>_full4k. The L1 mask is swapped through T1T2_MASK_<CONTRAST>.
# MC-GS: one trainer per card, each started only on a card with no compute process at all (candidates CARDS).
set -uo pipefail
J=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF20_v3_jiangyizhou
W=/data0/congcong/code/GS/mcgs_af16_wt
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
export PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID
CARDS=${CARDS:-"5 0 2 3 4"}; GPU_L1=${GPU_L1:-7}; THREADS=${THREADS:-12}
cd $J
log() { echo "[$(date +%H:%M)] $*" >> logs/mask_draws_af20.log; }
card_idle() {
  local bus
  bus=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | awk -F', ' -v i="$1" '$1==i{print $2}')
  ! nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader | grep -q "$bus"
}
l1() {  # draw contrast lambda
  local d=$1 c=$2 L=$3
  [ -f logs/l1espirit_${c}_d${d}_lam${L}.json ] && return
  env T1T2_MASK_${c}=$J/data/mask_af20_d${d}_3d.h5 \
    $PY -u scripts/recon_l1espirit.py --topic $J --contrast $c --gpu $GPU_L1 --lam $L --iters 100 \
        --suffix _d${d}_lam$L --threads $THREADS > logs/l1espirit_${c}_d${d}_lam${L}.log 2>&1
  log "l1 draw $d $c lam=$L rc=$?"
}
log "START (MC-GS candidate cards: $CARDS; L1 scoring card $GPU_L1; BART threads $THREADS)"
# three CPU chains, one per draw (the machine's load was already ~70 of 112 cores)
for d in b c d; do
  ( for L in 0.0007 0.001 0.0015; do l1 $d T1 $L; done
    for L in 0.002 0.003 0.005;   do l1 $d FLAIR $L; done
    log "L1_d${d}_DONE" ) &
done
(
  for d in b c d; do
    exp=af20_p3_joint_d${d}_full4k
    [ -f $W/results_arc/${exp}_score.json ] && { log "mcgs draw $d exists, skipped"; continue; }
    card=""
    while [ -z "$card" ]; do
      for g in $CARDS; do
        if card_idle $g; then card=$g; break; fi
      done
      [ -z "$card" ] && sleep 120
    done
    log "mcgs draw $d START card $card"
    ( cd $W && MASK_DRAW=$d ./run_af20_mcgs.sh $exp $card 4000; echo "[$(date +%H:%M)] mcgs draw $d rc=$?" >> $J/logs/mask_draws_af20.log ) &
    sleep 300        # the trainer holds its card (and so shows as busy) well within five minutes of its start
  done
  wait
  log "MCGS_DRAWS_DONE"
) &
wait
log "ALL_MASK_DRAWS_DONE"
