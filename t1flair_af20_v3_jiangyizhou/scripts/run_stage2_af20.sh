#!/bin/bash
# run_stage2_af20.sh -- the three learned / network arms that complete the nine-arm AF20 table:
# ZS-SSDU, 3D-DIP, 3D-INR, with exactly the AF16 topic's arguments (read back from its logs:
# ZS-SSDU epochs 100; DIP lr 3e-3, 5000 it; INR lr 5e-3 default, 1500 it).
# Two chains, one per card, each starting only when that card's MC-GS run has written its score:
#   card 6 after MC-GS joint        -> T1:    ZS-SSDU, DIP, INR
#   card 5 after GS-single FLAIR    -> FLAIR: ZS-SSDU, DIP, INR
# One trainer per card at all times. Each job is skipped if its result json already exists.
set -uo pipefail
J=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF20_v3_jiangyizhou
ARC=/data0/congcong/code/GS/mcgs_af16_wt/results_arc
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
export PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID
cd $J
log() { echo "[$(date +%H:%M)] $*" >> logs/stage2_af20.log; }
run() {   # gpu contrast name cmd...
  local g=$1 c=$2 n=$3; shift 3
  [ -f logs/${n}_${c}.json ] && { log "$n $c exists, skipped"; return; }
  log "$n $c START gpu $g"
  $PY -u "$@" --contrast $c --gpu $g > logs/${n}_${c}.log 2>&1
  log "$n $c rc=$?"
}
chain() { # gpu contrast wait_for_score_json
  local g=$1 c=$2 w=$3
  until [ -f "$w" ]; do sleep 120; done
  sleep 90                                   # let the MC-GS process release the card
  run $g $c zsssdu scripts/recon_zsssdu.py --epochs 100
  run $g $c dip    scripts/recon_dip_inr.py --method dip --iters 5000 --lr 3e-3 --eval_every 100
  run $g $c inr    scripts/recon_dip_inr.py --method inr --iters 1500 --eval_every 50
  log "CHAIN_$c DONE"
}
log "armed: T1 on card 6 after af20_p3_joint_full4k, FLAIR on card 5 after af20_p3_singleFLAIR_full4k"
chain 6 T1    $ARC/af20_p3_joint_full4k_score.json &
chain 5 FLAIR $ARC/af20_p3_singleFLAIR_full4k_score.json &
wait
log "ALL_STAGE2_AF20_DONE"
