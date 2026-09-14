#!/bin/bash
# run_stage2_af24.sh -- the three learned / network arms that complete the nine-arm AF24 table:
# ZS-SSDU, 3D-DIP, 3D-INR, with exactly the AF16 topic's arguments, as at AF20 (read back from its logs:
# ZS-SSDU epochs 100; DIP lr 3e-3, 5000 it; INR lr 5e-3 default, 1500 it).
# One chain per card, cards chosen at launch:
#   CARD_ZT1: ZS-SSDU T1    -> 3D-INR T1
#   CARD_ZFL: ZS-SSDU FLAIR -> 3D-INR FLAIR
#   CARD_DIP: 3D-DIP T1     -> 3D-DIP FLAIR   (run_all_af24.sh puts this chain behind the joint-TV sweep on its card)
# A job starts when card_ok (scripts/card_ok_af24.sh) holds for its card and memory, and is skipped if its result json
# exists. Memory: ZS-SSDU ~15.8 GB and INR ~16.4 GB measured 2026-09-09; DIP never measured, 40 GB assumed.
set -uo pipefail
J=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF24_v3_jiangyizhou
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
export PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID
cd $J
source scripts/card_ok_af24.sh
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> logs/stage2_af24.log; }
run() {   # gpu need_gb contrast name cmd...
  local g=$1 need=$2 c=$3 n=$4; shift 4
  [ -f logs/${n}_${c}.json ] && { log "$n $c exists, skipped"; return; }
  until card_ok $g $need; do sleep 60; done
  log "$n $c START gpu $g"
  $PY -u "$@" --contrast $c --gpu $g > logs/${n}_${c}.log 2>&1
  local rc=$?
  log "$n $c rc=$rc"
}
chain_zt1() { run $1 18 T1    zsssdu scripts/recon_zsssdu.py --epochs 100
              run $1 18 T1    inr    scripts/recon_dip_inr.py --method inr --iters 1500 --eval_every 50
              log "CHAIN_ZT1 gpu $1 DONE"; }
chain_zfl() { run $1 18 FLAIR zsssdu scripts/recon_zsssdu.py --epochs 100
              run $1 18 FLAIR inr    scripts/recon_dip_inr.py --method inr --iters 1500 --eval_every 50
              log "CHAIN_ZFL gpu $1 DONE"; }
chain_dip() { run $1 40 T1    dip    scripts/recon_dip_inr.py --method dip --iters 5000 --lr 3e-3 --eval_every 100
              run $1 40 FLAIR dip    scripts/recon_dip_inr.py --method dip --iters 5000 --lr 3e-3 --eval_every 100
              log "CHAIN_DIP gpu $1 DONE"; }
# sourced by run_all_af24.sh for the chain functions; run directly, all three chains start now
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  : "${CARD_ZT1:?}" "${CARD_ZFL:?}" "${CARD_DIP:?}"
  log "armed: gpu $CARD_ZT1 ZS-SSDU T1 -> INR T1 | gpu $CARD_ZFL ZS-SSDU FLAIR -> INR FLAIR | gpu $CARD_DIP DIP T1 -> DIP FLAIR"
  chain_zt1 $CARD_ZT1 & chain_zfl $CARD_ZFL & chain_dip $CARD_DIP &
  wait
  log "ALL_STAGE2_AF24_DONE"
fi
