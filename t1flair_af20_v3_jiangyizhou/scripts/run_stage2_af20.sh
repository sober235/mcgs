#!/bin/bash
# run_stage2_af20.sh -- the three learned / network arms that complete the nine-arm AF20 table:
# ZS-SSDU, 3D-DIP, 3D-INR, with exactly the AF16 topic's arguments (read back from its logs:
# ZS-SSDU epochs 100; DIP lr 3e-3, 5000 it; INR lr 5e-3 default, 1500 it).
#
# v2 (22:40): the first placement (behind the MC-GS runs on cards 6 and 5) ran ZS-SSDU T1 at ~530 s per
# epoch, ten times the AF16 rate, next to another user's long job; cards 0, 2 and 3 had meanwhile gone
# completely idle. Placement now, one chain per card, one trainer per card:
#   card 0: ZS-SSDU T1    -> 3D-INR T1
#   card 2: ZS-SSDU FLAIR -> 3D-INR FLAIR
#   card 3: 3D-DIP T1     -> 3D-DIP FLAIR
# Every job waits until its card has NO compute process at all before it starts (the rule the other
# sessions' schedulers follow too), and is skipped if its result json already exists.
set -uo pipefail
J=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF20_v3_jiangyizhou
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
export PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID
cd $J
log() { echo "[$(date +%H:%M)] $*" >> logs/stage2_af20.log; }
card_idle() {   # true only when the card with this index has no compute process
  # The listing goes into a variable first: under pipefail, `nvidia-smi | grep -q` fails whenever grep exits before
  # nvidia-smi has written everything (SIGPIPE), and the negation then reported busy cards as idle (2026-09-14).
  local bus apps
  bus=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | awk -F', ' -v i="$1" '$1==i{print $2}')
  [ -n "$bus" ] || return 1
  apps=$(nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader) || return 1
  ! grep -qF "$bus" <<< "$apps"
}
run() {   # gpu contrast name cmd...
  local g=$1 c=$2 n=$3; shift 3
  [ -f logs/${n}_${c}.json ] && { log "$n $c exists, skipped"; return; }
  until card_idle $g; do sleep 60; done
  log "$n $c START gpu $g"
  $PY -u "$@" --contrast $c --gpu $g > logs/${n}_${c}.log 2>&1
  log "$n $c rc=$?"
}
log "v2 armed: card 0 ZS-SSDU T1 -> INR T1 | card 2 ZS-SSDU FLAIR -> INR FLAIR | card 3 DIP T1 -> DIP FLAIR"
( run 0 T1    zsssdu scripts/recon_zsssdu.py --epochs 100
  run 0 T1    inr    scripts/recon_dip_inr.py --method inr --iters 1500 --eval_every 50
  log "CHAIN_card0 DONE" ) &
( run 2 FLAIR zsssdu scripts/recon_zsssdu.py --epochs 100
  run 2 FLAIR inr    scripts/recon_dip_inr.py --method inr --iters 1500 --eval_every 50
  log "CHAIN_card2 DONE" ) &
( run 3 T1    dip    scripts/recon_dip_inr.py --method dip --iters 5000 --lr 3e-3 --eval_every 100
  run 3 FLAIR dip    scripts/recon_dip_inr.py --method dip --iters 5000 --lr 3e-3 --eval_every 100
  log "CHAIN_card3 DONE" ) &
wait
log "ALL_STAGE2_AF20_DONE"
