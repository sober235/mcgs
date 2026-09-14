#!/bin/bash
# requeue_mcgs_draws_af20.sh DRAW... -- run the MC-GS joint arm of the given AF20 mask draws, one trainer per card.
# Written after the first queue put draws b, c and d all on card 5 (card_idle misread under pipefail, see
# run_mask_draws_af20.sh). Each draw starts only on a candidate card with no compute process, and the next draw
# is not placed until this one's trainer is confirmed on the card it was given.
set -uo pipefail
J=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF20_v3_jiangyizhou
W=/data0/congcong/code/GS/mcgs_af16_wt
CARDS=${CARDS:-"5 0 2 3 4"}
log() { echo "[$(date +%H:%M)] $*" >> $J/logs/mask_draws_af20.log; }
source <(sed -n "/^card_idle()/,/^}/p" $J/scripts/run_mask_draws_af20.sh)
bus_of() { nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | awk -F', ' -v i="$1" '$1==i{print $2}'; }
for d in "$@"; do
  exp=af20_p3_joint_d${d}_full4k
  [ -f $W/results_arc/${exp}_score.json ] && { log "requeue: draw $d already scored, skipped"; continue; }
  card=""
  while [ -z "$card" ]; do
    for g in $CARDS; do if card_idle $g; then card=$g; break; fi; done
    [ -z "$card" ] && sleep 120
  done
  log "requeue: mcgs draw $d START card $card"
  ( cd $W && MASK_DRAW=$d ./run_af20_mcgs.sh $exp $card 4000; rc=$?; echo "[$(date +%H:%M)] requeue: mcgs draw $d rc=$rc" >> $J/logs/mask_draws_af20.log ) &
  bus=$(bus_of $card); ok=0
  for t in $(seq 1 60); do                       # up to 10 min for the trainer to take its card
    sleep 10
    pid=$(pgrep -f "^/home/congcongliu/anaconda3/envs/GSTMI/bin/python -u arc/cycle.py --exp $exp " | head -1)
    [ -n "$pid" ] || continue
    if nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader > /tmp/.q_apps_$$ && grep -q "^$pid, $bus" /tmp/.q_apps_$$; then ok=1; break; fi
  done
  rm -f /tmp/.q_apps_$$
  [ $ok = 1 ] && log "requeue: draw $d confirmed on card $card (pid $pid)" || { log "requeue: draw $d NOT confirmed on card $card within 10 min - stopping the queue"; exit 1; }
done
wait
log "REQUEUE_DONE $*"
