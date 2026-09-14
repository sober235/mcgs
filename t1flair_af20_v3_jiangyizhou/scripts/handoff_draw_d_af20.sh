#!/bin/bash
# handoff_draw_d_af20.sh -- start MC-GS draw d on card 3 the moment draw c's trainer (pid given) exits there,
# so the freed card is not lost to another session's poller; if the card is taken in that gap, poll the
# candidates every 15 s. One trainer per card; placement confirmed as in requeue_mcgs_draws_af20.sh.
set -uo pipefail
C_PID=$1; CARD=3
J=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF20_v3_jiangyizhou
W=/data0/congcong/code/GS/mcgs_af16_wt
CARDS="3 5 0 2 4"
exp=af20_p3_joint_dd_full4k
log() { echo "[$(date +%H:%M)] handoff: $*" >> $J/logs/mask_draws_af20.log; }
source <(sed -n "/^card_idle()/,/^}/p" $J/scripts/run_mask_draws_af20.sh)
bus_of() { nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | awk -F', ' -v i="$1" '$1==i{print $2}'; }
while kill -0 $C_PID 2>/dev/null; do sleep 2; done
card=""
for t in $(seq 1 15); do card_idle $CARD && { card=$CARD; break; }; sleep 2; done   # the exiting process may hold the card a few seconds
while [ -z "$card" ]; do
  for g in $CARDS; do if card_idle $g; then card=$g; break; fi; done
  [ -z "$card" ] && sleep 15
done
log "mcgs draw d START card $card"
( cd $W && MASK_DRAW=d ./run_af20_mcgs.sh $exp $card 4000; rc=$?; echo "[$(date +%H:%M)] handoff: mcgs draw d rc=$rc" >> $J/logs/mask_draws_af20.log ) &
bus=$(bus_of $card); ok=0
for t in $(seq 1 60); do
  sleep 10
  pid=$(pgrep -f "^/home/congcongliu/anaconda3/envs/GSTMI/bin/python -u arc/cycle.py --exp $exp " | head -1)
  [ -n "$pid" ] || continue
  apps=$(nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader)
  if grep -q "^$pid, $bus" <<< "$apps"; then ok=1; break; fi
done
[ $ok = 1 ] && log "draw d confirmed on card $card (pid $pid)" || log "draw d NOT confirmed on card $card within 10 min"
wait
log "DONE"
