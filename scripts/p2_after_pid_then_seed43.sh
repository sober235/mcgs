#!/bin/bash
# p2_after_pid_then_seed43.sh CARD PID: wait (kill -0) for PID — chain E's bash on CARD — then the seed-43 repeat of the
# promoted grid recipe on CARD.
CARD=${1:?card}; PID=${2:?pid}
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
cd /data0/congcong/code/GS/mcgs_af12_wt && bash scripts/p2_seed43_of_best.sh "$CARD"
