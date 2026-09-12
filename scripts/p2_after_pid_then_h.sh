#!/bin/bash
# p2_after_pid_then_h.sh CARD PID: wait (kill -0) for PID — the runner on CARD — then chain H on CARD.
CARD=${1:?card}; PID=${2:?pid}
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
cd /data0/congcong/code/GS/mcgs_af12_wt && bash scripts/p2_chain_h.sh "$CARD"
