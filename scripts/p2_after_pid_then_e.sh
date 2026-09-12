#!/bin/bash
# p2_after_pid_then_e.sh CARD PID: wait (kill -0 polling) for PID — the runner currently on CARD — then chain E on CARD.
CARD=${1:?card}; PID=${2:?pid}
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
cd /data0/congcong/code/GS/mcgs_af12_wt && bash scripts/p2_chain_e.sh "$CARD"
