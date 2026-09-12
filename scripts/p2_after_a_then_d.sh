#!/bin/bash
# Waits for chain A (pid 3362907, card 3) to exit, then runs chain D (round 2 part 1) on card 3.
# PID looked up once at launch, then kill -0 polling (never pgrep -f from the tool shell).
while kill -0 3362907 2>/dev/null; do sleep 60; done
cd /data0/congcong/code/GS/mcgs_af12_wt && bash scripts/p2_chain_d.sh 3
