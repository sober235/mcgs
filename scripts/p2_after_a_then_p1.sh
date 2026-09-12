#!/bin/bash
# Waits for chain A (pid 3362907, card 3) to exit, then promotion 1 (rotation lr 0.001 at 4000) and chain D on card 3.
while kill -0 3362907 2>/dev/null; do sleep 60; done
cd /data0/congcong/code/GS/mcgs_af12_wt && bash scripts/p2_chain_p1.sh 3
