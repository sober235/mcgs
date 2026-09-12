#!/bin/bash
# Waits for chain C (pid 3363135, card 2) to exit, then runs chain D on card 2 too (three-way work sharing).
while kill -0 3363135 2>/dev/null; do sleep 60; done
cd /data0/congcong/code/GS/mcgs_af12_wt && bash scripts/p2_chain_d.sh 2
