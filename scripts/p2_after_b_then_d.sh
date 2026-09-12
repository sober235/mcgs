#!/bin/bash
# Waits for chain B (pid 3363036, card 5) to exit, then runs chain D on card 5 as well: the runner refuses to
# overwrite, so the two copies of chain D (cards 3 and 5) share its four experiments without duplication.
while kill -0 3363036 2>/dev/null; do sleep 60; done
cd /data0/congcong/code/GS/mcgs_af12_wt && bash scripts/p2_chain_d.sh 5
