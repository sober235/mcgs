#!/bin/bash
# Waits (file-based) for round 3's p2_rot002_slr003_full4k to be scored, then runs round 4's second promotion,
# rotation lr 0.004 at 4000 iterations with volumes, on card $1. The runner's guard keeps one trainer per card.
cd /data0/congcong/code/GS/mcgs_af12_wt
G=${1:?card}
while [ ! -e results_arc/p2_rot002_slr003_full4k_score.json ]; do sleep 60; done
SAVE_VOLUMES=1 ./autoresearch_jyz_af12_p2g.sh p2_rotlr004_full4k "$G" 4000 --set gaussian.rotation_lr=0.004
