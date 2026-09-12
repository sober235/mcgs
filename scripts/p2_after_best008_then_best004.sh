#!/bin/bash
# Waits for the 0.008 + scale lr promotion to be scored (file-based, no PID), then runs the 0.004 + scale lr promotion
# at 4000 iterations with volumes on card $1 (pre-registered 22:38: both stacked rows are within one floor at 1500,
# the better of the two at 4000 becomes P13). The runner's guard keeps one trainer per card.
cd /data0/congcong/code/GS/mcgs_af12_wt
G=${1:?card}
while [ ! -e results_arc/p2_best_rot008_slr005_full4k_score.json ]; do sleep 60; done
SAVE_VOLUMES=1 ./autoresearch_jyz_af12_p2g.sh p2_best_rot004_slr005_full4k "$G" 4000 --set gaussian.rotation_lr=0.004 --set gaussian.scale_lr=0.005
