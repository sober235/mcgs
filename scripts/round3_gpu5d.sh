#!/bin/bash
# Card 5 after ar4_kmid08: the seed floor on P8 (the runner now bakes db4 + 2 levels).
cd /data0/congcong/code/GS/mcgs_af12_wt
until [ -f results_arc/ar4_kmid08_score.json ] || grep -q -E 'Traceback|Killed' results_arc/ar4_kmid08.log 2>/dev/null; do sleep 90; done
./autoresearch_jyz_af12.sh ar1_P8_i1500_s43 5 1500 --set training.seed=43
