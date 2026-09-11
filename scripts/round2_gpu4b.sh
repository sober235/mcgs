#!/bin/bash
# Card 4: after the round-2 chain (ends with ar4_wavC085), the next capacity point.
cd /data0/congcong/code/GS/mcgs_af12_wt
until [ -f results_arc/ar4_wavC085_score.json ] || grep -q -E 'Traceback|Killed' results_arc/ar4_wavC085.log 2>/dev/null; do sleep 90; done
./autoresearch_jyz_af12.sh ar2_n60 4 1500 --set gaussian.initial_num_points=6000000
