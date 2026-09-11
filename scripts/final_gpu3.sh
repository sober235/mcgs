#!/bin/bash
# Card 3 after the sym4 promotion: P10 at 6000 iterations with volumes (the 1.5x-compute row, rule: final recipe only).
cd /data0/congcong/code/GS/mcgs_af12_wt
until [ -f results_arc/ar5_P10_sym4_full4k_score.json ] || grep -q -E 'Traceback|Killed' results_arc/ar5_P10_sym4_full4k.log 2>/dev/null; do sleep 90; done
SAVE_VOLUMES=1 ./autoresearch_jyz_af12.sh ar3_P10_i6000 3 6000
