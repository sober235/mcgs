#!/bin/bash
# Card 5 after ar5_db4_lev1_C10_full4k: promotion run of scale lr 0.008 on P9 (4000 iterations, volumes).
cd /data0/congcong/code/GS/mcgs_af12_wt
until [ -f results_arc/ar5_db4_lev1_C10_full4k_score.json ] || grep -q -E 'Traceback|Killed' results_arc/ar5_db4_lev1_C10_full4k.log 2>/dev/null; do sleep 90; done
SAVE_VOLUMES=1 ./autoresearch_jyz_af12.sh ar5_slr008_full4k 5 4000 --set gaussian.scale_lr=0.008
