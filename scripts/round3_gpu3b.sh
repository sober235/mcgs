#!/bin/bash
# Card 3 after ar5_n42_full4k: the warm-up knob at a value the guard allows (warmup < 0.5 * iters).
cd /data0/congcong/code/GS/mcgs_af12_wt
until [ -f results_arc/ar5_n42_full4k_score.json ] || grep -q -E 'Traceback|Killed' results_arc/ar5_n42_full4k.log 2>/dev/null; do sleep 90; done
./autoresearch_jyz_af12.sh ar4_warmup700 3 1500 --warmup 700
