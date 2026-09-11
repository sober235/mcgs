#!/bin/bash
# Card 3: after ar4_kcenter05, the promotion run of the 4.8M-point keep-marginal (4000 iterations, volumes), then
# the round-2 warm-up knob that was queued on this card.
cd /data0/congcong/code/GS/mcgs_af12_wt
until [ -f results_arc/ar4_kcenter05_score.json ] || grep -q -E 'Traceback|Killed' results_arc/ar4_kcenter05.log 2>/dev/null; do sleep 90; done
SAVE_VOLUMES=1 ./autoresearch_jyz_af12.sh ar5_n48_full4k 3 4000 --set gaussian.initial_num_points=4800000
./autoresearch_jyz_af12.sh ar4_warmup1000 3 1500 --warmup 1000
