#!/bin/bash
# Card 3 after the 4.8M promotion (not promoted): the warm-up knob left from round 2, then the capacity curve at
# the promotion budget between 3.6M (P6) and 4.8M: 4.2M at 4000 iterations with volumes.
cd /data0/congcong/code/GS/mcgs_af12_wt
./autoresearch_jyz_af12.sh ar4_warmup1000 3 1500 --warmup 1000
SAVE_VOLUMES=1 ./autoresearch_jyz_af12.sh ar5_n42_full4k 3 4000 --set gaussian.initial_num_points=4200000
