#!/bin/bash
# Card 3 (idle after the P10 promotion): P10 + scale lr 0.008 at the promotion budget (4000 iterations, volumes) —
# the union of the parent with the remaining positive lever, run in parallel with its 1500-iteration screen on card 4.
cd /data0/congcong/code/GS/mcgs_af12_wt
SAVE_VOLUMES=1 ./autoresearch_jyz_af12.sh ar5_P10_slr008_full4k 3 4000 --set gaussian.scale_lr=0.008
