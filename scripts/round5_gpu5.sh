#!/bin/bash
# Card 5 (idle after the P9 + scale-lr-0.008 promotion): P10 + scale lr 0.005 at 4000 iterations with volumes — the
# intermediate point in case 0.008 over-fits T1 at the full budget on P10 as it did on P9.
cd /data0/congcong/code/GS/mcgs_af12_wt
SAVE_VOLUMES=1 ./autoresearch_jyz_af12.sh ar5_P10_slr005_full4k 5 4000 --set gaussian.scale_lr=0.005
