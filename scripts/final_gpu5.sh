#!/bin/bash
# Card 5: P10 seed 43 at 4000 iterations with volumes (the 4000-iteration floor and the second seed of the 2x row).
cd /data0/congcong/code/GS/mcgs_af12_wt
SAVE_VOLUMES=1 ./autoresearch_jyz_af12.sh ar5_P10_s43_full4k 5 4000 --set training.seed=43
