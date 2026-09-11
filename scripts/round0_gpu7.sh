#!/bin/bash
# Round 0 on card 7 (shared with another user's light job): the parent P6 at the screening budget
# (1500 iterations, seed 42), then the same with seed 43 = the seed floor at that budget.
cd /data0/congcong/code/GS/mcgs_af12_wt
./autoresearch_jyz_af12.sh ar1_P6_i1500 7 1500
./autoresearch_jyz_af12.sh ar1_P6_i1500_s43 7 1500 --set training.seed=43
