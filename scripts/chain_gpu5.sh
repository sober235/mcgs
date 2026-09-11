#!/bin/bash
# Card 5 came free at 19:59 (clean card, ~4x the shared-card speed). Chain: the seed floor at the
# screening budget first (the gate every decision needs), then the round-1 knobs that were queued
# behind the slow chains on cards 6/7 (their waiters skip anything that already exists).
cd /data0/congcong/code/GS/mcgs_af12_wt
./autoresearch_jyz_af12.sh ar1_P6_i1500_s43 5 1500 --set training.seed=43
./autoresearch_jyz_af12.sh ar2_wav_down 5 1500 --set loss.lambda_wav=0.05 --set loss.lambda_wav_plain=0.07
./autoresearch_jyz_af12.sh ar2_geo007 5 1500 --lambda_geo 0.07
./autoresearch_jyz_af12.sh ar2_kq090 5 1500 --edge_tv_kappa_q 0.90
./autoresearch_jyz_af12.sh ar2_n48 5 1500 --set gaussian.initial_num_points=4800000
