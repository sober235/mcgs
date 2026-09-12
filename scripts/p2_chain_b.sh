#!/bin/bash
# Phase-2 chain B (card = $1): the seed floor at the screening budget first, then round-1 knobs.
cd /data0/congcong/code/GS/mcgs_af12_wt
G=${1:?card}
./autoresearch_jyz_af12_p2.sh p2_P10_i1500_s43 "$G" 1500 --set training.seed=43
./autoresearch_jyz_af12_p2.sh p2_geo_T107_F013 "$G" 1500 --set loss.lambda_geo_plain=0.07 --set loss.lambda_geo_contrast=0.13
./autoresearch_jyz_af12_p2.sh p2_gradnorm10 "$G" 1500 --set training.max_grad_norm=10.0
./autoresearch_jyz_af12_p2.sh p2_rotlr "$G" 1500 --set gaussian.rotation_lr=0.001
./autoresearch_jyz_af12_p2.sh p2_edge002 "$G" 1500 --lambda_edge 0.02
