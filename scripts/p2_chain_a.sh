#!/bin/bash
# Phase-2 chain A (card = $1): the screening parent first (its iteration-1 loss is the bit-identity check of the
# per-contrast lambda_geo commit against ar9_P10_flair_af11_full4k), then round-1 knobs. Design B config is the
# runner's default. The runner refuses to overwrite, so relaunching skips finished experiments.
cd /data0/congcong/code/GS/mcgs_af12_wt
G=${1:?card}
./autoresearch_jyz_af12_p2.sh p2_P10_i1500 "$G" 1500
./autoresearch_jyz_af12_p2.sh p2_geo_T113_F007 "$G" 1500 --set loss.lambda_geo_plain=0.13 --set loss.lambda_geo_contrast=0.07
./autoresearch_jyz_af12_p2.sh p2_tvvec "$G" 1500 --tv_vector
./autoresearch_jyz_af12_p2.sh p2_kfa_off "$G" 1500 --set loss.kspace_freq_adaptive=false
./autoresearch_jyz_af12_p2.sh p2_vtv002 "$G" 1500 --lambda_vtv 0.02
