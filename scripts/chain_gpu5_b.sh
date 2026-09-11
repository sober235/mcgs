#!/bin/bash
# Continuation of chain_gpu5.sh after lending card 5 to the AF16 phase-2 loop for one 30-minute promotion
# (agreed 20:05). Waits for the go flag (written by this session when the peer reports the card free), then
# runs the round-1 knobs on card 5.
cd /data0/congcong/code/GS/mcgs_af12_wt
FLAG=/data0/congcong/code/GS/mcgs_af12_wt/results_arc/.gpu5_go
until [ -f "$FLAG" ]; do sleep 60; done
./autoresearch_jyz_af12.sh ar2_wav_down 5 1500 --set loss.lambda_wav=0.05 --set loss.lambda_wav_plain=0.07
./autoresearch_jyz_af12.sh ar2_geo007 5 1500 --lambda_geo 0.07
./autoresearch_jyz_af12.sh ar2_kq090 5 1500 --edge_tv_kappa_q 0.90
./autoresearch_jyz_af12.sh ar2_n48 5 1500 --set gaussian.initial_num_points=4800000
