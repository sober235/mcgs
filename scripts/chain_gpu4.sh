#!/bin/bash
# Card 4 (free since 20:07): four round-1 knobs.
cd /data0/congcong/code/GS/mcgs_af12_wt
./autoresearch_jyz_af12.sh ar2_wav_up 4 1500 --set loss.lambda_wav=0.10 --set loss.lambda_wav_plain=0.14
./autoresearch_jyz_af12.sh ar2_geo013 4 1500 --lambda_geo 0.13
./autoresearch_jyz_af12.sh ar2_kq090 4 1500 --edge_tv_kappa_q 0.90
./autoresearch_jyz_af12.sh ar2_n48 4 1500 --set gaussian.initial_num_points=4800000
