#!/bin/bash
# Seed repeat of the FINAL recipe at the promoted budget (declared 00:00): 3.6M points, T1 wavelet 0.10 /
# FLAIR 0.07, lambda_geo 0.16, kappa_q 0.90, 4000 iterations, seed 43 -- the noise floor under the final
# claim. GPU 1 after the 2.4M control ar10_wT1010_wFL007_full4k.
cd /data0/congcong/code/GS/mcgs_af16_wt
until [ -f results_arc/ar10_wT1010_wFL007_full4k_score.json ]; do sleep 30; done; sleep 60
until [ -z "$(nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader | grep 1C:00)" ]; do sleep 30; done
echo "[$(date +%H:%M)] GPU 1 free -> ar12_n36_final_seed43_full4k" >> results_arc/ar6_waiters.log
./autoresearch_jyz_af16.sh ar12_n36_final_seed43_full4k 1 4000 --lambda_geo 0.16 --edge_tv_kappa_q 0.90 --set loss.lambda_wav=0.07 --set loss.lambda_wav_plain=0.10 --set gaussian.initial_num_points=3600000 --set training.seed=43
