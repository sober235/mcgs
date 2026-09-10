#!/bin/bash
# GPU 3 after ar8_P4_wav010 (declared 23:05): seed repeat of P5 (wavelet 0.07 / 0.07, 2.4M). The only floor so
# far (0.01 dB) was measured at 1.2M without the wavelet term, whose cycle spinning adds randomness; any
# "beats L1-ESPIRiT by a few hundredths" needs the floor at this recipe.
cd /data0/congcong/code/GS/mcgs_af16_wt
until [ -f results_arc/ar8_P4_wav010_score.json ]; do sleep 30; done; sleep 60
until [ -z "$(nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader | grep 1E:00)" ]; do sleep 30; done
echo "[$(date +%H:%M)] GPU 3 free -> ar9_P5_seed43" >> results_arc/ar6_waiters.log
./autoresearch_jyz_af16.sh ar9_P5_seed43 3 1500 --lambda_geo 0.16 --edge_tv_kappa_q 0.90 --set loss.lambda_wav=0.07 --set gaussian.initial_num_points=2400000 --set training.seed=43
