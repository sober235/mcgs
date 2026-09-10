#!/bin/bash
# Single-contrast arms at P6 (declared 01:20): v2 edge-TV (lambda_geo 0.10, kappa_q 0.85), 3.6M points, the
# contrast's own wavelet strength (T1 0.10, FLAIR 0.07), 4000 iterations, volumes -- behind the old-recipe singles.
cd /data0/congcong/code/GS/mcgs_af16_wt
REC="--lambda_geo 0.10 --edge_tv_kappa_q 0.85 --set gaussian.initial_num_points=3600000"
gpu_free() { [ -z "$(nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader | grep $1)" ]; }
(
  until [ -f results_arc/ar11_n36_singleT1_wav010_full4k_score.json ]; do sleep 30; done; sleep 60
  until gpu_free 1E:00; do sleep 30; done
  echo "[$(date +%H:%M)] GPU 3 free -> ar15_P6_singleT1_full4k" >> results_arc/ar6_waiters.log
  CONFIG=configs/t1flair_v3_af16_jyz_singleT1.yaml SAVE_VOLUMES=1 ./autoresearch_jyz_af16.sh ar15_P6_singleT1_full4k 3 4000 $REC --set loss.lambda_wav=0.10
) &
(
  until [ -f results_arc/ar11_n36_singleFLAIR_wav007_full4k_score.json ]; do sleep 30; done; sleep 60
  until gpu_free 1D:00; do sleep 30; done
  echo "[$(date +%H:%M)] GPU 2 free -> ar15_P6_singleFLAIR_full4k" >> results_arc/ar6_waiters.log
  CONFIG=configs/t1flair_v3_af16_jyz_singleFLAIR.yaml SAVE_VOLUMES=1 ./autoresearch_jyz_af16.sh ar15_P6_singleFLAIR_full4k 2 4000 $REC --set loss.lambda_wav=0.07
) &
wait
