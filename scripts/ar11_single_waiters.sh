#!/bin/bash
# The single-contrast arms at the final recipe and budget (declared 23:55): same model with both channels the
# same contrast, 3.6M points, lambda_geo 0.16, kappa_q 0.90, that contrast's wavelet strength, 4000 iters.
#   GPU 3 after ar9_P5_seed43            -> ar11_n36_singleT1_wav010_full4k    (wavelet 0.10)
#   GPU 2 after ar9_n36_wT1_010_wFL_007  -> ar11_n36_singleFLAIR_wav007_full4k (wavelet 0.07)
cd /data0/congcong/code/GS/mcgs_af16_wt
REC="--lambda_geo 0.16 --edge_tv_kappa_q 0.90 --set gaussian.initial_num_points=3600000"
gpu_free() { [ -z "$(nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader | grep $1)" ]; }
(
  until [ -f results_arc/ar9_P5_seed43_score.json ]; do sleep 30; done; sleep 60
  until gpu_free 1E:00; do sleep 30; done
  echo "[$(date +%H:%M)] GPU 3 free -> ar11_n36_singleT1_wav010_full4k" >> results_arc/ar6_waiters.log
  CONFIG=configs/t1flair_v3_af16_jyz_singleT1.yaml SAVE_VOLUMES=1 ./autoresearch_jyz_af16.sh ar11_n36_singleT1_wav010_full4k 3 4000 $REC --set loss.lambda_wav=0.10
) &
(
  until [ -f results_arc/ar9_n36_wT1_010_wFL_007_score.json ]; do sleep 30; done; sleep 60
  until gpu_free 1D:00; do sleep 30; done
  echo "[$(date +%H:%M)] GPU 2 free -> ar11_n36_singleFLAIR_wav007_full4k" >> results_arc/ar6_waiters.log
  CONFIG=configs/t1flair_v3_af16_jyz_singleFLAIR.yaml SAVE_VOLUMES=1 ./autoresearch_jyz_af16.sh ar11_n36_singleFLAIR_wav007_full4k 2 4000 $REC --set loss.lambda_wav=0.07
) &
wait
