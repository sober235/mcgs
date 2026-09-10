#!/bin/bash
# Round 9 (declared 23:00): 3.6M points -- on hyw, 3.6M + wavelet lifted the second contrast by +0.26 over
# 2.4M with T1 held by a strong enough T1 wavelet; FLAIR binds here at -0.041 dB.
#  GPU 0 after ar8_P4_wT1_006 -> ar9_P5_n36   : P5 (wavelet 0.07 / 0.07) at 3.6M points (single delta: capacity)
#  GPU 2 after ar8_P4_wT1_010 and ar8_P4_wav010 -> ar9_n36_wT1_010_wFL_<x> : 3.6M, T1 wavelet 0.10,
#        FLAIR wavelet 0.10 if ar8_P4_wav010's FLAIR beat P5's 34.461, else 0.07
cd /data0/congcong/code/GS/mcgs_af16_wt
BASE="--lambda_geo 0.16 --edge_tv_kappa_q 0.90 --set gaussian.initial_num_points=3600000"
gpu_free() { [ -z "$(nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader | grep $1)" ]; }
(
  until [ -f results_arc/ar8_P4_wT1_006_score.json ]; do sleep 30; done; sleep 60
  until gpu_free 1B:00; do sleep 30; done
  echo "[$(date +%H:%M)] GPU 0 free -> ar9_P5_n36" >> results_arc/ar6_waiters.log
  ./autoresearch_jyz_af16.sh ar9_P5_n36 0 1500 $BASE --set loss.lambda_wav=0.07
) &
(
  until [ -f results_arc/ar8_P4_wT1_010_score.json ] && [ -f results_arc/ar8_P4_wav010_score.json ]; do sleep 30; done; sleep 60
  until gpu_free 1D:00; do sleep 30; done
  FL=$(env PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python -c "
import json; d = json.load(open('results_arc/ar8_P4_wav010_score.json')); print('0.10' if d['contrast']['head_psnr'] > 34.461 else '0.07')")
  echo "[$(date +%H:%M)] GPU 2 free -> ar9_n36_wT1_010_wFL_${FL/./} (FLAIR wavelet $FL)" >> results_arc/ar6_waiters.log
  ./autoresearch_jyz_af16.sh ar9_n36_wT1_010_wFL_${FL/./} 2 1500 $BASE --set loss.lambda_wav=$FL --set loss.lambda_wav_plain=0.10
) &
wait
