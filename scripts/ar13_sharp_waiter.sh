#!/bin/bash
# Declared 00:20: at the promoted recipe MC-GS joint beats L1-ESPIRiT on head PSNR but not on head SSIM /
# HFEN / sharpness. With the wavelet prior carrying part of the regularisation, less edge-TV may restore
# edges: the final recipe with the v2 edge-TV (lambda_geo 0.10, kappa_q 0.85), 1500 iterations, compared with
# ar9_n36_wT1_010_wFL_007 (same recipe with lambda_geo 0.16 / kappa_q 0.90). GPU 0 after the final promotion.
cd /data0/congcong/code/GS/mcgs_af16_wt
until [ -f results_arc/ar10_n36_wT1010_wFL007_full4k_score.json ]; do sleep 30; done; sleep 60
until [ -z "$(nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader | grep 1B:00)" ]; do sleep 30; done
echo "[$(date +%H:%M)] GPU 0 free -> ar13_n36_final_v2tv" >> results_arc/ar6_waiters.log
./autoresearch_jyz_af16.sh ar13_n36_final_v2tv 0 1500 --lambda_geo 0.10 --edge_tv_kappa_q 0.85 --set loss.lambda_wav=0.07 --set loss.lambda_wav_plain=0.10 --set gaussian.initial_num_points=3600000
