#!/bin/bash
# GPU 0's next run once ar5_P1w_dtv010 is scored: T1 binds on P3, so the T1-only wavelet lever --
# P3 with lambda_wav_plain 0.06 (FLAIR stays at 0.03), the hyw loop's per-contrast setting.
cd /data0/congcong/code/GS/mcgs_af16_wt
until [ -f results_arc/ar5_P1w_dtv010_score.json ]; do sleep 30; done
sleep 60
until [ -z "$(nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader | grep 1B:00)" ]; do sleep 30; done
echo "[$(date +%H:%M)] GPU 0 free -> ar6_P3_wavT1_006" >> results_arc/ar6_waiters.log
./autoresearch_jyz_af16.sh ar6_P3_wavT1_006 0 1500 --lambda_geo 0.16 --edge_tv_kappa_q 0.90 --set loss.lambda_wav=0.03 --set loss.lambda_wav_plain=0.06 --set gaussian.initial_num_points=2400000
