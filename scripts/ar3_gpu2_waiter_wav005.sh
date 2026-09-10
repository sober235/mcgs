#!/bin/bash
# GPU 2 slot, third version (19:35): P1 + lambda_wav 0.03 now starts on GPU 0 as soon as Round 3 frees it
# (scripts/launch_r4_slots.sh), about 40 minutes before ar2_lam020_g2 frees GPU 2; so GPU 2 takes the
# lowest-priority Round 4 run instead: P1 + lambda_wav 0.05 (the hyw loop's "keep-T1" point).
cd /data0/congcong/code/GS/mcgs_af16_wt
until [ -f results_arc/ar2_lam020_g2_score.json ]; do sleep 30; done
sleep 30
echo "[$(date +%H:%M)] ar2_lam020_g2 done -> ar4_P1_wav005 on gpu2" >> results_arc/ar3_gpu2_waiter.log
./autoresearch_jyz_af16.sh ar4_P1_wav005 2 1500 --lambda_geo 0.16 --edge_tv_kappa_q 0.90 --set loss.lambda_wav=0.05
