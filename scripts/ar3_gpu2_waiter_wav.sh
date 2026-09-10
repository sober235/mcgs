#!/bin/bash
# Round 3, seventh slot (GPU 2) -- REPLACES ar3_gpu2_waiter.sh (19:25): the hyw loop's wavelet-sparsity term
# (c2c68e9, taken as 63980bb) became available and is the highest-value test; the smoothing-family
# follow-up it displaced is low value (P1 already sits on that family). Launch P1 + lambda_wav 0.03
# (the hyw loop's best: 35.37 / 35.12 there) when ar2_lam020_g2 frees the card.
cd /data0/congcong/code/GS/mcgs_af16_wt
until [ -f results_arc/ar2_lam020_g2_score.json ]; do sleep 30; done
sleep 30
echo "[$(date +%H:%M)] ar2_lam020_g2 done -> launching ar3_P1_wav003" >> results_arc/ar3_gpu2_waiter.log
./autoresearch_jyz_af16.sh ar3_P1_wav003 2 1500 --lambda_geo 0.16 --edge_tv_kappa_q 0.90 --set loss.lambda_wav=0.03
