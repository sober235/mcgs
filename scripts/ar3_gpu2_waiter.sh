#!/bin/bash
# Round 3, seventh slot (GPU 2): wait for ar2_lam020_g2, then launch the pre-declared follow-up.
# Rule (program doc, round 3): if lambda 0.20 beats lambda 0.16 (ar1_lam016: 34.611 / 33.771) on BOTH
# contrasts, run lambda 0.20 + kappa_q 0.90 (a stronger P1); otherwise run P1 + w_floor 0.20.
cd /data0/congcong/code/GS/mcgs_af16_wt
until [ -f results_arc/ar2_lam020_g2_score.json ]; do sleep 30; done
sleep 30
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
CHOICE=$(env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -c "
import json
d = json.load(open('results_arc/ar2_lam020_g2_score.json'))
t1, fl = d['plain']['head_psnr'], d['contrast']['head_psnr']
print('lam020' if (t1 > 34.611 and fl > 33.771) else 'wfloor')")
echo "[$(date +%H:%M)] ar2_lam020_g2 done -> choice=$CHOICE" >> results_arc/ar3_gpu2_waiter.log
if [ "$CHOICE" = "lam020" ]; then
  ./autoresearch_jyz_af16.sh ar3_P1b_lam020_kq090 2 1500 --lambda_geo 0.20 --edge_tv_kappa_q 0.90
else
  ./autoresearch_jyz_af16.sh ar3_P1_wfloor020 2 1500 --lambda_geo 0.16 --edge_tv_kappa_q 0.90 --edge_tv_wfloor 0.20
fi
