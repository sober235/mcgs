#!/bin/bash
# GPU 2's next run once ar5_P2_dtv020 is scored (pre-declared 22:00). FLAIR's own lever, lambda_dtv, was
# tested on P2 (lambda_dtv 0.10 on GPU 0, 0.20 on GPU 2). If either lifts FLAIR over P2 (33.924) by more
# than 0.05 dB without costing T1 more than 0.05 dB, move the better one onto P3; otherwise give the
# card to the T1 side: P3 with lambda_wav_plain 0.10 (the stronger T1-only wavelet).
cd /data0/congcong/code/GS/mcgs_af16_wt
until [ -f results_arc/ar5_P2_dtv020_score.json ] && [ -f results_arc/ar5_P1w_dtv010_score.json ]; do sleep 30; done
sleep 60
until [ -z "$(nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader | grep 1D:00)" ]; do sleep 30; done
DTV=$(env PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python -c "
import json
best = None
for lam, e in ((0.10, 'ar5_P1w_dtv010'), (0.20, 'ar5_P2_dtv020')):
    d = json.load(open(f'results_arc/{e}_score.json'))
    t1, fl = d['plain']['head_psnr'], d['contrast']['head_psnr']
    if fl - 33.924 > 0.05 and t1 - 34.975 > -0.05 and (best is None or fl > best[1]):
        best = (lam, fl)
print(best[0] if best else 'none')")
P3="--lambda_geo 0.16 --edge_tv_kappa_q 0.90 --set loss.lambda_wav=0.03 --set gaussian.initial_num_points=2400000"
if [ "$DTV" != "none" ]; then
  echo "[$(date +%H:%M)] dtv pays on P2 (lambda_dtv $DTV) -> ar6_P3_dtv on GPU 2" >> results_arc/ar6_waiters.log
  ./autoresearch_jyz_af16.sh ar6_P3_dtv${DTV/./} 2 1500 $P3 --lambda_dtv $DTV
else
  echo "[$(date +%H:%M)] dtv does not pay on P2 -> ar6_P3_wavT1_010 on GPU 2" >> results_arc/ar6_waiters.log
  ./autoresearch_jyz_af16.sh ar6_P3_wavT1_010 2 1500 $P3 --set loss.lambda_wav_plain=0.10
fi
