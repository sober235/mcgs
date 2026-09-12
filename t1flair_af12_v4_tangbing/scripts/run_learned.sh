#!/bin/bash
# ZS-SSDU / 3D-DIP / 3D-INR on the retrospective AF12 T1+FLAIR pair.
# Same hyper-parameters as the CE pre/post study (Results_20260908/CE_pre_post_contrast).
W=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_v4_tangbing
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
E="PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID"

env $E "$PY" -u "$W/scripts/recon_zsssdu.py"  --contrast T1 --gpu 3 --epochs 100 > "$W/logs/zsssdu_T1.log" 2>&1 &
env $E "$PY" -u "$W/scripts/recon_zsssdu.py"  --contrast FLAIR --gpu 4 --epochs 100 > "$W/logs/zsssdu_FLAIR.log" 2>&1 &
env $E "$PY" -u "$W/scripts/recon_dip_inr.py" --method dip --contrast T1 --gpu 5 --iters 5000 --lr 3e-3 --eval_every 100 > "$W/logs/dip_T1.log" 2>&1 &
env $E "$PY" -u "$W/scripts/recon_dip_inr.py" --method dip --contrast FLAIR --gpu 6 --iters 5000 --lr 3e-3 --eval_every 100 > "$W/logs/dip_FLAIR.log" 2>&1 &
( env $E "$PY" -u "$W/scripts/recon_dip_inr.py" --method inr --contrast T1 --gpu 7 --iters 1500 --eval_every 50 > "$W/logs/inr_T1.log" 2>&1
  env $E "$PY" -u "$W/scripts/recon_dip_inr.py" --method inr --contrast FLAIR --gpu 7 --iters 1500 --eval_every 50 > "$W/logs/inr_FLAIR.log" 2>&1 ) &
wait
echo "ALL LEARNED DONE"
