#!/bin/bash
# Waits for the FLAIR ESPIRiT map, then runs every arm that is NOT already covered by the
# T1+T2 topic. The T1 single-contrast arms are byte-identical there (same k-space, same mask,
# same maps, same seed) and are reused rather than recomputed; only the FLAIR arms and the
# T1+FLAIR joint run are new.
set -uo pipefail
W=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_retro
F=/data0/congcong/code/GS/Paired_ARC_GS_general_framework
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
E="PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID"

until grep -q 'ALL_DONE' "$W/logs/prep_03_csm.log"; do sleep 60; done
echo "=== CSM ready $(date -Is)"
env $E "$PY" "$W/scripts/prep_05_csm_to_h5.py" FLAIR > "$W/logs/prep_05_csm.log" 2>&1 || exit 1
echo "=== CSM converted, launching arms $(date -Is)"

env $E "$PY" -u "$W/scripts/recon_classical.py" --contrast FLAIR --gpu 2 > "$W/logs/classical_FLAIR.log" 2>&1 &
env $E "$PY" -u "$W/scripts/recon_zsssdu.py"  --contrast FLAIR --gpu 3 --epochs 100 > "$W/logs/zsssdu_FLAIR.log" 2>&1 &
env $E "$PY" -u "$W/scripts/recon_dip_inr.py" --method dip --contrast FLAIR --gpu 4 --iters 5000 --lr 3e-3 --eval_every 100 > "$W/logs/dip_FLAIR.log" 2>&1 &
env $E "$PY" -u "$W/scripts/recon_dip_inr.py" --method inr --contrast FLAIR --gpu 5 --iters 1500 --eval_every 50 > "$W/logs/inr_FLAIR.log" 2>&1 &
( cd $F && env $E "$PY" -u arc/cycle.py --exp t1flair_af12_joint --gpu 0 --iters 4000 \
    --lambda_geo 0.10 --warmup 500 --max_radius 20 --kspace_high 2.2 \
    --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 \
    --base_cfg configs/t1flair_af12_hyw.yaml --save_volumes > "$W/logs/gs_joint.log" 2>&1 ) &
( cd $F && env $E "$PY" -u arc/cycle.py --exp t1flair_af12_single_FLAIR --gpu 1 --iters 4000 \
    --lambda_geo 0.10 --warmup 500 --max_radius 20 --kspace_high 2.2 \
    --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 \
    --base_cfg configs/t1flair_af12_hyw_singleFLAIR.yaml --save_volumes > "$W/logs/gs_single_FLAIR.log" 2>&1 ) &
wait
echo "ALL T1FLAIR ARMS DONE $(date -Is)"
