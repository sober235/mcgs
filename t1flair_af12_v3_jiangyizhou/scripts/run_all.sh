#!/bin/bash
# Full pipeline for this subject: waits for both ESPIRiT maps, converts them, then runs all
# sixteen arms (eight methods x two contrasts). Nothing is reused from another subject here.
set -uo pipefail
W=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_v3_jiangyizhou
F=/data0/congcong/code/GS/Paired_ARC_GS_general_framework
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
E="PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID"

until grep -q 'ALL_DONE' "$W/logs/prep_03_csm.log"; do sleep 60; done
echo "=== both CSMs ready $(date -Is)"
env $E "$PY" "$W/scripts/prep_05_csm_to_h5.py" T1 FLAIR > "$W/logs/prep_05_csm.log" 2>&1 || exit 1
env $E GPU=0 "$PY" "$W/scripts/prep_06_align_check.py" > "$W/logs/prep_06_align.log" 2>&1 || true
echo "=== launching arms $(date -Is)"

# classical: cheap, run both contrasts back to back on one card
( env $E "$PY" -u "$W/scripts/recon_classical.py" --contrast T1    --gpu 6 > "$W/logs/classical_T1.log" 2>&1
  env $E "$PY" -u "$W/scripts/recon_classical.py" --contrast FLAIR --gpu 6 > "$W/logs/classical_FLAIR.log" 2>&1 ) &
( env $E "$PY" -u "$W/scripts/recon_zsssdu.py" --contrast T1    --gpu 7 --epochs 100 > "$W/logs/zsssdu_T1.log" 2>&1
  env $E "$PY" -u "$W/scripts/recon_zsssdu.py" --contrast FLAIR --gpu 7 --epochs 100 > "$W/logs/zsssdu_FLAIR.log" 2>&1 ) &
( env $E "$PY" -u "$W/scripts/recon_dip_inr.py" --method dip --contrast T1    --gpu 2 --iters 5000 --lr 3e-3 --eval_every 100 > "$W/logs/dip_T1.log" 2>&1
  env $E "$PY" -u "$W/scripts/recon_dip_inr.py" --method dip --contrast FLAIR --gpu 2 --iters 5000 --lr 3e-3 --eval_every 100 > "$W/logs/dip_FLAIR.log" 2>&1 ) &
( env $E "$PY" -u "$W/scripts/recon_dip_inr.py" --method inr --contrast T1    --gpu 3 --iters 1500 --eval_every 50 > "$W/logs/inr_T1.log" 2>&1
  env $E "$PY" -u "$W/scripts/recon_dip_inr.py" --method inr --contrast FLAIR --gpu 3 --iters 1500 --eval_every 50 > "$W/logs/inr_FLAIR.log" 2>&1 ) &
( cd $F && env $E "$PY" -u arc/cycle.py --exp t1flair_v3_af12_joint --gpu 0 --iters 4000 \
    --lambda_geo 0.10 --warmup 500 --max_radius 20 --kspace_high 2.2 \
    --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 \
    --base_cfg configs/t1flair_v3_af12_hyw.yaml --save_volumes > "$W/logs/gs_joint.log" 2>&1 )&
( cd $F && env $E "$PY" -u arc/cycle.py --exp t1flair_v3_af12_single_T1 --gpu 1 --iters 4000 \
    --lambda_geo 0.10 --warmup 500 --max_radius 20 --kspace_high 2.2 \
    --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 \
    --base_cfg configs/t1flair_v3_af12_hyw_singleT1.yaml --save_volumes > "$W/logs/gs_single_T1.log" 2>&1
  cd $F && env $E "$PY" -u arc/cycle.py --exp t1flair_v3_af12_single_FLAIR --gpu 1 --iters 4000 \
    --lambda_geo 0.10 --warmup 500 --max_radius 20 --kspace_high 2.2 \
    --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 \
    --base_cfg configs/t1flair_v3_af12_hyw_singleFLAIR.yaml --save_volumes > "$W/logs/gs_single_FLAIR.log" 2>&1 ) &
wait
echo "ALL V3 ARMS DONE $(date -Is)"
