#!/bin/bash
# L1-ESPIRiT FLAIR, downward extension: on this subject at AF16 the FLAIR curve falls from the first
# grid point (lambda 0.003: 34.503 > 0.005: 34.365), so the upward points (0.007, 0.01) were dropped
# and two lower points added.
J=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
export PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID
cd $J
for L in 0.002 0.0015; do
  [ -f logs/l1espirit_FLAIR_lam${L}.json ] && continue
  $PY -u scripts/recon_l1espirit.py --topic $J --contrast FLAIR --gpu 3 --lam $L --iters 100 --suffix _lam$L --threads 16 > logs/l1espirit_FLAIR_lam${L}.log 2>&1
  echo "[$(date +%H:%M)] l1 FLAIR lam=$L rc=$?" >> logs/classical_revisited.log
done
echo "[$(date +%H:%M)] CHAIN_D2_DONE" >> logs/classical_revisited.log
