#!/bin/bash
# Refinement of the L1-ESPIRiT lambda peaks found by run_classical_resweep_af12.sh (T1 peak at 0.0025 between
# 0.0015 and 0.004; FLAIR peak at 0.005 between 0.003 and 0.007): two points either side of each peak.
J=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_v3_jiangyizhou
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
export PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID
cd $J
run() { local c=$1 L=$2; [ -f logs/l1espirit_${c}_lam${L}.json ] && return
  nice -n 10 $PY -u scripts/recon_l1espirit.py --topic $J --contrast $c --gpu 6 --lam $L --iters 100 --suffix _lam$L --threads 32 > logs/sweeps/l1espirit_${c}_lam${L}.log 2>&1
  echo "[$(date +%H:%M)] l1 $c lam=$L rc=$?" >> logs/classical_resweep.log; }
run T1 0.002; run T1 0.003; run FLAIR 0.004; run FLAIR 0.006
echo "[$(date +%H:%M)] L1_REFINE_DONE" >> logs/classical_resweep.log
