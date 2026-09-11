#!/bin/bash
# run_classical_resweep_af12.sh -- lambda re-sweep of the three strongest classical arms ON THIS SUBJECT
# (jiangyizhou, AF12), the same privilege the MC-GS autoresearch loop has (autoresearch_jyz_af12.md,
# "The bar"). Until now the topic's lambdas were the huangyuwei-selected ones, transferred unchanged
# (README, "Classical arms revisited"). summarize.py keeps the best lambda per family and contrast.
# GPU chain (card 6, shared with another user's light job): converged TV 200 outer, then joint TV.
# CPU chain: L1-wavelet ESPIRiT (BART pics, 100 FISTA, nice 10, 32 threads).
J=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_v3_jiangyizhou
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
export PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID
cd $J
G=${GPU:-6}
chain_tv() {   # contrast lambdas...
  local c=$1; shift 1
  for L in "$@"; do
    [ -f logs/classical_${c}_o200_lam${L}.json ] && continue
    $PY -u scripts/recon_classical.py --contrast $c --gpu $G --methods cssense --admm_outer 200 --lam $L --suffix _o200_lam$L > logs/sweeps/cssense_${c}_o200_lam${L}.log 2>&1
    echo "[$(date +%H:%M)] tv $c lam=$L rc=$?" >> logs/classical_resweep.log
  done
}
chain_jtv() {  # lambdas...
  for L in "$@"; do
    [ -f logs/jcssense_o200_lam${L}.json ] && continue
    $PY -u scripts/recon_joint_cssense.py --topic $J --contrasts T1,FLAIR --gpu $G --admm_outer 200 --lam $L --suffix _o200_lam$L > logs/sweeps/jcssense_o200_lam${L}.log 2>&1
    echo "[$(date +%H:%M)] jtv lam=$L rc=$?" >> logs/classical_resweep.log
  done
}
chain_l1() {   # contrast lambdas...
  local c=$1; shift 1
  for L in "$@"; do
    [ -f logs/l1espirit_${c}_lam${L}.json ] && continue
    nice -n 10 $PY -u scripts/recon_l1espirit.py --topic $J --contrast $c --gpu $G --lam $L --iters 100 --suffix _lam$L --threads 32 > logs/sweeps/l1espirit_${c}_lam${L}.log 2>&1
    echo "[$(date +%H:%M)] l1 $c lam=$L rc=$?" >> logs/classical_resweep.log
  done
}
echo "[$(date +%H:%M)] start gpu=$G" >> logs/classical_resweep.log
( chain_tv T1 0.0005 0.0007 0.0014 0.002; chain_tv FLAIR 0.0004 0.0005 0.001 0.0014; chain_jtv 0.0005 0.0007 0.0014 0.002; echo "[$(date +%H:%M)] GPU_CHAIN_DONE" >> logs/classical_resweep.log ) &
( chain_l1 T1 0.0007 0.001 0.0025 0.004; chain_l1 FLAIR 0.003 0.007 0.01; echo "[$(date +%H:%M)] L1_CHAIN_DONE" >> logs/classical_resweep.log ) &
wait
echo "[$(date +%H:%M)] ALL_DONE" >> logs/classical_resweep.log
