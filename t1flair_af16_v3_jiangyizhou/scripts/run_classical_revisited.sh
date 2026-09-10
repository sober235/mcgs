#!/bin/bash
# run_classical_revisited.sh -- the corrected classical arms on jiangyizhou AF16 (ported 2026-09-10
# from the AF12 topics' db69313 audit): converged TV (200 outer ADMM), L1-wavelet ESPIRiT (BART
# pics, 100 FISTA, CPU) and joint vectorial TV (200 outer). Lambda is swept ON THIS SUBJECT against
# the ground truth -- the same privilege the MC-GS autoresearch loop has -- and summarize.py keeps
# the best lambda per contrast. Four chains, GPUs 2 and 3 (borrowed while idle), ~10 GB each.
J=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
export PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID
cd $J
chain_tv() {   # gpu contrast lambdas...
  local g=$1 c=$2; shift 2
  for L in "$@"; do
    [ -f logs/classical_${c}_o200_lam${L}.json ] && continue
    $PY -u scripts/recon_classical.py --contrast $c --gpu $g --methods cssense --admm_outer 200 --lam $L --suffix _o200_lam$L > logs/classical_${c}_o200_lam${L}.log 2>&1
    echo "[$(date +%H:%M)] tv $c lam=$L rc=$?" >> logs/classical_revisited.log
  done
}
chain_l1() {   # gpu contrast lambdas...
  local g=$1 c=$2; shift 2
  for L in "$@"; do
    [ -f logs/l1espirit_${c}_lam${L}.json ] && continue
    $PY -u scripts/recon_l1espirit.py --topic $J --contrast $c --gpu $g --lam $L --iters 100 --suffix _lam$L --threads 16 > logs/l1espirit_${c}_lam${L}.log 2>&1
    echo "[$(date +%H:%M)] l1 $c lam=$L rc=$?" >> logs/classical_revisited.log
  done
}
chain_jtv() {  # gpu lambdas...
  local g=$1; shift 1
  for L in "$@"; do
    [ -f logs/jcssense_o200_lam${L}.json ] && continue
    $PY -u scripts/recon_joint_cssense.py --topic $J --contrasts T1,FLAIR --gpu $g --admm_outer 200 --lam $L --suffix _o200_lam$L > logs/jcssense_o200_lam${L}.log 2>&1
    echo "[$(date +%H:%M)] jtv lam=$L rc=$?" >> logs/classical_revisited.log
  done
}
echo "[$(date +%H:%M)] start" >> logs/classical_revisited.log
( chain_tv 2 T1 0.0007 0.001 0.0014 0.002; chain_jtv 2 0.001 0.0014; echo "[$(date +%H:%M)] CHAIN_A_DONE" >> logs/classical_revisited.log ) &
( chain_tv 3 FLAIR 0.0005 0.0007 0.001 0.0014; echo "[$(date +%H:%M)] CHAIN_B_DONE" >> logs/classical_revisited.log ) &
( chain_l1 2 T1 0.001 0.0015 0.0025 0.004; echo "[$(date +%H:%M)] CHAIN_C_DONE" >> logs/classical_revisited.log ) &
( chain_l1 3 FLAIR 0.003 0.005 0.007 0.01; echo "[$(date +%H:%M)] CHAIN_D_DONE" >> logs/classical_revisited.log ) &
wait
echo "[$(date +%H:%M)] ALL_CLASSICAL_REVISITED_DONE" >> logs/classical_revisited.log
