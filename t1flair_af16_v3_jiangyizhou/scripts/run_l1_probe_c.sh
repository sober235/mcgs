#!/bin/bash
# Phase-3 acquisition probe (mcgs_af16_wt spec 2026-09-12, section 5; Track B step 4): L1-ESPIRiT FLAIR on the
# in-shutter probe mask data/mask_af16c_flair_3d.h5 through the T1T2_MASK_FLAIR override of t1t2_common.Pair,
# lambda 0.003, 0.002, 0.005 in that order, sequential, CPU (BART) under nice 10. The GPU argument only parks the
# loader's tensors. Outputs are moved under logs/probe_c/ and recon/probe_c/ (suffix _probe_c_lam*), which
# summarize.py's registered globs (logs/l1espirit_FLAIR_lam*.json) provably do not match, so the topic tables
# cannot pick them up. Restart-safe: a lambda whose json exists is skipped.
J=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
export PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID
export T1T2_MASK_FLAIR=$J/data/mask_af16c_flair_3d.h5
GPU=${GPU:-4}
cd $J
CH=logs/probe_c/l1espirit_probe_c_chain.log
for L in 0.003 0.002 0.005; do
  [ -f logs/probe_c/l1espirit_FLAIR_probe_c_lam${L}.json ] && continue
  echo "[$(date '+%F %H:%M')] start lam=$L gpu=$GPU T1T2_MASK_FLAIR=$T1T2_MASK_FLAIR" >> $CH
  nice -n 10 $PY -u scripts/recon_l1espirit.py --topic $J --contrast FLAIR --gpu $GPU --lam $L --iters 100 \
      --suffix _probe_c_lam$L --threads 32 > logs/probe_c/l1espirit_FLAIR_probe_c_lam${L}.log 2>&1
  rc=$?
  mv -n logs/l1espirit_FLAIR_probe_c_lam${L}.json logs/probe_c/ 2>/dev/null
  mv -n recon/l1espirit_probe_c_lam${L}_FLAIR_complex.npy recon/probe_c/ 2>/dev/null
  echo "[$(date '+%F %H:%M')] l1 FLAIR probe_c lam=$L rc=$rc" >> $CH
done
echo "[$(date '+%F %H:%M')] CHAIN_PROBE_C_DONE" >> $CH
