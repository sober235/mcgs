#!/bin/bash
# L1-ESPIRiT lambda scan on the CPU (BART has no GPU support in this build). One job at a time,
# nice 10, 8 threads: the cardiac line also runs BART on this host.
set -uo pipefail
T=/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro
cd $T
for L in 0.001 0.0015 0.0025 0.004; do
  [ -e logs/l1espirit_T1w_lam$L.json ] || ./scripts/py scripts/recon_l1espirit.py --topic $T \
      --contrast T1w --gpu 2 --lam $L --iters 100 --suffix _lam$L --threads 8 >> logs/l1espirit_T1w.log 2>&1
done
for L in 0.003 0.005 0.007 0.01; do
  [ -e logs/l1espirit_T2s_lam$L.json ] || ./scripts/py scripts/recon_l1espirit.py --topic $T \
      --contrast T2s --gpu 2 --lam $L --iters 100 --suffix _lam$L --threads 8 >> logs/l1espirit_T2s.log 2>&1
done
echo L1_DONE
