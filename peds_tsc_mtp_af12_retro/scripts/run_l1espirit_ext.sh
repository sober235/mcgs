#!/bin/bash
# L1-ESPIRiT lambda scan, EXTENSION (plan Task 7 step 6 edge rule): the best lambda of both contrasts
# landed on the smallest value of its grid, so the grid is extended downward by the grids' own ratio
# (~1.4x per step) until the optimum is interior. Waits for the main scan to finish first.
set -uo pipefail
T=/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro
cd $T
while pgrep -f "run_l1espirit.sh" > /dev/null; do sleep 30; done
for L in 0.0007 0.0005 0.00035; do
  [ -e logs/l1espirit_T1w_lam$L.json ] || ./scripts/py scripts/recon_l1espirit.py --topic $T \
      --contrast T1w --gpu 2 --lam $L --iters 100 --suffix _lam$L --threads 8 >> logs/l1espirit_T1w.log 2>&1
done
for L in 0.002 0.0014 0.001; do
  [ -e logs/l1espirit_T2s_lam$L.json ] || ./scripts/py scripts/recon_l1espirit.py --topic $T \
      --contrast T2s --gpu 2 --lam $L --iters 100 --suffix _lam$L --threads 8 >> logs/l1espirit_T2s.log 2>&1
done
echo L1_EXT_DONE
