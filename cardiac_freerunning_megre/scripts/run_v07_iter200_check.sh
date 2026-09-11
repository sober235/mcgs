#!/usr/bin/env bash
# Spec 5.6 fallback (plan Task 10 Step 4): the 60-vs-100 iteration check failed on V07. Re-run REF echo 0
# with 200 iterations and repeat the check (120 vs 200) before spending the other seven echoes.
set -euo pipefail
T=/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE
[ -f $T/recon/V07/ref_it100_e0.npy ] || cp $T/recon/V07/ref_e0.npy $T/recon/V07/ref_it100_e0.npy
cd $T/scripts
./py s06_recon.py V07 --set ref --echoes 0 --iters 200 --debug
./py s08_checks.py V07 --ref-iters 200
echo ITER200_CHECK_DONE
