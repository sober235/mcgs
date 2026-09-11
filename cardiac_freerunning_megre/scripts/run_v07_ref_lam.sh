#!/usr/bin/env bash
# Spec 5.6 as revised on 2026-09-11 (user-approved): REF regularisation relative to the largest eigenvalue of
# the REF normal operator. Choose lambda_rel on echo 0, then REF (8 echoes), the two checks and the figure.
set -euo pipefail
T=/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE
cd $T/scripts
./py s08b_lambda_select.py V07
REL=$(./py -c "import json; print(json.load(open('$T/recon/V07/ref_lambda.json'))['lam_rel'])")
./py s06_recon.py V07 --set ref --debug --lam "$REL" --lam-rel
./py s08_checks.py V07 --lam-rel "$REL"
./py s11_figures.py V07 --tag ref
echo V07_REF_DONE
