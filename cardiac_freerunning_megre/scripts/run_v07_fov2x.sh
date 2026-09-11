#!/usr/bin/env bash
# Spec 5.6/5.7 correction 5 (2026-09-11, 2x FOV): V07's coil maps, lambda selection, REF, checks, figures and the
# self-check on the 224^3 grid. Waits for phase 1 of the other subjects (one BART job at a time); stops at the
# first failure (spec 7 stop rules).
set -euo pipefail
T=/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE
P=$(cat "$T/logs/run_rest_phase1.pid")
while kill -0 "$P" 2>/dev/null; do sleep 60; done
echo "phase 1 finished, starting V07 on 224^3 at $(date +%F_%T)"
cd "$T/scripts"
./py s05b_coilmaps_2x.py V07
./py s08b_lambda_select.py V07
REL=$(./py -c "import json; print(json.load(open('$T/recon/V07/ref_lambda.json'))['lam_rel'])")
./py s06_recon.py V07 --set ref --debug --lam "$REL" --lam-rel
./py s08_checks.py V07 --lam-rel "$REL"
./py s11_figures.py V07 --tag ref
./py s11_figures.py V07 --tag ref --echoes 4 --center 61,64,57 --box --name V07_ref_heart.png
./py s10_selfcheck.py
echo "V07_FOV2X_DONE $(date +%F_%T)"
