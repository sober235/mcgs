#!/usr/bin/env bash
# Spec 5.8: phase-1 pipeline for one subject whose raw.h5 and spokes.npz exist (Tasks 6 and 8).
# Usage: run_subject.sh <Vxx> <iterations>
set -euo pipefail
S=$1
IT=$2
cd /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts
./py s05_coilmaps.py "$S"
./py s06_recon.py "$S" --set r0 --iters "$IT"
./py s11_figures.py "$S" --tag r0
./py s06_recon.py "$S" --set low --echoes 3 --iters "$IT"
./py s06_recon.py "$S" --set high --echoes 3 --iters "$IT"
./py s07_polarity.py "$S"
./py s06_recon.py "$S" --set ref --debug --iters "$IT"
./py s08_checks.py "$S" --ref-iters "$IT"
./py s11_figures.py "$S" --tag ref
echo "PIPELINE_DONE $S"
