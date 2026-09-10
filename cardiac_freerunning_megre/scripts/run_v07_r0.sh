#!/usr/bin/env bash
# Spec 5.5-5.6 on V07: coil maps, R0 (8 echoes), R0 figure.
set -euo pipefail
cd /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts
./py s05_coilmaps.py V07
./py s06_recon.py V07 --set r0
./py s11_figures.py V07 --tag r0
echo V07_R0_DONE
