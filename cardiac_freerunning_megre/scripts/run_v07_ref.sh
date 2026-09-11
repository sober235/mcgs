#!/usr/bin/env bash
# Spec 5.4(b) and 5.6 on V07: respiratory thirds, polarity, REF (8 echoes), checks, REF figure.
set -euo pipefail
cd /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts
./py s06_recon.py V07 --set low --echoes 3
./py s06_recon.py V07 --set high --echoes 3
./py s07_polarity.py V07
./py s06_recon.py V07 --set ref --debug
./py s08_checks.py V07
./py s11_figures.py V07 --tag ref
echo V07_REF_DONE
