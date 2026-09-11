#!/usr/bin/env bash
# 2026-09-11: continuation of run_v07_ref.sh after the approved criterion-(b) fix. The two respiratory
# thirds are reconstructed at echo 3 by a separate job (logs/thirds_e3_V07.log); wait for it, then run the
# polarity check, REF (8 echoes), the two checks and the REF figure. Stops at the first failure.
set -euo pipefail
T=/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE
while kill -0 "$(cat $T/logs/thirds_e3_V07.pid)" 2>/dev/null; do sleep 15; done
grep -q THIRDS_E3_DONE $T/logs/thirds_e3_V07.log
cd $T/scripts
./py s07_polarity.py V07
./py s06_recon.py V07 --set ref --debug
./py s08_checks.py V07
./py s11_figures.py V07 --tag ref
echo V07_REF_DONE
