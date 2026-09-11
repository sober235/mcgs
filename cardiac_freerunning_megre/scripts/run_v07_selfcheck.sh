#!/usr/bin/env bash
# Spec 5.7 self-check on V07 (5 lambda runs + 12 grid runs, then scoring).
set -euo pipefail
cd /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts
./py s10_selfcheck.py
echo V07_SELFCHECK_DONE
