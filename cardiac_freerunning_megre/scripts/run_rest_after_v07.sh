#!/usr/bin/env bash
# Task 12 continuation (2026-09-11 evening): once the V07 224^3 chain has ended, and only if it passed
# (V07_FOV2X_DONE in its log, which the self-check gates), freeze the parameters and run phase 2 for V03, V04, V08.
# A failed chain leaves the machine idle and the failure in logs/run_v07_fov2x.log (spec 7 stop rules).
set -uo pipefail
T=/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE
P=$(cat "$T/logs/run_v07_fov2x.pid")
while kill -0 "$P" 2>/dev/null; do sleep 120; done
if ! grep -q V07_FOV2X_DONE "$T/logs/run_v07_fov2x.log"; then
  echo "V07 chain did not finish cleanly; not starting phase 2 ($(date +%F_%T))"
  exit 1
fi
cd "$T/scripts"
./py freeze_params.py
bash "$T/scripts/run_rest.sh" 2
