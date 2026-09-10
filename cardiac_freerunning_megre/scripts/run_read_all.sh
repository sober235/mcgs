#!/usr/bin/env bash
# Spec 5.1: raw.h5 for the four subjects, one after the other (a few minutes each).
set -euo pipefail
T=/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts
for S in V07 V03 V04 V08; do
  "$T/py" "$T/s01_read.py" "$S"
done
echo ALL_READ_DONE
