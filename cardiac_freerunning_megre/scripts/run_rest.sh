#!/usr/bin/env bash
# Spec 5.8: V03, V04, V08 one after the other with the parameters frozen on V07. Usage: run_rest.sh <1|2|2a|2b>.
# Phase 1 (coil maps, R0, thirds; 112^3 grid) only needs the iteration count from V07's checks; phase 2 (polarity,
# REF, checks) reads ref_iters and ref_lam_rel from params_frozen.json. Stops at the first failure.
set -uo pipefail
T=/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE
PHASE=$1
if [ "$PHASE" = 1 ] || [ "$PHASE" = 2a ]; then
  IT=$("$T/scripts/py" -c "import json; print(json.load(open('$T/logs/checks_V07.json'))['ref_iters'])")
  REL=0
else
  IT=$("$T/scripts/py" -c "import json; print(json.load(open('$T/params_frozen.json'))['ref_iters'])")
  REL=$("$T/scripts/py" -c "import json; print(json.load(open('$T/params_frozen.json'))['ref_lam_rel'])")
fi
echo "phase $PHASE, iterations $IT, REF lambda_rel $REL"
for S in V03 V04 V08; do
  if bash "$T/scripts/run_subject.sh" "$S" "$PHASE" "$IT" "$REL" > "$T/logs/pipeline_${S}_phase${PHASE}.log" 2>&1; then
    echo "OK $S phase $PHASE $(date +%F_%T)"
  else
    echo "FAILED $S phase $PHASE (see logs/pipeline_${S}_phase${PHASE}.log)"
    exit 1
  fi
done
echo "ALL_DONE phase $PHASE"
