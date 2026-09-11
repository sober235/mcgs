#!/usr/bin/env bash
# Spec 5.8: V03, V04, V08 one after the other with the parameters frozen on V07 (params_frozen.json).
# Usage: run_rest.sh <1|2>. Stops at the first failure.
set -uo pipefail
T=/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE
PHASE=$1
IT=$("$T/scripts/py" -c "import json; print(json.load(open('$T/params_frozen.json'))['ref_iters'])")
REL=$("$T/scripts/py" -c "import json; print(json.load(open('$T/params_frozen.json'))['ref_lam_rel'])")
for S in V03 V04 V08; do
  if bash "$T/scripts/run_subject.sh" "$S" "$PHASE" "$IT" "$REL" > "$T/logs/pipeline_${S}_phase${PHASE}.log" 2>&1; then
    echo "OK $S phase $PHASE"
  else
    echo "FAILED $S phase $PHASE (see logs/pipeline_${S}_phase${PHASE}.log)"
    exit 1
  fi
done
echo "ALL_DONE phase $PHASE"
