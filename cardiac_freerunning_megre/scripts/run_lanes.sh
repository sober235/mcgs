#!/usr/bin/env bash
# Phase 2 for V03, V04, V08 in two lanes (user approved two concurrent BART jobs on 2026-09-12; 32 threads each).
#   lane A: V03 2b -> (wait for V08 2a) -> V08 2b        lane B: V04 2a -> V08 2a -> V04 2b
# 2a = 224^3 coil maps + polarity (V03's already done), 2b = REF, checks, figure. Each lane stops at its first failure.
set -uo pipefail
T=/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE
IT=$("$T/scripts/py" -c "import json; print(json.load(open('$T/params_frozen.json'))['ref_iters'])")
REL=$("$T/scripts/py" -c "import json; print(json.load(open('$T/params_frozen.json'))['ref_lam_rel'])")
step() {  # <Vxx> <phase>
  if bash "$T/scripts/run_subject.sh" "$1" "$2" "$IT" "$REL" > "$T/logs/pipeline_$1_phase$2.log" 2>&1; then
    echo "OK $1 phase $2 $(date +%F_%T)"
  else
    echo "FAILED $1 phase $2 (see logs/pipeline_$1_phase$2.log) $(date +%F_%T)"; return 1
  fi
}
lane_a() {
  step V03 2b || return 1
  until grep -q 'PHASE2a_DONE V08' "$T/logs/pipeline_V08_phase2a.log" 2>/dev/null; do
    grep -q 'FAILED V08 phase 2a' "$T/logs/run_lanes.log" 2>/dev/null && { echo "lane A: V08 2a failed, stopping"; return 1; }
    sleep 60
  done
  step V08 2b
}
lane_b() { step V04 2a && step V08 2a && step V04 2b; }
echo "lanes started $(date +%F_%T), iterations $IT, REF lambda_rel $REL"
lane_a & PA=$!
lane_b & PB=$!
wait $PA; RA=$?
wait $PB; RB=$?
echo "lanes ended $(date +%F_%T): lane A rc=$RA, lane B rc=$RB"
[ $RA -eq 0 ] && [ $RB -eq 0 ] && echo "ALL_DONE phase 2" || echo "PHASE2_INCOMPLETE"
