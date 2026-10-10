#!/usr/bin/env bash
# scripts/sg2x2_watch.sh QUEUE_LOG EXP_REGEX R_START_LINE   (run in the background; R_START_LINE = current line count of results_arc/sg2x2_runs.log)
# Wake the controller when the queue log says "queue drained"; a runs.log line written AFTER line R_START_LINE and
# matching EXP_REGEX has rc != 0 or "manifest FAILED"; a running sg2x2 trainer's log is idle > STALL_MIN minutes; or a
# running sg2x2 trainer has shared its card with another process for > SHARE_MIN minutes (2026-10-10: 8x slowdown).
# Read-only: never stops anything itself.
cd /data0/congcong/code/GS/mcgs_sg2x2_wt || exit 1
Q=${1:?queue log}; RX=${2:?exp regex}; R0=${3:?runs.log start line}
R=results_arc/sg2x2_runs.log
PYPAT='^/home/congcongliu/anaconda3/envs/GSTMI/bin/python -u arc/cycle.py --exp sg2x2_'
STALL_MIN=${STALL_MIN:-25}; SHARE_MIN=${SHARE_MIN:-10}; LIMIT_S=${LIMIT_S:-20000}
declare -A first_shared
t0=$(date +%s)
echo "watch up $(date '+%F %T'): queue log $Q, regex $RX, runs.log from line $((R0 + 1))"
while true; do
  if grep -q 'queue drained' "$Q" 2>/dev/null; then echo "QUEUE DRAINED $(date '+%F %T')"; break; fi
  new=$(tail -n +"$((R0 + 1))" "$R" 2>/dev/null)
  bad=$(echo "$new" | grep -E "${RX}.* rc=[1-9]" | head -3)
  if [[ -n "$bad" ]]; then echo "RUN FAILED: $bad"; break; fi
  mf=$(echo "$new" | grep -E "${RX}.* manifest FAILED" | head -3)
  if [[ -n "$mf" ]]; then echo "MANIFEST FAILED: $mf"; break; fi
  alert=""
  now=$(date +%s)
  while read -r pid rest; do
    [[ -z "$pid" ]] && continue
    exp=$(echo "$rest" | grep -oE -- '--exp [^ ]+' | awk '{print $2}')
    gpu=$(echo "$rest" | grep -oE -- '--gpu [0-9]+' | awk '{print $2}')
    log="results_arc/${exp}.log"
    [[ -f "$log" ]] || continue
    age=$(( now - $(stat -c %Y "$log") ))
    others=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu" 2>/dev/null | grep -v -x "$pid" | tr '\n' ' ')
    if (( age > STALL_MIN * 60 )); then
      alert+="STALL $exp (pid $pid, card $gpu) log idle $((age / 60)) min; other pids on the card: [${others}]; "
    fi
    if [[ -n "${others// /}" ]]; then
      [[ -z "${first_shared[$pid]}" ]] && first_shared[$pid]=$now
      if (( now - first_shared[$pid] > SHARE_MIN * 60 )); then
        alert+="SHARED $exp (pid $pid, card $gpu) for $(( (now - first_shared[$pid]) / 60 )) min with pids [${others}]; last line: $(grep -E '^\[iter' "$log" | tail -1 | cut -c1-24); "
      fi
    else
      unset "first_shared[$pid]"
    fi
  done < <(pgrep -af "$PYPAT")
  if [[ -n "$alert" ]]; then echo "$alert"; break; fi
  if (( now - t0 > LIMIT_S )); then echo "WATCH LIMIT reached $(date '+%F %T') (re-arm)"; break; fi
  sleep 120
done
echo "=== queue log (tail) ==="; tail -15 "$Q"
echo "=== runs.log since start ==="; tail -n +"$((R0 + 1))" "$R"
