#!/usr/bin/env bash
# sg2x2_queue.sh QUEUE_FILE -- queue runner for the shared-geometry 2x2 ablation (run_sg2x2.sh); only ever uses an EMPTY card.
# Queue lines: "[KEY=VAL ...] EXP CELL ITERS SEED" (blank / # lines ignored,
# re-read every pass, top first). A card is usable when it is in ALLOWED, has no compute process, < 1500 MiB used, no
# live shared card lock of another scheduler (<cardiac topic>/work/.gpulock.<card>) and no trainer of this loop.
# At most MAXRUN trainers of this loop at once, launches >= 90 s apart. Items with any output already are skipped.
# Touch results_arc/.sg2x2_queue_pause to hold launches. This runner never creates or removes lock files.
# A card is also skipped while any running fixed-lane queue of another line (`*_queue.py --lanes A,B`) claims it:
# such a queue launches on its lanes without checking occupancy, and two trainers on one card stall each other to
# zero progress on this machine (2026-10-09: three hours lost on cards 0/1/3, ledger ruling R17).
cd "$(dirname "$0")/.."
Q=${1:?queue file}
ALLOWED=${ALLOWED:-"0 1 2 3 4"}
MAXRUN=${MAXRUN:-5}
PYPAT='^/home/congcongliu/anaconda3/envs/GSTMI/bin/python -u arc/cycle.py --exp sg2x2_'
LOCKD=/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MCGS/work
log() { echo "$(date '+%F %T') $*"; }
claimed_lanes() {
  pgrep -af '^/home/congcongliu/anaconda3/envs/GSTMI/bin/python -u .*_queue\.py .*--lanes' 2>/dev/null \
    | grep -oE -- '--lanes [0-9,]+' | sed 's/--lanes //' | tr ',' '\n' | sort -u | tr '\n' ' '
}
free_card() {
  local claimed
  claimed=" $(claimed_lanes) "
  for g in $ALLOWED; do
    [[ "$claimed" == *" $g "* ]] && continue
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
    [[ -z "$used" ]] && continue
    procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" 2>/dev/null | grep -c .)
    mine=$(pgrep -f "$PYPAT.* --gpu $g " | grep -c .)
    locked=0
    if [[ -d "$LOCKD/.gpulock.$g" ]]; then
      owner=$(cat "$LOCKD/.gpulock.$g/pid" 2>/dev/null)
      if [[ -n "$owner" ]] && kill -0 "$owner" 2>/dev/null; then locked=1; fi
    fi
    if [[ "$used" -lt 1500 && "$procs" -eq 0 && "$mine" -eq 0 && "$locked" -eq 0 ]]; then echo "$g"; return 0; fi
  done
  return 1
}
log "queue runner up, queue=$Q allowed=[$ALLOWED] maxrun=$MAXRUN"
while true; do
  pending=0
  while read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    set -- $line
    ENVS=()
    while [[ "$1" == *=* && "$1" != -* ]]; do ENVS+=("$1"); shift; done
    EXP=$1
    [[ "$EXP" != sg2x2_* ]] && { log "bad queue line (not sg2x2_): $line"; continue; }
    if [[ -e "results_arc/${EXP}_score.json" || -e "results_arc/${EXP}.log" || -e "results_arc/${EXP}.launcher.log" ]]; then continue; fi
    pending=$((pending + 1))
    [[ -e results_arc/.sg2x2_queue_pause ]] && { log "paused"; break; }
    running=$(pgrep -f "$PYPAT" | grep -c .)
    [[ "$running" -ge "$MAXRUN" ]] && break
    card=$(free_card) || break
    shift
    log "launch $EXP on card $card: $*${ENVS[*]:+ env: ${ENVS[*]}}"
    setsid nohup env "${ENVS[@]}" bash ./run_sg2x2.sh "$EXP" "$card" "$@" > "results_arc/${EXP}.launcher.log" 2>&1 < /dev/null &
    pending=$((pending - 1))
    sleep 90
  done < "$Q"
  running=$(pgrep -f "$PYPAT" | grep -c .)
  if [[ "$pending" -eq 0 && "$running" -eq 0 ]]; then log "queue drained"; exit 0; fi
  sleep 60
done
