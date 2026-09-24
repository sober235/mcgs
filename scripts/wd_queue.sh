#!/usr/bin/env bash
# wd_queue.sh QUEUE_FILE -- queue runner for the widen-the-margin loop; only ever uses an EMPTY card.
# Queue lines: "[KEY=VAL ...] EXP ITERS AF ARM LAMBDA_EDGE [extra run_wd_mcgs.sh args...]" (blank / # lines ignored,
# re-read every pass, top first). A card is usable when it is in ALLOWED, has no compute process, < 1500 MiB used, no
# live shared card lock of another scheduler (<cardiac topic>/work/.gpulock.<card>) and no trainer of this loop.
# At most MAXRUN trainers of this loop at once, launches >= 90 s apart. Items with any output already are skipped.
# Touch results_arc/.wd_queue_pause to hold launches. This runner never creates or removes lock files.
cd "$(dirname "$0")/.."
Q=${1:?queue file}
ALLOWED=${ALLOWED:-"0 1 2 3 4 5 6 7"}
MAXRUN=${MAXRUN:-6}
PYPAT='^/home/congcongliu/anaconda3/envs/GSTMI/bin/python -u arc/cycle.py --exp wd_'
LOCKD=/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MCGS/work
log() { echo "$(date '+%F %T') $*"; }
free_card() {
  for g in $ALLOWED; do
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
    [[ "$EXP" != wd_* ]] && { log "bad queue line (not wd_): $line"; continue; }
    if [[ -e "results_arc/${EXP}_score.json" || -e "results_arc/${EXP}.log" || -e "results_arc/${EXP}.launcher.log" ]]; then continue; fi
    pending=$((pending + 1))
    [[ -e results_arc/.wd_queue_pause ]] && { log "paused"; break; }
    running=$(pgrep -f "$PYPAT" | grep -c .)
    [[ "$running" -ge "$MAXRUN" ]] && break
    card=$(free_card) || break
    shift
    log "launch $EXP on card $card: $*${ENVS[*]:+ env: ${ENVS[*]}}"
    setsid nohup env "${ENVS[@]}" bash ./run_wd_mcgs.sh "$EXP" "$card" "$@" > "results_arc/${EXP}.launcher.log" 2>&1 < /dev/null &
    pending=$((pending - 1))
    sleep 90
  done < "$Q"
  running=$(pgrep -f "$PYPAT" | grep -c .)
  if [[ "$pending" -eq 0 && "$running" -eq 0 ]]; then log "queue drained"; exit 0; fi
  sleep 60
done
