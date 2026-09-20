#!/usr/bin/env bash
# ca_queue.sh QUEUE_FILE -- queue runner for the hyper-parameter sensitivity study; only ever uses an EMPTY card.
# Queue lines: "[KEY=VAL ...] EXP ITERS [extra run_ca_mcgs.sh args...]" (blank lines and # comments ignored; re-read on
# every pass, top first). A card is usable when it is in ALLOWED (default "1 2"), carries no compute process, uses
# < 1500 MiB, has no live card lock of another scheduler (<cardiac topic>/work/.gpulock.<card> with a live owner) and
# no trainer of this study. One trainer per card, launches at least 90 s apart. Items whose score json, log or
# launcher log already exists are skipped (a crashed item is not retried automatically). Touch
# results_arc/.ca_queue_pause to hold launches. Exits 0 once every item has outputs.
# Tests: CA_DRY=1 prints the launches instead of starting them (a dry-launched card is taken out of ALLOWED for the
# pass); CA_ONCE=1 makes a single pass over the queue and exits.
cd "$(dirname "$0")/.."
Q=${1:?queue file}
ALLOWED=${ALLOWED:-"1 2"}
PYPAT='^/home/congcongliu/anaconda3/envs/GSTMI/bin/python -u arc/cycle.py --exp ca_'
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
log "queue runner up, queue=$Q allowed=[$ALLOWED] dry=${CA_DRY:-0}"
while true; do
  pending=0
  while read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    set -- $line
    ENVS=""
    while [[ "$1" == *=* && "$1" != -* ]]; do ENVS="$ENVS $1"; shift; done
    EXP=$1; IT=$2; CFG=$3; LEDGE=$4; shift 4; EXTRA="$*"
    [[ "$EXP" != ca_* ]] && { log "bad queue line (not ca_): $line"; continue; }
    if [[ -e "results_arc/${EXP}_score.json" || -e "results_arc/${EXP}.log" || -e "results_arc/${EXP}.launcher.log" ]]; then continue; fi
    pending=$((pending + 1))
    [[ -e results_arc/.ca_queue_pause ]] && { log "paused (results_arc/.ca_queue_pause exists)"; break; }
    card=$(free_card) || { log "no free card in [$ALLOWED]"; break; }
    if [[ "${CA_DRY:-0}" == "1" ]]; then
      log "DRY launch $EXP on card $card ($IT it) cfg=$CFG ledge=$LEDGE${ENVS:+ env:$ENVS} $EXTRA"
      ALLOWED=$(echo " $ALLOWED " | sed "s/ $card / /" | xargs)
      pending=$((pending - 1))
      continue
    fi
    log "launch $EXP on card $card ($IT it) cfg=$CFG ledge=$LEDGE${ENVS:+ env:$ENVS} $EXTRA"
    # detached (setsid + nohup): the run must not depend on this runner's process tree
    setsid nohup bash scripts/ca_launch_one.sh $EXP $card $IT $CFG $LEDGE $ENVS -- $EXTRA > results_arc/${EXP}.launcher.log 2>&1 < /dev/null &
    pending=$((pending - 1))
    sleep 90
  done < "$Q"
  if [[ "${CA_ONCE:-0}" == "1" ]]; then log "single pass done, pending=$pending"; exit 0; fi
  # drained = every item has outputs AND no trainer of this study is still running
  running=$(pgrep -f "$PYPAT" | grep -c .)
  if [[ "$pending" -eq 0 && "$running" -eq 0 ]]; then log "queue drained"; exit 0; fi
  sleep 60
done
