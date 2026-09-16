#!/usr/bin/env bash
# gauss_sched.sh QUEUE_FILE -- launcher for the Gaussian-mask autoresearch loop; only ever uses an EMPTY card.
# Queue lines: "[KEY=VAL ...] EXP ITERS [extra autoresearch_gauss.sh args...]" (blank lines and # comments ignored;
# re-read on every pass, so lines can be appended while it runs; top first). A card is usable when it is in ALLOWED,
# carries no compute process, uses < 1500 MiB, has no live card lock of another scheduler
# (<cardiac topic>/work/.gpulock.<card> with a live owner) and no trainer of this line. One trainer per card,
# launches at least 90 s apart. Items whose score json or launcher log already exists are skipped (a crashed item
# is not retried automatically). Touch results_arc/.gauss_sched_pause to hold launches. Runs forever; kill the
# tmux session `gauss_sched` to stop (the trainers are detached and survive that; the tmux SERVER process shows this
# script's command line in ps -- never kill it by pid).
cd "$(dirname "$0")/.."
Q=${1:?queue file}
ALLOWED=${ALLOWED:-"4 7 2 3 5 6"}
PYPAT='^/home/congcongliu/anaconda3/envs/GSTMI/bin/python -u arc/cycle.py --exp g_'
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
log "scheduler up, queue=$Q allowed=[$ALLOWED]"
while true; do
  while read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    set -- $line
    ENVS=""
    while [[ "$1" == *=* && "$1" != -* ]]; do ENVS="$ENVS $1"; shift; done
    EXP=$1; IT=$2; shift 2; EXTRA="$*"
    [[ "$EXP" != g_* ]] && { log "bad queue line (not g_): $line"; continue; }
    if [[ -e "results_arc/${EXP}_score.json" || -e "results_arc/${EXP}.log" || -e "results_arc/${EXP}.launcher.log" ]]; then continue; fi
    [[ -e results_arc/.gauss_sched_pause ]] && break
    card=$(free_card) || break
    log "launch $EXP on card $card ($IT it)${ENVS:+ env:$ENVS} $EXTRA"
    # detached (setsid + nohup): the run must not depend on this scheduler's tmux server -- killing that server on
    # 2026-09-16 15:02 took four trainers down with it
    setsid nohup bash scripts/gauss_launch_one.sh $EXP $card $IT $ENVS -- $EXTRA > results_arc/${EXP}.launcher.log 2>&1 < /dev/null &
    sleep 90
  done < "$Q"
  sleep 60
done
