#!/usr/bin/env bash
# p3_sched.sh QUEUE_FILE — phase-3 launcher that only ever uses an EMPTY card (user rule 2026-09-12: "优先选择空GPU").
# Queue lines: "[KEY=VAL ...] EXP ITERS [extra autoresearch_af16_p3.sh args...]" (leading KEY=VAL = env for that launch) (blank lines and # comments ignored; the file is
# re-read on every pass, so lines can be appended while it runs). A card is usable when it is in ALLOWED, carries no
# compute process, uses < 1500 MiB, is not claimed by a live AF12 chain (`bash scripts/p2_chain_X.sh <card>`), and has
# no trainer of this line. One trainer per card; launches at least 90 s apart. Items whose score json or launcher log
# already exists are skipped (a crashed item is therefore not retried automatically). Touch results_arc/.p3_sched_pause
# to hold launches. Runs forever; kill the tmux session `p3_sched` to stop.
cd "$(dirname "$0")/.."
Q=${1:?queue file}
ALLOWED=${ALLOWED:-"4 5 3 2 1"}
PYPAT='^/home/congcongliu/anaconda3/envs/GSTMI/bin/python -u arc/cycle.py --exp p3_'
log() { echo "$(date '+%F %T') $*"; }
free_card() {
  for g in $ALLOWED; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
    [[ -z "$used" ]] && continue
    procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" 2>/dev/null | grep -c .)
    claimed=$(pgrep -f "p2_chain_[a-z]\.sh $g\$" | grep -c .)
    mine=$(pgrep -f "$PYPAT.* --gpu $g " | grep -c .)
    if [[ "$used" -lt 1500 && "$procs" -eq 0 && "$claimed" -eq 0 && "$mine" -eq 0 ]]; then echo "$g"; return 0; fi
  done
  return 1
}
log "scheduler up, queue=$Q allowed=[$ALLOWED]"
while true; do
  launched=0
  while read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    set -- $line
    ENVS=""
    while [[ "$1" == *=* && "$1" != -* ]]; do ENVS="$ENVS $1"; shift; done   # leading KEY=VAL tokens: per-launch env
    EXP=$1; IT=$2; shift 2; EXTRA="$*"
    [[ "$EXP" != p3_* ]] && { log "bad queue line (not p3_): $line"; continue; }
    if [[ -e "results_arc/${EXP}_score.json" || -e "results_arc/${EXP}.log" ]]; then continue; fi
    [[ -e results_arc/.p3_sched_pause ]] && break
    card=$(free_card) || break
    log "launch $EXP on card $card ($IT it)${ENVS:+ env:$ENVS} $EXTRA"
    tmux new-session -d -s "p3_$EXP" "env $ENVS ./autoresearch_af16_p3.sh $EXP $card $IT $EXTRA > results_arc/${EXP}.launcher.log 2>&1"
    launched=1
    sleep 90
  done < "$Q"
  sleep 60
done
