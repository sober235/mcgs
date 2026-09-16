#!/usr/bin/env bash
# p5_sched.sh QUEUE -- stage 5 of the 16yo MTP AF12 line (the MC-GS lever round, user 2026-09-16 "启动mc-gs杠杆"); a copy of p4_sched2.sh with its own log, stop and pause files and the p5_ run prefix.
#
# One long-lived instance. It launches queue jobs in the background, never more than LINE_MAX of this
# line's trainings at once -- counted over EVERY arc/cycle.py run on a configs/peds_* config, so jobs
# started by other sessions or by peds_sched.sh count too (user rule 2026-09-14: at most two). A job
# goes on a card that had no compute process, utilisation <= 10 % and < 1500 MiB used at two polls
# POLL seconds apart; a card that just received a job is left alone for 10 minutes so its context
# can appear. One training per card.
#
# Queue lines: "[ITERS=n] <exp> <config> [extra cycle.py / --set args]"; blank lines and # comments
# are ignored. A line is pending while results_arc/<exp>.log does not exist (autoresearch_peds.sh
# refuses to overwrite an existing run, so a line can never run twice). Lines may be appended while
# the scheduler runs. Touch logs/.p5_pause to hold launches, logs/.p5_stop to end the scheduler
# (running jobs are not touched).
set -uo pipefail
T=/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro
W=/data0/congcong/code/GS/mcgs_peds_wt
Q=${1:?queue file}
ALLOWED=${ALLOWED:-"0 1 2 3 4 5 6 7"}
LINE_MAX=${LINE_MAX:-2}
OWN_MAX=${OWN_MAX:-2}                # this campaign owns both of the line's slots (claude-eb runs nothing here, 2026-09-16)
POLL=${POLL:-60}
LOG="$T/logs/p5_sched.log"
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

line_trainings() { pgrep -fc "^[^ ]*python[^ ]* -u arc/cycle.py --exp .*--base_cfg configs/peds_" || true; }   # anchored: a shell whose command line merely quotes this pattern must not count
own_trainings() { pgrep -fc "^[^ ]*python[^ ]* -u arc/cycle.py --exp p5_" || true; }

card_ok() {                          # listing into variables first: pipefail + grep -q misreads busy cards
  local g=$1 bus apps q util mem
  bus=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | awk -F', ' -v i="$g" '$1==i{print $2}')
  [[ -n "$bus" ]] || return 1
  apps=$(nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader) || return 1
  grep -qF "$bus" <<< "$apps" && return 1
  q=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i "$g") || return 1
  util=$(echo "$q" | cut -d, -f1 | tr -d ' '); mem=$(echo "$q" | cut -d, -f2 | tr -d ' ')
  [[ "$util" -le 10 && "$mem" -lt 1500 ]]
}

declare -A CLAIM
log "p5 scheduler up: queue=$Q allowed=[$ALLOWED] line_max=$LINE_MAX own_max=$OWN_MAX poll=$POLL"
while [[ ! -e "$T/logs/.p5_stop" ]]; do
  if [[ -e "$T/logs/.p5_pause" ]]; then sleep "$POLL"; continue; fi
  next=""
  while read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "${line:0:1}" == "#" ]] && continue
    read -r -a words <<< "$line"
    [[ "${words[0]}" == ITERS=* ]] && words=("${words[@]:1}")
    [[ -e "$W/results_arc/${words[0]}.log" ]] && continue
    next=$line; break
  done < "$Q"
  if [[ -z "$next" ]]; then sleep "$POLL"; continue; fi
  if [[ "$(line_trainings)" -ge "$LINE_MAX" || "$(own_trainings)" -ge "$OWN_MAX" ]]; then sleep "$POLL"; continue; fi
  card=""
  for g in $ALLOWED; do
    [[ "${CLAIM[$g]:-0}" -gt "$(date +%s)" ]] && continue
    card_ok "$g" || continue
    sleep "$POLL"
    card_ok "$g" && { card=$g; break; }
  done
  if [[ -z "$card" ]]; then sleep "$POLL"; continue; fi
  [[ "$(line_trainings)" -ge "$LINE_MAX" || "$(own_trainings)" -ge "$OWN_MAX" ]] && continue
  read -r -a words <<< "$next"
  iters=4000
  if [[ "${words[0]}" == ITERS=* ]]; then iters=${words[0]#ITERS=}; words=("${words[@]:1}"); fi
  exp=${words[0]}; cfg=${words[1]}; extra=("${words[@]:2}")
  ( cd "$W" && CONFIG="$cfg" SAVE_VOLUMES=1 exec ./autoresearch_peds.sh "$exp" "$card" "$iters" "${extra[@]}" ) \
      >> "$T/logs/p5_job_${exp}.log" 2>&1 < /dev/null &
  CLAIM[$card]=$(( $(date +%s) + 600 ))
  log "launch $exp on card $card, $iters iters, cfg $cfg, extra: ${extra[*]}"
  sleep 20                           # let cycle.py appear before the line count is read again
done
log "stop file seen, scheduler exits (running jobs untouched)"
