#!/usr/bin/env bash
# peds_sched.sh QUEUE -- run the queue one job at a time on an EMPTY allowed card.
# ALLOWED="0 1 3" picks the cards (stage 2 rule was card 2 only; stage 3 allows any empty card, at most
# two of this line's trainings at once, which is two instances with disjoint ALLOWED lists).
# TAG=r1a keeps two instances' logs and done-markers apart.
# Queue lines: "classical <c>" | "cssense <c> <lam>" | "jcssense <lam>" | "zsssdu <c>" | "dip <c>" |
#              "inr <c>" | "gs <joint|single_T1w|single_T2s>" | "p3 <exp> <config> [extra cycle.py args]".
# Blank lines and # comments are ignored.
# A line whose result file exists is skipped, so the scheduler can be killed and restarted at any time.
# Touch logs/.sched_pause to hold further launches. Exits when every queue line has its result.
set -uo pipefail
T=/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro
W=/data0/congcong/code/GS/mcgs_peds_wt
Q=${1:?queue file}
ALLOWED=${ALLOWED:-"2"}
TAG=${TAG:-}
SUF=${TAG:+_$TAG}
log() { echo "$(date '+%F %T') $*" | tee -a "$T/logs/sched$SUF.log"; }

free_card() {
  for g in $ALLOWED; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
    [[ -z "$used" ]] && continue
    procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" 2>/dev/null | grep -c .)
    if [[ "$used" -lt 1500 && "$procs" -eq 0 ]]; then echo "$g"; return 0; fi
  done
  return 1
}

done_file() {                      # done_file <job words...> -> the file that marks it finished
  case "$1" in
    classical) echo "$T/logs/classical_$2.json" ;;
    cssense)   echo "$T/logs/classical_$2_o200_lam$3.json" ;;
    jcssense)  echo "$T/logs/jcssense_o200_lam$2.json" ;;
    zsssdu)    echo "$T/logs/zsssdu_$2.json" ;;
    dip)       echo "$T/logs/dip_$2.json" ;;
    inr)       echo "$T/logs/inr_$2.json" ;;
    gs)        echo "$W/results_arc/peds_af12_$2_score.json" ;;
    p3)        echo "$W/results_arc/$2_score.json" ;;
    *)         echo "" ;;
  esac
}

command_for() {                    # command_for <card> <job words...>
  local g=$1; shift
  case "$1" in
    classical) echo "cd $T && ./scripts/py scripts/recon_classical.py --contrast $2 --gpu $g" ;;
    cssense)   echo "cd $T && ./scripts/py scripts/recon_classical.py --contrast $2 --gpu $g --methods cssense --admm_outer 200 --lam $3 --suffix _o200_lam$3" ;;
    jcssense)  echo "cd $T && ./scripts/py scripts/recon_joint_cssense.py --topic $T --contrasts T1w,T2s --gpu $g --admm_outer 200 --lam $2 --suffix _o200_lam$2" ;;
    zsssdu)    echo "cd $T && ./scripts/py scripts/recon_zsssdu.py --contrast $2 --gpu $g --epochs 100" ;;
    dip)       echo "cd $T && ./scripts/py scripts/recon_dip_inr.py --method dip --contrast $2 --gpu $g --iters 5000 --lr 3e-3 --eval_every 100" ;;
    inr)       echo "cd $T && ./scripts/py scripts/recon_dip_inr.py --method inr --contrast $2 --gpu $g --iters 1500 --eval_every 50" ;;
    gs)        local cfg=configs/peds_af12_T1w_T2s.yaml
               [[ "$2" == single_T1w ]] && cfg=configs/peds_af12_single_T1w.yaml
               [[ "$2" == single_T2s ]] && cfg=configs/peds_af12_single_T2s.yaml
               echo "cd $W && CONFIG=$cfg SAVE_VOLUMES=1 ./autoresearch_peds.sh peds_af12_$2 $g 4000" ;;
    p3)        local exp=$2 cfg=$3; shift 3
               echo "cd $W && CONFIG=$cfg SAVE_VOLUMES=1 ./autoresearch_peds.sh $exp $g ${ITERS:-4000} $*" ;;
  esac
}

log "scheduler up, queue=$Q allowed=[$ALLOWED]"
while true; do
  pending=0; ran=0
  while read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "${line:0:1}" == "#" ]] && continue
    set -- $line
    df=$(done_file "$@")
    [[ -n "$df" && -e "$df" ]] && continue
    [[ -z "$df" ]] && { log "bad queue line: $*"; continue; }
    pending=1
    [[ -e "$T/logs/.sched_pause" ]] && { log "paused"; break; }
    card=$(free_card) || break
    cmd=$(command_for "$card" "$@")
    jl="$T/logs/job_$(echo "$*" | tr ' ./' '___' | cut -c1-120).log"
    log "launch [$*] on card $card -> $jl"
    bash -c "$cmd" >> "$jl" 2>&1
    rc=$?
    st=MISSING; [[ -e "$df" ]] && st=present
    log "done [$*] rc=$rc result $st"
    printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$*" "$rc" "$st" >> "$T/logs/sched_status$SUF.tsv"
    ran=1
    break                          # re-read the queue from the top after every job
  done < "$Q"
  if [[ "$pending" -eq 0 ]]; then log "ALL DONE"; touch "$T/logs/DONE_SCHED$SUF"; break; fi
  [[ "$ran" -eq 0 ]] && sleep 60
done
