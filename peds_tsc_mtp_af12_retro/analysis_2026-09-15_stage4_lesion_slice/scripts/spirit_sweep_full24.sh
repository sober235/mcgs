#!/usr/bin/env bash
# spirit_sweep_full24.sh -- as spirit_sweep.sh but --calib full24 (kernel from the full k-space centre 24 x 24,
# the block the ESPIRiT maps came from; the sampled ACS here is only 24 x 8). L1-SPIRiT sweep on this topic (user 2026-09-15: L1-SPIRiT replaces L1-ESPIRiT in the comparison).
# One chain per contrast, run in parallel; each job waits for a card with no compute process, utilisation <= 10 %
# and < 1500 MiB used at two polls 60 s apart, and for this line's GPU jobs (GS trainings + SPIRiT runs, every
# session's) to be fewer than LINE_MAX. A job whose log json exists is skipped, so the sweep can be restarted.
# Order: the jyz optimum region first (ksize 5, lambda 1e-3..2e-3), the grid edges last.
set -uo pipefail
T=/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro
LINE_MAX=${LINE_MAX:-2}
LOG=$T/logs/spirit_sweep.log
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
GRID="5:0.002 5:0.001 3:0.002 3:0.001 5:0.004 3:0.004 5:0.0005 3:0.0005"   # the middle of the lambda grid
line_jobs() {
  local a b
  a=$(pgrep -fc "^[^ ]*python[^ ]* -u arc/cycle.py --exp .*--base_cfg configs/peds_" || true)
  b=$(pgrep -fc "^[^ ]*python[^ ]* (-u )?[^ ]*recon_spirit\.py --contrast (T1w|T2s) " || true)   # T1w/T2s = this topic (jyz runs use T1/FLAIR)
  echo $(( a + b ))
}
card_ok() {
  local g=$1 bus apps q util mem
  bus=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | awk -F', ' -v i="$g" '$1==i{print $2}')
  [[ -n "$bus" ]] || return 1
  apps=$(nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader) || return 1
  grep -qF "$bus" <<< "$apps" && return 1
  q=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i "$g") || return 1
  util=$(echo "$q" | cut -d, -f1 | tr -d ' '); mem=$(echo "$q" | cut -d, -f2 | tr -d ' ')
  [[ "$util" -le 10 && "$mem" -lt 1500 ]]
}
claim_dir=$T/logs/.spirit_claims; mkdir -p "$claim_dir"
claim() {                          # atomic: mkdir succeeds for exactly one chain; a claim older than 10 min is stale
  local d="$claim_dir/$1.lock"
  mkdir "$d" 2>/dev/null && return 0
  if [[ $(( $(date +%s) - $(stat -c %Y "$d" 2>/dev/null || echo 0) )) -ge 600 ]]; then
    rmdir "$d" 2>/dev/null; mkdir "$d" 2>/dev/null && return 0
  fi
  return 1
}
chain() {
  local c=$1 kl k lam sfx card
  for kl in $GRID; do
    k=${kl%%:*}; lam=${kl##*:}; sfx="_full24_ks${k}_lam${lam}"
    [[ -e "$T/logs/spirit_${c}${sfx}.json" ]] && continue
    card=""
    while [[ -z "$card" ]]; do
      if [[ "$(line_jobs)" -lt "$LINE_MAX" ]]; then
        for g in 0 1 2 3 4 5 6 7; do
          [[ -d "$claim_dir/$g.lock" && $(( $(date +%s) - $(stat -c %Y "$claim_dir/$g.lock") )) -lt 600 ]] && continue
          card_ok "$g" || continue
          sleep 60
          card_ok "$g" && [[ "$(line_jobs)" -lt "$LINE_MAX" ]] && claim "$g" && { card=$g; break; }
        done
      fi
      [[ -z "$card" ]] && sleep 60
    done
    log "$c full24 ks$k lam$lam START card $card"
    ( cd "$T" && ./scripts/py -u scripts/recon_spirit.py --contrast "$c" --gpu "$card" --ksize "$k" --lam "$lam" --calib full24 --suffix "$sfx" ) \
        > "$T/logs/spirit_${c}${sfx}.log" 2>&1
    log "$c full24 ks$k lam$lam rc=$? $(grep -oE 'head_psnr=[0-9.]+' "$T/logs/spirit_${c}${sfx}.log" | tail -1)"
  done
  log "CHAIN full24 $c DONE"
}
log "full24 sweep up: line_max=$LINE_MAX grid=[$GRID]"
chain T1w & sleep 90
chain T2s &
wait
log "ALL DONE full24"
