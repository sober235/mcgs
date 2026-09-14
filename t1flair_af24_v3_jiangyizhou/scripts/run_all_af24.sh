#!/bin/bash
# run_all_af24.sh -- every arm of the AF24 study, detached, one chain per card (cards chosen at launch, all distinct):
#   CARD_JOINT  MC-GS joint                       run_af24_mcgs.sh af24_p3_joint_full4k       (SAVE_VOLUMES=1, 4000 it)
#   CARD_SINGLE GS-single FLAIR -> GS-single T1   run_af24_mcgs.sh af24_p3_single{FLAIR,T1}_full4k (the AF20 order)
#   CARD_TV     classical base + converged-TV sweeps, and the L1-ESPIRiT scoring passes (BART itself runs on the CPU)
#   CARD_JTV    joint-TV sweep -> 3D-DIP T1 -> 3D-DIP FLAIR
#   CARD_ZT1    ZS-SSDU T1 -> 3D-INR T1
#   CARD_ZFL    ZS-SSDU FLAIR -> 3D-INR FLAIR
# Every GPU job waits for card_ok (scripts/card_ok_af24.sh). When every chain has exited, finalize_af24_jyz.sh registers
# the three MC-GS runs (after their processes exited, so the volume npz files are complete), rebuilds logs/summary.*,
# the README tables, the AF20 comparison and the figures.
# Launch (from a bash script file, detached, so the session's memory watchdog cannot kill it):
#   CARD_JOINT=.. CARD_SINGLE=.. CARD_TV=.. CARD_JTV=.. CARD_ZT1=.. CARD_ZFL=.. \
#     setsid nohup bash scripts/run_all_af24.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
J=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF24_v3_jiangyizhou
F=/data0/congcong/code/GS/mcgs_af16_wt
cd $J
: "${CARD_JOINT:?}" "${CARD_SINGLE:?}" "${CARD_TV:?}" "${CARD_JTV:?}" "${CARD_ZT1:?}" "${CARD_ZFL:?}"
CARDS="$CARD_JOINT $CARD_SINGLE $CARD_TV $CARD_JTV $CARD_ZT1 $CARD_ZFL"
[ "$(tr ' ' '\n' <<< "$CARDS" | sort -u | wc -l)" -eq 6 ] || { echo "cards must be distinct: $CARDS" >&2; exit 2; }
source scripts/card_ok_af24.sh
source scripts/run_stage2_af24.sh          # chain_zt1 / chain_zfl / chain_dip (its main block does not run when sourced)
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> logs/run_all_af24.log; }
gs() {   # gpu exp [T1|FLAIR]
  local g=$1 exp=$2 single=${3:-}
  [ -f "$F/results_arc/${exp}_score.json" ] && { log "$exp exists, skipped"; return; }
  until card_ok $g 38; do sleep 60; done
  log "$exp START gpu $g"
  (cd $F && SINGLE=$single SAVE_VOLUMES=1 ./run_af24_mcgs.sh $exp $g 4000)
  local rc=$?
  log "$exp rc=$rc"
}
log "START joint $CARD_JOINT | singles $CARD_SINGLE | tv $CARD_TV | jtv+dip $CARD_JTV | zsssdu+inr T1 $CARD_ZT1 FLAIR $CARD_ZFL"
( gs $CARD_JOINT af24_p3_joint_full4k; log "CHAIN_JOINT DONE" ) &
( gs $CARD_SINGLE af24_p3_singleFLAIR_full4k FLAIR; gs $CARD_SINGLE af24_p3_singleT1_full4k T1; log "CHAIN_SINGLE DONE" ) &
( until card_ok $CARD_TV 12 && card_ok $CARD_JTV 12; do sleep 60; done
  GPU_BASE=$CARD_TV GPU_TV=$CARD_TV GPU_L1=$CARD_TV GPU_JTV=$CARD_JTV bash scripts/run_classical_af24.sh
  log "CHAIN_CLASSICAL DONE (jtv chain finished on gpu $CARD_JTV)"
  chain_dip $CARD_JTV; log "CHAIN_DIP DONE" ) &
( chain_zt1 $CARD_ZT1 ) &
( chain_zfl $CARD_ZFL ) &
wait
log "ALL CHAINS EXITED"
bash scripts/finalize_af24_jyz.sh af24_p3_joint_full4k af24_p3_singleT1_full4k af24_p3_singleFLAIR_full4k >> logs/finalize_af24.log 2>&1
rc=$?
log "FINALIZE rc=$rc"
