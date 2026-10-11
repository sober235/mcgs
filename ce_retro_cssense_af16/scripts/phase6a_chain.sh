#!/usr/bin/env bash
# phase6a_chain.sh -- the smoke of the subtraction prior on the first empty card, then (only if it ran clean) the six phase-6a
# probes on up to three empty cards (2026-10-10). Never edit this file while it runs; write a new one.
set -u
N=/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois
W=/data0/congcong/code/GS/mcgs_ce_mask_wt/ce_retro_cssense_af16
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
cd $N
export RETRO_TOPIC=$N RETRO_TRAINER=$N/trainer/_trainer_e778bf2_ar PYTHONNOUSERSITE=1 PYTHONPATH=
echo "$(date '+%F %T') smoke queue"
WARMUP=20 $PY -u $W/scripts/ar_free_queue.py --jobs $W/ar_jobs_phase6a_smoke.tsv --max_run 1 --settle 10
LOGF=$RETRO_TRAINER/results_arc/ar_k0_smoke_r2.log
if [ ! -f logs/ar_k0_smoke_r2_pre.json ] || grep -q "Traceback" $LOGF || ! grep -q "\[sub\] iter=" $LOGF; then
  echo "$(date '+%F %T') SMOKE FAILED: see $LOGF"; exit 1
fi
echo "$(date '+%F %T') smoke ok: $(grep '\[sub\] iter=' $LOGF | tail -1)"
echo "$(date '+%F %T') phase 6a queue"
$PY -u $W/scripts/ar_free_queue.py --jobs $W/ar_jobs_phase6a.tsv --max_run 2 --settle 10
echo "$(date '+%F %T') PHASE6A_DONE"
