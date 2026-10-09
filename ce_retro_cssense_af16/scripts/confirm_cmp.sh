#!/usr/bin/env bash
# confirm_cmp.sh -- build the _cmp confirmation topic with the improved comparison arms and rerun the registered line's
# bookkeeping on it (gauge check of the replaced volumes, eleven-arm table, enhancement fidelity, per-slice ranking, lesion
# slices), the way confirm_ar2.sh did for the MC-GS winner. 2026-10-09 comparison-arm audit.
#   bash scripts/confirm_cmp.sh <out_topic> "<--replace ARM:SCAN=KEY ...>" "<one-line note>"
# e.g. bash scripts/confirm_cmp.sh /data0/.../CE_retro_cssense_af16_vdpois_cmp "--replace inr:pre=inr_i1_n256_sup6 --replace inr:post=..." "round 1"
set -u
OUT=${1:?out topic}; REPL=${2:-}; NOTE=${3:-}
SRC=/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois
S=/data0/congcong/code/GS/mcgs_ce_mask_wt/ce_retro_cssense_af16/scripts
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
E="env RETRO_TOPIC=$OUT RETRO_TRAINER=$SRC/trainer/_trainer_e778bf2_ar PYTHONNOUSERSITE=1 PYTHONPATH="
env PYTHONNOUSERSITE=1 PYTHONPATH= $PY $S/make_cmp_topic.py --src $SRC --out "$OUT" $REPL --note "$NOTE" || exit 1
cd "$OUT"
# the gauge check: every replaced volume re-scored through score_volume.py against the row make_cmp_topic.py installed
for spec in $(echo "$REPL" | tr ' ' '\n' | grep -v '^--replace$'); do
  arm=${spec%%:*}; rest=${spec#*:}; scan=${rest%%=*}
  case $arm in cgsense) js=logs/classical_${scan}.json;; *) js=logs/${arm}_${scan}.json;; esac
  $E $PY $S/score_volume.py --scan $scan --file recon/${arm}_${scan}_complex.npy --method $arm --check $js --gpu -1 2>&1 | tail -1
done > logs/gauge_check_cmp.txt
cat logs/gauge_check_cmp.txt
$E $PY $S/summarize.py > logs/summarize_cmp.out 2>&1
$E $PY $S/enhancement_fidelity.py > logs/enhancement_cmp.out 2>&1
$E $PY $S/readme_tables.py > logs/readme_tables_cmp.md 2>&1
$E $PY $S/find_best_slice.py > logs/find_best_slice_cmp.out 2>&1
$E $PY $S/best_slice_mcgs.py --axis 0 > logs/best_slice_mcgs_cmp.out 2>&1
$E $PY $S/lesion_slice_allmetrics.py > logs/lesion_allmetrics_cmp.out 2>&1
echo CONFIRM_DONE > logs/confirm_cmp_done
echo "done: $OUT/logs/readme_tables_cmp.md"
