#!/usr/bin/env bash
# confirm_cmp.sh -- build the _cmp confirmation topic with the improved comparison arms and rerun the registered line's
# bookkeeping on it (gauge check of the replaced volumes, eleven-arm table, enhancement fidelity, per-slice ranking, lesion
# slices), the way confirm_ar2.sh did for the MC-GS winner; FIGS=1 also renders the PNG figures (three planes, enhancement,
# lesion planes) with the tag _cmp, so no figure of record is overwritten. 2026-10-09/10 comparison-arm audit.
#   bash scripts/confirm_cmp.sh <out_topic> "<--replace ARM:SCAN=KEY ...>" "<one-line note>"
set -u
OUT=${1:?out topic}; REPL=${2:-}; NOTE=${3:-}
SRC=/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois
S=/data0/congcong/code/GS/mcgs_ce_mask_wt/ce_retro_cssense_af16/scripts
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
E="env RETRO_TOPIC=$OUT RETRO_TRAINER=$SRC/trainer/_trainer_e778bf2_ar PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16"
env PYTHONNOUSERSITE=1 PYTHONPATH= $PY $S/make_cmp_topic.py --src $SRC --out "$OUT" $REPL --note "$NOTE" || exit 1
cd "$OUT"
$E $PY $S/cmp_gauge_check.py > logs/gauge_check_cmp.txt 2>&1; tail -1 logs/gauge_check_cmp.txt
$E $PY $S/summarize.py > logs/summarize_cmp.out 2>&1
$E $PY $S/enhancement_fidelity.py > logs/enhancement_cmp.out 2>&1
$E $PY $S/readme_tables.py > logs/readme_tables_cmp.md 2>&1
$E $PY $S/find_best_slice.py > logs/find_best_slice_cmp.out 2>&1
$E $PY $S/best_slice_mcgs.py --axis 0 > logs/best_slice_mcgs_cmp.out 2>&1
$E $PY $S/lesion_slice_allmetrics.py > logs/lesion_allmetrics_cmp.out 2>&1
if [ "${FIGS:-0}" = 1 ]; then
  for s in pre post; do
    $E $PY $S/fig_three_planes_compact.py --scan $s --slices 201 131 48 --inset 48 --inset_corner tr bl bl --mask_error --gray_cbar \
      --cbar_planes 0 --tra_rot 1 --err_vmax 0.1 --with_zf --caps --l1_arm spirit --brightness 1.5 --tag _cmp > logs/fig_three_planes_${s}_cmp.out 2>&1
  done
  $E $PY $S/fig_enhancement.py --slices 201,131,48 --vmax 0.667 --suffix _cmp > logs/fig_enhancement_cmp.out 2>&1
  $E $PY $S/fig_lesion_planes_zoom.py --tra 204 --cor 118 --sag 44 --order plane --tag _cmp > logs/fig_lesion_planes_cmp.out 2>&1
fi
echo CONFIRM_DONE > logs/confirm_cmp_done
echo "done: $OUT/logs/readme_tables_cmp.md"
