#!/bin/bash
# postprocess.sh <hyw|jyz> <gpu>
#   Everything after the fifteen arms have finished: pull the MC-GS volumes out of the worktree's
#   npz archives, build the summary table, prove the gauge on one classical volume, draw the
#   three-plane panels and the butterfly slice (same slice index as the AF12 topic of the same
#   subject), the per-slice sweep, the AF12-vs-AF16 comparison, and the README tables.
set -uo pipefail
SUBJ=$1; GPU=$2
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
E="PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID"
case $SUBJ in
  hyw) R=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_retro;          R12=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_retro;          SLICE=154; FIGS=/home/congcongliu/figs/t1flair_af16 ;;
  jyz) R=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou; R12=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_v3_jiangyizhou; SLICE=168; FIGS=/home/congcongliu/figs/t1flair_v3_af16 ;;
  *) echo "usage: $0 <hyw|jyz> <gpu>"; exit 2 ;;
esac
S=$R/scripts; L=$R/logs
step() { echo "=== [$(date +%H:%M:%S)] $*"; }
step extract;      env $E $PY $S/extract_gs_volumes.py            > $L/post_extract.log 2>&1;      tail -3 $L/post_extract.log
step summarize;    env $E $PY $S/summarize.py                      > $L/post_summarize.log 2>&1;    grep -c . $L/summary.tsv
step gauge;        env $E $PY $S/check_gauge.py --contrast T1 --method cgsense --gpu $GPU > $L/post_gauge.log 2>&1; tail -1 $L/post_gauge.log
for c in T1 FLAIR; do
  step figures $c; env $E $PY $S/make_figures.py --contrast $c   > $L/post_figures_$c.log 2>&1;   tail -1 $L/post_figures_$c.log
  step butterfly $c; env $E $PY $S/fig_slice.py --contrast $c --axis 0 --index $SLICE --tag butterfly > $L/post_butterfly_$c.log 2>&1; tail -2 $L/post_butterfly_$c.log | cut -c1-200
done
step per-slice;    env $E $PY $S/find_best_slice.py                > $L/per_slice_sweep.log 2>&1;  grep "ranks 1st\|wrote" $L/per_slice_sweep.log
step compare;      env $E $PY $S/compare_topics.py                 > $L/compare_af12_af16.txt 2>&1; head -40 $L/compare_af12_af16.txt
step readme-tables; env $E $PY $S/readme_tables.py $R $R12         > $L/readme_tables.md 2>&1;      head -5 $L/readme_tables.md
mkdir -p $FIGS; cp $R/figures/*.png $FIGS/ 2>/dev/null; ls $FIGS | head -20
echo "POST_DONE $SUBJ"
