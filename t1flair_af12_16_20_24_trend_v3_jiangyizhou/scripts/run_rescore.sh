#!/bin/bash
# run_rescore.sh -- the eight re-score jobs (AF 12/16/20/24 x T1/FLAIR), CPU only, detached (the session memory watchdog kills
# attached background jobs). Outputs logs/rescore_af<AF>_<C>.{json,log}.
T=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_16_20_24_trend_v3_jiangyizhou
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
cd /data0/congcong/code/GS/mcgs_af16_wt
for af in 12 16 20 24; do for c in T1 FLAIR; do
  setsid nohup env PYTHONNOUSERSITE=1 PYTHONPATH= OMP_NUM_THREADS=2 nice -n 10 $PY $T/scripts/rescore_topic.py $af $c \
      > $T/logs/rescore_af${af}_${c}.log 2>&1 < /dev/null &
done; done
