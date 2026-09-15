#!/bin/bash
# run_slice_trend.sh -- slice_trend.py for T1 and FLAIR, detached, CPU (about 3 min each). Needs logs/rescore_*.json first.
T=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_16_20_24_trend_v3_jiangyizhou
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
cd /data0/congcong/code/GS/mcgs_af16_wt
for c in T1 FLAIR; do
  setsid nohup env PYTHONNOUSERSITE=1 PYTHONPATH= OMP_NUM_THREADS=4 nice -n 10 $PY $T/scripts/slice_trend.py $c \
      > $T/logs/slice_trend_$c.log 2>&1 < /dev/null &
done
