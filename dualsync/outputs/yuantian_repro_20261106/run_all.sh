#!/bin/bash
# Yuantian port, default parameters, 3 profiles x 2 registered slices = 6 GPU runs (user budget 6).
set -u
cd /data0/zhuoxu/code/DualSync
O=/data0/zhuoxu/code/DualSync/outputs/yuantian_repro_20261106
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
for P in sharp balanced legacy; do
  echo "=== $P start $(date -Is)" >> $O/progress.log
  /home/congcongliu/anaconda3/envs/pt1.10/bin/python main_yuantian.py --gpu 4 --variant ${P}_seed1234 \
    --profile $P --seed 1234 --output_dir $O --slices MSP-FSE-T/slice_06,DYF-FSE-T/slice_01 > $O/${P}.log 2>&1
  echo "=== $P exit=$? $(date -Is)" >> $O/progress.log
done
echo ALL_DONE >> $O/progress.log
