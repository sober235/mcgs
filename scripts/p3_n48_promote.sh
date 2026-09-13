#!/bin/bash
# Phase-3: the capacity keep at the promotion budget — P14 at 4.8M points, 4000 it, volumes.
cd /data0/congcong/code/GS/mcgs_af12_wt
SAVE_VOLUMES=1 setsid nohup ./autoresearch_jyz_af12_p3.sh p3_e0001_ptv001_n48_full4k 3 4000 \
  --lambda_edge 0.001 --lambda_ptv 0.01 --set gaussian.initial_num_points=4800000 \
  > results_arc/p3_n48_full4k_gpu3.out 2>&1 &
echo LAUNCHED
