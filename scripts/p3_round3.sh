#!/bin/bash
# Phase-3 round 3: capacity on the new base (the round-2 combination at 4.8M points), screening budget.
cd /data0/congcong/code/GS/mcgs_af12_wt
setsid nohup ./autoresearch_jyz_af12_p3.sh p3_e0001_ptv001_n48 4 1500 \
  --lambda_edge 0.001 --lambda_ptv 0.01 --set gaussian.initial_num_points=4800000 \
  > results_arc/p3_e0001_ptv001_n48_gpu4.out 2>&1 &
echo LAUNCHED
