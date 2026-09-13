#!/bin/bash
# Phase-3 round 1: bracket the common-edge lever on our own data, and the phase-TV dose curve.
cd /data0/congcong/code/GS/mcgs_af12_wt
setsid nohup ./autoresearch_jyz_af12_p3.sh p3_edge00005 0 1500 --lambda_edge 0.0005 > results_arc/p3_edge00005_gpu0.out 2>&1 &
sleep 3
setsid nohup ./autoresearch_jyz_af12_p3.sh p3_edge0005  1 1500 --lambda_edge 0.005  > results_arc/p3_edge0005_gpu1.out  2>&1 &
sleep 3
setsid nohup ./autoresearch_jyz_af12_p3.sh p3_ptv001    3 1500 --lambda_ptv 0.01    > results_arc/p3_ptv001_gpu3.out    2>&1 &
sleep 3
echo LAUNCHED_3
