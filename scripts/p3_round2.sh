#!/bin/bash
# Phase-3 round 2: the rest of the phase-TV dose curve and the first combination.
cd /data0/congcong/code/GS/mcgs_af12_wt
setsid nohup ./autoresearch_jyz_af12_p3.sh p3_ptv002        0 1500 --lambda_ptv 0.02                       > results_arc/p3_ptv002_gpu0.out        2>&1 &
sleep 3
setsid nohup ./autoresearch_jyz_af12_p3.sh p3_ptv003        1 1500 --lambda_ptv 0.03                       > results_arc/p3_ptv003_gpu1.out        2>&1 &
sleep 3
setsid nohup ./autoresearch_jyz_af12_p3.sh p3_e0001_ptv001  3 1500 --lambda_edge 0.001 --lambda_ptv 0.01   > results_arc/p3_e0001_ptv001_gpu3.out  2>&1 &
sleep 3
echo LAUNCHED_3
