#!/bin/bash
# Phase-3 round 0: the two reference rows and the two common-edge screens, one card each (0, 1, 3, 4 were free).
cd /data0/congcong/code/GS/mcgs_af12_wt
setsid nohup ./autoresearch_jyz_af12_p3.sh p3_P13_i1500     0 1500                          > results_arc/p3_P13_i1500_gpu0.out     2>&1 &
sleep 3
setsid nohup ./autoresearch_jyz_af12_p3.sh p3_P13_i1500_s43 1 1500 --set training.seed=43   > results_arc/p3_P13_i1500_s43_gpu1.out 2>&1 &
sleep 3
setsid nohup ./autoresearch_jyz_af12_p3.sh p3_edge0001      3 1500 --lambda_edge 0.001      > results_arc/p3_edge0001_gpu3.out      2>&1 &
sleep 3
setsid nohup ./autoresearch_jyz_af12_p3.sh p3_edge0002      4 1500 --lambda_edge 0.002      > results_arc/p3_edge0002_gpu4.out      2>&1 &
sleep 3
echo LAUNCHED
