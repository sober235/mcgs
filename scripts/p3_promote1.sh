#!/bin/bash
# Phase-3 promotion 1: the combination at the full budget, plus the two GS-single arms at the same recipe
# (rule 4: the FLAIR verdict is the sign of joint - single, which needs both).
cd /data0/congcong/code/GS/mcgs_af12_wt
ARGS="--lambda_edge 0.001 --lambda_ptv 0.01"
SAVE_VOLUMES=1 setsid nohup ./autoresearch_jyz_af12_p3.sh p3_e0001_ptv001_full4k 0 4000 $ARGS > results_arc/p3_e0001_ptv001_full4k_gpu0.out 2>&1 &
sleep 3
SAVE_VOLUMES=1 CONFIG=configs/t1flair_v3_af12_hyw_singleT1.yaml setsid nohup ./autoresearch_jyz_af12_p3.sh p3_e0001_ptv001_singleT1_full4k 1 4000 $ARGS > results_arc/p3_singleT1_gpu1.out 2>&1 &
sleep 3
SAVE_VOLUMES=1 CONFIG=configs/t1flair_v3_af12_hyw_singleFLAIR.yaml setsid nohup ./autoresearch_jyz_af12_p3.sh p3_e0001_ptv001_singleFLAIR_full4k 3 4000 $ARGS > results_arc/p3_singleFLAIR_gpu3.out 2>&1 &
sleep 3
echo LAUNCHED_3
