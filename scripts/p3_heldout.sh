#!/bin/bash
# Phase-3 held-out check: P14 unchanged on huangyuwei (T1FLAIR_AF12_retro), joint plus both GS-single arms.
cd /data0/congcong/code/GS/mcgs_af12_wt
ARGS="--lambda_edge 0.001 --lambda_ptv 0.01"
SAVE_VOLUMES=1 CONFIG=configs/t1flair_af12_hyw.yaml setsid nohup ./autoresearch_jyz_af12_p3.sh p3_hyw_P14_full4k 1 4000 $ARGS > results_arc/p3_hyw_joint_gpu1.out 2>&1 &
sleep 3
SAVE_VOLUMES=1 CONFIG=configs/t1flair_af12_hyw_singleFLAIR.yaml setsid nohup ./autoresearch_jyz_af12_p3.sh p3_hyw_P14_singleFLAIR_full4k 3 4000 $ARGS > results_arc/p3_hyw_sFL_gpu3.out 2>&1 &
sleep 3
echo LAUNCHED_2
