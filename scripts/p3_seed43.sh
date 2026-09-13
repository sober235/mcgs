#!/bin/bash
# Phase-3: the seed repeat of the promoted recipe P14 — the floor at 4000 in this arena, and half of a two-seed row.
cd /data0/congcong/code/GS/mcgs_af12_wt
SAVE_VOLUMES=1 setsid nohup ./autoresearch_jyz_af12_p3.sh p3_e0001_ptv001_s43_full4k 0 4000 \
  --lambda_edge 0.001 --lambda_ptv 0.01 --set training.seed=43 > results_arc/p3_s43_gpu0.out 2>&1 &
echo LAUNCHED
