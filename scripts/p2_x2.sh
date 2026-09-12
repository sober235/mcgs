#!/bin/bash
# The P13 two-seed average row (rule 4), on card $1 alongside whatever trains there: score_x2.py is a short scorer,
# not a trainer, and the 3-D SSIM must run on CUDA or the host memory watchdog kills it. Self-check first
# (members A A must reproduce A's own score json).
cd /data0/congcong/code/GS/mcgs_af12_wt
export PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${1:?card}
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
$PY scripts/score_x2.py --out _selfcheck_x2 --members p2_rotlr008_full4k p2_rotlr008_full4k > results_arc/_selfcheck_x2.log 2>&1
$PY scripts/score_x2.py --out p2_P13_x2_full4k --members p2_rotlr008_full4k p2_P13_s43_full4k > results_arc/p2_P13_x2_full4k.log 2>&1
$PY scripts/autoresearch_jyz_af12_row.py --exp p2_P13_x2_full4k --status x2-2xcompute --note "voxelwise mean of the magnitudes of seeds 42 and 43 (P13, design B), 2x compute"
