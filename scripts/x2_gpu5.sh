#!/bin/bash
# Card 5 after the seed-43 run: self-check of score_x2.py (members A A must reproduce A), then the P10 two-seed
# average row ar5_P10_x2_full4k. Runs on the card (CUDA) so the 3-D SSIM does not exhaust host memory.
cd /data0/congcong/code/GS/mcgs_af12_wt
until [ -f results_arc/ar5_P10_s43_full4k_score.json ] || grep -q -E 'Traceback|Killed' results_arc/ar5_P10_s43_full4k.log 2>/dev/null; do sleep 90; done
export PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
$PY scripts/score_x2.py --out _selfcheck_x2 --members ar5_best_C10_T112_full4k ar5_best_C10_T112_full4k > results_arc/_selfcheck_x2.log 2>&1
$PY scripts/score_x2.py --out ar5_P10_x2_full4k --members ar5_best_C10_T112_full4k ar5_P10_s43_full4k > results_arc/ar5_P10_x2_full4k.log 2>&1
$PY scripts/autoresearch_jyz_af12_row.py --exp ar5_P10_x2_full4k --status x2-2xcompute --note "voxelwise mean of the magnitudes of seeds 42 and 43 (P10), 2x compute"
