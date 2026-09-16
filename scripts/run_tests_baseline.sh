#!/usr/bin/env bash
# baseline test run of this worktree before any code change (card given as $1); output results_arc/tests_baseline.txt
cd "$(dirname "$0")/.."
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${1:-7} \
  TORCH_EXTENSIONS_DIR=/data0/congcong/code/GS/mcgs_gauss_wt.torch_ext \
  /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests -q --ignore=tests/smoke_test.py --ignore=tests/test_score_edge.py \
  > results_arc/tests_baseline.txt 2>&1
echo "TESTS_RC=$?" >> results_arc/tests_baseline.txt
