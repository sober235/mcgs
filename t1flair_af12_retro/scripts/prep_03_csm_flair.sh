#!/bin/bash
# FLAIR ESPIRiT map only: the T1 map is byte-identical to the one already computed for the
# T1+T2 topic (same CFL input, same recipe) and is hard-linked into data/, so it is not recomputed.
set -euo pipefail
export TOOLBOX_PATH=/home/congcongliu/src/bart
export PATH=$TOOLBOX_PATH:$PATH
R=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_retro
SRC=/data1/data1_congcong/data/MC_Jia/Train_SpiritDiffusion_CFL/vol1_huangyuwei
echo "=== FLAIR $(date -Is)"
time bart ecalib -t 0.001 -c 0.0 -r 24 -m 1 "$SRC/flair_train_huangyuwei_cc" "$R/data/csm_espirit_FLAIR"
echo "ALL_DONE $(date -Is)"
