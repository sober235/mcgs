#!/bin/bash
# ESPIRiT CSM for the T1/FLAIR pair -- frozen house recipe: ecalib -t 0.001 -c 0.0 -r 24 -m 1.
# Input CFL is already in BART dim order (X=readout, Y, Z, C).
set -euo pipefail
export TOOLBOX_PATH=/home/congcongliu/src/bart
export PATH=$TOOLBOX_PATH:$PATH
R=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_v3_jiangyizhou
SRC=/data1/data1_congcong/data/MC_Jia/Train_SpiritDiffusion_CFL/vol3_jiangyizhou
for pair in "T1:t1_fsp_train_jiangyizhou_cc" "FLAIR:flair_train_jiangyizhou_cc"; do
  tag=${pair%%:*}; f=${pair##*:}
  echo "=== $tag  $(date -Is)"
  time bart ecalib -t 0.001 -c 0.0 -r 24 -m 1 "$SRC/$f" "$R/data/csm_espirit_${tag}"
done
echo "ALL_DONE $(date -Is)"
