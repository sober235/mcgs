#!/bin/bash
# Phase-2 chain C (card = $1): the rule-7 bundle at the promotion budget (volumes), the high-band knob, then the
# 4000-iteration seed floor in design B.
cd /data0/congcong/code/GS/mcgs_af12_wt
G=${1:?card}
SAVE_VOLUMES=1 ./autoresearch_jyz_af12_p2.sh p2_bundle_slr005_i6000 "$G" 6000 --set gaussian.scale_lr=0.005
./autoresearch_jyz_af12_p2.sh p2_khigh20 "$G" 1500 --kspace_high 2.0
./autoresearch_jyz_af12_p2.sh p2_P10_s43_full4k "$G" 4000 --set training.seed=43
