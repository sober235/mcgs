#!/bin/bash
# Phase-2 delivery chain (card = $1): the runs the final recipe P13 (= P10 + gaussian.rotation_lr 0.008) needs.
#   p2_P13_s43_full4k          seed 43 of P13, design B  -> the floor at the new working point + half the two-seed row
#   p2_P13_singleT1_full4k     GS-single T1  (AF12 mask, unchanged in design B)   rule 5
#   p2_P13_singleFLAIR_full4k  GS-single FLAIR (AF11 mask, design B)              rule 5
#   p2_P13_af12_full4k         P13 in the ALL-AF12 config -> the T1 panel of the delivered figure
# All at 4000 iterations with volumes. Two copies of this chain on two cards share the work (the runner refuses to
# overwrite an existing experiment, and its guard keeps one trainer per card).
cd /data0/congcong/code/GS/mcgs_af12_wt
G=${1:?card}
R=(--set gaussian.rotation_lr=0.008)
SAVE_VOLUMES=1 ./autoresearch_jyz_af12_p2g.sh p2_P13_s43_full4k "$G" 4000 "${R[@]}" --set training.seed=43
SAVE_VOLUMES=1 CONFIG=configs/t1flair_v3_af12_hyw_singleT1.yaml ./autoresearch_jyz_af12_p2g.sh p2_P13_singleT1_full4k "$G" 4000 "${R[@]}"
SAVE_VOLUMES=1 CONFIG=configs/t1flair_v3_af12_hyw_singleFLAIR_af11.yaml ./autoresearch_jyz_af12_p2g.sh p2_P13_singleFLAIR_full4k "$G" 4000 "${R[@]}"
SAVE_VOLUMES=1 CONFIG=configs/t1flair_v3_af12_hyw.yaml ./autoresearch_jyz_af12_p2g.sh p2_P13_af12_full4k "$G" 4000 "${R[@]}"
