#!/bin/bash
# Phase-2 promotion 1 (card = $1): rotation lr 0.001 at 4000 iterations with volumes (rule 2), then chain D.
cd /data0/congcong/code/GS/mcgs_af12_wt
G=${1:?card}
SAVE_VOLUMES=1 ./autoresearch_jyz_af12_p2.sh p2_rotlr_full4k "$G" 4000 --set gaussian.rotation_lr=0.001
bash scripts/p2_chain_d.sh "$G"
