#!/bin/bash
# Phase-2 chain G (card = $1): round 2 part 4 item 2 — rotation lr 0.008 and rotation 0.004 + scale lr 0.005, 1500 each.
cd /data0/congcong/code/GS/mcgs_af12_wt
G=${1:?card}
./autoresearch_jyz_af12_p2g.sh p2_rotlr008 "$G" 1500 --set gaussian.rotation_lr=0.008
./autoresearch_jyz_af12_p2g.sh p2_rotlr004_slr005 "$G" 1500 --set gaussian.rotation_lr=0.004 --set gaussian.scale_lr=0.005
