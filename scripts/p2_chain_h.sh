#!/bin/bash
# Phase-2 chain H (card = $1): round 2 part 5 — rotation lr 0.016 and rotation 0.008 + scale lr 0.005 at 1500, then the
# promotion of the best optimiser-grid row at 4000 (p2_promote_best_grid.sh).
cd /data0/congcong/code/GS/mcgs_af12_wt
G=${1:?card}
./autoresearch_jyz_af12_p2g.sh p2_rotlr016 "$G" 1500 --set gaussian.rotation_lr=0.016
./autoresearch_jyz_af12_p2g.sh p2_rotlr008_slr005 "$G" 1500 --set gaussian.rotation_lr=0.008 --set gaussian.scale_lr=0.005
bash scripts/p2_promote_best_grid.sh "$G"
