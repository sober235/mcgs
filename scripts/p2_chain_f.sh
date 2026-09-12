#!/bin/bash
# Phase-2 chain F (card = $1): round 2 part 4, the optimiser grid — rotation lr 0.002 with scale lr 0.005 at the
# screening budget (the conditional items are queued by hand once p2_rotlr004 is read).
cd /data0/congcong/code/GS/mcgs_af12_wt
G=${1:?card}
./autoresearch_jyz_af12_p2g.sh p2_rotlr002_slr005 "$G" 1500 --set gaussian.rotation_lr=0.002 --set gaussian.scale_lr=0.005
