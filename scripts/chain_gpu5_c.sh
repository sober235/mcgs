#!/bin/bash
# Card 5 after the seed-floor run: round-2 single knobs (capacity down, budget, Charbonnier eps, w_floor).
# Gated on the flag so the card can be lent to the AF16 phase-2 loop first if it still wants it.
cd /data0/congcong/code/GS/mcgs_af12_wt
FLAG=/data0/congcong/code/GS/mcgs_af12_wt/results_arc/.gpu5_go
until [ -f "$FLAG" ] && [ -f results_arc/ar1_P6_i1500_s43_score.json ]; do sleep 60; done
./autoresearch_jyz_af12.sh ar2_n24 5 1500 --set gaussian.initial_num_points=2400000
./autoresearch_jyz_af12.sh ar3_i2500 5 2500
./autoresearch_jyz_af12.sh ar2_eps003 5 1500 --tv_eps 0.03
./autoresearch_jyz_af12.sh ar2_wfloor02 5 1500 --edge_tv_wfloor 0.20
