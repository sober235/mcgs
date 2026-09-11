#!/bin/bash
# Card 3: after the 4.8M promotion run, the combination of the two kept knobs (4.8M points + wavelet 2 levels) at
# 4000 iterations with volumes, then the warm-up knob left from round 2.
cd /data0/congcong/code/GS/mcgs_af12_wt
until [ -f results_arc/ar5_n48_full4k_score.json ] || grep -q -E 'Traceback|Killed' results_arc/ar5_n48_full4k.log 2>/dev/null; do sleep 90; done
SAVE_VOLUMES=1 ./autoresearch_jyz_af12.sh ar5_n48_wavlev2_full4k 3 4000 --set gaussian.initial_num_points=4800000 --set loss.wav_levels=2
./autoresearch_jyz_af12.sh ar4_warmup1000 3 1500 --warmup 1000
