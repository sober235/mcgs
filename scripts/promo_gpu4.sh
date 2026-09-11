#!/bin/bash
# Card 4: after ar4_kcenter015, the promotion run of the wavelet-2-levels keep (4000 iterations, volumes), then the
# remaining round-2 knob on this card, the 1-level follow-up, and the 6.0M capacity point.
cd /data0/congcong/code/GS/mcgs_af12_wt
until [ -f results_arc/ar4_kcenter015_score.json ] || grep -q -E 'Traceback|Killed' results_arc/ar4_kcenter015.log 2>/dev/null; do sleep 90; done
SAVE_VOLUMES=1 ./autoresearch_jyz_af12.sh ar5_wavlev2_full4k 4 4000 --set loss.wav_levels=2
./autoresearch_jyz_af12.sh ar4_wavC085 4 1500 --set loss.lambda_wav=0.085
./autoresearch_jyz_af12.sh ar4_wavlev1 4 1500 --set loss.wav_levels=1
./autoresearch_jyz_af12.sh ar2_n60 4 1500 --set gaussian.initial_num_points=6000000
