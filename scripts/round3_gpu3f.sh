#!/bin/bash
# Card 3 after ar6_scale_lr003: promotion run of the three-way combination (db4 + 1 level + FLAIR wavelet 0.085,
# 4000 iterations, volumes), then the last round-3 knob on this card.
cd /data0/congcong/code/GS/mcgs_af12_wt
until [ -f results_arc/ar6_scale_lr003_score.json ] || grep -q -E 'Traceback|Killed' results_arc/ar6_scale_lr003.log 2>/dev/null; do sleep 90; done
SAVE_VOLUMES=1 ./autoresearch_jyz_af12.sh ar5_db4_lev1_C085_full4k 3 4000 --set loss.wav_levels=1 --set loss.lambda_wav=0.085
./autoresearch_jyz_af12.sh ar6_headfrac03 3 1500 --set gaussian.init_head_frac=0.03
