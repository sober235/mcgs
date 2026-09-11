#!/bin/bash
# Card 4 after ar6_wavlev1_wavC085: the promotion run of the db4 + 1-level combination (4000 it, volumes), then the
# rest of round 3 on this card.
cd /data0/congcong/code/GS/mcgs_af12_wt
until [ -f results_arc/ar6_wavlev1_wavC085_score.json ] || grep -q -E 'Traceback|Killed' results_arc/ar6_wavlev1_wavC085.log 2>/dev/null; do sleep 90; done
SAVE_VOLUMES=1 ./autoresearch_jyz_af12.sh ar5_db4_lev1_full4k 4 4000 --set loss.wav_name=db4 --set loss.wav_levels=1
./autoresearch_jyz_af12.sh ar6_db6 4 1500 --set loss.wav_name=db6
./autoresearch_jyz_af12.sh ar6_sym4 4 1500 --set loss.wav_name=sym4
./autoresearch_jyz_af12.sh ar6_wavT1_012 4 1500 --set loss.lambda_wav_plain=0.12
./autoresearch_jyz_af12.sh ar6_waveps 4 1500 --set loss.wav_eps=0.001
./autoresearch_jyz_af12.sh ar6_density_lr02 4 1500 --set gaussian.density_lr=0.02
