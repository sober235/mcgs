#!/bin/bash
# Card 5 after ar6_db4_lev1_C12: promotion runs (4000 iterations, volumes) of the FLAIR-0.10 three-way row and of
# scale lr 0.005 alone.
cd /data0/congcong/code/GS/mcgs_af12_wt
until [ -f results_arc/ar6_db4_lev1_C12_score.json ] || grep -q -E 'Traceback|Killed' results_arc/ar6_db4_lev1_C12.log 2>/dev/null; do sleep 90; done
SAVE_VOLUMES=1 ./autoresearch_jyz_af12.sh ar5_db4_lev1_C10_full4k 5 4000 --set loss.wav_levels=1 --set loss.lambda_wav=0.10
SAVE_VOLUMES=1 ./autoresearch_jyz_af12.sh ar5_slr005_full4k 5 4000 --set gaussian.scale_lr=0.005
