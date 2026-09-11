#!/bin/bash
# Card 5 after ar6_db4_lev1_C085_T112: FLAIR 0.12 under the 1-level prior (next point of the curve), then the
# promotion runs (4000 iterations, volumes) of the three-way row with FLAIR 0.10 and of scale lr 0.003 alone.
cd /data0/congcong/code/GS/mcgs_af12_wt
until [ -f results_arc/ar6_db4_lev1_C085_T112_score.json ] || grep -q -E 'Traceback|Killed' results_arc/ar6_db4_lev1_C085_T112.log 2>/dev/null; do sleep 90; done
./autoresearch_jyz_af12.sh ar6_db4_lev1_C12 5 1500 --set loss.wav_levels=1 --set loss.lambda_wav=0.12
SAVE_VOLUMES=1 ./autoresearch_jyz_af12.sh ar5_db4_lev1_C10_full4k 5 4000 --set loss.wav_levels=1 --set loss.lambda_wav=0.10
SAVE_VOLUMES=1 ./autoresearch_jyz_af12.sh ar5_slr003_full4k 5 4000 --set gaussian.scale_lr=0.003
