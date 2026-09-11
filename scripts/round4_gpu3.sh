#!/bin/bash
# Card 3 after ar6_headfrac03: promotion run of T1 wavelet 0.12 on P9 (4000 iterations, volumes).
cd /data0/congcong/code/GS/mcgs_af12_wt
until [ -f results_arc/ar6_headfrac03_score.json ] || grep -q -E 'Traceback|Killed' results_arc/ar6_headfrac03.log 2>/dev/null; do sleep 90; done
SAVE_VOLUMES=1 ./autoresearch_jyz_af12.sh ar5_T112_full4k 3 4000 --set loss.lambda_wav_plain=0.12
