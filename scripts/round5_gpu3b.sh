#!/bin/bash
# Card 3 (idle): the deferred sym4 promotion on P10 (4000 iterations, volumes).
cd /data0/congcong/code/GS/mcgs_af12_wt
SAVE_VOLUMES=1 ./autoresearch_jyz_af12.sh ar5_P10_sym4_full4k 3 4000 --set loss.wav_name=sym4
