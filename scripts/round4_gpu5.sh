#!/bin/bash
# Card 5 (idle after the P8 seed floor): two single steps beyond the best screening row (db4 + 1 level + FLAIR 0.085).
cd /data0/congcong/code/GS/mcgs_af12_wt
./autoresearch_jyz_af12.sh ar6_db4_lev1_C10 5 1500 --set loss.wav_levels=1 --set loss.lambda_wav=0.10
./autoresearch_jyz_af12.sh ar6_db4_lev1_C085_T112 5 1500 --set loss.wav_levels=1 --set loss.lambda_wav=0.085 --set loss.lambda_wav_plain=0.12
