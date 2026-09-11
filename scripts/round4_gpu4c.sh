#!/bin/bash
# Card 4 after ar6_best_C10_T112: scale lr 0.008 (next point of the curve), T1 0.14 on the union row, scale lr 0.005
# on the union row, then the remaining single knobs.
cd /data0/congcong/code/GS/mcgs_af12_wt
until [ -f results_arc/ar6_best_C10_T112_score.json ] || grep -q -E 'Traceback|Killed' results_arc/ar6_best_C10_T112.log 2>/dev/null; do sleep 90; done
./autoresearch_jyz_af12.sh ar6_slr008 4 1500 --set gaussian.scale_lr=0.008
./autoresearch_jyz_af12.sh ar6_best_C10_T114 4 1500 --set loss.wav_levels=1 --set loss.lambda_wav=0.10 --set loss.lambda_wav_plain=0.14
./autoresearch_jyz_af12.sh ar6_best_C10_T112_slr005 4 1500 --set loss.wav_levels=1 --set loss.lambda_wav=0.10 --set loss.lambda_wav_plain=0.12 --set gaussian.scale_lr=0.005
./autoresearch_jyz_af12.sh ar6_sym4 4 1500 --set loss.wav_name=sym4
./autoresearch_jyz_af12.sh ar6_waveps 4 1500 --set loss.wav_eps=0.001
./autoresearch_jyz_af12.sh ar6_density_lr02 4 1500 --set gaussian.density_lr=0.02
