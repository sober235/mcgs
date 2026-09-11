#!/bin/bash
# Card 4 after ar6_db6: the scale-lr follow-ups (0.005, and 0.003 combined with the best three-way row), then the
# remaining round-3 knobs.
cd /data0/congcong/code/GS/mcgs_af12_wt
until [ -f results_arc/ar6_db6_score.json ] || grep -q -E 'Traceback|Killed' results_arc/ar6_db6.log 2>/dev/null; do sleep 90; done
./autoresearch_jyz_af12.sh ar6_slr005 4 1500 --set gaussian.scale_lr=0.005
./autoresearch_jyz_af12.sh ar6_db4_lev1_C085_slr003 4 1500 --set loss.wav_levels=1 --set loss.lambda_wav=0.085 --set gaussian.scale_lr=0.003
./autoresearch_jyz_af12.sh ar6_sym4 4 1500 --set loss.wav_name=sym4
./autoresearch_jyz_af12.sh ar6_wavT1_012 4 1500 --set loss.lambda_wav_plain=0.12
./autoresearch_jyz_af12.sh ar6_waveps 4 1500 --set loss.wav_eps=0.001
./autoresearch_jyz_af12.sh ar6_density_lr02 4 1500 --set gaussian.density_lr=0.02
