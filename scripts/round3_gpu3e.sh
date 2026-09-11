#!/bin/bash
# Card 3 after ar6_density_lr01: the three-way combination (db4 + 1 level + FLAIR wavelet 0.085) at the screening
# budget, then the remaining round-3 knobs on this card.
cd /data0/congcong/code/GS/mcgs_af12_wt
until [ -f results_arc/ar6_density_lr01_score.json ] || grep -q -E 'Traceback|Killed' results_arc/ar6_density_lr01.log 2>/dev/null; do sleep 90; done
./autoresearch_jyz_af12.sh ar6_db4_lev1_C085 3 1500 --set loss.wav_name=db4 --set loss.wav_levels=1 --set loss.lambda_wav=0.085
./autoresearch_jyz_af12.sh ar6_scale_lr003 3 1500 --set gaussian.scale_lr=0.003
./autoresearch_jyz_af12.sh ar6_headfrac03 3 1500 --set gaussian.init_head_frac=0.03
