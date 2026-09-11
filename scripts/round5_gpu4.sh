#!/bin/bash
# Card 4 after ar6_slr012: scale lr 0.008 on P10 (the runner now bakes P10), then the remaining single knobs on P10.
cd /data0/congcong/code/GS/mcgs_af12_wt
until [ -f results_arc/ar6_slr012_score.json ] || grep -q -E 'Traceback|Killed' results_arc/ar6_slr012.log 2>/dev/null; do sleep 90; done
./autoresearch_jyz_af12.sh ar7_P10_slr008 4 1500 --set gaussian.scale_lr=0.008
./autoresearch_jyz_af12.sh ar7_P10_sym4 4 1500 --set loss.wav_name=sym4
./autoresearch_jyz_af12.sh ar7_P10_waveps 4 1500 --set loss.wav_eps=0.001
./autoresearch_jyz_af12.sh ar7_P10_density_lr02 4 1500 --set gaussian.density_lr=0.02
