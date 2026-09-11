#!/bin/bash
# Round 3 on card 4: waits for ar2_n60, then pre-registered single knobs on P7 at the screening budget.
cd /data0/congcong/code/GS/mcgs_af12_wt
until [ -f results_arc/ar2_n60_score.json ] || grep -q -E 'Traceback|Killed' results_arc/ar2_n60.log 2>/dev/null; do sleep 90; done
./autoresearch_jyz_af12.sh ar6_wavT1_012 4 1500 --set loss.lambda_wav_plain=0.12
./autoresearch_jyz_af12.sh ar6_waveps 4 1500 --set loss.wav_eps=0.001
./autoresearch_jyz_af12.sh ar6_density_lr02 4 1500 --set gaussian.density_lr=0.02
