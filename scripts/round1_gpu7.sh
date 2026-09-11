#!/bin/bash
# Round 1 on card 7: waits for round 0 (the seed-floor run's score json), then two pre-registered single
# knobs against P6 at the screening budget.
cd /data0/congcong/code/GS/mcgs_af12_wt
until [ -f results_arc/ar1_P6_i1500_s43_score.json ] || grep -q -E "Traceback|Killed" results_arc/ar1_P6_i1500_s43.log 2>/dev/null; do sleep 120; done
./autoresearch_jyz_af12.sh ar2_wav_up 7 1500 --set loss.lambda_wav=0.10 --set loss.lambda_wav_plain=0.14
./autoresearch_jyz_af12.sh ar2_geo013 7 1500 --lambda_geo 0.13
