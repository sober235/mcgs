#!/bin/bash
# Round 2 on card 4: waits for ar2_n48 (end of the round-1 chain there), then pre-registered single knobs.
cd /data0/congcong/code/GS/mcgs_af12_wt
until [ -f results_arc/ar2_n48_score.json ] || grep -q -E 'Traceback|Killed' results_arc/ar2_n48.log 2>/dev/null; do sleep 90; done
./autoresearch_jyz_af12.sh ar4_wavlev2 4 1500 --set loss.wav_levels=2
./autoresearch_jyz_af12.sh ar4_kcenter015 4 1500 --kspace_center 0.15
./autoresearch_jyz_af12.sh ar4_wavC085 4 1500 --set loss.lambda_wav=0.085
