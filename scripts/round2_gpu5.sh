#!/bin/bash
# Round 2 on card 5: waits for ar2_wfloor02 (end of the round-1 chain there), then pre-registered single knobs.
cd /data0/congcong/code/GS/mcgs_af12_wt
until [ -f results_arc/ar2_wfloor02_score.json ] || grep -q -E 'Traceback|Killed' results_arc/ar2_wfloor02.log 2>/dev/null; do sleep 90; done
./autoresearch_jyz_af12.sh ar4_db4 5 1500 --set loss.wav_name=db4
./autoresearch_jyz_af12.sh ar4_warmup250 5 1500 --warmup 250
./autoresearch_jyz_af12.sh ar4_kmid08 5 1500 --kspace_mid 0.8
