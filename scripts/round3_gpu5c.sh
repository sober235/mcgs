#!/bin/bash
# Card 5: waits for ar4_warmup250, then the listed runs (autoresearch_jyz_af12.md, round 3).
cd /data0/congcong/code/GS/mcgs_af12_wt
until [ -f results_arc/ar4_warmup250_score.json ] || grep -q -E 'Traceback|Killed' results_arc/ar4_warmup250.log 2>/dev/null; do sleep 90; done
SAVE_VOLUMES=1 ./autoresearch_jyz_af12.sh ar5_db4_full4k 5 4000 --set loss.wav_name=db4
./autoresearch_jyz_af12.sh ar4_kmid08 5 1500 --kspace_mid 0.8
