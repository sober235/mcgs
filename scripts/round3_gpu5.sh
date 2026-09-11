#!/bin/bash
# Card 5 after ar3_i6000: the P7 screening baseline (the runner now bakes wav_levels=2), then the round-2 knobs
# that were queued on this card, now on top of P7.
cd /data0/congcong/code/GS/mcgs_af12_wt
until [ -f results_arc/ar3_i6000_score.json ] || grep -q -E 'Traceback|Killed' results_arc/ar3_i6000.log 2>/dev/null; do sleep 90; done
./autoresearch_jyz_af12.sh ar1_P7_i1500 5 1500
./autoresearch_jyz_af12.sh ar4_db4 5 1500 --set loss.wav_name=db4
./autoresearch_jyz_af12.sh ar4_warmup250 5 1500 --warmup 250
./autoresearch_jyz_af12.sh ar4_kmid08 5 1500 --kspace_mid 0.8
