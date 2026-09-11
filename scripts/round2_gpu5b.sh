#!/bin/bash
# Card 5 after the round-1 chain (ar2_wfloor02): first the 6000-iteration budget run of P6 with volumes (if the
# 4000-iteration parent is not converged this is a free gain and the run can serve as the new parent), then
# the round-2 single knobs that were queued for this card.
cd /data0/congcong/code/GS/mcgs_af12_wt
until [ -f results_arc/ar2_wfloor02_score.json ] || grep -q -E 'Traceback|Killed' results_arc/ar2_wfloor02.log 2>/dev/null; do sleep 90; done
SAVE_VOLUMES=1 ./autoresearch_jyz_af12.sh ar3_i6000 5 6000
./autoresearch_jyz_af12.sh ar4_db4 5 1500 --set loss.wav_name=db4
./autoresearch_jyz_af12.sh ar4_warmup250 5 1500 --warmup 250
./autoresearch_jyz_af12.sh ar4_kmid08 5 1500 --kspace_mid 0.8
