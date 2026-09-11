#!/bin/bash
# Round 2 on card 3: waits for ar2_geo007 (end of the round-1 chain there), then pre-registered single knobs.
cd /data0/congcong/code/GS/mcgs_af12_wt
until [ -f results_arc/ar2_geo007_score.json ] || grep -q -E 'Traceback|Killed' results_arc/ar2_geo007.log 2>/dev/null; do sleep 90; done
./autoresearch_jyz_af12.sh ar4_wavlev4 3 1500 --set loss.wav_levels=4
./autoresearch_jyz_af12.sh ar4_kcenter05 3 1500 --kspace_center 0.5
./autoresearch_jyz_af12.sh ar4_warmup1000 3 1500 --warmup 1000
