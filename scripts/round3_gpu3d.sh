#!/bin/bash
# Card 3: waits for ar4_warmup700, then the listed runs (autoresearch_jyz_af12.md, round 3).
cd /data0/congcong/code/GS/mcgs_af12_wt
until [ -f results_arc/ar4_warmup700_score.json ] || grep -q -E 'Traceback|Killed' results_arc/ar4_warmup700.log 2>/dev/null; do sleep 90; done
SAVE_VOLUMES=1 ./autoresearch_jyz_af12.sh ar5_wavlev1_full4k 3 4000 --set loss.wav_levels=1
./autoresearch_jyz_af12.sh ar6_density_lr01 3 1500 --set gaussian.density_lr=0.01
./autoresearch_jyz_af12.sh ar6_scale_lr003 3 1500 --set gaussian.scale_lr=0.003
./autoresearch_jyz_af12.sh ar6_headfrac03 3 1500 --set gaussian.init_head_frac=0.03
