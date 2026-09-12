#!/bin/bash
# Phase-2 chain E (card = $1): round 2 part 3 — rotation lr 0.004 screen, promotion of rotation lr 0.002 at 4000
# (volumes), then the wavelet-shape screens that chain D did not reach. Two copies share the work (overwrite refusal).
cd /data0/congcong/code/GS/mcgs_af12_wt
G=${1:?card}
./autoresearch_jyz_af12_p2g.sh p2_rotlr004 "$G" 1500 --set gaussian.rotation_lr=0.004
SAVE_VOLUMES=1 ./autoresearch_jyz_af12_p2g.sh p2_rotlr002_full4k "$G" 4000 --set gaussian.rotation_lr=0.002
./autoresearch_jyz_af12_p2g.sh p2_wav_lev2_w05 "$G" 1500 --set loss.wav_levels=2 --set 'loss.wav_level_weights=[1.0, 0.5]'
./autoresearch_jyz_af12_p2g.sh p2_wav_ddd05 "$G" 1500 --set 'loss.wav_band_weights={ddd: 0.5}'
./autoresearch_jyz_af12_p2g.sh p2_wav_ddd20 "$G" 1500 --set 'loss.wav_band_weights={ddd: 2.0}'
./autoresearch_jyz_af12_p2g.sh p2_wav_ax05 "$G" 1500 --set 'loss.wav_band_weights={daa: 0.5, ada: 0.5, aad: 0.5}'
