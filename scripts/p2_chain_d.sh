#!/bin/bash
# Phase-2 chain D (card = $1): round 2 part 2 (optimiser family around the rotation-lr keep) first, then round 2
# part 1 (the wavelet prior's shape, commit d22be4b). Runner default = design B. The runner refuses to overwrite,
# so two copies of this chain on two cards share the experiments; a refused launch just moves on.
cd /data0/congcong/code/GS/mcgs_af12_wt
G=${1:?card}
./autoresearch_jyz_af12_p2g.sh p2_rotlr002 "$G" 1500 --set gaussian.rotation_lr=0.002
./autoresearch_jyz_af12_p2g.sh p2_rotlr0005 "$G" 1500 --set gaussian.rotation_lr=0.0005
./autoresearch_jyz_af12_p2g.sh p2_rotlr_slr005 "$G" 1500 --set gaussian.rotation_lr=0.001 --set gaussian.scale_lr=0.005
./autoresearch_jyz_af12_p2g.sh p2_wav_lev2_w05 "$G" 1500 --set loss.wav_levels=2 --set 'loss.wav_level_weights=[1.0, 0.5]'
./autoresearch_jyz_af12_p2g.sh p2_wav_ddd05 "$G" 1500 --set 'loss.wav_band_weights={ddd: 0.5}'
./autoresearch_jyz_af12_p2g.sh p2_wav_ddd20 "$G" 1500 --set 'loss.wav_band_weights={ddd: 2.0}'
./autoresearch_jyz_af12_p2g.sh p2_wav_ax05 "$G" 1500 --set 'loss.wav_band_weights={daa: 0.5, ada: 0.5, aad: 0.5}'
