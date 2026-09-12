#!/bin/bash
# Phase-2 chain D (card = $1): round 2 part 1, the wavelet prior's shape (per-level / per-sub-band weights,
# commit d22be4b). Runner default = design B. The runner refuses to overwrite, so relaunching skips finished runs.
cd /data0/congcong/code/GS/mcgs_af12_wt
G=${1:?card}
./autoresearch_jyz_af12_p2.sh p2_wav_lev2_w05 "$G" 1500 --set loss.wav_levels=2 --set 'loss.wav_level_weights=[1.0, 0.5]'
./autoresearch_jyz_af12_p2.sh p2_wav_ddd05 "$G" 1500 --set 'loss.wav_band_weights={ddd: 0.5}'
./autoresearch_jyz_af12_p2.sh p2_wav_ddd20 "$G" 1500 --set 'loss.wav_band_weights={ddd: 2.0}'
./autoresearch_jyz_af12_p2.sh p2_wav_ax05 "$G" 1500 --set 'loss.wav_band_weights={daa: 0.5, ada: 0.5, aad: 0.5}'
