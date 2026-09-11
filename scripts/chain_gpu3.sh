#!/bin/bash
# Card 3 (free since 20:07): the screening baseline first, then two round-1 knobs.
cd /data0/congcong/code/GS/mcgs_af12_wt
./autoresearch_jyz_af12.sh ar1_P6_i1500 3 1500
./autoresearch_jyz_af12.sh ar2_wav_down 3 1500 --set loss.lambda_wav=0.05 --set loss.lambda_wav_plain=0.07
./autoresearch_jyz_af12.sh ar2_geo007 3 1500 --lambda_geo 0.07
