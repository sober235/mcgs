#!/bin/bash
# Round 1 on card 6: waits for the classical GPU chain to finish (its log line GPU_CHAIN_DONE), then two
# pre-registered single knobs against P6 at the screening budget (autoresearch_jyz_af12.md, "Allowed and queued").
cd /data0/congcong/code/GS/mcgs_af12_wt
L=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_v3_jiangyizhou/logs/classical_resweep.log
until grep -q GPU_CHAIN_DONE $L 2>/dev/null; do sleep 120; done
./autoresearch_jyz_af12.sh ar2_wav_down 6 1500 --set loss.lambda_wav=0.05 --set loss.lambda_wav_plain=0.07
./autoresearch_jyz_af12.sh ar2_geo007 6 1500 --lambda_geo 0.07
