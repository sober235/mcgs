#!/usr/bin/env bash
# Chains the L1-SPIRiT sweep across the AF12 and AF20 topics once the AF16 one has finished.
# Same grid everywhere (6 lambdas x 2 kernel sizes per contrast) so every topic's optimum can be
# checked to be interior to its own grid -- the check the cross-AF review flagged AF16's Joint
# SENSE for failing.
set -u
R=/data0/congcong/code/GS/Results_20260908
A=$R/T1FLAIR_AF16_v3_jiangyizhou
until grep -q "sweep done: FLAIR" "$A/logs/spirit/sweep_af16.log" 2>/dev/null; do sleep 60; done
for t in T1FLAIR_AF12_v3_jiangyizhou T1FLAIR_AF20_v3_jiangyizhou; do
  for c in T1 FLAIR; do
    echo "=== $t $c $(date +%H:%M)"
    bash "$R/$t/scripts/run_spirit_sweep.sh" "$c" 2 6 7
  done
done
echo "ALL SPIRIT SWEEPS DONE"
