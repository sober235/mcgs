#!/bin/bash
# Round 4, the four runs that do not depend on Round 3's winners. Each takes over the card of one
# Round 3 run the moment that run's score json lands (two runs per card, as before).
#   GPU 0: after ar3_P1_dtv005 -> ar4_P1_wav003        (P1 + lambda_wav 0.03)
#          after ar3_P1_dtv010 -> ar4_P1_wav003_n24    (P1 + lambda_wav 0.03 + 2.4M points)
#   GPU 1: after ar3_P1_dtv020 -> ar4_v2_wav003        (v2 parent + lambda_wav 0.03: the term alone on jyz)
#          after ar3_P1_eps002 -> ar4_v2_wav003_n24    (the hyw loop's 2.4M + wavelet recipe, transferred)
# The two GPU 3 slots (P1 + wavelet + Round 3's best lambda_dtv / eps) are launched by hand.
cd /data0/congcong/code/GS/mcgs_af16_wt
P1="--lambda_geo 0.16 --edge_tv_kappa_q 0.90"
slot() {   # after_exp gpu exp args...
  local after=$1 g=$2 exp=$3; shift 3
  until [ -f results_arc/${after}_score.json ]; do sleep 30; done
  sleep 30
  echo "[$(date +%H:%M)] $after done -> $exp on gpu$g" >> results_arc/launch_r4_slots.log
  ./autoresearch_jyz_af16.sh $exp $g 1500 "$@"
}
slot ar3_P1_dtv005 0 ar4_P1_wav003     $P1 --set loss.lambda_wav=0.03 &
slot ar3_P1_dtv010 0 ar4_P1_wav003_n24 $P1 --set loss.lambda_wav=0.03 --set gaussian.initial_num_points=2400000 &
slot ar3_P1_dtv020 1 ar4_v2_wav003         --set loss.lambda_wav=0.03 &
slot ar3_P1_eps002 1 ar4_v2_wav003_n24     --set loss.lambda_wav=0.03 --set gaussian.initial_num_points=2400000 &
wait
