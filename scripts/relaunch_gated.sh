#!/bin/bash
# relaunch_gated.sh (2026-09-10 20:06) -- after the 19:01 stall of all seven runs, relaunch one run at a
# time: the next starts only when the previous one has written iteration >= 100 to its joint_train.log,
# i.e. has passed the window in which the stall hit. A run that does not reach iteration 100 within
# 15 minutes stops the launcher with an ALERT line (results_arc/relaunch_gated.log).
cd /data0/congcong/code/GS/mcgs_af16_wt
P1="--lambda_geo 0.16 --edge_tv_kappa_q 0.90"
L=results_arc/relaunch_gated.log
run() {  # gpu exp args...
  local g=$1 exp=$2; shift 2
  setsid nohup ./autoresearch_jyz_af16.sh $exp $g 1500 "$@" > /dev/null 2>&1 < /dev/null &
  echo "[$(date +%H:%M:%S)] START $exp gpu$g" >> $L
  local t0=$(date +%s)
  while true; do
    sleep 20
    if grep -qE "^\[iter +(1[0-9]{2}|[2-9][0-9]{2}|1[0-9]{3})/" results_arc/$exp/$exp/joint_train.log 2>/dev/null; then
      echo "[$(date +%H:%M:%S)] PASS $exp iter>=100 after $(( $(date +%s) - t0 )) s" >> $L; return 0
    fi
    if [ $(( $(date +%s) - t0 )) -gt 900 ]; then
      echo "[$(date +%H:%M:%S)] ALERT $exp did not reach iter 100 in 15 min -- launcher stopped" >> $L; exit 1
    fi
  done
}
run 0 ar4_P1_wav003      $P1 --set loss.lambda_wav=0.03
run 1 ar4_P1_wav003_n24  $P1 --set loss.lambda_wav=0.03 --set gaussian.initial_num_points=2400000
run 3 ar4_v2_wav003          --set loss.lambda_wav=0.03
run 0 ar3_P1_dtv010      $P1 --lambda_dtv 0.10
run 1 ar3_P1_eps002      $P1 --tv_eps 0.02
run 3 ar4_v2_wav003_n24      --set loss.lambda_wav=0.03 --set gaussian.initial_num_points=2400000
run 2 ar3_P1_dtv020      $P1 --lambda_dtv 0.20
echo "[$(date +%H:%M:%S)] ALL_LAUNCHED" >> $L
