#!/bin/bash
# stall_watch.sh -- every 2 min, for each of this worktree's running trainers, alert if its
# joint_train.log has not been written for more than 10 min (a line normally lands every 2-4 min),
# unless the last line is the final iteration (then it is scoring). One line per state change.
W=/data0/congcong/code/GS/mcgs_af16_wt
declare -A warned
while true; do
  now=$(date +%s)
  for p in $(pgrep -f "arc/cycle.py --exp"); do
    [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$W" ] || continue
    exp=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -o -- "--exp [^ ]*" | cut -d' ' -f2)
    f=$W/results_arc/$exp/$exp/joint_train.log
    [ -f "$f" ] || continue
    last=$(tail -1 "$f" | grep -o "iter *[0-9]*/[0-9]*" | tr -d ' ')
    it=${last#iter}; it=${it%%/*}; mx=${last#*/}
    age=$(( now - $(stat -c %Y "$f") ))
    if [ -n "$it" ] && [ "$it" = "$mx" ]; then continue; fi
    if [ $age -gt 600 ]; then
      [ -z "${warned[$exp]}" ] && echo "STALL? $exp: no training-log write for ${age}s (last: $last, pid $p)" && warned[$exp]=1
    elif [ -n "${warned[$exp]}" ]; then
      echo "RECOVERED $exp (last: $last)"; unset "warned[$exp]"
    fi
  done
  sleep 120
done
