#!/usr/bin/env bash
# Read the four free-running .dat files sequentially once so the twix reader hits the page cache
# (its seek-per-ADC pattern is slow on a cold disk). Read-only.
for S in V07 V03 V04 V08; do
  f=$(readlink -f /data1/data1_congcong/data/3D/cardiac_freerunning_multiecho_5D/$S/freerunning_8echo.dat)
  t0=$(date +%s); nice -n 10 cat "$f" > /dev/null; t1=$(date +%s)
  echo "$S $(( $(stat -c %s "$f") / 1000000 )) MB in $((t1 - t0)) s"
done
echo PREFETCH_DONE
