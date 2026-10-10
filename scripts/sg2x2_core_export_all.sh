#!/usr/bin/env bash
# The launcher that produced the 2026-10-10 core-test export (08:06:54-08:29:46, all 14 subjects; kept for provenance:
# a re-run refuses, every output is an exclusive create). Run scripts/sg2x2_core_export.py once per subject (one process each, three at a time), each with its own log
# (append, never truncate). A failing subject does not stop the others; failures are listed in export_failed.txt.
cd /data0/congcong/code/GS/mcgs_sg2x2_wt || exit 1
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
OUT=/data2/congcong/data/sg2x2_core_h5
mkdir -p "$OUT/logs"
SUBJECTS="jiangyizhou_20230705 huangyuwei_20230705 jiangdian_20230705 sunxinyi_20230705 tangbing_20230705
liaozhenghan_20231030 zhoushuo_20231030 shiyixuan_20231101 zhangyangming_20231101 yangkun_20231102
zhangxiqian_20231102 chenshuaibao_20231106 lishengbo_20231106 litianyu_20231107"
run_one() {
  local s=$1
  echo "$(date '+%F %T') start $s" >> "$OUT/logs/export_${s}.log"
  if env PYTHONNOUSERSITE=1 PYTHONPATH= OMP_NUM_THREADS=16 nice -n 10 "$PY" scripts/sg2x2_core_export.py "$s" \
       >> "$OUT/logs/export_${s}.log" 2>&1; then
    echo "$(date '+%F %T') done $s" >> "$OUT/logs/export_${s}.log"
  else
    echo "$(date '+%F %T') FAILED $s" | tee -a "$OUT/logs/export_${s}.log" >> "$OUT/logs/export_failed.txt"
  fi
}
n=0
for s in $SUBJECTS; do
  run_one "$s" &
  n=$((n + 1))
  if (( n % 3 == 0 )); then wait; fi
done
wait
echo "$(date '+%F %T') ALL EXPORTS FINISHED" >> "$OUT/logs/export_all.log"
