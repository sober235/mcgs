#!/usr/bin/env bash
# Refresh one AF12 topic: summary table, README tables, per-slice sweep (ours = MC-GS joint v3), and the compact
# figures at v3's best slices read from logs/best_slice_v2.json (axis-0 panel + three-plane panel per contrast).
# usage: refresh_topic.sh <topic dir name> <figure prefix> <c1,c2>
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
export PYTHONNOUSERSITE=1 PYTHONPATH=
R=/data0/congcong/code/GS/Results_20260908
S=$R/T1T2_AF12_retro/scripts
SCR=/data0/congcong/code/GS/Results_20260908/T1T2_AF12_retro/scripts
topic=$1; prefix=$2; cs=$3
$PY $R/$topic/scripts/summarize.py > $R/$topic/logs/summarize.log 2>&1 || { echo "summarize failed: $topic"; exit 1; }
$PY $SCR/readme_tables.py $topic
$PY $S/find_best_slice_v2.py --topic $R/$topic --contrasts $cs > $R/$topic/logs/best_slice_v2.log 2>&1 || { echo "sweep failed: $topic"; exit 1; }
for c in ${cs//,/ }; do
  i0=$($PY -c "import json; b=json.load(open('$R/$topic/logs/best_slice_v2.json'))['$c']; print(max((r for r in b if r['axis']==0), key=lambda r: r['ours'])['index'])")
  $PY $S/fig_slice_v2.py --topic $R/$topic --prefix $prefix --contrast $c --axis 0 --index $i0 | head -1
  $PY $S/fig_planes_v2.py --topic $R/$topic --prefix $prefix --contrast $c | tail -1
done
echo "refreshed $topic $(date +%H:%M)"
