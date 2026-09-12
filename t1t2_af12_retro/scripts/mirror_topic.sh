#!/usr/bin/env bash
# Mirror one topic's README, summary + sweep logs and its current v3 figures (axis-0 best slice + three planes per contrast)
# into the framework repo and stage them. usage: mirror_topic.sh <topic dir name> <repo mirror dir> <figure prefix> <c1,c2>
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
R=/data0/congcong/code/GS/Results_20260908; F=/data0/congcong/code/GS/Paired_ARC_GS_general_framework
topic=$1; M=$2; prefix=$3; cs=$4
cd $F || exit 1
mkdir -p $M/logs $M/figures
rsync -a $R/$topic/README.md $M/
rsync -a $R/$topic/logs/summary.tsv $R/$topic/logs/summary.json $R/$topic/logs/best_slice_v2.json $R/$topic/logs/best_slice_v2.log $M/logs/
figs=()
for c in ${cs//,/ }; do
  i0=$($PY -c "import json; b=json.load(open('$R/$topic/logs/best_slice_v2.json'))['$c']; print(max((r for r in b if r['axis']==0), key=lambda r: r['ours'])['index'])")
  figs+=("${prefix}_${c}_best_axis0_${i0}.png" "${prefix}_${c}_best3planes.png")
done
for f in "${figs[@]}"; do rsync -a $R/$topic/figures/$f $M/figures/ || echo "missing figure $f"; done
git add -f $M/README.md $M/logs/summary.tsv $M/logs/summary.json $M/logs/best_slice_v2.json $M/logs/best_slice_v2.log $(printf "$M/figures/%s " "${figs[@]}")
git status --short $M
