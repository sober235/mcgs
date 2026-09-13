#!/bin/bash
# finalize_af20_jyz.sh JOINT_EXP [SINGLE_T1_EXP] [SINGLE_FLAIR_EXP]
#   Registers the MC-GS runs of this topic, rebuilds logs/summary.{tsv,json}, refreshes the README
#   tables (AF16 column = the same arm in T1FLAIR_AF16_v3_jiangyizhou) and draws the compact
#   three-plane figures at the AF16 figure slices, so the two studies can be laid side by side.
#   Keys are the AF16 phase-3 keys (gs_joint_p3 / gs_single_p3) on purpose: readme_tables.py matches
#   arms across topics by key, so the AF16 column lines up with the same recipe.
set -uo pipefail
J=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF20_v3_jiangyizhou
A16=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
E="env PYTHONNOUSERSITE=1 PYTHONPATH="
JOINT=$1; ST1=${2:-}; SFL=${3:-}
cd $J
$E $PY scripts/add_gs_arm.py --exp "$JOINT" --key gs_joint_p3 --label "MC-GS" --map T1:plain,FLAIR:contrast || exit 1
[ -n "$ST1" ] && { $E $PY scripts/add_gs_arm.py --exp "$ST1" --key gs_single_p3 --label "GS-single" --map T1:plain || exit 1; }
[ -n "$SFL" ] && { $E $PY - "$SFL" <<'PY' || exit 1
# the single-FLAIR run registers under the same key as the single-T1 run: merge its FLAIR channel in
import json, sys
from pathlib import Path
import numpy as np
R = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF20_v3_jiangyizhou")
exp = sys.argv[1]
p = R / "logs/extra_gs_arms.json"; arms = json.load(open(p)) if p.exists() else []
with np.load(f"/data0/congcong/code/GS/mcgs_af16_wt/results_arc/{exp}_volumes.npz") as z:
    np.save(R / "recon/gs_single_p3_FLAIR_mag.npy", np.asarray(z["recT1"], dtype=np.float32))
    if "recT1_complex" in z:
        np.save(R / "recon/gs_single_p3_FLAIR_complex.npy", np.asarray(z["recT1_complex"], dtype=np.complex64))
arms = [a for a in arms if not (a["key"] == "gs_single_p3" and "FLAIR" in a["map"])]
arms.append(dict(key="gs_single_p3", label="GS-single", exp=exp, map={"FLAIR": "plain"}))
json.dump(arms, open(p, "w"), indent=1); print("registered single-FLAIR", exp)
PY
}
$E $PY scripts/summarize.py > logs/post_summarize.log 2>&1 || { tail -5 logs/post_summarize.log; exit 1; }
tail -25 logs/post_summarize.log
$E $PY scripts/readme_tables.py $J $A16 > logs/readme_tables.md 2>&1 && $E $PY scripts/refresh_readme_tables.py $J
$E $PY scripts/compare_topics.py > logs/compare_af16_af20.txt 2>&1; cat logs/compare_af16_af20.txt
# the compact figure needs all nine arms (the script asserts it); stage 2 supplies ZS-SSDU, DIP and INR
if ls recon/zsssdu_{T1,FLAIR}_complex.npy recon/dip_{T1,FLAIR}_complex.npy recon/inr_{T1,FLAIR}_complex.npy \
      recon/gs_single_p3_{T1,FLAIR}_mag.npy >/dev/null 2>&1; then
  for c in T1 FLAIR; do
    $E $PY scripts/fig_three_planes_compact.py --contrast $c --slices 174 110 47 --mask_error \
        --mcgs_key gs_joint_p3 --single_key gs_single_p3 --tag _p3 | tail -3
  done
else
  echo "figures skipped: not all nine arms have volumes yet (stage 2 or a GS-single run is still pending)"
fi
echo "FINALIZE_AF20_DONE $JOINT ${ST1:-} ${SFL:-}"
