#!/bin/bash
# finalize_af16_jyz.sh JOINT_EXP [SINGLE_T1_EXP] [SINGLE_FLAIR_EXP]
#   Registers the autoresearch winners of worktree mcgs_af16_wt as extra arms of this topic
#   (scripts/add_gs_arm.py: volumes into recon/, arm into logs/extra_gs_arms.json), then rebuilds
#   logs/summary.{tsv,json}, the README tables and the two visual-gate butterfly figures.
#   Keys: gs_joint_ar (MC-GS joint, autoresearch) and gs_single_ar (MC-GS single, autoresearch).
set -uo pipefail
J=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
E="env PYTHONNOUSERSITE=1 PYTHONPATH="
JOINT=$1; ST1=${2:-}; SFL=${3:-}
cd $J
$E $PY scripts/add_gs_arm.py --exp "$JOINT" --key gs_joint_ar --label "MC-GS joint (autoresearch)" --map T1:plain,FLAIR:contrast || exit 1
[ -n "$ST1" ] && { $E $PY scripts/add_gs_arm.py --exp "$ST1" --key gs_single_ar --label "MC-GS single (autoresearch)" --map T1:plain || exit 1; }
[ -n "$SFL" ] && { $E $PY - "$SFL" <<'PY' || exit 1
# the single-FLAIR run registers under the same key as the single-T1 run: merge its FLAIR channel in
import json, sys
from pathlib import Path
import numpy as np
R = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou")
exp = sys.argv[1]
p = R / "logs/extra_gs_arms.json"; arms = json.load(open(p)) if p.exists() else []
with np.load(f"/data0/congcong/code/GS/mcgs_af16_wt/results_arc/{exp}_volumes.npz") as z:
    np.save(R / "recon/gs_single_ar_FLAIR_mag.npy", np.asarray(z["recT1"], dtype=np.float32))
# one FLAIR entry under gs_single_ar, whatever ran before (the T1 entry of the same key is kept)
arms = [a for a in arms if not (a["key"] == "gs_single_ar" and "FLAIR" in a["map"])]
arms.append(dict(key="gs_single_ar", label="MC-GS single (autoresearch)", exp=exp, map={"FLAIR": "plain"}))
json.dump(arms, open(p, "w"), indent=1); print("registered single-FLAIR", exp)
PY
}
$E $PY scripts/summarize.py > logs/post_summarize.log 2>&1 || { tail -5 logs/post_summarize.log; exit 1; }
$E $PY scripts/readme_tables.py $J /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_v3_jiangyizhou > logs/readme_tables.md 2>&1 && $PY scripts/refresh_readme_tables.py $J
for c in T1 FLAIR; do $E $PY scripts/fig_butterfly_final.py --contrast $c --gs_key gs_joint_ar | tail -2; done
mkdir -p /home/congcongliu/figs/t1flair_v3_af16
echo "FINALIZE_DONE $JOINT ${ST1:-} ${SFL:-}"
