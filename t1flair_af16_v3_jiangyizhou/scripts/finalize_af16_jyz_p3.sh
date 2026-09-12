#!/bin/bash
# finalize_af16_jyz_p3.sh JOINT_EXP [SINGLE_T1_EXP] [SINGLE_FLAIR_EXP]
#   Phase-3 counterpart of finalize_af16_jyz.sh. Registers the phase-3 winner under NEW keys
#   gs_joint_p3 / gs_single_p3 -- the phase-1 keys gs_joint_ar / gs_single_ar stay P6, which is what
#   the user quotes externally (CLAUDE.md, 2026-09-12) -- then rebuilds the tables and draws the
#   compact three-plane figures with the phase-3 arm in the MC-GS column (suffix _p3).
set -uo pipefail
J=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
E="env PYTHONNOUSERSITE=1 PYTHONPATH="
JOINT=$1; ST1=${2:-}; SFL=${3:-}
cd $J
$E $PY scripts/add_gs_arm.py --exp "$JOINT" --key gs_joint_p3 --label "MC-GS (phase 3)" --map T1:plain,FLAIR:contrast || exit 1
[ -n "$ST1" ] && { $E $PY scripts/add_gs_arm.py --exp "$ST1" --key gs_single_p3 --label "GS-single (phase 3)" --map T1:plain || exit 1; }
[ -n "$SFL" ] && { $E $PY - "$SFL" <<'PY' || exit 1
# the single-FLAIR run registers under the same key as the single-T1 run: merge its FLAIR channel in
import json, sys
from pathlib import Path
import numpy as np
R = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou")
exp = sys.argv[1]
p = R / "logs/extra_gs_arms.json"; arms = json.load(open(p)) if p.exists() else []
with np.load(f"/data0/congcong/code/GS/mcgs_af16_wt/results_arc/{exp}_volumes.npz") as z:
    np.save(R / "recon/gs_single_p3_FLAIR_mag.npy", np.asarray(z["recT1"], dtype=np.float32))
    if "recT1_complex" in z:
        np.save(R / "recon/gs_single_p3_FLAIR_complex.npy", np.asarray(z["recT1_complex"], dtype=np.complex64))
arms = [a for a in arms if not (a["key"] == "gs_single_p3" and "FLAIR" in a["map"])]
arms.append(dict(key="gs_single_p3", label="GS-single (phase 3)", exp=exp, map={"FLAIR": "plain"}))
json.dump(arms, open(p, "w"), indent=1); print("registered single-FLAIR", exp)
PY
}
$E $PY scripts/summarize.py > logs/post_summarize_p3.log 2>&1 || { tail -5 logs/post_summarize_p3.log; exit 1; }
$E $PY scripts/readme_tables.py $J /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_v3_jiangyizhou > logs/readme_tables.md 2>&1 && $PY scripts/refresh_readme_tables.py $J
for c in T1 FLAIR; do
  $E $PY scripts/fig_three_planes_compact.py --contrast $c --slices 174 110 47 --mask_error \
      --mcgs_key gs_joint_p3 --single_key gs_single_p3 --tag _p3 | tail -4
done
echo "FINALIZE_P3_DONE $JOINT ${ST1:-} ${SFL:-}"
