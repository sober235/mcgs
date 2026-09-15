#!/bin/bash
# finalize_diffmask.sh KIND JOINT_EXP [SINGLE_T1_EXP] [SINGLE_FLAIR_EXP]
#   Registers this topic's MC-GS runs, rebuilds logs/summary.{tsv,json}, refreshes the README tables against the
#   HOUSE vd-Poisson topic T1FLAIR_AF16_v3_jiangyizhou (the diff-mask study's comparison column -- same subject, same
#   budget, same recipe, only the mask pattern differs), and draws the compact three-plane figures at that topic's
#   figure-of-record slices so the two can be laid side by side.
#   Keys are the AF16 phase-3 keys (gs_joint_p3 / gs_single_p3) on purpose: readme_tables.py matches arms across
#   topics by key, so the comparison column lines up with the same recipe.
set -uo pipefail
KIND=$1; shift
J=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_${KIND}_v3_jiangyizhou
H=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
E="env PYTHONNOUSERSITE=1 PYTHONPATH="
JOINT=$1; ST1=${2:-}; SFL=${3:-}
cd $J
$E $PY scripts/add_gs_arm.py --exp "$JOINT" --key gs_joint_p3 --label "MC-GS" --map T1:plain,FLAIR:contrast || exit 1
[ -n "$ST1" ] && { $E $PY scripts/add_gs_arm.py --exp "$ST1" --key gs_single_p3 --label "GS-single" --map T1:plain || exit 1; }
[ -n "$SFL" ] && { $E $PY - "$J" "$SFL" <<'PYEOF' || exit 1
# the single-FLAIR run registers under the same key as the single-T1 run: merge its FLAIR channel in
import json, sys
from pathlib import Path
import numpy as np
R = Path(sys.argv[1]); exp = sys.argv[2]
p = R / "logs/extra_gs_arms.json"; arms = json.load(open(p)) if p.exists() else []
with np.load(f"/data0/congcong/code/GS/mcgs_diffmask_wt/results_arc/{exp}_volumes.npz") as z:
    np.save(R / "recon/gs_single_p3_FLAIR_mag.npy", np.asarray(z["recT1"], dtype=np.float32))
    if "recT1_complex" in z:
        np.save(R / "recon/gs_single_p3_FLAIR_complex.npy", np.asarray(z["recT1_complex"], dtype=np.complex64))
arms = [a for a in arms if not (a["key"] == "gs_single_p3" and "FLAIR" in a["map"])]
arms.append(dict(key="gs_single_p3", label="GS-single", exp=exp, map={"FLAIR": "plain"}))
json.dump(arms, open(p, "w"), indent=1); print("registered single-FLAIR", exp)
PYEOF
}
$E $PY scripts/summarize.py > logs/post_summarize.log 2>&1 || { tail -5 logs/post_summarize.log; exit 1; }
tail -25 logs/post_summarize.log
$E $PY scripts/readme_tables.py $J $H > logs/readme_tables.md 2>&1 && $E $PY scripts/refresh_readme_tables.py $J
# the compact figure needs every arm it draws; stage 2 supplies ZS-SSDU, DIP and INR
if ls recon/zsssdu_{T1,FLAIR}_complex.npy recon/dip_{T1,FLAIR}_complex.npy recon/inr_{T1,FLAIR}_complex.npy \
      recon/gs_single_p3_{T1,FLAIR}_mag.npy >/dev/null 2>&1; then
  for c in T1 FLAIR; do
    $E $PY scripts/fig_three_planes_compact.py --contrast $c --slices 174 110 47 --mask_error \
        --mcgs_key gs_joint_p3 --single_key gs_single_p3 --tag _p3 | tail -3
    $E $PY scripts/fig_three_planes_compact.py --contrast $c --slices 180 124 76 --inset 48 --inset_corner tr bl bl \
        --mask_error --gray_cbar --cbar_planes 0 --tra_rot 1 --err_vmax 0.1 --pdf \
        --mcgs_key gs_joint_p3 --single_key gs_single_p3 --tag _p3 | tail -3
  done
else
  echo "figures skipped: not all arms have volumes yet"
fi
echo "FINALIZE_DIFFMASK_DONE $KIND $JOINT ${ST1:-} ${SFL:-}"
