#!/bin/bash
# finalize_noise.sh JOINT_EXP SINGLE_T1_EXP SINGLE_FLAIR_EXP -- after run_all_noise.py's queue is empty.
#   1. register the three GS runs under the AF16 phase-3 keys (gs_joint_p3 / gs_single_p3), so every table lines
#      up by key with the zero-noise topic T1FLAIR_AF16_v3_jiangyizhou;
#   2. MC-INR: its score.py (gauge units against the clean AF16 topic), then this topic's score_mcinr_arm.py;
#   3. the render-only diagnostic of the GS arms (logs/gs_render_<c>.json);
#   4. summarize.py, README tables against the zero-noise topic, compare_noise.py for the three head metrics;
#   5. figures: the noise figure and the compact three-plane figures (one render at a time).
# GPU=<card> (default 1) is the card used by the two GPU scoring steps.
set -uo pipefail
N=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_noise2x_v3_jiangyizhou
H=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou
M=/data0/congcong/code/GS/mcinr_niessen2025
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
PY_TCNN=/home/congcongliu/anaconda3/envs/GR_pt1.10/bin/python
GPU=${GPU:-1}
JOINT=$1; ST1=$2; SFL=$3
export PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID
cd "$N" || exit 1

"$PY" scripts/add_gs_arm.py --exp "$JOINT" --key gs_joint_p3 --label "MC-GS" --map T1:plain,FLAIR:contrast || exit 1
"$PY" scripts/add_gs_arm.py --exp "$ST1" --key gs_single_p3 --label "GS-single" --map T1:plain || exit 1
"$PY" - "$N" "$SFL" <<'PYEOF' || exit 1
# the single-FLAIR run registers under the same key as the single-T1 run: merge its FLAIR channel in
import json, sys
from pathlib import Path
import numpy as np
R = Path(sys.argv[1]); exp = sys.argv[2]
p = R / "logs/extra_gs_arms.json"; arms = json.load(open(p)) if p.exists() else []
with np.load(f"/data0/congcong/code/GS/mcgs_noise_wt/results_arc/{exp}_volumes.npz") as z:
    np.save(R / "recon/gs_single_p3_FLAIR_mag.npy", np.asarray(z["recT1"], dtype=np.float32))
    np.save(R / "recon/gs_single_p3_FLAIR_complex.npy", np.asarray(z["recT1_complex"], dtype=np.complex64))
arms = [a for a in arms if not (a["key"] == "gs_single_p3" and "FLAIR" in a["map"])]
arms.append(dict(key="gs_single_p3", label="GS-single", exp=exp, map={"FLAIR": "plain"}))
json.dump(arms, open(p, "w"), indent=1); print("registered single-FLAIR", exp)
PYEOF

if [ -f /data0/congcong/code/GS/Results_20260908/T1FLAIR_MCINR_Niessen2025_v3_jiangyizhou/recon/af16_noise2x/meta.json ]; then
  # the MC-INR repository runs everything in its own env (its CLAUDE.md); score.py is CPU-only
  (cd "$M" && "$PY_TCNN" -u scripts/score.py --af 16 --tag _noise2x) || { echo "MC-INR score.py failed"; exit 1; }
  "$PY" -u scripts/score_mcinr_arm.py --gpu "$GPU" --af 16 --tag _noise2x || exit 1
else
  echo "MC-INR run missing: not scored"
fi

"$PY" -u scripts/score_gs_render.py --gpu "$GPU" --joint "$JOINT" --single_T1 "$ST1" --single_FLAIR "$SFL" || exit 1

"$PY" scripts/summarize.py > logs/post_summarize.log 2>&1 || { tail -5 logs/post_summarize.log; exit 1; }
tail -30 logs/post_summarize.log
"$PY" scripts/readme_tables.py "$N" "$H" > logs/readme_tables.md 2>&1 && "$PY" scripts/refresh_readme_tables.py "$N"
for m in head_psnr head_ssim head_hfen; do
  "$PY" scripts/compare_noise.py --metric $m --out logs/compare_noise_$m.md > /dev/null || exit 1
done

CUDA_VISIBLE_DEVICES= "$PY" scripts/fig_noise_data.py || exit 1
for c in T1 FLAIR; do
  "$PY" scripts/fig_three_planes_compact.py --contrast $c --slices 174 110 47 --mask_error \
      --mcgs_key gs_joint_p3 --single_key gs_single_p3 --tag _p3_mcinr --l1_arm spirit --with_zf --caps | tail -2
  "$PY" scripts/fig_three_planes_compact.py --contrast $c --slices 180 124 76 --inset 48 --inset_corner tr bl bl \
      --mask_error --gray_cbar --cbar_planes 0 --tra_rot 1 --err_vmax 0.1 --pdf \
      --mcgs_key gs_joint_p3 --single_key gs_single_p3 --tag _p3_mcinr --l1_arm spirit --with_zf --caps | tail -2
done
echo "FINALIZE_NOISE_DONE $JOINT $ST1 $SFL"
