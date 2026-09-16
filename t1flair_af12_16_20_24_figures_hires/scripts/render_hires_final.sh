#!/usr/bin/env bash
# render_hires_final.sh -- the figure of record of the four AF topics (12 / 16 / 20 / 24), re-rendered at
# 200 dpi (6453 px wide = 911 dpi at a 180 mm page width, user 2026-09-16) with the zero-filling column
# and the caps typography, plus the vector PDF whose image panels are rasterised at 600 dpi.
# Two renders at a time: one 600-dpi PDF render holds ~100 GB of RAM.
# Flags per topic are the ones run_spirit_figures.sh and the AF24 README record; only --png_dpi is new.
set -u
R=/data0/congcong/code/GS/Results_20260908
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
C="--slices 180 124 76 --inset 48 --inset_corner tr bl bl --mask_error --gray_cbar --cbar_planes 0 --tra_rot 1 --err_vmax 0.1 --pdf --png_dpi 200 --l1_arm spirit --with_zf --caps"

one() {  # topic script contrast extra...
  local t=$1 s=$2 c=$3; shift 3
  ( cd "$R/$t" && mkdir -p logs/hires && PYTHONNOUSERSITE=1 PYTHONPATH= nice -n 5 $PY "scripts/$s" \
      --contrast "$c" $C "$@" > "logs/hires/fig_${c}.log" 2>&1 )
  echo "done $t $c rc=$?"
}
A12=(T1FLAIR_AF12_v3_jiangyizhou fig_three_planes_inset.py)
A16=(T1FLAIR_AF16_v3_jiangyizhou fig_three_planes_compact.py)
A20=(T1FLAIR_AF20_v3_jiangyizhou fig_three_planes_compact.py)
A24=(T1FLAIR_AF24_v3_jiangyizhou fig_three_planes_compact.py)
K12="--mcgs_key gs_joint_p14 --single_key gs_single_p14 --tag _p14_mcinr"
K3="--mcgs_key gs_joint_p3 --single_key gs_single_p3 --tag _p3_mcinr"

for c in T1 FLAIR; do
  one "${A12[@]}" $c $K12 &
  one "${A16[@]}" $c $K3  &
  wait
  one "${A20[@]}" $c $K3 --with_mcinr &
  one "${A24[@]}" $c $K3 --with_mcinr &
  wait
done
echo "HIRES_FINAL_DONE $(date +%H:%M)"
