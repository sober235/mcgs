#!/usr/bin/env bash
# render_zf_caps_af12_16_20.sh -- the ZF-after-GT + caps figure set for AF 12 / 16 / 20 (user 2026-09-16), at most two
# renders at a time: one 600-dpi PDF render holds ~100 GB of RAM, and four in parallel took the shared server to 6 GB free.
set -u
R=/data0/congcong/code/GS/Results_20260908
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
C="--slices 180 124 76 --inset 48 --inset_corner tr bl bl --mask_error --gray_cbar --cbar_planes 0 --tra_rot 1 --err_vmax 0.1 --pdf --l1_arm spirit --with_zf --caps"
one() {  # topic script contrast extra...
  local t=$1 s=$2 c=$3; shift 3
  ( cd "$R/$t" && PYTHONNOUSERSITE=1 PYTHONPATH= $PY "scripts/$s" --contrast "$c" $C "$@" > "logs/fig_zf_caps_$c.log" 2>&1 )
  echo "done $t $c"
}
# pairs: two at a time
one T1FLAIR_AF12_v3_jiangyizhou fig_three_planes_inset.py   T1    --mcgs_key gs_joint_p14 --single_key gs_single_p14 --tag _p14_mcinr &
one T1FLAIR_AF16_v3_jiangyizhou fig_three_planes_compact.py T1    --mcgs_key gs_joint_p3  --single_key gs_single_p3  --tag _p3_mcinr &
wait
one T1FLAIR_AF20_v3_jiangyizhou fig_three_planes_compact.py T1    --mcgs_key gs_joint_p3  --single_key gs_single_p3  --tag _p3_mcinr --with_mcinr &
one T1FLAIR_AF12_v3_jiangyizhou fig_three_planes_inset.py   FLAIR --mcgs_key gs_joint_p14 --single_key gs_single_p14 --tag _p14_mcinr &
wait
one T1FLAIR_AF16_v3_jiangyizhou fig_three_planes_compact.py FLAIR --mcgs_key gs_joint_p3  --single_key gs_single_p3  --tag _p3_mcinr &
one T1FLAIR_AF20_v3_jiangyizhou fig_three_planes_compact.py FLAIR --mcgs_key gs_joint_p3  --single_key gs_single_p3  --tag _p3_mcinr --with_mcinr &
wait
echo "ZF_CAPS_RENDERS_DONE"
