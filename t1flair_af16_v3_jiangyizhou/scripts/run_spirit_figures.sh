#!/usr/bin/env bash
# run_spirit_figures.sh -- each topic's figure of record re-rendered with the L1 column filled by the
# k-space arm (_spirit) and with both L1 arms side by side (_bothl1). Existing figures are never
# touched: --l1_arm appends its own suffix. Each topic also gets a plain re-render under a _repro tag
# that must come out pixel-identical to its figure of record -- that is what proves the command here
# is the one that made it (AF16 was checked by hand already: identical, 0 pixels differing).
# AF12 renders with fig_three_planes_inset.py: only there did the inset layout stay a separate script,
# its fig_three_planes_compact.py being the older 174/110/47 one.
# The three topics run concurrently (CPU matplotlib, ~6 min per render with the 600-dpi PDF).
set -u
R=/data0/congcong/code/GS/Results_20260908
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
COMMON="--slices 180 124 76 --inset 48 --inset_corner tr bl bl --mask_error --gray_cbar --cbar_planes 0 --tra_rot 1 --err_vmax 0.1"

topic() {  # dir  script  tag  extra...
  local t=$1 s=$2 tag=$3; shift 3
  cd "$R/$t" || return 1
  for c in T1 FLAIR; do
    for arm in spirit both; do
      PYTHONNOUSERSITE=1 PYTHONPATH= $PY "scripts/$s" $COMMON --contrast $c --tag "$tag" "$@" \
          --l1_arm $arm --pdf
    done
  done
  PYTHONNOUSERSITE=1 PYTHONPATH= $PY "scripts/$s" $COMMON --contrast T1 --tag "${tag}_repro" "$@"
  echo "TOPIC_DONE $t"
}

topic T1FLAIR_AF12_v3_jiangyizhou fig_three_planes_inset.py   _p14_mcinr \
      --mcgs_key gs_joint_p14 --single_key gs_single_p14 > "$R/T1FLAIR_AF16_v3_jiangyizhou/logs/spirit/fig_af12.log" 2>&1 &
topic T1FLAIR_AF16_v3_jiangyizhou fig_three_planes_compact.py _p3_mcinr \
      --mcgs_key gs_joint_p3 --single_key gs_single_p3 > "$R/T1FLAIR_AF16_v3_jiangyizhou/logs/spirit/fig_af16.log" 2>&1 &
topic T1FLAIR_AF20_v3_jiangyizhou fig_three_planes_compact.py _p3_mcinr \
      --mcgs_key gs_joint_p3 --single_key gs_single_p3 --with_mcinr > "$R/T1FLAIR_AF16_v3_jiangyizhou/logs/spirit/fig_af20.log" 2>&1 &
wait
echo "SPIRIT_FIGURES_DONE"
