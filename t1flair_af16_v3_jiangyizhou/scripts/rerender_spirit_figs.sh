#!/usr/bin/env bash
# Re-render the six delivered _spirit figures after the AF-label gauge fix (2026-09-15):
# the title now reads AF_vs_full_rect (whole PE plane / points this contrast has data on, the
# user's 2026-09-14 definition = BART Acc), not AF_vs_own_support. Only FLAIR changes.
set -u
R=/data0/congcong/code/GS/Results_20260908
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
C="--slices 180 124 76 --inset 48 --inset_corner tr bl bl --mask_error --gray_cbar --cbar_planes 0 --tra_rot 1 --err_vmax 0.1 --pdf --l1_arm spirit"
go() { local t=$1 s=$2; shift 2; cd "$R/$t" || return 1
       for c in T1 FLAIR; do PYTHONNOUSERSITE=1 PYTHONPATH= $PY "scripts/$s" $C --contrast $c "$@"; done; echo "DONE $t"; }
go T1FLAIR_AF12_v3_jiangyizhou fig_three_planes_inset.py   --tag _p14_mcinr --mcgs_key gs_joint_p14 --single_key gs_single_p14 > $R/T1FLAIR_AF16_v3_jiangyizhou/logs/spirit/relabel_af12.log 2>&1 &
go T1FLAIR_AF16_v3_jiangyizhou fig_three_planes_compact.py --tag _p3_mcinr  --mcgs_key gs_joint_p3  --single_key gs_single_p3  > $R/T1FLAIR_AF16_v3_jiangyizhou/logs/spirit/relabel_af16.log 2>&1 &
go T1FLAIR_AF20_v3_jiangyizhou fig_three_planes_compact.py --tag _p3_mcinr  --mcgs_key gs_joint_p3  --single_key gs_single_p3 --with_mcinr > $R/T1FLAIR_AF16_v3_jiangyizhou/logs/spirit/relabel_af20.log 2>&1 &
wait
echo "RELABEL_DONE"
