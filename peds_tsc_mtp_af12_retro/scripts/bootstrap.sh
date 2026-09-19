#!/bin/bash
# bootstrap.sh -- copy the AF16 jiangyizhou arm scripts into this topic and retarget them.
# Rewrites: topic path, framework worktree path, contrast names T1->T1w / FLAIR->T2s, the served
# figures directory, the mask file names and the experiment-name prefix. The scoring code is NOT
# touched; step 3 of the plan diffs it against the source to prove that.
set -euo pipefail
SRC=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou
OUT=/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro
cd "$SRC/scripts"
FILES="t1t2_common.py recon_classical.py recon_joint_cssense.py recon_l1espirit.py recon_zsssdu.py \
recon_dip_inr.py extract_gs_volumes.py add_gs_arm.py summarize.py readme_tables.py \
refresh_readme_tables.py check_gauge.py make_figures.py fig_row_final.py \
fig_three_planes_compact.py best_slice_mcgs.py fig_slice.py slice_quality.py"
cp $FILES "$OUT/scripts/"
cd "$OUT/scripts"
sed -i \
  -e 's|Results_20260908/T1FLAIR_AF16_v3_jiangyizhou|Results_20260908/Peds_TSC_MTP_AF12_retro|g' \
  -e 's|/data0/congcong/code/GS/mcgs_af16_wt|/data0/congcong/code/GS/mcgs_peds_wt|g' \
  -e 's|mcgs_af16_wt|mcgs_peds_wt|g' \
  -e 's|figs/t1flair_v3_af16|figs/peds_tsc_af12|g' \
  -e 's|t1flair_v3_af16_|peds_af12_|g' \
  -e 's|data/mask_af16_3d\.h5|data/mask_af12_union_3d.h5|g' \
  -e 's|data/mask_vdpois_256x176_af16\.mat|data/mask_af12_2d.mat|g' \
  -e 's|"T1", "FLAIR"|"T1w", "T2s"|g' \
  -e 's|\["T1", "FLAIR"\]|["T1w", "T2s"]|g' \
  -e 's|T1:plain,FLAIR:contrast|T1w:plain,T2s:contrast|g' \
  -e 's|default="T1"|default="T1w"|g' \
  -e 's|\bAF 16\b|AF 12|g' -e 's|\bAF16\b|AF12|g' -e 's|retrospective AF 16|retrospective AF 12|g' \
  -e 's|jiangyizhou|16yo MTP|g' \
  $FILES
echo "--- residual mentions of the source topic (expected: none)"
grep -n "T1FLAIR_AF16\|mcgs_af16_wt\|huangyuwei\|jiangyizhou\|t1flair_v3_af16" $FILES || echo "(none)"
echo "--- residual bare contrast literals"
grep -n '"T1"\|"FLAIR"\|_FLAIR\b' $FILES || echo "(none)"
echo BOOTSTRAP_OK
