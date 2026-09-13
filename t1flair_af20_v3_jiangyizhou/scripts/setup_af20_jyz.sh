#!/bin/bash
# setup_af20_jyz.sh -- clone the jiangyizhou AF16 topic into an AF20 one (2026-09-13).
# Exactly one thing differs from T1FLAIR_AF16_v3_jiangyizhou: the vd-Poisson mask (nominal AF 20,
# accelx 5 instead of 4; every other generator setting is the house convention). Same subject, same
# CFL k-space, same two ESPIRiT maps, same ground truth, same gauge, same arms.
# The k-space and the maps are NOT copied: data/ksp_*.h5 and data/csm_espirit_*.h5 are symlinks into
# the AF12 topic, exactly as the AF16 topic does it, and prep_04_export.py asserts the symlink so it
# can never write through them.
# Run once, from anywhere. Afterwards: prep_02_mask.m (MATLAB), then prep_04_export.py.
set -euo pipefail
SRC=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou
A12=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_v3_jiangyizhou
OUT=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF20_v3_jiangyizhou
SELF=$(basename "$0")
mkdir -p "$OUT"/{scripts,logs,figures,data,recon}
cp "$SRC"/scripts/*.py "$SRC"/scripts/*.m "$SRC"/scripts/*.sh "$OUT/scripts/"
cd "$OUT/scripts"
# the AF16 acquisition probe and the phase-3 export point at files that will never exist here
rm -f prep_02c_mask_af16c.m prep_04c_export_af16c.py run_l1_probe_c.sh run_l1_flair_down.sh \
      export_phase3_bundle.py finalize_af16_jyz.sh finalize_af16_jyz_p3.sh
# this script is written by hand, not cloned: never let the rewrite below touch it
FILES=$(ls *.py *.m *.sh | grep -v "^${SELF}$")
# mcgs_af16_wt is the FRAMEWORK worktree path (it now carries the branch autoresearch-jyz-af20);
# it must survive the af16 -> af20 rewrite, so park it behind a token first.
sed -i -e 's|mcgs_af16_wt|@@WORKTREE@@|g' \
       -e 's|T1FLAIR_AF16_v3_jiangyizhou|T1FLAIR_AF20_v3_jiangyizhou|g' \
       -e 's|af16|af20|g' -e 's|AF16|AF20|g' -e 's|AF 16|AF 20|g' \
       -e 's|@@WORKTREE@@|mcgs_af16_wt|g' $FILES
for f in *af16*; do [ -e "$f" ] && mv "$f" "${f//af16/af20}"; done
# the mask itself: nominal AF 20 = accelx 5 (the generator takes accelx = AF / accely, accely = 4)
sed -i -e 's|for af = \[16\]|for af = [20]|' \
       -e '1s|.*|% AF=20 house-convention vd-Poisson mask on the (ky,kz) = (256,176) PE plane.|' \
       -e '2s|.*|% Same generator/settings as the AF12 and AF16 masks: accely=4, accelx=AF/4, calib=24,|' \
       prep_02_mask.m
# the README table column compares against AF16 (the same subject one acceleration down), not AF12
sed -i -e 's|(+ the AF12 twin)|(+ the AF16 twin)|' -e 's|<af12_topic_dir>|<af16_topic_dir>|' \
       -e 's|the AF12 head-only PSNR|the AF16 head-only PSNR|' -e 's|T, T12 = |T, T16 = |' \
       -e 's|\bT12\b|T16|g' -e 's|rows12|rows16|g' -e 's|AF12 column|AF16 column|' \
       -e 's|head PSNR @AF12|head PSNR @AF16|' -e 's|AF20 − AF12|AF20 − AF16|' \
       -e 's|(AF12: whole / head)|(AF16: whole / head)|' -e 's|\ba12\b|a16|g' -e 's|\bd12\b|d16|g' \
       -e 's|\bj12\b|j16|g' -e 's|\bs12\b|s16|g' readme_tables.py
# run_classical_revisited{,_ext}.sh carry the AF16 lambda grids; run_classical_af20.sh replaces them
rm -f run_classical_revisited.sh run_classical_revisited_ext.sh
echo "--- residual AF16 references (expect only the worktree path mcgs_af16_wt):"
grep -n "AF16\|af16" *.py *.m *.sh | grep -v "mcgs_af16_wt" || echo "(none)"
# this subject's own k-space and ESPIRiT maps are mask-independent: reuse them through symlinks
for f in ksp_T1.h5 ksp_FLAIR.h5 csm_espirit_T1.h5 csm_espirit_FLAIR.h5; do
  ln -sfn "$A12/data/$f" "$OUT/data/$f"
done
ls -la "$OUT/data"
echo "SETUP_AF20_SCRIPTS_OK"
