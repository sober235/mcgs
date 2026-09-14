#!/bin/bash
# setup_af24_jyz.sh -- clone the jiangyizhou AF20 topic into an AF24 one (2026-09-14).
# Exactly one thing differs from T1FLAIR_AF20_v3_jiangyizhou: the vd-Poisson mask. The acceleration follows the user's
# definition of 2026-09-14 (AF = PE-plane points / sampled points that carry that contrast's data): T1, whose acquired
# support is the whole plane, is at exactly 24x (1877 of 45056 points); FLAIR shares the mask and, with its vendor
# shutter leaving the corners unacquired, lands above 24x. The mask comes from the house generator at the house
# settings, only accelx differs (prep_02_mask_pool_af24.m draws a pool, prep_03_pick_mask_af24.py takes a draw with
# exactly 1877 points). Same subject, CFL k-space, ESPIRiT maps, ground truth, gauge and arms; the MC-GS recipe stays
# the AF16 phase-3 recipe, frozen as at AF20.
# The k-space and the maps are NOT copied: data/ksp_*.h5 and data/csm_espirit_*.h5 are symlinks into the AF12 topic,
# exactly as the AF16 and AF20 topics do it.
# Run once. Afterwards: prep_02_mask_pool_af24.m (MATLAB), prep_03_pick_mask_af24.py, prep_04_export.py.
set -euo pipefail
SRC=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF20_v3_jiangyizhou
A12=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_v3_jiangyizhou
OUT=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF24_v3_jiangyizhou
mkdir -p "$OUT"/{scripts,logs,figures,data,recon}
cd "$OUT/scripts"
# not cloned: this script's AF20 twin; the AF20 mask generator (its accelx = AF/4 rule is replaced by the pool and the
# pick); the AF20 mask-draw variance study; the MC-INR registration (no MC-INR run at AF24); postprocess.sh and
# sched_af20.py, which point at the retrospective 2026-09-09 topics (huangyuwei, the pre-phase-3 MC-GS arguments) and
# would launch the wrong recipe; run_classical_af20.sh and run_stage2_af20.sh, whose jobs and lambda grids now live in
# scripts/run_all_af24.py, the one launcher of this topic (sched_af24.py and the two cloned launchers were removed by
# hand after the first run of this script)
SKIP="setup_af20_jyz.sh prep_02_mask.m prep_02b_mask_draws.m prep_04b_export_draws.py run_mask_draws_af20.sh
      requeue_mcgs_draws_af20.sh handoff_draw_d_af20.sh score_mask_draws.py score_mcinr_arm.py postprocess.sh sched_af20.py
      run_classical_af20.sh run_stage2_af20.sh"
CLONED=""
for f in "$SRC"/scripts/*.py "$SRC"/scripts/*.m "$SRC"/scripts/*.sh; do
  b=$(basename "$f")
  case " $(echo $SKIP) " in *" $b "*) continue ;; esac
  t=${b//af20/af24}
  if [ -e "$t" ]; then echo "keep existing $t (not overwritten)"; continue; fi
  cp -p "$f" "$t"
  CLONED="$CLONED $t"
done
# mcgs_af16_wt is the FRAMEWORK worktree path (it now carries the branch autoresearch-jyz-af24); park it behind a
# token so the rewrite cannot touch it. Only the cloned files are rewritten, never the hand-written AF24 scripts.
sed -i -e 's|mcgs_af16_wt|@@WORKTREE@@|g' \
       -e 's|T1FLAIR_AF20_v3_jiangyizhou|T1FLAIR_AF24_v3_jiangyizhou|g' \
       -e 's|af20|af24|g' -e 's|AF20|AF24|g' -e 's|AF 20|AF 24|g' \
       -e 's|@@WORKTREE@@|mcgs_af16_wt|g' $CLONED
echo "--- residual AF20 references in the cloned files (expect none):"
grep -n "AF20\|af20\|AF 20" $CLONED || echo "(none)"
echo "--- AF16 references to review by hand (the recipe / figure of record stay AF16; the comparison twin becomes AF20):"
grep -ln "AF16\|af16\|AF 16" $CLONED | grep -v "^$" || true
# this subject's own k-space and ESPIRiT maps are mask-independent: reuse them through symlinks
for f in ksp_T1.h5 ksp_FLAIR.h5 csm_espirit_T1.h5 csm_espirit_FLAIR.h5; do
  ln -sfn "$A12/data/$f" "$OUT/data/$f"
done
ls -la "$OUT/data"
echo "SETUP_AF24_SCRIPTS_OK"
