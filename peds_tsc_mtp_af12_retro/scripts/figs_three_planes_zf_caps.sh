#!/usr/bin/env bash
# Three-plane comparison figures (transverse 46 / coronal 146 / sagittal 161, the both-contrast all-first sagittal slice of
# 2026-09-16) in the 2026-09-16 typography: ZF column after GT, plane names capitalised, method names upper case;
# L1 column = L1-SPIRiT; error maps inside the head only; PNG + 600-dpi PDF. CPU only, one contrast at a time (a 600-dpi
# PDF render holds ~100 GB of RAM).  MCGS_KEY / T1W_SINGLE / T2S_SINGLE / TAG may be overridden in the environment.
#   MCGS_KEY=gs_joint_p5pre  -> the stage-5 preview arm (p5_ptv001_i6000, not registered; recon/gs_joint_p5pre_*)
cd "$(dirname "$0")/.."
MCGS_KEY=${MCGS_KEY:-gs_joint_p5pre}; T1W_SINGLE=${T1W_SINGLE:-gs_single_p3f}; T2S_SINGLE=${T2S_SINGLE:-gs_single_ar}
TAG=${TAG:-_p5pre_bothfirst161}
for c in T1w T2s; do
  single=$T1W_SINGLE; [[ $c == T2s ]] && single=$T2S_SINGLE
  CUDA_VISIBLE_DEVICES= ./scripts/py scripts/fig_three_planes_inset.py --contrast $c --slices 46 146 161 \
    --inset 48 48 16 --inset_center 157 -1 -1 --inset_corner tr bl bl --mask_error --gray_cbar --cbar_planes 0 --pdf \
    --l1 spirit --mcgs_key "$MCGS_KEY" --single_key "$single" --with_zf --caps --tag "$TAG" || exit 1
done
echo FIGS_THREE_PLANES_ZF_CAPS_DONE
