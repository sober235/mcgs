#!/usr/bin/env bash
# the same selection as the zoom-row figures (box chosen by the gap to the best non-GS column) but with the zoom ON the panel:
# a corner inset half the panel wide, grey window at 0.8 x P99.9 (user 2026-09-18: brighter) (--inset_frac 0.5, the 0.38 of the figures of record was too small to read).
# T1: transverse 126, 32x32 at (146, 29).  FLAIR: transverse 205, 32x32 at (148, 146).
cd "$(dirname "$0")/.."
PY="env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES= /home/congcongliu/anaconda3/envs/GSTMI/bin/python"
$PY scripts/fig_masks_transverse_stack.py --contrast T1 --tra 126 --inset 32 --inset_center 146 29 --inset_frac 0.5 --grey_scale 0.8 --pdf --tag _p3_zf_caps_nongsmax || exit 1
$PY scripts/fig_masks_transverse_stack.py --contrast FLAIR --tra 205 --inset 32 --inset_center 148 146 --inset_frac 0.5 --grey_scale 0.8 --pdf --tag _p3_zf_caps_nongsmax || exit 1
echo FIGS_INSET_NONGSMAX_DONE
