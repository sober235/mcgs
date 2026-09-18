#!/usr/bin/env bash
# figs_masks_stack_maxgap.sh -- the stacked three-pattern figure on the slice of LARGEST whole-slice margin over the comparison
# methods, chosen per contrast (the two figures are separate): T1 transverse 104 (three patterns first on PSNR/SSIM/NMSE,
# smallest PSNR margin +0.178 dB against +0.122 at 205), FLAIR transverse 148 (+0.063 against +0.050 at 205); slice ranking from
# find_slice_all_masks.json. Zoom rows with the box of the widest gap to the best non-GS column on that slice
# (find_zoom_box_visible.py --rank nongs): T1 32x32 at (107, 34), FLAIR 32x32 at (141, 139).
cd "$(dirname "$0")/.."
PY="env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES= /home/congcongliu/anaconda3/envs/GSTMI/bin/python"
$PY scripts/fig_masks_transverse_stack.py --contrast T1 --tra 104 --zoom_rows 32 --inset_center 107 34 --pdf --tag _p3_zf_caps_maxgap || exit 1
$PY scripts/fig_masks_transverse_stack.py --contrast FLAIR --tra 148 --zoom_rows 32 --inset_center 141 139 --pdf --tag _p3_zf_caps_maxgap || exit 1
echo FIGS_MAXGAP_DONE
