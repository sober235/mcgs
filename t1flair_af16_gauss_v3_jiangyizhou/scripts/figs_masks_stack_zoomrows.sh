#!/usr/bin/env bash
# figs_masks_stack_zoomrows.sh -- the stacked three-pattern figure at transverse 205 with dedicated zoom rows (2026-09-17: the corner
# zoom was too small and its box showed too little difference). Boxes per contrast from find_zoom_box_visible.py / eval_zoom_box.py:
# T1 40x40 at (160, 131), FLAIR 32x32 at (148, 146) -- MC-GS first over every column (GS-single included) on box PSNR and SSIM in all
# three pattern rows, with the widest gap to the best non-GS column. Step 1 re-checks that the default render is unchanged.
cd "$(dirname "$0")/.."
PY="env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES= /home/congcongliu/anaconda3/envs/GSTMI/bin/python"
$PY scripts/fig_masks_transverse_stack.py --contrast T1 --tra 205 --tag _p3_zf_caps_defaultcheck || exit 1
$PY - <<'PYEOF' || exit 1
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
a = np.asarray(Image.open("figures/t1flair_v3_af16_masks_stack_T1_tra205_headerr_p3_zf_caps.png"))
b = np.asarray(Image.open("figures/t1flair_v3_af16_masks_stack_T1_tra205_headerr_p3_zf_caps_defaultcheck.png"))
print("DEFAULT RENDER UNCHANGED:", a.shape == b.shape and np.array_equal(a, b))
PYEOF
rm -f figures/t1flair_v3_af16_masks_stack_T1_tra205_headerr_p3_zf_caps_defaultcheck.png \
      /home/congcongliu/figs/t1flair_v3_af16_diffmask/t1flair_v3_af16_masks_stack_T1_tra205_headerr_p3_zf_caps_defaultcheck.png
$PY scripts/fig_masks_transverse_stack.py --contrast T1 --tra 205 --zoom_rows 40 --inset_center 160 131 --pdf --tag _p3_zf_caps_box160_131 || exit 1
$PY scripts/fig_masks_transverse_stack.py --contrast FLAIR --tra 205 --zoom_rows 32 --inset_center 148 146 --pdf --tag _p3_zf_caps_box148_146 || exit 1
echo FIGS_ZOOMROWS_DONE
