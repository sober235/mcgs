#!/usr/bin/env bash
# the stacked three-pattern figure at transverse 205 with the zoom box of the largest weakest-row lead found on slices 203-206
# (find_zoom_box_all_masks.py --step 1: centre 138, 76; smallest box-PSNR lead +0.219 dB against +0.154 for the record box)
cd "$(dirname "$0")/.."
for c in T1 FLAIR; do
  env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES= /home/congcongliu/anaconda3/envs/GSTMI/bin/python scripts/fig_masks_transverse_stack.py \
    --contrast $c --tra 205 --inset 48 --inset_center 138 76 --pdf --tag _p3_zf_caps_box138_76 || exit 1
done
echo FIGS_BOX138_DONE
