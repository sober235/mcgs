#!/usr/bin/env bash
# figs_masks_stack_inset.sh -- (1) proves the default render of fig_masks_transverse_stack.py is unchanged by the 2026-09-17
# zoom options (T1 at 205 re-rendered under a scratch tag and compared pixel by pixel with the committed figure), then
# (2) renders the stacked three-pattern figure with the 48 x 48 zoom (centre 124, 76 = the figure-of-record box) at transverse
# 205 and 180, T1 and FLAIR, PNG + PDF. CPU only, one render at a time.
cd "$(dirname "$0")/.."
PY="env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES= /home/congcongliu/anaconda3/envs/GSTMI/bin/python"
$PY scripts/fig_masks_transverse_stack.py --contrast T1 --tra 205 --tag _p3_zf_caps_defaultcheck || exit 1
$PY - <<'PYEOF' || exit 1
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
a = np.asarray(Image.open("figures/t1flair_v3_af16_masks_stack_T1_tra205_headerr_p3_zf_caps.png"))
b = np.asarray(Image.open("figures/t1flair_v3_af16_masks_stack_T1_tra205_headerr_p3_zf_caps_defaultcheck.png"))
same = a.shape == b.shape and np.array_equal(a, b)
print("DEFAULT RENDER UNCHANGED:", same, a.shape, b.shape, "" if same else f"differing pixels {int((a != b).any(-1).sum())}")
PYEOF
rm -f figures/t1flair_v3_af16_masks_stack_T1_tra205_headerr_p3_zf_caps_defaultcheck.png \
      /home/congcongliu/figs/t1flair_v3_af16_diffmask/t1flair_v3_af16_masks_stack_T1_tra205_headerr_p3_zf_caps_defaultcheck.png
for tra in 205 180; do
  for c in T1 FLAIR; do
    $PY scripts/fig_masks_transverse_stack.py --contrast $c --tra $tra --inset 48 --pdf || exit 1
  done
done
echo FIGS_MASKS_STACK_INSET_DONE
