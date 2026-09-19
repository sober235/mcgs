#!/usr/bin/env bash
# The three-mask generalisation figure (vd-Poisson / Gaussian / pseudo-radial, one frozen MC-GS recipe)
# with only the seven methods of the 2026-09-18 cut: CS-SENSE, Joint SENSE, ZS-SSDU, 3D-INR, MC-INR,
# GS-single, MC-GS. GT keeps its column; ZF, CG-SENSE, L1-SPIRiT and 3D-DIP are dropped -- so the file
# name drops the _zf of the record figure's tag. Everything else is the 2026-09-18 figure of record
# unchanged: same slices and boxes (T1 transverse 126, 32x32 at (146, 29); FLAIR transverse 205,
# 32x32 at (148, 146)), --inset_frac 0.5, --grey_scale 0.8, same metrics and resolution.
cd "$(dirname "$0")/.."
PY="env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES= nice -n 5 /home/congcongliu/anaconda3/envs/GSTMI/bin/python"
K=(--keep CS-SENSE "Joint SENSE" ZS-SSDU 3D-INR MC-INR GS-single MC-GS)
mkdir -p logs/sel7
$PY scripts/fig_masks_transverse_stack.py --contrast T1    --tra 126 --inset 32 --inset_center 146 29  \
    --inset_frac 0.5 --grey_scale 0.8 --pdf --tag _p3_caps_nongsmax "${K[@]}" > logs/sel7/masks_stack_T1.log 2>&1 &
$PY scripts/fig_masks_transverse_stack.py --contrast FLAIR --tra 205 --inset 32 --inset_center 148 146 \
    --inset_frac 0.5 --grey_scale 0.8 --pdf --tag _p3_caps_nongsmax "${K[@]}" > logs/sel7/masks_stack_FLAIR.log 2>&1 &
wait
echo "MASKS_STACK_SEL7_DONE $(date +%H:%M)"
grep -h "^wrote\|text under a zoom" logs/sel7/masks_stack_*.log
