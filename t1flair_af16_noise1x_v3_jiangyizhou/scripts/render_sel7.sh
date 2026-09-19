#!/usr/bin/env bash
# render_sel7.sh -- this topic's figure of record with only the seven methods the user asked for
# (2026-09-18): CS-SENSE, Joint SENSE, ZS-SSDU, 3D-INR, MC-INR, GS-single, MC-GS. GT keeps its column
# (the error maps are against it, the sampling mask sits under it); ZF, CG-SENSE, L1-SPIRiT / L1-ESPIRiT
# and 3D-DIP are dropped. measurement-noise robustness, 1 sigma0; MC-INR on by default.
# Everything else is the figure of record unchanged: same slices, inset, grey window, error scale,
# orientation, metrics and resolution. The in-panel bold now marks the best of the seven shown; the
# numbers themselves do not depend on which other arms are on the figure.
set -u
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
cd "$(dirname "$0")/.."
mkdir -p logs/sel7
for c in T1 FLAIR; do
  PYTHONNOUSERSITE=1 PYTHONPATH= nice -n 5 $PY scripts/fig_three_planes_compact.py --contrast "$c" \
    --slices 180 124 76 --inset 48 --inset_corner tr bl bl --mask_error --gray_cbar --cbar_planes 0 --tra_rot 1 --err_vmax 0.1 --pdf --caps --mcgs_key gs_joint_p3 --single_key gs_single_p3 --tag _p3_mcinr \
    --keep CS-SENSE "Joint SENSE" ZS-SSDU 3D-INR MC-INR GS-single MC-GS > "logs/sel7/fig_${c}.log" 2>&1 &
done
wait
echo "SEL7_DONE $(date +%H:%M)"
grep -h '^wrote' logs/sel7/fig_*.log
