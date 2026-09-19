#!/usr/bin/env bash
# render_sel7.sh -- this topic's figure of record with only the seven methods the user asked for
# (2026-09-18): CS-SENSE, Joint SENSE, ZS-SSDU, 3D-INR, MC-INR, GS-single, MC-GS. GT keeps its column
# (the error maps are against it, the sampling mask sits under it); ZF, CG-SENSE, L1-SPIRiT / L1-ESPIRiT
# and 3D-DIP are dropped. 16yo TSC T1w + T2*, the figure of the paper (three planes through the right subependymal nodules, user 2026-09-17/18).
# Everything else is the figure of record unchanged: same slices, inset, grey window, error scale,
# orientation, metrics and resolution. The in-panel bold now marks the best of the seven shown; the
# numbers themselves do not depend on which other arms are on the figure.
set -u
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
cd "$(dirname "$0")/.."
mkdir -p logs/sel7
for c in T1w T2s; do
  single=gs_single_p3f; [[ $c == T2s ]] && single=gs_single_ar
  PYTHONNOUSERSITE=1 PYTHONPATH= nice -n 5 $PY scripts/fig_three_planes_inset.py --contrast "$c" \
    --slices 46 149 171 --inset 48 48 16 --inset_center 157 -1 -1 --inset_corner tr bl bl --mask_error --gray_cbar --cbar_planes 0 --pdf --l1 spirit --caps --mcgs_key gs_joint_p5 --tag _lesionB_allfirst_p5 --single_key "$single" \
    --keep CS-SENSE "Joint SENSE" ZS-SSDU 3D-INR MC-INR GS-single MC-GS > "logs/sel7/fig_${c}.log" 2>&1 &
done
wait
echo "SEL7_DONE $(date +%H:%M)"
grep -h '^wrote' logs/sel7/fig_*.log
