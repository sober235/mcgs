#!/usr/bin/env bash
# render_sel7.sh -- the AF 20 figure of record with only the seven methods the user asked for
# (2026-09-18): CS-SENSE, Joint SENSE, ZS-SSDU, 3D-INR, MC-INR, GS-single, MC-GS. GT keeps its
# column (the error maps are against it, the sampling mask sits under it); ZF, CG-SENSE,
# L1-SPIRiT and 3D-DIP are dropped. Everything else is render_hires_final.sh unchanged:
# same slices, inset, grey window, error colour scale, orientation, metrics, 200 dpi PNG and
# the vector PDF whose panels are rasterised at 600 dpi.
# The in-panel bold now marks the best of the seven shown; the numbers themselves are unchanged.
set -u
R=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF20_v3_jiangyizhou
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
TITLE_FS=${TITLE_FS:-22}
KEEP=(CS-SENSE "Joint SENSE" ZS-SSDU 3D-INR MC-INR GS-single MC-GS)

cd "$R" && mkdir -p logs/sel7
for c in T1 FLAIR; do
  PYTHONNOUSERSITE=1 PYTHONPATH= nice -n 5 $PY scripts/fig_three_planes_compact.py \
    --contrast "$c" --slices 180 124 76 --inset 48 --inset_corner tr bl bl --mask_error \
    --gray_cbar --cbar_planes 0 --tra_rot 1 --err_vmax 0.1 --pdf --png_dpi 200 \
    --caps --title_fs "$TITLE_FS" --mcgs_key gs_joint_p3 --single_key gs_single_p3 \
    --tag _p3_mcinr --with_mcinr --keep "${KEEP[@]}" > "logs/sel7/fig_${c}.log" 2>&1 &
done
wait
echo "SEL7_DONE $(date +%H:%M)"
grep -h '^wrote' logs/sel7/fig_*.log
