#!/usr/bin/env bash
# The T1-row / FLAIR-row figure with the added-noise 3-D surface bottom-left, cut to the seven methods
# of 2026-09-18 (CS-SENSE, Joint SENSE, ZS-SSDU, 3D-INR, MC-INR, GS-single, MC-GS; GT keeps its column,
# with the sampling mask under T1 and the noise surface under FLAIR).
# The zoom box is chosen from the lead of MC-GS over the best OTHER column, so it would move when the
# column set changes: this script first reproduces the figure of record to read the box it picked, then
# renders the cut with that same box, so both figures show the same piece of anatomy.
set -u
cd "$(dirname "$0")/.."
PY="env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES= nice -n 5 /home/congcongliu/anaconda3/envs/GSTMI/bin/python"
mkdir -p logs/sel7
out=$($PY scripts/fig_tra_t1_flair_zoom3d.py --slice 170 2>&1 | tee logs/sel7/noise3d_record_rerun.log)
box=$(sed -n 's/.*centre (j, k) = (\([0-9]*\), \([0-9]*\)).*/\1 \2/p' <<<"$out")
[[ -n $box ]] || { echo "could not read the box centre"; exit 1; }
echo "box of the figure of record: $box"
$PY scripts/fig_tra_t1_flair_zoom3d.py --slice 170 --box $box --pdf \
  --keep CS-SENSE "Joint SENSE" ZS-SSDU 3D-INR MC-INR GS-single MC-GS 2>&1 | tee logs/sel7/noise3d_sel7.log
echo "NOISE3D_SEL7_DONE $(date +%H:%M)"
