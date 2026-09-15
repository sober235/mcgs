#!/usr/bin/env bash
# The figures of record with the L1 column swapped to L1-SPIRiT (user 2026-09-15): the compact three-plane
# figure of the final configuration (_p3f) and its AF16-layout version with the 48-voxel zoom (_af16style),
# both contrasts. Same slices / keys / options as the 2026-09-15 originals; only --l1 (default spirit) and the tag differ.
# T1w's tuned single is gs_single_p3f; T2s's own weights were never tuned, so gs_single_ar is its tuned single.
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=
for c in T1w T2s; do
  single=gs_single_p3f; [ "$c" = T2s ] && single=gs_single_ar
  ./scripts/py scripts/fig_three_planes_compact.py --contrast $c --slices 53 160 152 --mask_error \
      --mcgs_key gs_joint_p3f --single_key $single --tag _p3f_spirit > logs/fig_compact_${c}_spirit.log 2>&1 &
  ./scripts/py scripts/fig_three_planes_inset.py --contrast $c --slices 53 160 152 --inset 48 --inset_corner tr bl bl \
      --mask_error --gray_cbar --cbar_planes 0 --pdf --mcgs_key gs_joint_p3f --single_key $single \
      --tag _af16style_spirit > logs/fig_inset_${c}_spirit.log 2>&1 &
done
wait
echo "$(date '+%F %T') figs_spirit done" >> logs/fig_compact_T1w_spirit.log
