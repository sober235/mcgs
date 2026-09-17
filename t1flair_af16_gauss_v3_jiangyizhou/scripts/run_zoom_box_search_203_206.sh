#!/usr/bin/env bash
# step-1 zoom-box search on the four transverse slices where MC-GS is first on all three patterns and both contrasts (203-206)
cd "$(dirname "$0")/.."
for t in 203 204 205 206; do
  env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 /home/congcongliu/anaconda3/envs/GSTMI/bin/python \
    scripts/find_zoom_box_all_masks.py --tra $t --step 1 --top 5 > logs/find_zoom_box_all_masks_tra${t}_step1.txt 2>&1 &
done
wait
echo ZOOM_BOX_SEARCH_DONE
