#!/usr/bin/env bash
# after an ar queue: for every scored ar_* MC-GS run listed in a jobs file, the lesion-slice all-metric scan with that run as MC-GS
JOBS=$1
until grep -q "QUEUE DONE" /data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois/logs/ar/queue.log && [ $(grep -c "QUEUE DONE" /data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois/logs/ar/queue.log) -ge $2 ]; do sleep 60; done
cd /data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois_ar
for e in $(grep -v "^#" $JOBS | awk -F'\t' '$2=="joint"{print $1}'); do
  [ -f /data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois/recon/${e}_post_complex.npy ] || { echo "missing $e"; continue; }
  env RETRO_TOPIC=/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois_ar PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python scripts/lesion_slice_allmetrics.py --mcgs_file "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois/recon/${e}_{scan}_complex.npy" --out logs/lesion_allmetrics_${e}.json > logs/lesion_allmetrics_${e}.out 2>&1
  echo "$e $(head -1 logs/lesion_allmetrics_${e}.out)"
done
echo SCORE_DONE
