#!/usr/bin/env bash
# Stage 1 + 1b for every acquired MTP contrast that has no stage-1b output yet (10 contrasts = 5 echoes x
# 2 flip angles). Same GPU policy as run_stage1.sh (GPU 2 only if empty at each launch, else CPU); ecalib on
# the CPU with 8 threads and nice 10. Sequential, detached by the caller.
set -uo pipefail
T=/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_3D
cd "$T/scripts"
TE=(3.25 9.29 15.33 21.37 27.41)
dev() {
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==2 {print $2}')
  if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$uuid"; then echo cpu; else echo cuda:0; fi
}
echo "all contrasts start $(date +%F_%T)"
for fa in 16 4; do
  for e in 0 1 2 3 4; do
    c="FA${fa}_TE${TE[$e]}"
    if [ -f "$T/data/$c/sense_grappa.npy" ]; then echo "skip $c (done)"; continue; fi
    D=$(dev); if [ "$D" = cpu ]; then export CUDA_VISIBLE_DEVICES=; else export CUDA_VISIBLE_DEVICES=2; fi
    echo "== $c device=$D $(date +%F_%T)"
    [ -f "$T/data/$c/meta.json" ] || ./py prep_mtp_grappa.py --fa "$fa" --echo "$e" --device "$D" || { echo "FAILED prep $c"; exit 1; }
    CUDA_VISIBLE_DEVICES= ./py fig_mtp_grappa.py --contrast "$c" || { echo "FAILED fig $c"; exit 1; }
    ./py stage1b_sense.py --contrast "$c" --device "$D" || { echo "FAILED stage1b $c"; exit 1; }
  done
done
echo "ALL_CONTRASTS_DONE $(date +%F_%T)"
