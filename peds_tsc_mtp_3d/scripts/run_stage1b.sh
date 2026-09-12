#!/usr/bin/env bash
# Stage 1b for both contrasts: ESPIRiT maps (BART on the CPU, 8 threads, nice 10), SENSE combination and
# CG-SENSE on the GPU. Same GPU policy as run_stage1.sh (GPU 2 only if empty, else CPU).
set -uo pipefail
T=/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_3D
cd "$T/scripts"
uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==2 {print $2}')
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$uuid"; then export CUDA_VISIBLE_DEVICES=; DEV=cpu; else export CUDA_VISIBLE_DEVICES=2; DEV=cuda:0; fi
echo "stage 1b start $(date +%F_%T) device=$DEV"
for c in FA16_TE3.25 FA4_TE27.41; do
  ./py stage1b_sense.py --contrast "$c" --device "$DEV" "$@" || { echo "FAILED $c"; exit 1; }
done
echo "STAGE1B_DONE $(date +%F_%T)"
