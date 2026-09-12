#!/usr/bin/env bash
# Stage 1 for the two most different MTP contrasts, one after the other, then the figures.
# GPU policy (user requirement, 2026-09-12): only a card with no compute process at launch, and never GPU 1
# (the user's own nnUNet training), GPU 0 (AF16 loop), GPUs 3-5 (AF12 loop) or 6-7 (other users). Candidate
# list: 2. If it is busy the run goes to the CPU with 8 threads.
set -uo pipefail
T=/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_3D
S=$T/scripts
cd "$S"
free_gpu() {
  for g in 2; do
    uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$g" '$1==g {print $2}')
    if ! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$uuid"; then echo "$g"; return; fi
  done
  echo ""
}
G=$(free_gpu)
if [ -n "$G" ]; then export CUDA_VISIBLE_DEVICES=$G; DEV=cuda:0; else export CUDA_VISIBLE_DEVICES=; DEV=cpu; fi
echo "stage 1 start $(date +%F_%T) device=$DEV CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
./py prep_mtp_grappa.py --fa 16 --echo 0 --device "$DEV" "$@" || { echo "FAILED FA16 echo0"; exit 1; }
./py prep_mtp_grappa.py --fa 4  --echo 4 --device "$DEV" "$@" || { echo "FAILED FA4 echo4"; exit 1; }
CUDA_VISIBLE_DEVICES= ./py fig_mtp_grappa.py --contrast FA16_TE3.25 || { echo "FAILED figure FA16"; exit 1; }
CUDA_VISIBLE_DEVICES= ./py fig_mtp_grappa.py --contrast FA4_TE27.41 || { echo "FAILED figure FA4"; exit 1; }
echo "STAGE1_DONE $(date +%F_%T)"
