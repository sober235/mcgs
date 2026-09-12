#!/usr/bin/env bash
# Wait for run_all_contrasts.sh to finish cleanly, then export every finished contrast to the dataset root.
T=/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_3D
P=$(cat "$T/logs/run_all_contrasts.pid")
while kill -0 "$P" 2>/dev/null; do sleep 60; done
if ! grep -q ALL_CONTRASTS_DONE "$T/logs/run_all_contrasts.log"; then echo "batch did not finish cleanly; exporting what exists anyway $(date +%F_%T)"; fi
cd "$T/scripts" && CUDA_VISIBLE_DEVICES= ./py export_gt_dataset.py && echo "EXPORT_DONE $(date +%F_%T)"
