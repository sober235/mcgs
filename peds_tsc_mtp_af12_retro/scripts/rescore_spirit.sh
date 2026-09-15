#!/usr/bin/env bash
# Rescore the L1-SPIRiT arm alone on both gauges and merge it into logs/rescore.json (primary, partitions
# 12..92) and logs/rescore_full.json (the cross-check); the other rows are untouched (deterministic gauge).
# 2026-09-15, after the sweep of scripts/spirit_sweep{,_full24}.sh and scripts/summarize.py.
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=
{
  echo "$(date '+%F %T') rescore --only spirit: primary + full, 2 jobs each"
  ./scripts/py scripts/rescore.py --only spirit --jobs 2 &
  ./scripts/py scripts/rescore.py --only spirit --jobs 2 --full &
  wait
  echo "$(date '+%F %T') done rc=$?"
} >> logs/rescore_spirit.log 2>&1
