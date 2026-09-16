#!/usr/bin/env bash
# 50-iteration smoke of the gauss-loop runner on card $1: runner, CUDA extension build, scorer and ledger row.
cd "$(dirname "$0")/.."
env WARMUP=20 ./autoresearch_gauss.sh g_smoke "${1:-4}" 50 > results_arc/g_smoke.launcher.log 2>&1
echo "SMOKE_RC=$?" >> results_arc/g_smoke.launcher.log
