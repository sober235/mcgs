#!/usr/bin/env bash
# Spec 5.8 for one subject whose raw.h5 and spokes.npz exist (Tasks 6 and 8), in two phases because criterion
# (b) of the polarity check needs boxes placed on this subject's R0 image (spec 5.4, 2026-09-11):
#   phase 1: coil maps, R0 (8 echoes) and its figures, the two respiratory thirds at echo 3
#   phase 2 (2x FOV, correction 5): 224^3 coil maps, polarity (112^3 thirds), REF (8 echoes on 224^3, lambda relative
#   to this subject's 224^3 lambda_max), the two checks, REF figure
# Usage: run_subject.sh <Vxx> <1|2|2a|2b> <iterations> <REF lambda_rel>
#   2a = the 224^3 coil maps and the polarity check only (not gated by the V07 self-check); 2b = the rest of phase 2
set -euo pipefail
S=$1
PHASE=$2
IT=$3
REL=$4
cd /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts
if [ "$PHASE" = 1 ]; then
  ./py s05_coilmaps.py "$S"
  ./py s06_recon.py "$S" --set r0 --iters "$IT"
  ./py s11_figures.py "$S" --tag r0
  ./py orient_views.py "$S"
  ./py s06_recon.py "$S" --set low --echoes 3 --iters "$IT"
  ./py s06_recon.py "$S" --set high --echoes 3 --iters "$IT"
fi
if [ "$PHASE" = 2 ] || [ "$PHASE" = 2a ]; then
  ./py s05b_coilmaps_2x.py "$S"
  ./py s07_polarity.py "$S"
fi
if [ "$PHASE" = 2 ] || [ "$PHASE" = 2b ]; then
  ./py s06_recon.py "$S" --set ref --debug --iters "$IT" --lam "$REL" --lam-rel
  ./py s08_checks.py "$S" --ref-iters "$IT" --lam-rel "$REL"
  ./py s11_figures.py "$S" --tag ref
fi
echo "PHASE${PHASE}_DONE $S"
