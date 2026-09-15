#!/usr/bin/env bash
# run_spirit_sweep.sh -- L1-SPIRiT lambda / kernel-size sweep, the same privilege every other
# classical arm here has (lambda swept on this subject against the ground truth).
# The job list is partitioned across the given GPUs and each GPU runs ITS list sequentially:
# one job per card at a time (a second big job on a busy card hung this machine twice in this
# campaign), and no lock file to race on.
# usage: run_spirit_sweep.sh <contrast> <gpu0> [gpu1 ...]
set -u
R=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_v3_jiangyizhou
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
C=${1:?contrast}; shift
GPUS=("$@")
mkdir -p "$R/logs/spirit"

JOBS=()
for ks in 3 5; do
  for lam in 0 0.0005 0.001 0.002 0.004 0.008; do JOBS+=("$ks $lam"); done
done

for gi in "${!GPUS[@]}"; do
  g=${GPUS[$gi]}
  (
    for ji in "${!JOBS[@]}"; do
      [ $((ji % ${#GPUS[@]})) -eq "$gi" ] || continue
      set -- ${JOBS[$ji]}; ks=$1; lam=$2
      sfx="_ks${ks}_lam${lam}"
      echo "gpu $g -> $C $sfx"
      PYTHONNOUSERSITE=1 PYTHONPATH= $PY "$R/scripts/recon_spirit.py" --contrast "$C" --gpu "$g" \
          --ksize "$ks" --lam "$lam" --iters 150 --chunk 24 --suffix "$sfx" \
          > "$R/logs/spirit/${C}${sfx}.log" 2>&1
    done
  ) &
done
wait
echo "sweep done: $C"
