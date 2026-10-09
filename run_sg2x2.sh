#!/usr/bin/env bash
# run_sg2x2.sh EXP GPU CELL [ITERS] [SEED] [extra arc/cycle.py args...]
#   One cell of the shared-geometry 2x2 ablation (docs/experiments/2026-10-09-mcgs-shared-geometry-2x2-ablation.md).
#   CELL = A | B_T1 | B_FLAIR | C | D_T1 | D_FLAIR (scripts/sg2x2_cells.py is the only place the cells are defined).
#   Env: NPTS (3600000), INIT (support_file | head), SUPPORT (configs/sg2x2/init_support_af16_jyz.npz),
#        SAVE_VOLUMES (1), WARMUP (500). Extra args come last and override (repeated flags: last wins).
#   Refuses to overwrite an existing run of the same name. Writes the manifest + regulariser audit afterwards.
set -uo pipefail
if [[ $# -lt 3 ]]; then echo "usage: $0 EXP GPU CELL [ITERS] [SEED] [extra cycle.py args...]" >&2; exit 2; fi
EXP=$1; GPU=$2; CELL=$3; ITERS=${4:-4000}; SEED=${5:-42}; shift 3; [[ $# -gt 0 ]] && shift; [[ $# -gt 0 ]] && shift
[[ "$EXP" == sg2x2_* ]] || { echo "EXP must start with sg2x2_: $EXP" >&2; exit 2; }
case "$CELL" in A|B_T1|B_FLAIR|C|D_T1|D_FLAIR) ;; *) echo "CELL must be A, B_T1, B_FLAIR, C, D_T1 or D_FLAIR: $CELL" >&2; exit 2 ;; esac
ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
NPTS=${NPTS:-3600000}; INIT=${INIT:-support_file}; SUPPORT=${SUPPORT:-$ROOT/configs/sg2x2/init_support_af16_jyz.npz}
WARMUP=${WARMUP:-500}
cd "$ROOT"
for path in "results_arc/${EXP}.log" "results_arc/${EXP}_score.json" "results_arc/${EXP}"; do
  if [[ -e "$path" ]]; then echo "refusing to overwrite existing output: $path" >&2; exit 1; fi
done
mapfile -t CELLARGS < <(env PYTHONNOUSERSITE=1 PYTHONPATH= "$PY" scripts/sg2x2_cells.py "$CELL" --npts "$NPTS" --seed "$SEED" \
  --support "$SUPPORT" --init "$INIT" --iters "$ITERS" --warmup "$WARMUP")
[[ ${#CELLARGS[@]} -gt 0 ]] || { echo "scripts/sg2x2_cells.py produced no arguments for $CELL (a process substitution hides its exit code)" >&2; exit 2; }
VOL=(); if [[ "${SAVE_VOLUMES:-1}" == "1" ]]; then VOL=(--save_volumes); fi
mkdir -p results_arc
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID \
  "$PY" -u arc/cycle.py --exp "$EXP" --gpu "$GPU" "${CELLARGS[@]}" "${VOL[@]}" "$@" > "results_arc/${EXP}.log" 2>&1
RC=$?
echo "[$(date '+%F %H:%M')] $EXP cell=$CELL seed=$SEED npts=$NPTS init=$INIT iters=$ITERS extra=[$*] rc=$RC" >> results_arc/sg2x2_runs.log
if [[ $RC -eq 0 ]]; then
  env PYTHONNOUSERSITE=1 PYTHONPATH= "$PY" scripts/sg2x2_manifest.py "$EXP" "$CELL" --seed "$SEED" --npts "$NPTS" --init "$INIT" \
    --support "$SUPPORT" --iters "$ITERS" >> "results_arc/${EXP}.log" 2>&1 || echo "[$(date '+%F %H:%M')] $EXP manifest FAILED" >> results_arc/sg2x2_runs.log
fi
exit $RC
