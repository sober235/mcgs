#!/usr/bin/env bash
# One MC-GS / GS-single run for the coupling-attribution study
# (spec docs/superpowers/specs/2026-09-20-coupling-attribution-design.md):
# jiangyizhou T1+FLAIR at nominal AF 16, house mask (T1 15.19x / FLAIR 16.49x).
#   run_ca_mcgs.sh EXP GPU ITERS CONFIG LAMBDA_EDGE [extra arc/cycle.py args...]
#   SAVE_VOLUMES=1 dumps the recon npz.  WARMUP (default 500) for short smokes.
#
# The recipe block below is byte-identical to run_hp_mcgs.sh except that the base config and
# --lambda_edge are positionals: CONFIG selects the arm (joint / singleT1 / singleFLAIR) and
# LAMBDA_EDGE selects whether the common-edge cross-gradient prior is on (0.001) or off (0).
# Everything else -- capacity, edge-TV, wavelet, phase TV, warm-up, iterations, seed -- is frozen.
set -uo pipefail
if [[ $# -lt 5 ]]; then echo "usage: $0 EXP GPU ITERS CONFIG LAMBDA_EDGE [extra cycle.py args...]" >&2; exit 2; fi
EXP=$1; GPU=$2; ITERS=$3; CONFIG=$4; LEDGE=$5; shift 5
if [[ "$EXP" != ca_* ]]; then
  echo "EXP must start with ca_ (keeps this study apart from the hp_ and AF12/16/20/24 lines): $EXP" >&2; exit 2
fi
if [[ ! -f "$(dirname "$0")/$CONFIG" ]]; then echo "no such config: $CONFIG" >&2; exit 2; fi
WARMUP=${WARMUP:-500}
ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
LOG="$ROOT/results_arc/${EXP}.log"
for path in "$LOG" "$ROOT/results_arc/${EXP}_score.json" "$ROOT/results_arc/${EXP}"; do
  if [[ -e "$path" ]]; then echo "refusing to overwrite existing output: $path" >&2; exit 1; fi
done
VOL=()
if [[ "${SAVE_VOLUMES:-0}" == "1" ]]; then VOL=(--save_volumes); fi
mkdir -p "$ROOT/results_arc"
cd "$ROOT"
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID TORCH_EXTENSIONS_DIR=/data0/congcong/code/GS/mcgs_hparam_wt.torch_ext \
  "$PY" -u arc/cycle.py --exp "$EXP" --gpu "$GPU" --iters "$ITERS" \
  --lambda_geo 0.10 --warmup "$WARMUP" --max_radius 20 --kspace_high 2.2 \
  --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 \
  --set gaussian.init_mode=head --set gaussian.init_scale_mode=3nn \
  --set loss.edge_tv_kappa_source=zf \
  --set gaussian.initial_num_points=3600000 \
  --set loss.wav_name=db4 --set loss.wav_levels=1 \
  --set loss.lambda_wav_plain=0.12 --set loss.lambda_wav=0.10 \
  --lambda_edge "$LEDGE" --lambda_ptv 0.03 \
  --base_cfg "$CONFIG" "${VOL[@]}" "$@" > "$LOG" 2>&1
RC=$?
echo "[$(date '+%F %H:%M')] $EXP cfg=$CONFIG ledge=$LEDGE rc=$RC  $(grep -o 'head_psnr[^,]*' "$ROOT/results_arc/${EXP}_score.json" 2>/dev/null | head -2 | tr '\n' ' ')" >> "$ROOT/results_arc/ca_runs.log"
exit $RC
