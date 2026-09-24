#!/usr/bin/env bash
# One run of the widen-the-margin loop (branch widen-mcgs-af16-20-24): jiangyizhou T1+FLAIR at AF 16 / 20 / 24.
#   run_wd_mcgs.sh EXP GPU ITERS AF ARM LAMBDA_EDGE [extra arc/cycle.py args...]
#   AF = 16 | 20 | 24 (picks that topic's house mask config); ARM = joint | T1 | FLAIR (T1 / FLAIR = GS-single).
#   SAVE_VOLUMES=1 dumps the recon npz.  WARMUP (default 500) for short smokes.
#
# The recipe block is byte-identical to run_ca_mcgs.sh (the AF16 phase-3 recipe); LAMBDA_EDGE is positional because the
# user decided 2026-09-24 that GS-single runs at lambda_edge 0 (its own best: the common-edge prior hurts the single arm)
# and only the joint arm's cross-contrast knobs are tuned. Extra args come last and override (repeated flags -> last wins).
set -uo pipefail
if [[ $# -lt 6 ]]; then echo "usage: $0 EXP GPU ITERS AF ARM LAMBDA_EDGE [extra cycle.py args...]" >&2; exit 2; fi
EXP=$1; GPU=$2; ITERS=$3; AF=$4; ARM=$5; LEDGE=$6; shift 6
[[ "$EXP" == wd_* ]] || { echo "EXP must start with wd_: $EXP" >&2; exit 2; }
case "$AF" in 16|20|24) ;; *) echo "AF must be 16, 20 or 24: $AF" >&2; exit 2 ;; esac
case "$ARM" in
  joint) CONFIG=configs/t1flair_v3_af${AF}_jyz.yaml ;;
  T1|FLAIR) CONFIG=configs/t1flair_v3_af${AF}_jyz_single${ARM}.yaml ;;
  *) echo "ARM must be joint, T1 or FLAIR: $ARM" >&2; exit 2 ;;
esac
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
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID \
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
echo "[$(date '+%F %H:%M')] $EXP af=$AF arm=$ARM ledge=$LEDGE extra=[$*] rc=$RC" >> "$ROOT/results_arc/wd_runs.log"
exit $RC
