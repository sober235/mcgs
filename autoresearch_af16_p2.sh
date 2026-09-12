#!/usr/bin/env bash
# One phase-2 autoresearch experiment on the jiangyizhou T1+FLAIR AF16 pair (see autoresearch_af16_p2.md).
#   autoresearch_af16_p2.sh EXP GPU ITERS [extra arc/cycle.py args...]
#
# ISOLATION from the AF12 experiment (user request, 2026-09-11): this line is AF16 only. The base config is
# hard-coded here and there is NO CONFIG environment override, EXP must start with p2_, and rows go to
# autoresearch_af16_p2.tsv. Nothing in this line can touch the AF12 configs, runs or ledger.
#
# PARENT = P6, the final recipe of the first jyz AF16 loop (autoresearch_jyz_af16.md, Verdict 2026-09-11 03:50):
# the v2 edge-TV working point + wavelet sparsity T1 0.10 / FLAIR 0.07 + 3.6M points. Every candidate differs
# from the parent only by the extra args it passes (cycle.py flags repeat -> last wins; --set applies in argv order).
# SAVE_VOLUMES=1 dumps the recon npz (promotion runs only). WARMUP=20 for smoke runs.
set -uo pipefail
if [[ $# -lt 3 ]]; then
  echo "usage: $0 EXP GPU ITERS [extra cycle.py args...]" >&2; exit 2
fi
EXP=$1; GPU=$2; ITERS=$3; shift 3
if [[ "$EXP" != p2_* ]]; then
  echo "EXP must start with p2_ (phase-2 namespace, keeps this line apart from the AF12 work): $EXP" >&2; exit 2
fi
WARMUP=${WARMUP:-500}
ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
# AF16 jyz only, deliberately not overridable by a path. SINGLE=T1|FLAIR picks that contrast's
# single-arm config by name, so the AF12 configs stay unreachable from this line.
case "${SINGLE:-joint}" in
  joint) CONFIG=configs/t1flair_v3_af16_jyz.yaml ;;
  T1)    CONFIG=configs/t1flair_v3_af16_jyz_singleT1.yaml ;;
  FLAIR) CONFIG=configs/t1flair_v3_af16_jyz_singleFLAIR.yaml ;;
  *)     echo "SINGLE must be T1, FLAIR or unset: ${SINGLE}" >&2; exit 2 ;;
esac
LOG="$ROOT/results_arc/${EXP}.log"
for path in "$LOG" "$ROOT/results_arc/${EXP}_score.json" "$ROOT/results_arc/${EXP}"; do
  if [[ -e "$path" ]]; then echo "refusing to overwrite existing output: $path" >&2; exit 1; fi
done
VOL=()
if [[ "${SAVE_VOLUMES:-0}" == "1" ]]; then VOL=(--save_volumes); fi
cd "$ROOT"
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID TORCH_EXTENSIONS_DIR=/data0/congcong/code/GS/mcgs_af16_wt.torch_ext \
  "$PY" -u arc/cycle.py --exp "$EXP" --gpu "$GPU" --iters "$ITERS" \
  --lambda_geo 0.10 --warmup "$WARMUP" --max_radius 20 --kspace_high 2.2 \
  --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 \
  --set gaussian.init_mode=head --set gaussian.init_scale_mode=3nn \
  --set loss.edge_tv_kappa_source=zf \
  --set loss.lambda_wav=0.07 --set loss.lambda_wav_plain=0.10 \
  --set gaussian.initial_num_points=3600000 \
  --base_cfg "$CONFIG" "${VOL[@]}" "$@" > "$LOG" 2>&1
RC=$?
STATUS=pending
if [[ $RC -ne 0 ]]; then STATUS=crash; fi
env PYTHONNOUSERSITE=1 PYTHONPATH= "$PY" scripts/autoresearch_af16_p2_row.py --exp "$EXP" --status "$STATUS" --note "rc=$RC"
exit $RC
