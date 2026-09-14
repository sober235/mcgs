#!/usr/bin/env bash
# One GS experiment on the 16-year-old MTP AF12 pair (spec 2026-09-12-16yo-mtp-stage2-af12-design.md).
#   autoresearch_peds.sh EXP GPU ITERS [extra arc/cycle.py args...]
# The baked-in working point is P10 (jiangyizhou AF12). The two wavelet weights come from the config,
# never from --set. CONFIG= picks joint / single_T1w / single_T2s. SAVE_VOLUMES=1 dumps the npz.
# GPU policy for this line: card 2 only, and only when it is empty.
set -uo pipefail
if [[ $# -lt 3 ]]; then echo "usage: $0 EXP GPU ITERS [extra cycle.py args...]" >&2; exit 2; fi
EXP=$1; GPU=$2; ITERS=$3; shift 3
WARMUP=${WARMUP:-500}
ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
CONFIG=${CONFIG:-configs/peds_af12_T1w_T2s.yaml}
LOG="$ROOT/results_arc/${EXP}.log"
for path in "$LOG" "$ROOT/results_arc/${EXP}_score.json" "$ROOT/results_arc/${EXP}"; do
  if [[ -e "$path" ]]; then echo "refusing to overwrite existing output: $path" >&2; exit 1; fi
done
VOL=()
if [[ "${SAVE_VOLUMES:-0}" == "1" ]]; then VOL=(--save_volumes); fi
cd "$ROOT"
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID \
    TORCH_EXTENSIONS_DIR=/data0/congcong/code/GS/mcgs_peds_wt.torch_ext \
  "$PY" -u arc/cycle.py --exp "$EXP" --gpu "$GPU" --iters "$ITERS" \
  --lambda_geo 0.10 --warmup "$WARMUP" --max_radius 20 --kspace_high 2.2 \
  --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 \
  --set gaussian.init_mode=head --set gaussian.init_scale_mode=3nn \
  --set loss.edge_tv_kappa_source=zf --set gaussian.initial_num_points=3600000 \
  --set loss.wav_name=db4 --set loss.wav_levels=1 \
  --base_cfg "$CONFIG" "${VOL[@]}" "$@" > "$LOG" 2>&1
RC=$?
STATUS=pending
if [[ $RC -ne 0 ]]; then STATUS=crash; fi
# The table is judged in the primary region (head inside the central 80 partitions), so score the run
# there before the ledger row is written; without this the row's margins stay blank.
TOPIC=/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro
MAP="T1w:plain,T2s:contrast"
case "$CONFIG" in *single_T1w*) MAP="T1w:plain" ;; *single_T2s*) MAP="T2s:plain" ;; esac
if [[ $RC -eq 0 && -f "$ROOT/results_arc/${EXP}_volumes.npz" ]]; then
  "$TOPIC/scripts/py" "$TOPIC/scripts/score_candidate.py" --exp "$EXP" --map "$MAP" >> "$LOG" 2>&1
fi
env PYTHONNOUSERSITE=1 PYTHONPATH= "$PY" scripts/autoresearch_peds_row.py --exp "$EXP" --status "$STATUS" --note "rc=$RC cfg=$CONFIG"
exit $RC
