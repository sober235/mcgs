#!/usr/bin/env bash
# The PHASE-2 FINAL RECIPE, P13, on the jiangyizhou T1+FLAIR pair in DESIGN B (see autoresearch_jyz_af12_p2.md).
# P13 = P10 + gaussian.rotation_lr 0.008 (promotion 5, 2026-09-13 01:05, p2_rotlr008_full4k: 36.968 / 36.683;
# 0.004 and 0.016 are nulls against it). P10 = P6 + wavelet db4 on 1 level, T1 0.12 / FLAIR 0.10. Reproduce a
# run with:  ./autoresearch_jyz_af12_P13.sh <EXP> <GPU> 4000   (SAVE_VOLUMES=1 for volumes; CONFIG= for the
# all-AF12 or single-contrast variants). The screening runner autoresearch_jyz_af12_p2g.sh keeps the P10 parent.
# T1 on the AF12 mask, FLAIR on the AF11 mask for MC-GS only; every comparison arm stays at AF12. Same P10 recipe
# baked in as autoresearch_jyz_af12.sh; the only difference is the default CONFIG (the design-B variant).
#   autoresearch_jyz_af12.sh EXP GPU ITERS [extra arc/cycle.py args...]
# The PARENT is P10 = P9 + FLAIR wavelet 0.10 + T1 wavelet 0.12 (promotion 8, 2026-09-12 05:55, ar5_best_C10_T112_full4k).
# P9 = P8 + wavelet prior on 1 level + FLAIR wavelet 0.085 (promotion 6, 2026-09-12 04:20,
# ar5_db4_lev1_C085_full4k). P8 = P7 + db4 wavelet family (promotion 4, 2026-09-12 02:25, ar5_db4_full4k).
# P7 = P6 + wavelet prior restricted to 2 decomposition levels (promotion 2, 2026-09-11 23:55,
# ar5_wavlev2_full4k). P6 was the final recipe of the jyz AF16 loop (autoresearch_jyz_af16.md, promotion 4),
# run unchanged on the AF12 mask as ar17_P6_tf_jyz_af12_full4k: frozen working point + ZF-head-support
# init, 3-NN scales, kappa from ZF, 3.6M points, wavelet T1 0.10 / FLAIR 0.07, edge-TV lambda_geo 0.10,
# kappa_q 0.85. Every experiment differs from it only by the extra args it passes (cycle.py flags
# repeat -> last wins; --set key=value overrides apply in argv order -> last wins).
# Never overwrites an existing experiment. SAVE_VOLUMES=1 dumps the recon npz (promotion runs only).
set -uo pipefail
if [[ $# -lt 3 ]]; then
  echo "usage: $0 EXP GPU ITERS [extra cycle.py args...]" >&2; exit 2
fi
EXP=$1; GPU=$2; ITERS=$3; shift 3
WARMUP=${WARMUP:-500}
ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
CONFIG=${CONFIG:-configs/t1flair_v3_af12_hyw_flair_af11.yaml}   # design B: FLAIR mask AF 11, T1 unchanged (jyz AF12 topic data)
LOG="$ROOT/results_arc/${EXP}.log"
for path in "$LOG" "$ROOT/results_arc/${EXP}_score.json" "$ROOT/results_arc/${EXP}"; do
  if [[ -e "$path" ]]; then echo "refusing to overwrite existing output: $path" >&2; exit 1; fi
done
VOL=()
if [[ "${SAVE_VOLUMES:-0}" == "1" ]]; then VOL=(--save_volumes); fi
cd "$ROOT"
# One trainer per card, ANY loop's: wait while another arc/cycle.py trainer sits on this card (the parallel AF16
# phase-3 loop launches into the gaps between our runs; agreed split 2026-09-12 18:35: this loop cards 2 and 3,
# that loop 4 and 5). The pattern matches cycle.py's argv only, never this runner's.
while pgrep -f -- "arc/cycle\.py --exp [^ ]+ --gpu ${GPU}( |\$)" >/dev/null 2>&1; do
  echo "[runner] card ${GPU} carries another trainer; waiting 60 s before launching ${EXP}" >&2
  sleep 60
done
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID TORCH_EXTENSIONS_DIR=/data0/congcong/code/GS/mcgs_af12_wt.torch_ext \
  "$PY" -u arc/cycle.py --exp "$EXP" --gpu "$GPU" --iters "$ITERS" \
  --lambda_geo 0.10 --warmup "$WARMUP" --max_radius 20 --kspace_high 2.2 \
  --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 \
  --set gaussian.init_mode=head --set gaussian.init_scale_mode=3nn \
  --set loss.edge_tv_kappa_source=zf --set gaussian.initial_num_points=3600000 \
  --set loss.lambda_wav=0.10 --set loss.lambda_wav_plain=0.12 --set loss.wav_levels=1 --set loss.wav_name=db4 \
  --set gaussian.rotation_lr=0.008 \
  --base_cfg "$CONFIG" "${VOL[@]}" "$@" > "$LOG" 2>&1
RC=$?
STATUS=pending
if [[ $RC -ne 0 ]]; then STATUS=crash; fi
env PYTHONNOUSERSITE=1 PYTHONPATH= "$PY" scripts/autoresearch_jyz_af12_row.py --exp "$EXP" --status "$STATUS" --note "rc=$RC cfg=$CONFIG"
exit $RC
