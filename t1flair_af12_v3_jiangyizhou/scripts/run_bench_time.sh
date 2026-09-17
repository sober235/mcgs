#!/usr/bin/env bash
# run_bench_time.sh -- reconstruction-time benchmark on ONE exclusive card, sequential.
# stage 1: fast arms harness (ZF, CG-SENSE, CS-SENSE converged, Joint SENSE; loading and scoring timed apart), then
#          L1-SPIRiT with the kernel / lambda of record (its script times the solver alone), --save 0 --suffix _bench.
# stage 2: spot checks of two long arms against their recorded one-card-per-job times: MC-GS P14 (same argv as the run of
#          record, new --exp, no --save_volumes) and ZS-SSDU T1 (--suffix _bench). Nothing of record is overwritten.
# Foreign processes on the card are listed before and after every job so a shared-card timing is visible.
# usage: run_bench_time.sh <gpu>
set -u
R=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_v3_jiangyizhou
W=/data0/congcong/code/GS/mcgs_af12_wt
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
G=${1:?gpu}
export PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$R/logs/bench"
probe() { echo "  card $G apps: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$G" | tr '\n' ';')"; }
run() { local name=$1; shift; echo "== $name start $(date -Is)"; probe; "$@"; echo "== $name rc=$? $(date -Is)"; probe; }

run fast_arms "$PY" -u "$R/scripts/bench_fast_arms.py" --gpu "$G"
run spirit_T1    "$PY" -u "$R/scripts/recon_spirit.py" --contrast T1    --gpu "$G" --ksize 5 --lam 0.002 --iters 150 --chunk 24 --suffix _bench --save 0
run spirit_FLAIR "$PY" -u "$R/scripts/recon_spirit.py" --contrast FLAIR --gpu "$G" --ksize 5 --lam 0.001 --iters 150 --chunk 24 --suffix _bench --save 0
echo "BENCH_STAGE1_DONE $(date -Is)"
cd "$W" || exit 1
run mcgs_p14 "$PY" -u arc/cycle.py --exp p3_e0001_ptv001_full4k_bench --gpu "$G" --iters 4000 --lambda_geo 0.10 --warmup 500 --max_radius 20 --kspace_high 2.2 --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 --set gaussian.init_mode=head --set gaussian.init_scale_mode=3nn --set loss.edge_tv_kappa_source=zf --set gaussian.initial_num_points=3600000 --set loss.lambda_wav=0.10 --set loss.lambda_wav_plain=0.12 --set loss.wav_levels=1 --set loss.wav_name=db4 --set gaussian.rotation_lr=0.008 --base_cfg configs/t1flair_v3_af12_hyw.yaml --lambda_edge 0.001 --lambda_ptv 0.01
run zsssdu_T1 "$PY" -u "$R/scripts/recon_zsssdu.py" --contrast T1 --gpu "$G" --epochs 100 --suffix _bench
echo "BENCH_STAGE2_DONE $(date -Is)"
