#!/usr/bin/env bash
# Rescore the hp_n36 checkpoint (jyz AF16 gs_joint_p3 anchor) with the unchanged arc/cycle.py and compare
# with the registered head PSNR 35.706153 / 34.928080 (tolerance 1e-4 dB). About 4 minutes on one GPU.
# Usage: scripts/regress_brain_pin.sh GPU
set -euo pipefail
GPU=${1:?usage: scripts/regress_brain_pin.sh GPU}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
CKPT=/data0/congcong/code/GS/mcgs_hparam_wt/results_arc/hp_n36/hp_n36/checkpoints/latest.pth
EXP=pin_hp_n36_$(date +%Y%m%d_%H%M%S)
cd "$ROOT"
mkdir -p results_arc
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID "$PY" arc/cycle.py --exp "$EXP" --gpu "$GPU" \
  --iters 4000 --lambda_geo 0.10 --warmup 500 --max_radius 20 --kspace_high 2.2 \
  --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.10 --tv_eps 0.04 \
  --set gaussian.init_mode=head --set gaussian.init_scale_mode=3nn --set loss.edge_tv_kappa_source=zf \
  --set gaussian.initial_num_points=3600000 --set loss.wav_name=db4 --set loss.wav_levels=1 \
  --set loss.lambda_wav_plain=0.12 --set loss.lambda_wav=0.10 --lambda_edge 0.001 --lambda_ptv 0.03 \
  --base_cfg configs/t1flair_v3_af16_jyz.yaml --from_ckpt "$CKPT" > "results_arc/$EXP.log" 2>&1
env PYTHONNOUSERSITE=1 PYTHONPATH= "$PY" - "$ROOT/results_arc/${EXP}_score.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
ref = {"plain": 35.706153059238126, "contrast": 34.92807957136013}
ok = True
for ch, r in ref.items():
    v = d[ch]["head_psnr"]
    print(f"{ch}: head_psnr {v:.6f} vs {r:.6f} (diff {v - r:+.2e})")
    ok = ok and abs(v - r) <= 1e-4
print("PIN PASS" if ok else "PIN FAIL")
sys.exit(0 if ok else 1)
EOF
echo "score json: $ROOT/results_arc/${EXP}_score.json"
