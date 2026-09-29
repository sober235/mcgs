#!/usr/bin/env bash
# Brain regression gate, run after any shared change; about 10 minutes on one GPU. Runs from any worktree and
# writes only into that worktree (results_arc/, results_spec/).
#   1. CKPT: the hp_n36 checkpoint (jyz AF16 gs_joint_p3 anchor) still has the md5 below.
#   2. PIN: the unchanged arc/cycle.py rescores it to head PSNR 35.706153 / 34.928080 (tolerance 1e-4 dB).
#   3. EQUIV: arc/run_spec.py, on a copy of specs/brain_hp_n36_equiv.yaml whose results_root is this worktree's
#      results_spec/, gives the same 16 metrics as step 2 on both channels (the KEYS of
#      tests/test_scoring_equivalence.py, rel 1e-9 / abs 1e-12) and reads no code default (fallbacks == []).
# Prints PIN PASS / PIN FAIL, then EQUIV PASS / EQUIV FAIL; exits non-zero on any failure. Afterwards
# `pytest tests/test_scoring_equivalence.py` checks the same pair and that the run_spec json was written by the
# committed code of HEAD.
# Usage: scripts/regress_brain_pin.sh GPU
set -euo pipefail
GPU=${1:?usage: scripts/regress_brain_pin.sh GPU}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
CKPT=/data0/congcong/code/GS/mcgs_hparam_wt/results_arc/hp_n36/hp_n36/checkpoints/latest.pth
CKPT_MD5=34bc8b2e41f53f42a3e79c9814ce01f6
STAMP=$(date +%Y%m%d_%H%M%S)
EXP=pin_hp_n36_$STAMP
cd "$ROOT"
md5=$(md5sum "$CKPT" | awk '{print $1}')
if [ "$md5" != "$CKPT_MD5" ]; then
  echo "CKPT FAIL: $CKPT has md5 $md5, expected $CKPT_MD5"
  exit 1
fi
echo "checkpoint md5 $md5 OK"
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

SPEC_EXP=brainpin_equiv_$STAMP
SPEC_COPY=$ROOT/results_spec/brain_equiv_$STAMP.yaml
mkdir -p results_spec
env PYTHONNOUSERSITE=1 PYTHONPATH= "$PY" - "$ROOT/specs/brain_hp_n36_equiv.yaml" "$SPEC_COPY" "$ROOT/results_spec" <<'EOF'
import sys, yaml
src, dst, results_root = sys.argv[1:]
spec = yaml.safe_load(open(src))
spec["spec"]["results_root"] = results_root          # the only change: results stay in this worktree
with open(dst, "x") as f:
    yaml.safe_dump(spec, f, sort_keys=True)
EOF
echo "spec copy: $SPEC_COPY"
env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID "$PY" arc/run_spec.py --spec "$SPEC_COPY" \
  --exp "$SPEC_EXP" --gpu "$GPU" > "results_spec/$SPEC_EXP.log" 2>&1 \
  || { echo "EQUIV FAIL: arc/run_spec.py exited non-zero, see $ROOT/results_spec/$SPEC_EXP.log"; exit 1; }
echo "run_spec json: $ROOT/results_spec/${SPEC_EXP}_score.json"
env PYTHONNOUSERSITE=1 PYTHONPATH= "$PY" - "$ROOT" "$ROOT/results_arc/${EXP}_score.json" \
  "$ROOT/results_spec/${SPEC_EXP}_score.json" <<'EOF'
import ast, json, math, sys
root, cyc_path, spc_path = sys.argv[1:]
src = open(f"{root}/tests/test_scoring_equivalence.py").read()
keys = next(ast.literal_eval(n.value) for n in ast.parse(src).body
            if isinstance(n, ast.Assign) and [getattr(t, "id", None) for t in n.targets] == ["KEYS"])
assert len(keys) == 16, keys
cyc, spc = json.load(open(cyc_path)), json.load(open(spc_path))
bad = [f"{ch}.{k}: cycle {cyc[ch][k]!r} vs run_spec {spc[ch][k]!r}" for ch in ("plain", "contrast") for k in keys
       if not math.isclose(cyc[ch][k], spc[ch][k], rel_tol=1e-9, abs_tol=1e-12)]
if spc["fallbacks"] != []:
    bad.append(f"run_spec read code defaults: {spc['fallbacks']}")
print(f"compared {len(keys)} keys x 2 channels; fallbacks {spc['fallbacks']}")
print("EQUIV PASS" if not bad else "EQUIV FAIL")
for b in bad:
    print("  " + b)
sys.exit(1 if bad else 0)
EOF
