#!/bin/bash
set -euo pipefail
SRC=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_retro
OUT=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_retro
F=/data0/congcong/code/GS/mcgs_af16_wt
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
test ! -e "$OUT" || { echo "exists: $OUT"; exit 1; }
mkdir -p "$OUT"/{scripts,logs,figures,data,recon}
cp "$SRC"/scripts/*.py "$SRC"/scripts/*.sh "$SRC"/scripts/*.m "$OUT/scripts/"
cd "$OUT/scripts"
rm -f setup_subject.sh prep_03_csm.sh prep_03_csm_flair.sh run_all_flair.sh run_learned.sh prep_05_csm_to_h5.py compare_topics.py
FILES=$(ls *.py *.m)
sed -i -e 's|Results_20260908/T1FLAIR_AF12_retro|Results_20260908/T1FLAIR_AF16_retro|g' \
  -e 's|/data0/congcong/code/GS/Paired_ARC_GS_general_framework|/data0/congcong/code/GS/mcgs_af16_wt|g' \
  -e 's|mask_af12_3d|mask_af16_3d|g' -e 's|mask_vdpois_256x176_af12|mask_vdpois_256x176_af16|g' \
  -e 's|t1flair_af12_|t1flair_af16_|g' -e 's|figs/t1flair_af12|figs/t1flair_af16|g' \
  -e 's|for af = \[12\]|for af = [16]|' \
  -e 's|AF=12|AF=16|g' -e 's|AF 12|AF 16|g' -e 's|AF12|AF16|g' $FILES
echo "--- residual af12 mentions in scripts:"; grep -n -i "af12\|AF 12\|Paired_ARC_GS_general_framework" $FILES || echo "(none)"
# prep_04_export.py: NEVER rewrite the k-space h5 (it is a symlink into the AF12 topic)
env PYTHONNOUSERSITE=1 PYTHONPATH= "$PY" - <<'PY'
from pathlib import Path
p = Path("prep_04_export.py"); s = p.read_text()
old = '''        out = f"{R}/data/ksp_{tag}.h5"
        arr = np.ascontiguousarray(np.moveaxis(np.asarray(k), 3, 0))   # (C,X,Y,Z)
        with h5py.File(out, "w") as h:
            h.create_dataset("kspace", data=arr, chunks=(1, 32, 256, 176))
        rep["contrasts"][tag]["ksp_h5"] = out
        print(f"[{tag}] wrote {out}  shape={arr.shape}  "
              f"support={sup.mean():.4f} sampled={eff.mean():.4f} "
              f"AF_rect={1/eff.mean():.2f} AF_own={sup.mean()/eff.mean():.2f}", flush=True)
        del arr, k
'''
new = '''        out = f"{R}/data/ksp_{tag}.h5"
        # AF16 topic: the coil-first k-space h5 is REUSED from the AF12 topic through a symlink
        # (same CFL source; it does not depend on the mask). It is never rewritten here -- a write
        # would go through the symlink into the AF12 topic's file.
        assert os.path.islink(out) and os.path.exists(out), f"expected a symlink to the AF12 k-space at {out}"
        rep["contrasts"][tag]["ksp_h5"] = out
        rep["contrasts"][tag]["ksp_h5_target"] = os.path.realpath(out)
        print(f"[{tag}] reusing {out} -> {os.path.realpath(out)}  "
              f"support={sup.mean():.4f} sampled={eff.mean():.4f} "
              f"AF_rect={1/eff.mean():.2f} AF_own={sup.mean()/eff.mean():.2f}", flush=True)
        del k
'''
assert s.count(old) == 1, "export block not found exactly once"
p.write_text(s.replace(old, new)); print("patched prep_04_export.py")
PY
for f in ksp_T1.h5 ksp_FLAIR.h5 csm_espirit_T1.h5 csm_espirit_FLAIR.h5; do ln -s "$SRC/data/$f" "$OUT/data/$f"; done
ls -la "$OUT/data"
echo "--- MATLAB mask"
cat prep_02_mask.m
matlab -nodisplay -nosplash -nodesktop -r "run('$OUT/scripts/prep_02_mask.m')" > "$OUT/logs/prep_02_mask.log" 2>&1 || true
tail -5 "$OUT/logs/prep_02_mask.log"
test -f "$OUT/data/mask_vdpois_256x176_af16.mat" || { echo "MASK NOT WRITTEN"; exit 1; }
echo "--- export"
env PYTHONNOUSERSITE=1 PYTHONPATH= "$PY" "$OUT/scripts/prep_04_export.py" > "$OUT/logs/prep_04_export.log" 2>&1 || { tail -20 "$OUT/logs/prep_04_export.log"; exit 1; }
grep -v "^{" "$OUT/logs/prep_04_export.log" | head -5
ls -la "$OUT/data/mask_af16_3d.h5"
echo "--- configs in the worktree"
cd "$F"
for c in "" "_singleFLAIR"; do
  sed -e 's|Results_20260908/T1FLAIR_AF12_retro|Results_20260908/T1FLAIR_AF16_retro|g' \
      -e 's|mask_af12_3d|mask_af16_3d|g' -e 's|mask_vdpois_256x176_af12|mask_vdpois_256x176_af16|g' \
      -e 's|acceleration_factor: 12|acceleration_factor: 16|g' \
      -e 's|t1flair_af12_|t1flair_af16_|g' \
      -e 's|AF=12|AF=16|g' -e 's|AF 12|AF 16|g' -e 's|AF12|AF16|g' \
      "configs/t1flair_af12_hyw${c}.yaml" > "configs/t1flair_af16_hyw${c}.yaml"
done
env PYTHONNOUSERSITE=1 PYTHONPATH= "$PY" - <<'PY'
import yaml, copy, json, re
F = "/data0/congcong/code/GS/mcgs_af16_wt"
OUT = "/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_retro"
rep = json.load(open(f"{OUT}/logs/prep_04_export.json"))
n, tot = rep["mask2d_samples"], 256 * 176
t1, fl = rep["contrasts"]["T1"], rep["contrasts"]["FLAIR"]
# rewrite the AF12-specific numbers in the base config header
p = f"{F}/configs/t1flair_af16_hyw.yaml"; s = open(p).read()
old_start = s.index("# Undersampling:"); old_end = s.index("#   presampled: false")
new = (f"# Undersampling: retrospective, house-convention vd-Poisson on the (ky,kz) plane, ONE mask for\n"
       f"#   both contrasts (scripts/prep_02_mask.m -> data/mask_vdpois_256x176_af16.mat, then broadcast\n"
       f"#   along the readout in scripts/prep_04_export.py -> data/mask_af16_3d.h5).\n"
       f"#   accely 4 / accelx 4 / calib 24 / ellipse 0 / pp 1 (nominal AF 16, same generator as AF12).\n"
       f"#   {n} / {tot} PE points = {100*n/tot:.3f} %, AF {rep['mask2d_AF_rect']:.3f} over the full PE rectangle.\n"
       f"#   Effective per contrast (mask AND that contrast's own acquired support):\n"
       f"#     T1  sampled {100*t1['sampled_frac_of_plane']:.3f} % of the plane -> AF {t1['AF_vs_full_rect']:.2f} vs the full grid, {t1['AF_vs_own_support']:.2f} vs its own support.\n"
       f"#     FLAIR sampled {100*fl['sampled_frac_of_plane']:.3f} % of the plane -> AF {fl['AF_vs_full_rect']:.2f} vs the full grid, {fl['AF_vs_own_support']:.2f} vs its own (elliptical)\n"
       f"#         support -- the variable-density mask is denser where the shutter keeps data.\n"
       f"#   k-space h5 and ESPIRiT maps are symlinks into T1FLAIR_AF12_retro/data (mask-independent, reused).\n")
s = s[:old_start] + new + s[old_end:]
s = s.replace("t1flair_af16_hyw.yaml — NEW DATA: 3D T1 + true FLAIR, one subject, retrospective AF=16.",
              "t1flair_af16_hyw.yaml — same data as t1flair_af12_hyw.yaml, retrospective AF=16 (2026-09-09).")
open(p, "w").write(s)
d = yaml.safe_load(open(p))
sd = copy.deepcopy(d)
sd['data']['contrast'] = copy.deepcopy(d['data']['plain']); sd['data']['contrast']['name'] = 'T1'
sd['output']['exp_name'] = "t1flair_af16_single_T1"; sd['output']['output_dir'] = "results_arc/t1flair_af16_single_T1"
yaml.safe_dump(sd, open(f"{F}/configs/t1flair_af16_hyw_singleT1.yaml", "w"), sort_keys=False)
for name in ["t1flair_af16_hyw", "t1flair_af16_hyw_singleFLAIR", "t1flair_af16_hyw_singleT1"]:
    c = yaml.safe_load(open(f"{F}/configs/{name}.yaml"))
    print(name, "|", c['data']['plain']['name'], c['data']['contrast']['name'], "|", c['data']['plain']['mask_path'].split('/')[-1],
          "| known_zero plain/contrast:", c['data']['plain'].get('known_zero'), c['data']['contrast'].get('known_zero'), "|", c['output']['exp_name'])
print("mask:", n, "samples,", f"{100*n/tot:.3f} %", "AF_rect", f"{rep['mask2d_AF_rect']:.3f}")
print("T1:", json.dumps({k: round(v, 4) for k, v in t1.items() if isinstance(v, float)}))
print("FLAIR:", json.dumps({k: round(v, 4) for k, v in fl.items() if isinstance(v, float)}))
PY
echo "SETUP_OK"
