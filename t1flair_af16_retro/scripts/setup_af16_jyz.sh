#!/bin/bash
set -euo pipefail
SRC=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_retro
A12=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_v3_jiangyizhou
OUT=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou
F=/data0/congcong/code/GS/mcgs_af16_wt
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
test ! -e "$OUT" || { echo "exists: $OUT"; exit 1; }
mkdir -p "$OUT"/{scripts,logs,figures,data,recon}
cp "$SRC"/scripts/*.py "$SRC"/scripts/*.m "$OUT/scripts/"
cd "$OUT/scripts"
FILES=$(ls *.py *.m)
sed -i -e 's|Results_20260908/T1FLAIR_AF16_retro|Results_20260908/T1FLAIR_AF16_v3_jiangyizhou|g' \
  -e 's|vol1_huangyuwei|vol3_jiangyizhou|g' -e 's|_train_huangyuwei_cc|_train_jiangyizhou_cc|g' \
  -e 's|t1flair_af16_|t1flair_v3_af16_|g' -e 's|figs/t1flair_af16|figs/t1flair_v3_af16|g' $FILES
echo "--- residual hyw mentions:"; grep -n "huangyuwei\|T1FLAIR_AF16_retro\|figs/t1flair_af16\b" $FILES || echo "(none)"
# this subject's own k-space and ESPIRiT maps, mask-independent, reused from its AF12 topic
for f in ksp_T1.h5 ksp_FLAIR.h5 csm_espirit_T1.h5 csm_espirit_FLAIR.h5; do ln -s "$A12/data/$f" "$OUT/data/$f"; done
# ONE mask for every subject (the AF12 topics' masks are byte-identical): copy the hyw AF16 mask
cp "$SRC/data/mask_vdpois_256x176_af16.mat" "$OUT/data/"
md5sum "$SRC/data/mask_vdpois_256x176_af16.mat" "$OUT/data/mask_vdpois_256x176_af16.mat"
ls -la "$OUT/data"
env PYTHONNOUSERSITE=1 PYTHONPATH= "$PY" "$OUT/scripts/prep_04_export.py" > "$OUT/logs/prep_04_export.log" 2>&1 || { tail -20 "$OUT/logs/prep_04_export.log"; exit 1; }
grep "reusing" "$OUT/logs/prep_04_export.log" | cut -c1-260
cmp <(env PYTHONNOUSERSITE=1 PYTHONPATH= "$PY" -c "import h5py;print(h5py.File('$SRC/data/mask_af16_3d.h5')['mask'][:].sum())") \
    <(env PYTHONNOUSERSITE=1 PYTHONPATH= "$PY" -c "import h5py;print(h5py.File('$OUT/data/mask_af16_3d.h5')['mask'][:].sum())") && echo "mask3d sums equal"
echo "--- configs"
cd "$F"
for c in "" "_singleFLAIR" "_singleT1"; do
  sed -e 's|Results_20260908/T1FLAIR_AF16_retro|Results_20260908/T1FLAIR_AF16_v3_jiangyizhou|g' \
      -e 's|vol1_huangyuwei|vol3_jiangyizhou|g' -e 's|_train_huangyuwei_cc|_train_jiangyizhou_cc|g' \
      -e 's|t1flair_af16_|t1flair_v3_af16_|g' -e 's|t1flair_af16_hyw|t1flair_v3_af16_jyz|g' \
      "configs/t1flair_af16_hyw${c}.yaml" > "configs/t1flair_v3_af16_jyz${c}.yaml"
done
env PYTHONNOUSERSITE=1 PYTHONPATH= "$PY" - <<'PY'
import yaml
F = "/data0/congcong/code/GS/mcgs_af16_wt"
for name in ["t1flair_v3_af16_jyz", "t1flair_v3_af16_jyz_singleFLAIR", "t1flair_v3_af16_jyz_singleT1"]:
    c = yaml.safe_load(open(f"{F}/configs/{name}.yaml"))
    for ch in ("plain", "contrast"):
        d = c['data'][ch]
        assert "v3_jiangyizhou" in d['data_path'] and "v3_jiangyizhou" in d['csm_path'] and "v3_jiangyizhou" in d['mask_path'], (name, ch)
    print(name, "|", c['data']['plain']['name'], c['data']['contrast']['name'], "| known_zero:", c['data']['plain'].get('known_zero'), c['data']['contrast'].get('known_zero'), "|", c['output']['exp_name'])
PY
echo "SETUP_JYZ_OK"
