#!/bin/bash
# setup_subject.sh <vol_dir> <subject> <short_tag>
#   e.g.  setup_subject.sh vol3_jiangyizhou jiangyizhou v3
# Clones this topic's scripts and configs onto another subject of the MC_Jia CFL set.
# Everything else -- geometry, mask, recipe, gauge -- is identical by construction.
set -euo pipefail
VOL=$1; SUBJ=$2; TAG=$3
SRC=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_retro
OUT=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_${TAG}_${SUBJ}
F=/data0/congcong/code/GS/Paired_ARC_GS_general_framework

mkdir -p "$OUT"/{scripts,logs,figures,data,recon}
cp "$SRC"/scripts/*.py "$SRC"/scripts/*.sh "$SRC"/scripts/*.m "$OUT/scripts/"
cd "$OUT/scripts"
sed -i "s|Results_20260908/T1FLAIR_AF12_retro|Results_20260908/T1FLAIR_AF12_${TAG}_${SUBJ}|g" *.py *.sh *.m
sed -i "s|vol1_huangyuwei|${VOL}|g"            *.py *.sh *.m
sed -i "s|_train_huangyuwei_cc|_train_${SUBJ}_cc|g" *.py *.sh *.m
sed -i "s|t1flair_af12_|t1flair_${TAG}_af12_|g" *.py *.sh
# figures must land in their own directory, or a clone overwrites the parent topic's panels
sed -i "s|figs/t1flair_af12|figs/t1flair_${TAG}_af12|g" *.py
rm -f setup_subject.sh prep_03_csm_flair.sh run_all_flair.sh

# configs: both contrasts are new here, so the T1 arm is NOT reused
for c in "" "_singleFLAIR"; do
  sed -e "s|Results_20260908/T1FLAIR_AF12_retro|Results_20260908/T1FLAIR_AF12_${TAG}_${SUBJ}|g" \
      -e "s|_train_huangyuwei_cc|_train_${SUBJ}_cc|g" \
      -e "s|vol1_huangyuwei|${VOL}|g" \
      -e "s|t1flair_af12_|t1flair_${TAG}_af12_|g" \
      "$F/configs/t1flair_af12_hyw${c}.yaml" > "$F/configs/t1flair_${TAG}_af12_hyw${c}.yaml"
done
# single-T1 config (this subject's T1 has to be reconstructed too)
python3 - "$F" "$TAG" <<'PY'
import sys, yaml, copy, pathlib
F, TAG = sys.argv[1], sys.argv[2]
d = yaml.safe_load(open(f"{F}/configs/t1flair_{TAG}_af12_hyw.yaml"))
sd = copy.deepcopy(d)
sd['data']['plain'] = copy.deepcopy(d['data']['plain'])
sd['data']['contrast'] = copy.deepcopy(d['data']['plain']); sd['data']['contrast']['name'] = 'T1'
sd['output']['exp_name'] = f"t1flair_{TAG}_af12_single_T1"
sd['output']['output_dir'] = f"results_arc/t1flair_{TAG}_af12_single_T1"
yaml.safe_dump(sd, open(f"{F}/configs/t1flair_{TAG}_af12_hyw_singleT1.yaml", "w"), sort_keys=False)
print("wrote singleT1 config")
PY
echo "topic ready: $OUT"
