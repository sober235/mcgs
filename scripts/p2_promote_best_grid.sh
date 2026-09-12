#!/bin/bash
# p2_promote_best_grid.sh CARD: pick the best optimiser-grid row at 1500 iterations (largest min_margin over the bar
# among the rotation-lr x scale-lr screens, status keep) and run that recipe at 4000 iterations with volumes, through
# the guarded P10 runner with the values explicit. Pre-registered rule (autoresearch_jyz_af12_p2.md, round 2 part 4/5).
cd /data0/congcong/code/GS/mcgs_af12_wt
G=${1:?card}
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
read -r EXP ARGS < <(PYTHONNOUSERSITE=1 PYTHONPATH= "$PY" - <<'PYEOF'
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts")
from autoresearch_jyz_af12_row import best_competitors
B = best_competitors()
grid = {  # exp -> (rotation lr, scale lr)
    "p2_rotlr": (0.001, 0.002), "p2_rotlr002": (0.002, 0.002), "p2_rotlr004": (0.004, 0.002), "p2_rotlr008": (0.008, 0.002),
    "p2_rotlr016": (0.016, 0.002), "p2_rotlr_slr005": (0.001, 0.005), "p2_rotlr002_slr005": (0.002, 0.005),
    "p2_rotlr004_slr005": (0.004, 0.005), "p2_rotlr008_slr005": (0.008, 0.005), "p2_rotlr016_slr005": (0.016, 0.005),
}
rows = {l.split("\t")[0]: l.split("\t")[1] for l in Path("autoresearch_jyz_af12.tsv").read_text().splitlines()[1:]}
best = None
for exp, (rl, sl) in grid.items():
    sj = Path("results_arc") / f"{exp}_score.json"
    if not sj.exists() or rows.get(exp) != "keep":
        continue
    d = json.loads(sj.read_text())
    if int(d["iters"]) != 1500:
        continue
    m = min(d["plain"]["head_psnr"] - B["T1"]["head_psnr"], d["contrast"]["head_psnr"] - B["FLAIR"]["head_psnr"])
    if best is None or m > best[0]:
        best = (m, exp, rl, sl)
if best is None:
    sys.exit("no grid row")
m, exp, rl, sl = best
name = f"p2_best_rot{str(rl).replace('0.', '')}_slr{str(sl).replace('0.', '')}_full4k"
print(name, f"--set gaussian.rotation_lr={rl} --set gaussian.scale_lr={sl}")
print(f"[promote_best_grid] best 1500 row = {exp} (min_margin {m:+.3f}) -> {name}", file=sys.stderr)
PYEOF
)
echo "[promote_best_grid] launching $EXP with: $ARGS"
# shellcheck disable=SC2086
SAVE_VOLUMES=1 ./autoresearch_jyz_af12_p2g.sh "$EXP" "$G" 4000 $ARGS
