"""scaffold_level.py NEW_LABEL MULT PREFIX REF_LABEL -- clone the 1 sigma0 noise topic into a new noise level (2026-09-17).

Creates Results_20260908/T1FLAIR_AF16_<NEW_LABEL>_v3_jiangyizhou/ (data symlinks, prep_04_export.json, scripts with the
level strings replaced, MULT set, the nested check pointed at REF_LABEL's measurements). Refuses an existing topic.
The runner run_noise_mcgs.sh must accept PREFIX and configs/t1flair_v3_<PREFIX>_jyz*.yaml must exist (worktree).
   scaffold_level.py noise0p5x 0.5 af16n05 noise1x      (done 2026-09-17)
   scaffold_level.py noise0p75x 0.75 af16n075 noise1x
   scaffold_level.py noise0p25x 0.25 af16n025 noise0p5x
Then: README, python scripts/make_noisy_kspace.py, launch_noise.sh --dry_run, launch.
"""
import os, re, shutil, sys
from pathlib import Path
new, mult, prefix, ref = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
RES = Path("/data0/congcong/code/GS/Results_20260908")
SRC = RES / "T1FLAIR_AF16_noise1x_v3_jiangyizhou"
DST = RES / f"T1FLAIR_AF16_{new}_v3_jiangyizhou"
REF = RES / f"T1FLAIR_AF16_{ref}_v3_jiangyizhou"
assert not DST.exists(), DST
ref_mult = {"noise2x": 2.0, "noise1x": 1.0, "noise0p5x": 0.5, "noise0p25x": 0.25, "noise0p75x": 0.75}[ref]
ratio = ref_mult / mult
lev = f"{mult:g} sigma0"
levu = f"{mult:g}σ0"
for d in ("data", "logs", "recon", "figures", "scripts"):
    (DST / d).mkdir(parents=True)
for p in sorted((SRC / "data").iterdir()):
    if p.is_symlink():
        os.symlink(os.readlink(p), DST / "data" / p.name)
(DST / "logs/prep_04_export.json").write_text(
    (SRC / "logs/prep_04_export.json").read_text().replace(str(SRC), str(DST)))


# level-independent helpers: copied unchanged (they name every level themselves)
VERBATIM = ("compare_noise_levels.py", "check_scale_levels.py", "best_planes_noise.py", "six_cells_now.py",
            "scaffold_level.py")


def conv(t):
    t = t.replace("T1FLAIR_AF16_noise1x_v3_jiangyizhou", f"T1FLAIR_AF16_{new}_v3_jiangyizhou")
    t = t.replace("t1flair_v3_af16_noise1x", f"t1flair_v3_af16_{new}")
    t = t.replace("_noise1x", f"_{new}")
    t = t.replace("mcinr_noise1x", f"mcinr_{new}")
    t = t.replace("af16n1_", f"{prefix}_").replace('"af16n1"', f'"{prefix}"')
    t = t.replace("1 sigma0", lev).replace("1σ0", levu)
    t = t.replace("per-sample std = 1 x the data's own noise", f"per-sample std = {mult:g} x the data's own noise")
    return t


for p in sorted((SRC / "scripts").iterdir()):
    if p.suffix not in (".py", ".sh") or p.name in VERBATIM:
        continue
    t = conv(p.read_text())
    if p.name == "make_noisy_kspace.py":
        assert "SEED, MULT = 20260916, 1.0" in t
        t = t.replace("SEED, MULT = 20260916, 1.0", f"SEED, MULT = 20260916, {mult:g}")
        t = t.replace('TWO_X = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_noise2x_v3_jiangyizhou")   # nested check',
                      f'TWO_X = Path("{REF}")   # nested check: the {ref_mult:g} sigma0 level (same draws)\nRATIO = {ratio!r}   # its multiplier / this one')
        t = t.replace("    sigma_raw_2x = 2.0 * float(sigma_raw)", "    sigma_raw_2x = RATIO * float(sigma_raw)")
        t = t.replace("        worst = max(worst, float((np.abs(2.0 * d1 - d2) / bound).max()))",
                      "        worst = max(worst, float((np.abs(RATIO * d1 - d2) / bound).max()))")
        t = t.replace('checks["nested_vs_2x"]', f'checks["nested_vs_{ref.replace("noise", "")}"]')
        t = t.replace("    # same seed and draws as the 2 sigma0 level: n_2x = 2 n_1x exactly in double precision (scaling by 2 is exact)",
                      f"    # same seed and draws as the {ref_mult:g} sigma0 level: n_ref = RATIO n_this before single-precision rounding")
        t = t.replace("       nested_vs_2x       (" + lev + " level, spec addendum 2026-09-16) same seed and draws as the 2 sigma0 level, so\n"
                      "                          2 (meas_1x - clean) = meas_2x - clean exactly before single-precision rounding; checked\n"
                      "                          sample by sample against a rounding bound 4e-7 (|clean| + 10 sigma_raw_2x), max ratio <= 1",
                      f"       nested_vs_{ref.replace('noise', '')}      ({lev} level, spec addendum 2026-09-17) same seed and draws as the {ref_mult:g} sigma0 level, so\n"
                      f"                          RATIO (meas_this - clean) = meas_ref - clean before single-precision rounding (RATIO = {ratio:g});\n"
                      "                          checked sample by sample against a rounding bound 4e-7 (|clean| + 10 sigma_raw_ref), max ratio <= 1")
        for must in (f"MULT = 20260916, {mult:g}", "RATIO = ", "RATIO * d1 - d2", str(REF)):
            assert must in t, must
    if p.name == "finalize_noise.sh":   # new levels also write the all-level tables (added 2026-09-17)
        t = t.replace('  "$PY" scripts/compare_noise.py --metric $m --out logs/compare_noise_$m.md > /dev/null || exit 1\n',
                      '  "$PY" scripts/compare_noise.py --metric $m --out logs/compare_noise_$m.md > /dev/null || exit 1\n'
                      '  "$PY" scripts/compare_noise_levels.py --metric $m --out logs/compare_noise_levels_$m.md > /dev/null || exit 1\n')
        assert "compare_noise_levels.py" in t
    dst = DST / "scripts" / p.name
    dst.write_text(t)
    dst.chmod(p.stat().st_mode)
for name in VERBATIM:
    shutil.copy2(SRC / "scripts" / name, DST / "scripts" / name)
left = [(p.name, m) for p in (DST / "scripts").iterdir() if p.is_file()
        for m in re.findall(r"noise1x|af16n1_|\"af16n1\"|1 sigma0|1σ0|noise2x|2 sigma0|2σ0", p.read_text())
        if p.name not in VERBATIM]
print("created", DST)
print("leftover level strings (outside the level-comparison scripts):", sorted(set(left)))
