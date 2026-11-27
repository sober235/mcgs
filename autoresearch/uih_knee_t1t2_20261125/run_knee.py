#!/usr/bin/env python3
"""run_knee.py -- MC-GS on the UIH 3D knee T1 + T2 topic built by prep_knee.py (AF12).

Rows (one training each):
  full        the joint MC-GS model (shared geometry, per-contrast complex densities)
  single_T1   GS-single on T1 alone (SingleContrastTrainer, its own geometry)
  single_T2   GS-single on T2 alone

Recipe = the final brain T1+T2 recipe (tangbing stage three, 2026-11-14), unchanged and NOT tuned on the knee:
V32 + T2 wavelet 0.08 + rotation_lr 0.008 + scale_lr 0.004, and for the joint model lambda_edge 0.001 with
edge_kappa 100. The single-contrast rows take the same settings with that contrast's own wavelet weight and every
cross-contrast weight at 0, as in the framework ablation (abl_framework.py: nojoint rows).

Code base: the trainer of mcgs_sg2x2_wt at 54f8a2a, copied into ROOT/trainer and checked file by file against the
source worktree (the same procedure as abl_framework.py). Nothing is written outside ROOT; an existing output is
never overwritten; a claimed row is skipped. No clock value or duration is written (time rule).

  gpu.py run --gpu N -- @python autoresearch/uih_knee_t1t2_20261125/run_knee.py --subject 6 --rows full
  gpu.py run --gpu N -- @python autoresearch/uih_knee_t1t2_20261125/run_knee.py --subject 6 --rows full --smoke
  gpu.py run --cpu   -- @python autoresearch/uih_knee_t1t2_20261125/run_knee.py --subject 6 --table
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path("/data0/congcong/code/GS/mcgs_sg2x2_wt")
SRC_HEAD = "54f8a2a3bef402e785314b748f730a9623ca9d43"
CODE_DIRS = ("arc", "paired_gs", "diag_artefact")
OUT = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_uih_knee_t1t2_20261125")
RESD = Path("/data0/congcong/code/GS/Results_20260908")
AF = 12
V32 = ["--iters", "4000", "--lambda_geo", "0.07", "--warmup", "500", "--max_radius", "20", "--kspace_high", "2.2",
       "--edge_tv", "--edge_tv_kappa_q", "0.85", "--edge_tv_wfloor", "0.10", "--tv_eps", "0.04",
       "--set", "gaussian.init_mode=head", "--set", "gaussian.init_scale_mode=3nn",
       "--set", "loss.edge_tv_kappa_source=zf", "--set", "gaussian.initial_num_points=2400000",
       "--set", "loss.lambda_wav=0.05", "--set", "loss.lambda_wav_plain=0.06"]
FINAL = ["--set", "loss.lambda_wav=0.08", "--set", "gaussian.rotation_lr=0.008", "--set", "gaussian.scale_lr=0.004"]
JOINT = ["--lambda_edge", "0.001", "--edge_kappa", "100"]
CROSS_OFF = ["--lambda_edge", "0", "--lambda_dtv", "0", "--lambda_vtv", "0"]
WAV = {"T1": "0.06", "T2": "0.08"}
ROWS = ("full", "single_T1", "single_T2")
KEYS = ("head_psnr", "head_ssim", "head_hfen", "nmse")
log = lambda *a: print(*a, flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("--subject", type=int, required=True)
ap.add_argument("--rows", nargs="+", default=list(ROWS), choices=ROWS)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--smoke", action="store_true", help="30 iterations, no volumes, into OUT/smoke")
ap.add_argument("--table", action="store_true")
ap.add_argument("--tag", default="", help="tuning candidate name, becomes part of the experiment name")
ap.add_argument("--no_volumes", action="store_true")
ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="trainer arguments appended after the recipe (must be last)")
args = ap.parse_args()
SID = "s%02d" % args.subject
TOPIC = RESD / f"KNEE_UIH_T1T2_AF{AF}_S{args.subject:02d}"
CFG0 = TOPIC / f"configs/knee_uih_t1t2_af{AF}.yaml"
root = OUT / "smoke" if args.smoke else OUT
TR, RES, CFG = root / "trainer", root / "trainer/results_arc", root / "configs"
TAG = f"knee_{SID}" + ("" if args.seed == 42 else f"_seed{args.seed}") + (f"_{args.tag}" if args.tag else "")
assert not args.extra or args.tag, "--extra needs --tag (the untagged names are the unchanged recipe)"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*a):
    return subprocess.run(["git", "-C", str(SRC), *a], check=True, capture_output=True, text=True).stdout


def ensure_trainer():
    assert git("rev-parse", "HEAD").strip() == SRC_HEAD, "source worktree moved"
    assert git("status", "--porcelain", "--", *CODE_DIRS).strip() == "", "source worktree has local changes"
    if not TR.exists():
        root.mkdir(parents=True, exist_ok=True)
        tmp = root / f"trainer.tmp{os.getpid()}"
        for d in CODE_DIRS:
            shutil.copytree(SRC / d, tmp / d, ignore=shutil.ignore_patterns("__pycache__", "results_arc"))
        try:
            tmp.rename(TR)
        except OSError:
            shutil.rmtree(tmp)
    tracked = [f for f in git("ls-files", "--", *CODE_DIRS).splitlines() if f]
    bad = [f for f in tracked if not (TR / f).is_file() or sha(TR / f) != sha(SRC / f)]
    assert not bad, f"trainer copy differs from the source worktree: {bad[:10]}"
    RES.mkdir(exist_ok=True)
    return len(tracked)


def single_cfg(tag):
    import yaml
    cfg = yaml.safe_load(open(CFG0))
    slot = "plain" if cfg["data"]["plain"]["name"] == tag else "contrast"
    block = cfg["data"][slot]
    assert block["name"] == tag
    cfg["data"] = {"plain": block}
    CFG.mkdir(parents=True, exist_ok=True)
    p = CFG / f"{TAG}_single_{tag}.yaml"
    text = yaml.safe_dump(cfg, sort_keys=False)
    if p.exists():
        assert p.read_text() == text, f"{p} exists with different content"
    else:
        tmp = p.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(text)
        os.replace(tmp, p)
    return p


def row_cmd(row, gpu):
    recipe = [*V32, *FINAL]
    if args.smoke:
        recipe[recipe.index("--iters") + 1] = "30"
        recipe[recipe.index("--warmup") + 1] = "10"
    vol = [] if args.smoke or args.no_volumes else ["--save_volumes"]
    seed = ["--set", f"training.seed={args.seed}"]
    exp = f"{TAG}_{row}"
    cyc = [sys.executable, "-u", "arc/cycle.py", "--exp", exp, "--gpu", gpu]
    if row == "full":
        return exp, [*cyc, *recipe, *JOINT, "--base_cfg", str(CFG0), *vol, *seed, *args.extra]
    tag = row.split("_")[1]
    return exp, [*cyc, *recipe, "--base_cfg", str(single_cfg(tag)), *vol, "--trainer", "single", *CROSS_OFF,
                 "--set", f"loss.lambda_wav={WAV[tag]}", "--set", f"loss.lambda_wav_plain={WAV[tag]}", *seed, *args.extra]


def load(p):
    return json.load(open(p)) if Path(p).exists() else None


def table():
    f, a, b = (load(RES / f"{TAG}_{r}_score.json") for r in ROWS)
    lines = [f"### UIH knee {SID.upper()}, T1 + T2, AF{AF}, seed {args.seed}\n",
             "| model | T1 PSNR | SSIM | HFEN | NMSE | T2 PSNR | SSIM | HFEN | NMSE |", "|---|---|---|---|---|---|---|---|---|"]

    def cells(s):
        return [f"{s['head_psnr']:.3f}", f"{s['head_ssim']:.4f}", f"{s['head_hfen']:.4f}", f"{s['nmse']:.5f}"] if s else ["(not finished)", "", "", ""]
    lines.append("| MC-GS (joint) | " + " | ".join(cells(f and f["plain"]) + cells(f and f["contrast"])) + " |")
    lines.append("| GS-single | " + " | ".join(cells(a and a["plain"]) + cells(b and b["plain"])) + " |")
    if f and a and b:
        d1, d2 = f["plain"]["head_psnr"] - a["plain"]["head_psnr"], f["contrast"]["head_psnr"] - b["plain"]["head_psnr"]
        lines.append(f"\nMC-GS minus GS-single, head PSNR: T1 {d1:+.3f} dB, T2 {d2:+.3f} dB")
    for name, s in (("full", f), ("single_T1", a), ("single_T2", b)):
        if s:
            zf = {k: v for k, v in s.items() if "zf" in k.lower()}
            lines.append(f"\n{name}: N={s.get('npts')}, {s.get('vram_gb', 0):.1f} GB" + (f", zf keys {sorted(zf)}" if zf else ""))
    text = "\n".join(lines)
    log(text)
    return text


art = Path(os.environ.get("MCGS_ARTIFACT_DIR", root / "artifacts"))
art.mkdir(parents=True, exist_ok=True)
if args.table:
    (art / f"table_{TAG}.md").write_text(table() + "\n")
    for p in sorted(RES.glob(f"{TAG}_*_score.json")):
        shutil.copy2(p, art / p.name)
    sys.exit(0)

vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
assert vis.isdigit(), f"needs exactly one GPU from gpu.py --gpu, got CUDA_VISIBLE_DEVICES={vis!r}"
assert CFG0.is_file(), f"{CFG0} missing: run prep_knee.py first"
n = ensure_trainer()
(root / "claims").mkdir(exist_ok=True)
log(f"root {root}; trainer copy verified ({n} tracked files, source {SRC_HEAD[:7]}); physical GPU {vis}; rows {args.rows}")
failed = []
for row in args.rows:
    exp, cmd = row_cmd(row, vis)
    try:
        (root / "claims" / exp).mkdir()
    except FileExistsError:
        log(f"-- {exp}: already claimed, skipped")
        continue
    outs = [RES / f"{exp}.log", RES / f"{exp}_score.json", RES / exp, RES / f"{exp}_volumes.npz"]
    assert not any(p.exists() for p in outs), f"refusing to overwrite outputs of {exp}"
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH="", CUDA_DEVICE_ORDER="PCI_BUS_ID")
    log(f"\n===== {exp}\n{' '.join(cmd)}")
    with open(outs[0], "w") as lf:
        rc = subprocess.run(cmd, cwd=TR, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    text = open(outs[0]).read()
    phys = [l for l in text.splitlines() if l.startswith("[physics")]
    log("\n".join(phys))
    if rc == 0 and outs[1].exists():
        s = json.load(open(outs[1]))
        shutil.copy2(outs[1], art / outs[1].name)
        c = f" / {s['contrast']['head_psnr']:.3f}" if s.get("contrast") else ""
        log(f"===== {exp} done: head PSNR {s['plain']['head_psnr']:.3f}{c}, N={s['npts']}, {s['vram_gb']:.1f} GB")
    else:
        failed.append(exp)
        log(f"===== {exp} FAILED rc={rc}; log tail:\n{''.join(text.splitlines(True)[-25:])}")
log("\n")
table()
log(f"DONE; failed rows: {failed or 'none'}")
sys.exit(1 if failed else 0)
