#!/usr/bin/env python3
"""indep_single.py -- a genuinely independent GS-single for the T1 + FLAIR acceleration study (jiangyizhou,
AF 12 / 20 / 24; AF16 already has it as fw_t1flair_nojoint_{T1,FLAIR} of mcgs_framework_ablation_20261110).

Why: the GS-single registered in these topics (gs_single_p14 / gs_single_p3) is the JOINT trainer fed the same scan
in both channels, with the common-edge prior still on and, for FLAIR, T1's wavelet weight on the kept channel. The
user asked (2026-11-21) for a standalone single-contrast arm and for the record figure to carry it.

Row "<af>:<T1|FLAIR>": that contrast alone in SingleContrastTrainer (arc/cycle.py --trainer single): its own
positions, scales, rotations and densities, no second contrast loaded, the registered recipe of that AF with this
contrast's own wavelet weight, every cross-contrast weight 0, the same number of Gaussians (3.6M). This is the
"w/o joint modelling" definition of ce_af16_ablation_20261110/abl_framework.py, unchanged.
Row "<af>:full": control, the registered joint recipe replayed in the same trainer copy (must reproduce the
registered MC-GS score, so that MC-GS and the independent single are compared inside one code base).

Code base: branch exp/shared-geometry-2x2-ablation-20261009 at 54f8a2a (server worktree SRC), copied into
OUT/trainer and checked file by file. No source file is edited. Nothing is written outside OUT; nothing is
overwritten; a claimed row is skipped. One seed (42). Neither GT nor any ROI enters a reconstruction.

  gpu.py run --gpu N -- @python autoresearch/t1flair_af_indep_single_20261121/indep_single.py --smoke --rows 12:T1
  gpu.py run --gpu N -- @python autoresearch/t1flair_af_indep_single_20261121/indep_single.py --rows 12:T1
  gpu.py run --cpu   -- @python autoresearch/t1flair_af_indep_single_20261121/indep_single.py --table
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

GS = Path("/data0/congcong/code/GS")
SRC = GS / "mcgs_sg2x2_wt"
SRC_HEAD = "54f8a2a3bef402e785314b748f730a9623ca9d43"
CODE_DIRS = ("arc", "paired_gs", "diag_artefact")
OUT = GS / "3DGS/outputs/codex_artifacts/mcgs_t1flair_af_indep_single_20261121"
FW = GS / "3DGS/outputs/codex_artifacts/mcgs_framework_ablation_20261110/trainer/results_arc"   # AF16, existing

COMMON = ["--iters", "4000", "--lambda_geo", "0.10", "--warmup", "500", "--max_radius", "20", "--kspace_high", "2.2",
          "--edge_tv", "--edge_tv_kappa_q", "0.85", "--edge_tv_wfloor", "0.10", "--tv_eps", "0.04",
          "--set", "gaussian.init_mode=head", "--set", "gaussian.init_scale_mode=3nn",
          "--set", "loss.edge_tv_kappa_source=zf", "--set", "gaussian.initial_num_points=3600000"]
# AF12 phase 3 "P14" (p3_e0001_ptv001_full4k) and the AF16 phase-3 recipe used unchanged at AF20 / AF24
P14 = [*COMMON, "--set", "loss.lambda_wav=0.10", "--set", "loss.lambda_wav_plain=0.12", "--set", "loss.wav_levels=1",
       "--set", "loss.wav_name=db4", "--set", "gaussian.rotation_lr=0.008", "--lambda_edge", "0.001", "--lambda_ptv", "0.01"]
P3 = [*COMMON, "--set", "loss.wav_name=db4", "--set", "loss.wav_levels=1", "--set", "loss.lambda_wav_plain=0.12",
      "--set", "loss.lambda_wav=0.10", "--lambda_edge", "0.001", "--lambda_ptv", "0.03"]
AFS = {
    "12": dict(cfg=GS / "mcgs_af12_wt/configs/t1flair_v3_af12_hyw.yaml", recipe=P14,
               reg=GS / "mcgs_af12_wt/results_arc", joint="p3_e0001_ptv001_full4k",
               single="p3_e0001_ptv001_single{c}_full4k"),
    "20": dict(cfg=GS / "mcgs_af16_wt/configs/t1flair_v3_af20_jyz.yaml", recipe=P3,
               reg=GS / "mcgs_af16_wt/results_arc", joint="af20_p3_joint_full4k", single="af20_p3_single{c}_full4k"),
    "24": dict(cfg=GS / "mcgs_af16_wt/configs/t1flair_v3_af24_jyz.yaml", recipe=P3,
               reg=GS / "mcgs_af16_wt/results_arc", joint="af24_p3_joint_full4k", single="af24_p3_single{c}_full4k"),
}
CROSS_OFF = ["--lambda_edge", "0", "--lambda_dtv", "0", "--lambda_vtv", "0"]
KINDS = ("T1", "FLAIR", "full")
ALL_ROWS = [f"{a}:{k}" for a in AFS for k in KINDS]
KEYS = ("head_psnr", "head_ssim", "head_hfen", "nmse")
log = lambda *a: print(*a, flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("--rows", nargs="+", default=[r for r in ALL_ROWS if not r.endswith(":full")], choices=ALL_ROWS)
ap.add_argument("--smoke", action="store_true", help="30 iterations, no volumes, into OUT/smoke")
ap.add_argument("--table", action="store_true", help="print the result table (CPU), run nothing")
args = ap.parse_args()
root = OUT / "smoke" if args.smoke else OUT
TR, RES, CFG = root / "trainer", root / "trainer/results_arc", root / "configs"


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
        except OSError:                      # another worker won the race
            shutil.rmtree(tmp)
    tracked = [f for f in git("ls-files", "--", *CODE_DIRS).splitlines() if f]
    bad = [f for f in tracked if not (TR / f).is_file() or sha(TR / f) != sha(SRC / f)]
    assert not bad, f"trainer copy differs from the source worktree: {bad[:10]}"
    RES.mkdir(exist_ok=True)
    return len(tracked)


def registered_argv(p):
    """the recipe part of a registered run's argv (experiment name, GPU, config and --save_volumes removed)"""
    argv, out, i = json.load(open(p))["argv"], [], 0
    while i < len(argv):
        if argv[i] in ("--exp", "--gpu", "--base_cfg"):
            i += 2
            continue
        if argv[i] != "--save_volumes":
            out.append(argv[i])
        i += 1
    return out


def check_recipes():
    for af, t in AFS.items():
        got = registered_argv(t["reg"] / f"{t['joint']}_score.json")
        assert got == t["recipe"], f"AF{af}: registered joint argv differs from the recipe here:\n{got}\n{t['recipe']}"


def single_cfg(af, c):
    """a data.plain-only config for one contrast: the AF's joint config with the other contrast removed"""
    import yaml
    cfg = yaml.safe_load(open(AFS[af]["cfg"]))
    d = cfg["data"]
    slot = {d["plain"]["name"]: "plain", d["contrast"]["name"]: "contrast"}[c]
    cfg["data"] = {"plain": d[slot]}
    CFG.mkdir(parents=True, exist_ok=True)
    p = CFG / f"af{af}_single_{c}.yaml"
    text = yaml.safe_dump(cfg, sort_keys=False)
    if p.exists():
        assert p.read_text() == text, f"{p} exists with different content"
    else:
        tmp = p.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(text)
        os.replace(tmp, p)
    return p, slot


def wav(recipe, key):
    return [a.split("=", 1)[1] for a in recipe if a.startswith(f"loss.{key}=")][-1]


def row_cmd(row, gpu):
    af, kind = row.split(":")
    t = AFS[af]
    recipe = list(t["recipe"])
    if args.smoke:
        recipe[recipe.index("--iters") + 1] = "30"
        recipe[recipe.index("--warmup") + 1] = "10"
    vol = [] if args.smoke else ["--save_volumes"]
    exp = f"is_af{af}_{'joint_control' if kind == 'full' else 'indep_' + kind}"
    cyc = [sys.executable, "-u", "arc/cycle.py", "--exp", exp, "--gpu", gpu]
    if kind == "full":
        return exp, [*cyc, *recipe, "--base_cfg", str(t["cfg"]), *vol]
    cfg, slot = single_cfg(af, kind)
    w = wav(recipe, "lambda_wav_plain" if slot == "plain" else "lambda_wav")        # this contrast's own weight
    return exp, [*cyc, *recipe, "--base_cfg", str(cfg), *vol, "--trainer", "single", *CROSS_OFF,
                 "--set", f"loss.lambda_wav={w}", "--set", f"loss.lambda_wav_plain={w}"]


def load(p):
    return json.load(open(p)) if Path(p).exists() else None


def table():
    """per AF and contrast: registered MC-GS, joint control here, registered GS-single, independent single"""
    out = dict(rows=[])
    lines = ["| AF | contrast | MC-GS registered | MC-GS control (this code) | GS-single registered | GS-single independent "
             "| indep - registered single | MC-GS - indep (PSNR / SSIM / NMSE) |", "|---|---|---|---|---|---|---|---|"]
    fmt = lambda s: "(none)" if s is None else f"{s['head_psnr']:.3f} / {s['head_ssim']:.4f} / {s['nmse']:.5f}"
    for af in ("12", "16", "20", "24"):
        for c in ("T1", "FLAIR"):
            if af == "16":
                j = load(GS / "mcgs_af16_wt/results_arc/p3_r3_e0001_ptv003_full4k_score.json")
                ctl = load(FW / "fw_t1flair_full_score.json")
                rs = load(GS / f"mcgs_af16_wt/results_arc/p3_r3_single{c}_full4k_score.json")
                ind = load(FW / f"fw_t1flair_nojoint_{c}_score.json")
            else:
                t = AFS[af]
                j = load(t["reg"] / f"{t['joint']}_score.json")
                ctl = load(RES / f"is_af{af}_joint_control_score.json")
                rs = load(t["reg"] / f"{t['single'].format(c=c)}_score.json")
                ind = load(RES / f"is_af{af}_indep_{c}_score.json")
            slot = "plain" if c == "T1" else "contrast"
            assert j[f"{slot}_name"] == c and rs["plain_name"] == c
            J, C, S, I = j[slot], ctl and ctl[slot], rs["plain"], ind and ind["plain"]
            if ind:
                assert ind["plain_name"] == c and ind["contrast"] is None
            d1 = f"{I['head_psnr'] - S['head_psnr']:+.3f} dB" if I else ""
            d2 = (f"{J['head_psnr'] - I['head_psnr']:+.3f} dB / {J['head_ssim'] - I['head_ssim']:+.4f} / "
                  f"{J['nmse'] - I['nmse']:+.5f}") if I else ""
            lines.append(f"| {af} | {c} | {fmt(J)} | {fmt(C)} | {fmt(S)} | {fmt(I)} | {d1} | {d2} |")
            out["rows"].append(dict(af=af, contrast=c, mcgs_registered=J and {k: J[k] for k in KEYS},
                                    mcgs_control=C and {k: C[k] for k in KEYS},
                                    single_registered={k: S[k] for k in KEYS},
                                    single_independent=I and {k: I[k] for k in KEYS}))
    text = "\n".join(lines)
    log(text)
    return text, out


if args.table:
    art = Path(os.environ.get("MCGS_ARTIFACT_DIR", root / "artifacts"))
    art.mkdir(parents=True, exist_ok=True)
    text, out = table()
    (art / "table.md").write_text("cells: head PSNR (dB) / head SSIM / head NMSE, trainer scorer, seed 42\n\n" + text + "\n")
    (art / "table.json").write_text(json.dumps(out, indent=1))
    for p in sorted(RES.glob("is_*_score.json")):
        shutil.copy2(p, art / p.name)
    sys.exit(0)

vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
assert vis.isdigit(), f"needs exactly one GPU from gpu.py --gpu, got CUDA_VISIBLE_DEVICES={vis!r}"
check_recipes()
n = ensure_trainer()
(root / "claims").mkdir(exist_ok=True)
art = Path(os.environ.get("MCGS_ARTIFACT_DIR", root / "artifacts"))
art.mkdir(parents=True, exist_ok=True)
log(f"root {root}; trainer copy verified ({n} tracked files, source {SRC_HEAD[:7]}); physical GPU {vis}; rows {args.rows}")
record = dict(root=str(root), source=str(SRC), source_head=SRC_HEAD, gpu=vis, smoke=args.smoke,
              configs={af: dict(path=str(t["cfg"]), sha256=sha(t["cfg"])) for af, t in AFS.items()}, runs=[])
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
    log(f"\n===== {exp}\n{' '.join(cmd)}")             # no server clock or durations in logs (time rule)
    with open(outs[0], "w") as lf:
        rc = subprocess.run(cmd, cwd=TR, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    run = dict(row=row, exp=exp, cmd=cmd, rc=rc)
    text = open(outs[0]).read()
    if rc == 0 and outs[1].exists():
        s = json.load(open(outs[1]))
        run.update(npts=s["npts"], vram_gb=s["vram_gb"], plain_name=s.get("plain_name"),
                   plain={k: s["plain"][k] for k in KEYS}, contrast=s["contrast"] and {k: s["contrast"][k] for k in KEYS})
        shutil.copy2(outs[1], art / outs[1].name)
        c = f" / {s['contrast']['head_psnr']:.3f}" if s["contrast"] else ""
        log(f"===== {exp} done: head PSNR {s['plain']['head_psnr']:.3f}{c}, N={s['npts']}, {s['vram_gb']:.1f} GB")
    else:
        failed.append(exp)
        log(f"===== {exp} FAILED rc={rc}; log tail:\n{''.join(text.splitlines(True)[-20:])}")
    record["runs"].append(run)
    (art / f"record_gpu{vis}{'_smoke' if args.smoke else ''}_{args.rows[0].replace(':', '_')}.json").write_text(json.dumps(record, indent=2))
log(f"DONE; failed rows: {failed or 'none'}")
sys.exit(1 if failed else 0)
