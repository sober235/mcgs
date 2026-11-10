#!/usr/bin/env python3
"""abl_sept.py -- leave-one-out ablation of the MC-GS recipe on the September CE retro data
(Results_20260908/CE_retro_cssense_af16, BART Poisson-disc AF16, the data where MC-GS wins all six panels).

Reference = the registered run ce_retro_af16_joint (trainer e778bf2, 4000 iterations, 3.6 M points, seed 42).
Every row replays that run's recorded argv (read from its score json, not retyped here) and appends the row's
overrides; argparse keeps the last value of a repeated option and cycle.py applies --set in order, so the
appended values win. Nothing else changes: same trainer snapshot, same config file, same data, same seed.

Rows (what each removes):
  full_rerun        nothing -- rerun noise of the reference
  no_edge_coupling  lambda_edge 0: the only explicit cross-contrast loss (common-edge prior). Shared geometry stays.
  no_freq_weight    loss.kspace_freq_adaptive false: uniform k-space weights (recipe: centre 0.3 / mid 1 / high 2.2)
  no_wavelet        lambda_wav 0: the db4 one-level wavelet L1 prior
  no_phase_tv       lambda_ptv 0: phase-only TV
  iso_tv            loss.edge_tv false + lambda_ptv 0: plain TV instead of edge-weighted TV. The trainer refuses
                    phase TV without edge TV, so compare this row with no_phase_tv, not with the reference.
  no_tv             lambda_geo 0: no magnitude TV at all (lambda_geo is the TV weight in this trainer)
  points_600k       600 k initial points instead of 3.6 M
  init_uniform      gaussian.init_mode uniform: points over the whole box instead of the ZF head support
  init_scale_fixed  gaussian.init_scale_mode fixed: one fixed width instead of the 3-nearest-neighbour width
Not ablated: shared_warmup_iters and loss.lambda_diff are inert in e778bf2 (the code that read them was deleted).
The joint-vs-single row already exists (ce_retro_af16_single_{pre,post}) and is only read by `--table`.

Isolation: the September topic is never written. The trainer snapshot is copied (without results_arc) to
ROOT/trainer and verified file by file against the source; cycle.py writes to <its own root>/results_arc.
Existing outputs are never overwritten; a row with a claim directory is skipped, so two workers can share a list.

GPU: gpu.py sets CUDA_VISIBLE_DEVICES to the one authorised physical id; cycle.py sets CUDA_DEVICE_ORDER=PCI_BUS_ID
and CUDA_VISIBLE_DEVICES=--gpu itself, so the wrapper passes --gpu = that same id.

  gpu.py run --gpu 2 -- @python autoresearch/ce_af16_ablation_20261110/abl_sept.py --smoke
  gpu.py run --gpu 2 -- @python autoresearch/ce_af16_ablation_20261110/abl_sept.py
  gpu.py run --cpu   -- @python autoresearch/ce_af16_ablation_20261110/abl_sept.py --table
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SEPT = Path("/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16")
SRC_TR = SEPT / "trainer/_trainer_e778bf2"
REF = "ce_retro_af16_joint"
OUT = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_ce_sept_ablation_20261110")
ROWS = {
    "full_rerun": [],
    "no_edge_coupling": ["--lambda_edge", "0"],
    "no_freq_weight": ["--set", "loss.kspace_freq_adaptive=false"],
    "no_wavelet": ["--set", "loss.lambda_wav=0", "--set", "loss.lambda_wav_plain=0"],
    "no_phase_tv": ["--lambda_ptv", "0"],
    "iso_tv": ["--set", "loss.edge_tv=false", "--lambda_ptv", "0"],
    "no_tv": ["--lambda_geo", "0"],
    "points_600k": ["--set", "gaussian.initial_num_points=600000"],
    "init_uniform": ["--set", "gaussian.init_mode=uniform"],
    "init_scale_fixed": ["--set", "gaussian.init_scale_mode=fixed"],
}
KEYS = ("head_psnr", "head_ssim", "head_hfen", "nmse", "sharp")

ap = argparse.ArgumentParser()
ap.add_argument("--rows", nargs="+", default=list(ROWS), choices=list(ROWS))
ap.add_argument("--smoke", action="store_true", help="30 iterations, no volumes, into OUT/smoke")
ap.add_argument("--table", action="store_true", help="print the result table (CPU), run nothing")
args = ap.parse_args()
log = lambda *a: print(*a, flush=True)
root = OUT / "smoke" if args.smoke else OUT
TR = root / "trainer"
RES = TR / "results_arc"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def files(base):
    return sorted(p.relative_to(base) for p in base.rglob("*")
                  if p.is_file() and "results_arc" not in p.parts and "__pycache__" not in p.parts)


def ensure_trainer():
    """Copy the pinned trainer once (atomic rename), then verify every file against the source."""
    if not TR.exists():
        root.mkdir(parents=True, exist_ok=True)
        tmp = root / f"trainer.tmp{os.getpid()}"
        shutil.copytree(SRC_TR, tmp, ignore=shutil.ignore_patterns("results_arc", "__pycache__"))
        try:
            tmp.rename(TR)
        except OSError:                      # another worker won the race
            shutil.rmtree(tmp)
    src, dst = files(SRC_TR), files(TR)
    assert src == dst, f"trainer copy file list differs: {set(src) ^ set(dst)}"
    bad = [str(r) for r in src if sha(SRC_TR / r) != sha(TR / r)]
    assert not bad, f"trainer copy differs from the pinned snapshot: {bad}"
    RES.mkdir(exist_ok=True)
    return len(src)


def ref_argv():
    """The registered run's own argv, minus --exp/--gpu (and, for smoke, with a short schedule)."""
    argv = json.load(open(SRC_TR / f"results_arc/{REF}_score.json"))["argv"]
    out, i = [], 0
    while i < len(argv):
        if argv[i] in ("--exp", "--gpu"):
            i += 2
            continue
        out.append(argv[i]); i += 1
    assert out[out.index("--iters") + 1] == "4000" and "--save_volumes" in out and str(SEPT) in " ".join(out), out
    if args.smoke:
        out[out.index("--iters") + 1] = "30"
        out[out.index("--warmup") + 1] = "10"
        out.remove("--save_volumes")
    return out


def pick(score):
    return {ch: {k: score[ch][k] for k in KEYS} for ch in ("plain", "contrast")}


def table():
    def load(p):
        return json.load(open(p)) if Path(p).exists() else None
    ref = load(SRC_TR / f"results_arc/{REF}_score.json")
    rows = [("reference (Sept)", ref)]
    sp, so = (load(SRC_TR / f"results_arc/ce_retro_af16_single_{s}_score.json") for s in ("pre", "post"))
    if sp and so:
        rows.append(("GS-single (Sept)", dict(plain=sp["plain"], contrast=so["plain"])))
    rows += [(n, load(RES / f"abl_{n}_score.json")) for n in ROWS]
    log(f"{'row':20s} | {'pre PSNR':>8s} {'d':>7s} {'SSIM':>7s} {'HFEN':>7s} | {'post PSNR':>9s} {'d':>7s} {'SSIM':>7s} {'HFEN':>7s}")
    for name, s in rows:
        if s is None:
            log(f"{name:20s} | (not finished)")
            continue
        p, c, rp, rc = s["plain"], s["contrast"], ref["plain"], ref["contrast"]
        log(f"{name:20s} | {p['head_psnr']:8.3f} {p['head_psnr'] - rp['head_psnr']:+7.3f} {p['head_ssim']:7.4f} "
            f"{p['head_hfen']:7.4f} | {c['head_psnr']:9.3f} {c['head_psnr'] - rc['head_psnr']:+7.3f} "
            f"{c['head_ssim']:7.4f} {c['head_hfen']:7.4f}")


if args.table:
    table()
    sys.exit(0)

vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
assert vis.isdigit(), f"needs exactly one GPU from gpu.py --gpu, got CUDA_VISIBLE_DEVICES={vis!r}"
n = ensure_trainer()
base = ref_argv()
(root / "claims").mkdir(exist_ok=True)
art = Path(os.environ.get("MCGS_ARTIFACT_DIR", root / "artifacts"))
art.mkdir(parents=True, exist_ok=True)
log(f"root {root}; trainer copy verified ({n} files); physical GPU {vis}; rows {args.rows}")
log(f"reference argv: {' '.join(base)}")
record = dict(root=str(root), sept=str(SEPT), gpu=vis, smoke=args.smoke, reference_argv=base, runs=[])
failed = []
for name in args.rows:
    exp = f"abl_{name}"
    try:
        (root / "claims" / exp).mkdir()
    except FileExistsError:
        log(f"-- {exp}: already claimed, skipped")
        continue
    outs = [RES / f"{exp}.log", RES / f"{exp}_score.json", RES / exp, RES / f"{exp}_volumes.npz"]
    assert not any(p.exists() for p in outs), f"refusing to overwrite outputs of {exp}"
    cmd = [sys.executable, "-u", "arc/cycle.py", "--exp", exp, "--gpu", vis, *base, *ROWS[name]]
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH="", CUDA_DEVICE_ORDER="PCI_BUS_ID")
    log(f"\n===== {time.strftime('%F %T')} {exp}: {' '.join(ROWS[name]) or '(no override)'}")
    t0 = time.time()
    with open(outs[0], "w") as lf:
        rc = subprocess.run(cmd, cwd=TR, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    dt = time.time() - t0
    run = dict(row=name, exp=exp, override=ROWS[name], cmd=cmd, rc=rc, seconds=dt)
    if rc == 0 and outs[1].exists():
        s = json.load(open(outs[1]))
        run.update(score=pick(s), npts=s["npts"], vram_gb=s["vram_gb"])
        shutil.copy2(outs[1], art / outs[1].name)
        log(f"===== {exp} done in {dt:.0f} s: pre {s['plain']['head_psnr']:.3f} / post {s['contrast']['head_psnr']:.3f} "
            f"head PSNR, N={s['npts']}, {s['vram_gb']:.1f} GB")
    else:
        failed.append(exp)
        tail = "".join(open(outs[0]).readlines()[-15:])
        log(f"===== {exp} FAILED rc={rc} after {dt:.0f} s; log tail:\n{tail}")
    record["runs"].append(run)
    (art / f"record_gpu{vis}{'_smoke' if args.smoke else ''}.json").write_text(json.dumps(record, indent=2))
log("\n")
table()
log(f"DONE; failed rows: {failed or 'none'}")
sys.exit(1 if failed else 0)
