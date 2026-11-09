#!/usr/bin/env python3
"""sept_cmp_support.py -- rerun 3D-DIP and 3D-INR on the September CE retro data (CE_retro_cssense_af16, BART
Poisson-disc AF16) with the October comparison-arm audit winners, which add an output support:

  dip : --support 6 --dip_base 96 --tv 0.03 --iters 12000 --lr 3e-3 --eval_every 200   (suffix _sup_b96_tv0.03)
  inr : --support 6 --inr_nmax 512 --tv 0.03 --iters 1500 --eval_every 50             (suffix _i10_n512_sup6_tv0.03)

These are the exact command lines of Results_20260908/CE_retro_cssense_af16_vdpois/logs/cmp/*_sup*_tv0.03.log,
picked there by each arm's own pre-registered rule (held-out k-space residual, no GT). Support = the zero-filled
head support (|ZF| > 0.05 P99.9) dilated 6 voxels: acquired data only, no GT / ROI. It removes the unconstrained
signal outside the coil coverage, where 97 % of the registered September DIP's error energy sits.

The recon script is the audited one (mcgs_ce_mask_wt @ b01fadf, scripts/recon_dip_inr.py), run unchanged. It reads
<topic>/sim and writes <topic>/recon and <topic>/logs, so this wrapper builds a NEW topic directory whose sim/ is a
symlink to the September sim/ and whose recon/ and logs/ are new; the September topic is never written. Existing
outputs are never overwritten.

GPU: gpu.py sets CUDA_VISIBLE_DEVICES to the one authorised physical id; recon_dip_inr.py sets
CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES=--gpu itself, so the wrapper passes --gpu = that same id
(nvidia-smi indices are PCI-bus order), i.e. the child lands on exactly the card gpu.py granted.

  gpu.py run --gpu 2 -- @python autoresearch/ce_af16_lesion_figs_20261107/sept_cmp_support.py --scan pre
  gpu.py run --gpu 2 -- @python ... --scan pre --methods dip --smoke      # 2-iteration plumbing check, own topic dir
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
SCRIPTS = Path("/data0/congcong/code/GS/mcgs_ce_mask_wt/ce_retro_cssense_af16/scripts")
OUT = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_ce_sept_cmp_20261109")
RUNS = {"dip": (["--iters", "12000", "--lr", "3e-3", "--eval_every", "200", "--support", "6", "--dip_base", "96",
                 "--tv", "0.03"], "_sup_b96_tv0.03"),
        "inr": (["--eval_every", "50", "--inr_nmax", "512", "--support", "6", "--iters", "1500", "--tv", "0.03"],
                "_i10_n512_sup6_tv0.03")}

ap = argparse.ArgumentParser()
ap.add_argument("--scan", required=True, choices=["pre", "post"])
ap.add_argument("--methods", nargs="+", default=["dip", "inr"], choices=["dip", "inr"])
ap.add_argument("--smoke", action="store_true", help="2 iterations, eval every iteration, into OUT/smoke")
args = ap.parse_args()
log = lambda *a: print(*a, flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
assert vis.isdigit(), f"needs exactly one GPU from gpu.py --gpu, got CUDA_VISIBLE_DEVICES={vis!r}"
topic = OUT / "smoke" if args.smoke else OUT
for d in ("recon", "logs"):
    (topic / d).mkdir(parents=True, exist_ok=True)
if not (topic / "sim").exists():
    (topic / "sim").symlink_to(SEPT / "sim")
assert (topic / "sim").resolve() == (SEPT / "sim").resolve(), "sim/ must be the September sim"
script = SCRIPTS / "recon_dip_inr.py"
rev = subprocess.run(["git", "-C", str(SCRIPTS), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(["git", "-C", str(SCRIPTS), "status", "--porcelain", "--", "."], capture_output=True, text=True).stdout
assert not dirty.strip(), f"audited scripts dir has local changes:\n{dirty}"
log(f"topic {topic} (sim -> {SEPT / 'sim'}); script {script} @ {rev}; physical GPU {vis}")

art = Path(os.environ.get("MCGS_ARTIFACT_DIR", topic / "logs"))
art.mkdir(parents=True, exist_ok=True)
record = dict(topic=str(topic), sept_topic=str(SEPT), script=str(script), script_sha256=sha(script), scripts_rev=rev,
              gpu=vis, scan=args.scan, smoke=args.smoke, runs=[])
for m in args.methods:
    extra, suffix = RUNS[m]
    if args.smoke:
        extra = [a for a in extra]
        extra[extra.index("--iters") + 1] = "2"
        extra[extra.index("--eval_every") + 1] = "1"
        suffix += "_smoke"
    out_npy = topic / f"recon/{m}{suffix}_{args.scan}_complex.npy"
    out_json = topic / f"logs/{m}{suffix}_{args.scan}.json"
    assert not out_npy.exists() and not out_json.exists(), f"refusing to overwrite {out_npy} / {out_json}"
    cmd = [sys.executable, "-u", str(script), "--method", m, "--scan", args.scan, "--gpu", vis, *extra, "--suffix", suffix]
    env = dict(os.environ, RETRO_TOPIC=str(topic), CUDA_DEVICE_ORDER="PCI_BUS_ID")
    log(f"\n===== {time.strftime('%F %T')} {m} {args.scan}: {' '.join(cmd)}")
    t0 = time.time()
    rc = subprocess.run(cmd, env=env).returncode
    log(f"===== {m} {args.scan} exit {rc} after {time.time() - t0:.0f} s")
    assert rc == 0, f"{m} {args.scan} failed"
    shutil.copy2(out_json, art / out_json.name)
    record["runs"].append(dict(method=m, suffix=suffix, cmd=cmd, seconds=time.time() - t0, volume=str(out_npy),
                               volume_sha256=sha(out_npy), score=json.load(open(out_json))["rows"][0]))
(art / f"record_{args.scan}{'_smoke' if args.smoke else ''}.json").write_text(json.dumps(record, indent=2))
log("DONE")
