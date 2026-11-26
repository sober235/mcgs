#!/usr/bin/env python3
"""reexport_gs.py -- re-export the volumes of ONE finished GS row from its saved model (no training).

Why: dm_vol4_tangbing_uniform_joint_volumes.npz is truncated (the row was stopped while it saved its volumes when the
worker was moved between cards; its score file and checkpoints are complete). run_gs.row_state counts a row as done by
its score file, so the relaunch skipped the row.

What it does (one GPU from gpu.py --gpu, as run_gs.py):
  1. runs the trainer with the row's own recorded argv, --exp <exp>_reexp, --from_ckpt <row>/checkpoints/latest.pth;
  2. requires the new score to equal the recorded one (|head PSNR diff| <= 0.005 dB on every scored channel) and the
     new npz to pass the zip test with all 12 members;
  3. only then keeps the damaged file as <exp>_volumes.npz.truncated and puts the new volumes in its place.
The row's score file, log and checkpoints are not touched. No server clock is read or printed (time rule).

    gpu.py run --gpu N -- @python autoresearch/diffmask_testset_20261122/reexport_gs.py --row vol4_tangbing:uniform:joint
"""
import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ts_common import OUT   # noqa: E402
import run_gs               # noqa: E402

log = lambda *a: print(*a, flush=True)
MEMBERS = {f"{k}{c}{s}.npy" for k in ("rec", "gt", "zf") for c in ("T1", "FL") for s in ("", "_complex")}


def zip_ok(p):
    try:
        with zipfile.ZipFile(p) as z:
            return z.testzip() is None and set(z.namelist()) == MEMBERS
    except (OSError, zipfile.BadZipFile):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True, choices=run_gs.FORMAL_ROWS)
    a = ap.parse_args()
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    assert vis.isdigit(), f"needs exactly one GPU from gpu.py --gpu, got CUDA_VISIBLE_DEVICES={vis!r}"
    tr, n = run_gs.ensure_trainer(OUT)
    res = tr / "results_arc"
    exp = run_gs.exp_name(a.row)
    new = exp + "_reexp"
    old_score = json.load(open(res / f"{exp}_score.json"))
    vol = res / f"{exp}_volumes.npz"
    assert vol.exists() and not zip_ok(vol), f"{vol} is readable; nothing to repair"
    ck = res / exp / exp / "checkpoints/latest.pth"
    assert ck.is_file(), ck
    for o in (res / f"{new}.log", res / f"{new}_score.json", res / new, res / f"{new}_volumes.npz"):
        assert not o.exists(), f"refusing to overwrite {o}"
    argv = list(old_score["argv"])
    argv[argv.index("--exp") + 1] = new
    argv[argv.index("--gpu") + 1] = vis
    assert "--save_volumes" in argv and "--from_ckpt" not in argv
    cmd = [sys.executable, "-u", "arc/cycle.py", *argv, "--from_ckpt", str(ck)]
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH="", CUDA_DEVICE_ORDER="PCI_BUS_ID")
    log(f"trainer copy verified ({n} tracked files); physical GPU {vis}\n{' '.join(cmd)}")
    with open(res / f"{new}.log", "w") as lf:
        rc = subprocess.run(cmd, cwd=tr, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    assert rc == 0, f"trainer rc={rc}; see {res / (new + '.log')} (not echoed: trainer logs may carry clocks)"
    new_score = json.load(open(res / f"{new}_score.json"))
    worst = 0.0
    for ch in ("plain", "contrast"):
        for k in ("head_psnr", "head_ssim", "nmse"):
            d = new_score[ch][k] - old_score[ch][k]
            log(f"{ch:8s} {k:9s} recorded {old_score[ch][k]:.6f} re-export {new_score[ch][k]:.6f} diff {d:+.2e}")
            if k == "head_psnr":
                worst = max(worst, abs(d))
    nv = res / f"{new}_volumes.npz"
    ok = worst <= 0.005 and zip_ok(nv)
    log(f"largest |head PSNR diff| {worst:.2e} dB; new volumes zip test {'passed' if zip_ok(nv) else 'FAILED'}")
    if not ok:
        log("REEXPORT_MISMATCH: nothing replaced")
        sys.exit(2)
    vol.rename(res / f"{exp}_volumes.npz.truncated")
    os.link(nv, vol)                      # same bytes under the row's name; the _reexp files stay as the record
    log(f"REEXPORT_DONE: {vol.name} replaced (damaged file kept as {vol.name}.truncated)")


if __name__ == "__main__":
    main()
