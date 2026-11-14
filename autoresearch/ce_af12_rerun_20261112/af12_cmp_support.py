#!/usr/bin/env python3
"""af12_cmp_support.py -- 3D-DIP and 3D-INR with output support on the AF12 rerun (user, 2026-11-13).

The AF16 figure of record shows 3D-DIP / 3D-INR rerun with the October audit winners, which add an output support
(../ce_af16_lesion_figs_20261107/sept_cmp_support.py, user's choice 2026-11-09). The user asked for the same at AF12
(2026-11-13, "That does indeed require a re-run."). This is that wrapper with one change, the mask:

  * recon script: the audited one (mcgs_ce_mask_wt @ b01fadf, scripts/recon_dip_inr.py), byte copy;
  * t1t2_common.py: the audited copy with only the mask file name replaced (mask_af16_3d -> mask_af12_3d, three
    places, each text must occur exactly once); diag_artefact/: byte copy. They are written once to OUT/scripts
    and compared byte for byte on every later start;
  * sim/: symlink to the AF12 topic's sim/ (AF12 mask; September k-space, coil maps and truth);
  * arguments: those of the AF16 support run, unchanged:
      dip : --support 6 --dip_base 96 --tv 0.03 --iters 12000 --lr 3e-3 --eval_every 200   (suffix _sup_b96_tv0.03)
      inr : --support 6 --inr_nmax 512 --tv 0.03 --iters 1500 --eval_every 50             (suffix _i10_n512_sup6_tv0.03)
    Support = zero-filled head support (|ZF| > 0.05 P99.9) dilated 6 voxels: acquired data only, no GT / ROI.
New topic OUT; the AF12 and September topics are never written; existing outputs are refused.

GPU: recon_dip_inr.py sets CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES=--gpu itself, so the wrapper passes
the physical id that gpu.py granted (as sept_cmp_support.py does).

Clock values: the recon script prints elapsed seconds and stores them in its json. Its raw output goes to
OUT/logs/<method><suffix>_<scan>.log; what this wrapper prints has those values cut, and the json copied to
MCGS_ARTIFACT_DIR has "seconds" set to null (the sha256 of the untouched server file is recorded).

  gpu.py run --cpu -- @python autoresearch/ce_af12_rerun_20261112/af12_cmp_support.py --check
  gpu.py run --gpu ID -- @python autoresearch/ce_af12_rerun_20261112/af12_cmp_support.py --scan pre --methods dip
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

RES = Path("/data0/congcong/code/GS/Results_20260908")
T12 = RES / "CE_retro_cssense_af12"
S16 = RES / "CE_retro_cssense_af16"
TR = S16 / "trainer/_trainer_e778bf2"
SCRIPTS = Path("/data0/congcong/code/GS/mcgs_ce_mask_wt/ce_retro_cssense_af16/scripts")
REV = "b01fadfa0e8bbf5237b22329624c3cf6535efb6d"
OUT = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_ce_af12_cmp_20261113")
RUNS = {"dip": (["--iters", "12000", "--lr", "3e-3", "--eval_every", "200", "--support", "6", "--dip_base", "96",
                 "--tv", "0.03"], "_sup_b96_tv0.03"),
        "inr": (["--eval_every", "50", "--inr_nmax", "512", "--support", "6", "--iters", "1500", "--tv", "0.03"],
                "_i10_n512_sup6_tv0.03")}
PORT = [("mask sim/mask/mask_af16_3d.h5 (key mask);", "mask sim/mask/mask_af12_3d.h5 (key mask);"),
        ('_mp = R / f"sim/mask/mask_af16_3d_{contrast}.h5"', '_mp = R / f"sim/mask/mask_af12_3d_{contrast}.h5"'),
        ('with h5py.File(_mp if _mp.exists() else R / "sim/mask/mask_af16_3d.h5", "r") as f:',
         'with h5py.File(_mp if _mp.exists() else R / "sim/mask/mask_af12_3d.h5", "r") as f:')]
CUT = re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*s\)")          # "(123s)" / "(123 s)": elapsed seconds of the server run
log = lambda *a: print(*a, flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tree(d):
    return {str(p.relative_to(d)): sha(p) for p in sorted(d.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}


def prepare():
    """Create OUT (sim link, ported scripts) on first use; afterwards verify it byte for byte."""
    rev = subprocess.run(["git", "-C", str(SCRIPTS), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(SCRIPTS), "status", "--porcelain", "--", "."],
                           capture_output=True, text=True).stdout
    assert rev == REV, f"audited scripts at {rev}, expected {REV}"
    assert not dirty.strip(), f"audited scripts dir has local changes:\n{dirty}"
    common = (SCRIPTS / "t1t2_common.py").read_text()
    for old, new in PORT:
        assert common.count(old) == 1, f"t1t2_common.py: expected exactly one occurrence of\n{old}\nfound {common.count(old)}"
        common = common.replace(old, new)
    assert "mask_af16" not in common
    want = {"recon_dip_inr.py": (SCRIPTS / "recon_dip_inr.py").read_bytes(), "t1t2_common.py": common.encode()}
    for d in ("recon", "logs", "scripts"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    if not (OUT / "sim").is_symlink():
        (OUT / "sim").symlink_to(T12 / "sim")
    assert (OUT / "sim").resolve() == (T12 / "sim").resolve(), "sim/ must be the AF12 topic's sim"
    for name, data in want.items():
        p = OUT / "scripts" / name
        if not p.exists():
            p.write_bytes(data)
        assert p.read_bytes() == data, f"{p} differs from the expected port"
    if not (OUT / "scripts/diag_artefact").exists():
        shutil.copytree(SCRIPTS / "diag_artefact", OUT / "scripts/diag_artefact",
                        ignore=shutil.ignore_patterns("__pycache__"))
    assert tree(OUT / "scripts/diag_artefact") == tree(SCRIPTS / "diag_artefact"), "diag_artefact copy differs"
    setup = dict(af12_topic=str(T12), sim_link=str(T12 / "sim"), audited_scripts=str(SCRIPTS), rev=rev,
                 port_t1t2_common=PORT, runs={m: dict(args=a, suffix=s) for m, (a, s) in RUNS.items()},
                 sha256={**{f"scripts/{n}": hashlib.sha256(d).hexdigest() for n, d in want.items()},
                         **{f"audited/{n}": sha(SCRIPTS / n) for n in want},
                         "sim/mask/mask_af12_3d.h5": sha(T12 / "sim/mask/mask_af12_3d.h5")})
    sp = OUT / "logs/setup.json"
    if not sp.exists():
        sp.write_text(json.dumps(setup, indent=2))
    assert json.loads(sp.read_text()) == json.loads(json.dumps(setup)), "logs/setup.json differs from this setup"
    log(f"prepare OK: {OUT} (sim -> {T12 / 'sim'}); scripts {SCRIPTS} @ {rev[:7]}, t1t2_common ported in 3 places")
    return setup


def check():
    """CPU: the ported loader reads the AF12 mask and the September truth / scales."""
    import h5py
    import numpy as np
    os.environ.update(RETRO_TOPIC=str(OUT), RETRO_TRAINER=str(TR))
    sys.path.insert(0, str(OUT / "scripts"))
    import torch
    import t1t2_common as C
    assert Path(C.__file__).resolve().parent == (OUT / "scripts").resolve() and C.R == OUT, (C.__file__, C.R)
    with h5py.File(T12 / "sim/mask/mask_af12_3d.h5", "r") as f:
        m12 = f["mask"][:] > 0.5
    with h5py.File(S16 / "sim/mask/mask_af16_3d.h5", "r") as f:
        m16 = f["mask"][:] > 0.5
    rows = {}
    for s in C.SCANS:
        P = C.Pair(s, torch.device("cpu"))
        m3 = P.m3.numpy()
        assert (m3 == m12).all(), f"{s}: loader mask is not mask_af12_3d.h5"
        assert not (m3 == m16).all()
        npts = int(P.m2.sum())
        with np.load(S16 / "sim/gt.npz") as g16, np.load(OUT / "sim/gt.npz") as g12:
            same_gt = bool(np.array_equal(g16[s], g12[s]))
            same_scale = float(g16[f"scale_{s}"]) == float(g12[f"scale_{s}"]) == P.scale
        assert npts == 3403 and same_scale and same_gt, (npts, same_scale, same_gt)
        rows[s] = dict(points=npts, af=round(P.m2.numel() / npts, 4), coils=P.NC, grid=list(P.G), scale=P.scale,
                       truth_as_september=same_gt)
        log(f"[{s}] {rows[s]}")
        del P
    (OUT / "logs/check.json").write_text(json.dumps(rows, indent=2))
    log("CHECK OK")


def run(scan, methods):
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    assert vis.isdigit(), f"needs exactly one GPU from gpu.py --gpu, got CUDA_VISIBLE_DEVICES={vis!r}"
    assert (OUT / "logs/check.json").exists(), "run --check first"
    art = Path(os.environ.get("MCGS_ARTIFACT_DIR", OUT / "logs"))
    art.mkdir(parents=True, exist_ok=True)
    script = OUT / "scripts/recon_dip_inr.py"
    for m in methods:
        extra, suffix = RUNS[m]
        out_npy = OUT / f"recon/{m}{suffix}_{scan}_complex.npy"
        out_json = OUT / f"logs/{m}{suffix}_{scan}.json"
        raw = OUT / f"logs/{m}{suffix}_{scan}.log"
        for p in (out_npy, out_json, raw):
            assert not p.exists(), f"refusing to overwrite {p}"
        cmd = [sys.executable, "-u", str(script), "--method", m, "--scan", scan, "--gpu", vis, *extra, "--suffix", suffix]
        env = dict(os.environ, RETRO_TOPIC=str(OUT), RETRO_TRAINER=str(TR), CUDA_DEVICE_ORDER="PCI_BUS_ID")
        log(f"\n===== {m} {scan} on physical GPU {vis}: {' '.join(cmd)}")
        af_ok = False
        with open(raw, "w") as f:
            p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in p.stdout:
                f.write(line)
                f.flush()
                print(CUT.sub("", line), end="", flush=True)
                a = re.search(r"AF_rect=([0-9.]+)", line)
                if a:
                    af_ok = 11.9 < float(a.group(1)) < 12.1
                    if not af_ok:
                        p.kill()
                        log(f"STOP: AF_rect {a.group(1)} is not the AF12 mask")
            rc = p.wait()
        log(f"===== {m} {scan} exit {rc}")
        assert rc == 0 and af_ok, f"{m} {scan} failed (rc {rc}, AF12 mask seen {af_ok})"
        js = json.loads(out_json.read_text())
        for r in js["rows"]:
            r["seconds"] = None
        (art / out_json.name).write_text(json.dumps(js, indent=2))
        rec = dict(method=m, suffix=suffix, scan=scan, gpu=vis, cmd=cmd, volume=str(out_npy),
                   volume_sha256=sha(out_npy), json_sha256_server=sha(out_json),
                   json_copy_note="copy has rows[].seconds = null; otherwise identical", score=js["rows"][0])
        for d in {OUT / "logs", art}:
            (d / f"record_{m}_{scan}.json").write_text(json.dumps(rec, indent=2))
        r = js["rows"][0]
        log(f"[{scan}] {m}{suffix} head_psnr={r['head_psnr']:.3f} head_ssim={r['head_ssim']:.4f} "
            f"psnr={r['psnr']:.3f} ssim={r['ssim']:.4f}")
    log("DONE")


ap = argparse.ArgumentParser()
ap.add_argument("--check", action="store_true", help="CPU: create / verify OUT and test the ported loader")
ap.add_argument("--scan", choices=["pre", "post"])
ap.add_argument("--methods", nargs="+", default=["dip", "inr"], choices=["dip", "inr"])
args = ap.parse_args()
prepare()
if args.check:
    check()
else:
    assert args.scan, "--scan is required"
    run(args.scan, args.methods)
