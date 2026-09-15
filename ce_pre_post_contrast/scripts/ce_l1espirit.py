#!/usr/bin/env python3
"""ce_l1espirit.py -- L1-wavelet ESPIRiT (BART pics -l1, 100 FISTA iterations, CPU) on the UIH prospective CE pair.

Same call as the jyz/hyw topics' recon_l1espirit.py: bart pics -l1 -r LAM -i 100 -S ksp maps out. Differences,
all forced by this data set:
  * the maps are this topic's bundled coil maps (SoS == 1 inside the support, 0 outside -- the ESPIRiT form),
    the same maps every other column of this topic uses, and the k-space is divided by ce_classical.py's
    P99.9 |ZF| scale;
  * the sampling pattern is passed explicitly (-p), so a --mask fit run never sees the held-out lines;
  * BART's data scaling is frozen per scan (-w, the value BART estimates on the full data), so a lambda means
    the same thing in the 95% selection runs and in the 100% refit.
Lambda is chosen on the held-out residual of --mask fit runs (ce_select_lambda.py), then refit with --mask full.

usage: ce_l1espirit.py --scan pre --mask fit --lams 5e-4,1e-3 [--threads 24] [--cleanup]
Writes <out>/l1espirit_i<iters>_lam<lam>_<mask>_<scan>_{complex.npy,meta.json}; existing metas are skipped.
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--scan", required=True, choices=["pre", "post"])
ap.add_argument("--mask", required=True, choices=["fit", "full"])
ap.add_argument("--lams", required=True, help="comma-separated")
ap.add_argument("--iters", type=int, default=100)
ap.add_argument("--threads", type=int, default=24)
ap.add_argument("--out", required=True)
ap.add_argument("--bart", default="/home/congcongliu/src/bart/bart")
ap.add_argument("--cleanup", action="store_true", help="remove this scan's cfl work directory when done")
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/congcongliu/src/bart/python")
import ce_common as C  # noqa: E402
import cfl  # noqa: E402

torch.set_num_threads(16)
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
WORK = OUT / f"work_{args.scan}"
WORK.mkdir(parents=True, exist_ok=True)
log = lambda *a: print(*a, flush=True)
lams = [float(v) for v in args.lams.split(",")]


def tag(lam):
    return f"l1espirit_i{args.iters}_lam{lam:g}_{args.mask}_{args.scan}"


todo = [lam for lam in lams if not (OUT / f"{tag(lam)}_meta.json").exists()]
t0 = time.time()
if todo:
    y, S, m2 = C.load_scan(args.scan)
    fit_plane, ho_plane = C.load_split()
    scale = C.classical_scale(args.scan)
    NX = y.shape[1]
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads))

    def bart_dims(a):                                   # ours (coil, x, y, z) -> BART (x, y, z, coil)
        return np.ascontiguousarray(a.numpy().transpose(1, 2, 3, 0))

    if not (WORK / "maps.cfl").exists():
        cfl.writecfl(str(WORK / "maps"), bart_dims(S))
    for mk, plane in (("full", m2), ("fit", fit_plane)):
        if not (WORK / f"ksp_{mk}.cfl").exists():
            cfl.writecfl(str(WORK / f"ksp_{mk}"), bart_dims(y * plane / scale))
            cfl.writecfl(str(WORK / f"pat_{mk}"), np.broadcast_to(plane.numpy(), (NX,) + tuple(plane.shape)).astype(np.complex64))
    log(f"[{args.scan}] cfl inputs ready in {WORK}  ({time.time() - t0:.0f}s)")

    wfile = WORK / "bart_inverse_scaling.txt"
    if not wfile.exists():
        res = subprocess.run([args.bart, "pics", "-d4", "-l1", "-r", "0.001", "-i", "1", "-p", str(WORK / "pat_full"),
                              str(WORK / "ksp_full"), str(WORK / "maps"), str(WORK / "probe")],
                             capture_output=True, text=True, env=env)
        m = re.search(r"Inverse scaling of the data:\s*([0-9.eE+-]+)", res.stdout + res.stderr)
        if res.returncode != 0 or not m:
            raise SystemExit(f"could not read BART's data scaling (rc={res.returncode}):\n{res.stderr[-2000:]}")
        wfile.write_text(m.group(1))
        for f in WORK.glob("probe.*"):
            f.unlink()
    W = float(wfile.read_text())
    log(f"[{args.scan}] BART inverse data scaling (full data, frozen): {W:.6g}")

    for lam in todo:
        t1 = time.time()
        out = WORK / f"out_{tag(lam)}"
        cmd = [args.bart, "pics", "-l1", "-r", f"{lam:g}", "-i", str(args.iters), "-p", str(WORK / f"pat_{args.mask}"),
               "-w", f"{W:.10g}", "-S", str(WORK / f"ksp_{args.mask}"), str(WORK / "maps"), str(out)]
        log(" ".join(cmd))
        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if res.returncode != 0:
            raise SystemExit(f"bart pics failed rc={res.returncode}\n{res.stderr[-2000:]}")
        t_bart = time.time() - t1
        x = torch.from_numpy(np.ascontiguousarray(np.squeeze(cfl.readcfl(str(out))).astype(np.complex64)))
        assert tuple(x.shape) == tuple(y.shape[1:]), (x.shape, y.shape)
        x = x * scale
        r = C.kres_metrics(x, y, S, {"full": m2, "fit": fit_plane, "ho": ho_plane})
        alpha = r["full"]["alpha"]
        np.save(OUT / f"{tag(lam)}_complex.npy", (alpha * x).numpy().astype(np.complex64))
        meta = dict(method="l1espirit", scan=args.scan, mask=args.mask, lam=lam, iters=args.iters, bart_w=W,
                    scale=scale, alpha_re=float(np.real(alpha)), alpha_im=float(np.imag(alpha)),
                    kres=r["full"]["kres"], kres_fit=r["fit"]["kres"], kres_heldout=r["ho"]["kres"],
                    heldout_is_unseen=args.mask == "fit", shape=list(x.shape), threads=args.threads,
                    seconds_bart=t_bart, seconds=time.time() - t1, bart_stdout_tail=res.stdout[-600:])
        C.dump_json(meta, OUT / f"{tag(lam)}_meta.json")
        for f in WORK.glob(f"out_{tag(lam)}.*"):
            f.unlink()
        log(f"[{tag(lam)}] kres full {meta['kres']:.5f} fit {meta['kres_fit']:.5f} held-out {meta['kres_heldout']:.5f}"
            f"  |alpha| {abs(alpha):.4f}  bart {t_bart:.0f}s")
if args.cleanup and WORK.exists():
    for f in WORK.iterdir():
        f.unlink()
    WORK.rmdir()
    log(f"removed {WORK}")
log(f"DONE {time.time() - t0:.0f}s")
