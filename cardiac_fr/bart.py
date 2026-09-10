"""Thin wrappers around the BART command line (CPU build v1.0.00) and BART's own cfl reader.

Verified on a 16^3 phantom (2026-09-10): with -S the pics image is in data units (independent of -w;
bart nufft of sens * image reproduces the measured k-space with scale 1); without -S it is divided by
BART's internal scaling. Non-Cartesian -l1 (FISTA) diverges unless -e sets the step size.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BART = "/home/congcongliu/src/bart/bart"
sys.path.insert(0, "/home/congcongliu/src/bart/python")
import cfl  # noqa: E402  (column-major .cfl/.hdr)

_SCALING = re.compile(r"Inverse scaling of the data:\s*([0-9.eE+-]+)")


def write_cfl(base, arr) -> None:
    cfl.writecfl(str(base), np.asarray(arr, dtype=np.complex64))


def read_cfl(base) -> np.ndarray:
    return cfl.readcfl(str(base))


def run(args, log=None) -> str:
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("CARDIAC_FR_THREADS", "32"))
    t0 = time.time()
    res = subprocess.run([BART, *map(str, args)], capture_output=True, text=True, env=env)
    out = res.stdout + res.stderr
    if log is not None:
        with open(log, "a") as fh:
            fh.write(f"$ bart {' '.join(map(str, args))}\n{out}[{time.time() - t0:.1f} s, rc={res.returncode}]\n")
    if res.returncode != 0:
        raise RuntimeError(f"bart {args[0]} failed (rc={res.returncode}):\n{out[-2000:]}")
    return out


def bart_ksp(spokes: np.ndarray) -> np.ndarray:
    """(n_spoke, n_coil, n_samp) -> BART non-Cartesian layout (1, n_samp, n_spoke, n_coil)."""
    return np.ascontiguousarray(np.asarray(spokes).transpose(2, 0, 1))[None]


def bart_traj(k: np.ndarray) -> np.ndarray:
    """(3, n_samp, n_spoke) coordinates in grid units -> complex64 (BART stores trajectories as cfl)."""
    return np.asarray(k).astype(np.complex64)


def parse_scaling(output: str) -> float:
    m = _SCALING.search(output)
    if m is None:
        raise ValueError("no 'Inverse scaling of the data' line (run pics with -d5)")
    return float(m.group(1))


def pics(ksp, sens, out, traj, reg: str, lam: float, iters: int, w: float | None = None,
         debug: bool = False, log=None) -> dict:
    args = ["pics", "-t", traj, "-l1" if reg == "l1" else "-l2", "-r", lam, "-i", iters, "-S"]
    if reg == "l1":
        args.append("-e")
    if w is not None:
        args += ["-w", repr(float(w))]
    if debug:
        args.append("-d5")
    t0 = time.time()
    txt = run([*args, ksp, sens, out], log=log)
    return {"seconds": time.time() - t0, "scaling": parse_scaling(txt) if debug else None}


def coil_maps(traj, ksp, work, out_sens, grid: int, calib: int, log=None) -> None:
    """Spec 5.5: inverse NUFFT to coil images, unitary FFT, ESPIRiT with the brain-topic flags."""
    work = Path(work)
    run(["nufft", "-i", "-x", f"{grid}:{grid}:{grid}", traj, ksp, work / "coilimg"], log=log)
    run(["fft", "-u", "7", work / "coilimg", work / "kgrid"], log=log)
    run(["ecalib", "-t", "0.001", "-c", "0.0", "-r", calib, "-m", "1", work / "kgrid", out_sens], log=log)


def forward(traj, img, sens, out, work, log=None) -> None:
    """k-space predicted by image `img`: sens * img, then NUFFT onto `traj`."""
    tmp = Path(work) / f"{Path(out).name}_coilimg"
    run(["fmac", img, sens, tmp], log=log)
    run(["nufft", traj, tmp, out], log=log)
