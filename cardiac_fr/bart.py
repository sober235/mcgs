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


_MAXEIG = re.compile(r"Maximum eigenvalue:\s*([0-9.eE+-]+)")


def parse_max_eigenvalue(output: str) -> float:
    m = _MAXEIG.search(output)
    if m is None:
        raise ValueError("no 'Maximum eigenvalue' line (run pics with -e -d5)")
    return float(m.group(1))


def max_eigenvalue(ksp, sens, traj, work, log=None) -> float:
    """Largest eigenvalue of pics' normal operator A^H A for this trajectory and these coil maps (BART's own
    power iteration via -e, printed at -d5). It does not depend on the data. On radial data it is far above 1
    (Cartesian SENSE with normalised maps: about 1), so regularisation weights are set relative to it."""
    txt = run(["pics", "-t", traj, "-l1", "-r", "1e-9", "-i", "1", "-e", "-d5", ksp, sens,
               Path(work) / "maxeig_probe"], log=log)
    return parse_max_eigenvalue(txt)


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


def dims(base) -> tuple:
    """Array dimensions from the .hdr file, without reading the data."""
    lines = Path(f"{base}.hdr").read_text().split("\n")
    return tuple(int(v) for v in lines[lines.index("# Dimensions") + 1].split())


def crop_center(arr: np.ndarray, n: int) -> np.ndarray:
    """Central n^3 of the first three axes: an image on a 2x FOV grid -> the protocol FOV. BART centres the image on
    index N/2 for every grid size, so the crop is [(N - n)/2, (N + n)/2) on each axis (test_cardiac_fr_bart.py).
    Note the normalisation: BART's NUFFT scales like a unitary FFT, so on a grid with 8x the voxels the forward
    operator is 8^-1/2 smaller and a -S reconstruction is 2.83x brighter in data units; the metrics are scale-free
    and held-out predictions use the same grid's operator, so nothing in the pipeline depends on this."""
    sl = tuple(slice((s - n) // 2, (s + n) // 2) for s in arr.shape[:3])
    return arr[sl]


def embed_center(arr: np.ndarray, n: int) -> np.ndarray:
    """Zero-pad the first three axes to n^3 around the same centre (the inverse of crop_center)."""
    out = np.zeros((n, n, n) + arr.shape[3:], dtype=arr.dtype)
    sl = tuple(slice((n - s) // 2, (n + s) // 2) for s in arr.shape[:3])
    out[sl] = arr
    return out


def interp_maps(sens_in, out, grid: int, work, log=None) -> None:
    """Coil maps on a coarser grid of the same FOV -> grid^3 by Fourier interpolation: unitary FFT, centred
    zero-padding, unitary inverse FFT, scaled by (grid/n)^1.5 so the values at the original nodes are unchanged.
    Spec 5.5 correction 5 (2026-09-11): ecalib's per-voxel step costs about 1 ms per voxel, hours on 224^3."""
    work = Path(work)
    n = dims(sens_in)[0]
    run(["fft", "-u", "7", sens_in, work / "interp_k"], log=log)
    run(["resize", "-c", "0", grid, "1", grid, "2", grid, work / "interp_k", work / "interp_kpad"], log=log)
    run(["fft", "-u", "-i", "7", work / "interp_kpad", work / "interp_img"], log=log)
    run(["scale", repr((grid / n) ** 1.5), work / "interp_img", out], log=log)
