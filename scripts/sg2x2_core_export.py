#!/usr/bin/env python3
"""Export core-test subjects from the UIH h5 set into the trainer's format (inventory section 3.3).

    env PYTHONNOUSERSITE=1 PYTHONPATH= OMP_NUM_THREADS=24 nice -n 10 $PY scripts/sg2x2_core_export.py SUBJECT [SUBJECT ...]

Per contrast: read `kspace` (RO, coil, SPE, PE) -- the readout axis is stored in image domain -- move the coil axis
first, centred orthonormal 1-D FFT along the readout, crop the SPE axis to its acquired rows [NY - 176, NY), swap the
two phase-encode axes -> (coil, RO, PE, SPE) = (12, 256, 256, 176) pure k-space, the pilot's frame; write ksp_<C>.h5
(key `kspace`). Then the pilot's ESPIRiT recipe: write the k-space as a BART CFL (RO, PE, SPE, coil) under <out>/bart/,
run `bart ecalib -t 0.001 -c 0.0 -r 24 -m 1`, write the maps coil first as csm_espirit_<C>.h5 (key `map`).

Checks (the export stops on a failure and writes no export.json): nothing acquired outside the cropped rows; the
k-space DC within one sample of the centre of the (PE, SPE) plane; T1 acquires the whole cropped plane; ESPIRiT
sum-of-squares within 1e-3 of 1 on at least 99.9 % of the voxels. Every number goes to <out>/export.json.

Never overwrites and never removes a file. The CFL intermediates under <out>/bart/ (2.2 GB per contrast) are kept;
delete them by hand when the maps are no longer needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from sg2x2_core_subjects import CONTRASTS, SUBJECTS, subject_paths  # noqa: E402

N_SPE = 176
BART = "/home/congcongliu/src/bart/bart"
ECALIB = ("ecalib", "-t", "0.001", "-c", "0.0", "-r", "24", "-m", "1")


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fft1c(x: np.ndarray, axis: int) -> np.ndarray:
    """centred orthonormal 1-D FFT (the one-axis form of paired_gs.data.transforms.fft3c)"""
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=axis), axis=axis, norm="ortho"), axes=axis)


def convert(raw: np.ndarray, n_spe: int = N_SPE) -> tuple[np.ndarray, dict]:
    """raw (RO, coil, NY, PE) with the readout in image domain -> k-space (coil, RO, PE, SPE), and the numbers checked"""
    if raw.ndim != 4:
        raise ValueError(f"expected a 4-D (RO, coil, NY, PE) array, got shape {raw.shape}")
    ny, npe = raw.shape[2], raw.shape[3]
    lo = ny - n_spe
    if lo < 0:
        raise ValueError(f"NY {ny} is smaller than the {n_spe} acquired SPE rows")
    k = fft1c(np.moveaxis(raw, 1, 0), axis=1).astype(np.complex64)      # (coil, RO, NY, PE)
    plane = np.sqrt((np.abs(k) ** 2).sum(axis=(0, 1)))                   # (NY, PE)
    pmax = float(plane.max())
    outside = float(plane[:lo].max()) / pmax if lo > 0 else 0.0
    k = np.ascontiguousarray(np.swapaxes(k[:, :, lo:, :], 2, 3))          # (coil, RO, PE, SPE)
    block = np.swapaxes(plane[lo:], 0, 1)                                 # (PE, SPE)
    dc = np.unravel_index(int(np.argmax(block)), block.shape)
    info = {"raw_shape": list(raw.shape), "ny": int(ny), "lo": int(lo), "shape": list(k.shape),
            "outside_rel_max": outside, "acquired_frac": float((block > 1e-12 * pmax).mean()),
            "dc_pe_spe": [int(dc[0]), int(dc[1])], "centre_pe_spe": [int(npe // 2), int(n_spe // 2)]}
    return k, info


def write_cfl(base: str, arr: np.ndarray) -> None:
    """BART CFL (column-major complex64) plus header; refuses to overwrite"""
    for ext in (".cfl", ".hdr"):
        if os.path.exists(base + ext):
            raise FileExistsError(base + ext)
    np.asarray(arr, dtype=np.complex64).ravel(order="F").tofile(base + ".cfl")
    with open(base + ".hdr", "x") as f:
        f.write("# Dimensions\n" + " ".join(str(d) for d in arr.shape) + " \n")


def read_cfl(base: str) -> np.ndarray:
    with open(base + ".hdr") as f:
        dims = [int(v) for v in f.read().splitlines()[1].split()]
    return np.fromfile(base + ".cfl", dtype=np.complex64).reshape(dims, order="F")


def espirit(k: np.ndarray, base: str, bart: str = BART, ecalib: tuple = ECALIB) -> tuple[np.ndarray, dict]:
    """ESPIRiT maps (coil, RO, PE, SPE) of k (coil, RO, PE, SPE); the BART files <base>_ksp and <base>_map are kept"""
    for ext in (".cfl", ".hdr"):
        if os.path.exists(base + "_map" + ext):
            raise FileExistsError(base + "_map" + ext)
    write_cfl(base + "_ksp", np.moveaxis(k, 0, 3))                       # (RO, PE, SPE, coil)
    t0 = time.time()
    env = dict(os.environ, TOOLBOX_PATH=str(Path(bart).parent))
    r = subprocess.run([bart, *ecalib, base + "_ksp", base + "_map"], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"bart ecalib failed (rc {r.returncode}): {r.stderr[-800:]}")
    maps = read_cfl(base + "_map")
    maps = np.ascontiguousarray(np.moveaxis(maps.reshape(maps.shape[:4]), 3, 0)).astype(np.complex64)
    sos = np.sqrt((np.abs(maps) ** 2).sum(axis=0))
    info = {"seconds": round(time.time() - t0, 1), "cmd": " ".join([bart, *ecalib]),
            "sos_min": float(sos.min()), "sos_max": float(sos.max()),
            "sos_frac_within_1e-3": float((np.abs(sos - 1.0) < 1e-3).mean())}
    return maps, info


def write_h5(path: str, key: str, arr: np.ndarray) -> str:
    """write arr under key into a NEW h5 file (mode "x" refuses an existing one); returns the file's md5"""
    chunks = (1, min(32, arr.shape[1]), arr.shape[2], arr.shape[3])
    with h5py.File(path, "x") as h:
        h.create_dataset(key, data=arr, chunks=chunks)
    return md5(path)


def export_pair(src: dict, out: str, n_spe: int = N_SPE, bart: str = BART, ecalib: tuple = ECALIB) -> dict:
    """src {"T1": h5, "FLAIR": h5} -> <out>/ksp_<C>.h5, <out>/csm_espirit_<C>.h5, <out>/bart/, <out>/export.json"""
    report = os.path.join(out, "export.json")
    if os.path.exists(report):
        raise FileExistsError(report)
    os.makedirs(os.path.join(out, "bart"), exist_ok=True)
    rep = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "n_spe": n_spe, "contrasts": {}}
    for c in CONTRASTS:
        with h5py.File(src[c], "r") as f:
            raw = f["kspace"][...]
        k, info = convert(raw, n_spe)
        del raw
        problems = []
        if info["outside_rel_max"] > 1e-6:
            problems.append(f"data outside the cropped rows (relative max {info['outside_rel_max']:.3g})")
        if any(abs(a - b) > 1 for a, b in zip(info["dc_pe_spe"], info["centre_pe_spe"])):
            problems.append(f"k-space DC {info['dc_pe_spe']} is not at the centre {info['centre_pe_spe']}")
        if c == "T1" and info["acquired_frac"] < 0.9999:
            problems.append(f"T1 acquires only {info['acquired_frac']:.4f} of the cropped plane")
        if problems:
            raise RuntimeError(f"{src[c]} ({c}): " + "; ".join(problems))
        info["src"], info["src_md5"] = src[c], md5(src[c])
        info["ksp"] = os.path.join(out, f"ksp_{c}.h5")
        info["ksp_md5"] = write_h5(info["ksp"], "kspace", k)
        maps, einfo = espirit(k, os.path.join(out, "bart", c), bart, ecalib)
        if einfo["sos_frac_within_1e-3"] < 0.999:
            raise RuntimeError(f"{c}: ESPIRiT sum-of-squares off 1 on {1 - einfo['sos_frac_within_1e-3']:.4f} of voxels")
        info["csm"] = os.path.join(out, f"csm_espirit_{c}.h5")
        info["csm_md5"] = write_h5(info["csm"], "map", maps)
        info["espirit"] = einfo
        rep["contrasts"][c] = info
        print(f"[export] {out} {c}: lo={info['lo']} dc={info['dc_pe_spe']} acquired={info['acquired_frac']:.4f} "
              f"ecalib {einfo['seconds']} s, SoS within 1e-3 on {einfo['sos_frac_within_1e-3']:.4f}", flush=True)
        del k, maps
    rep["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(report, "x") as f:
        json.dump(rep, f, indent=2)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subjects", nargs="+", choices=sorted(SUBJECTS))
    a = ap.parse_args()
    for s in a.subjects:
        p = subject_paths(s)
        rep = export_pair(p["src"], p["out"])
        print(f"[export] {s} ({p['role']}) done: {p['report']} "
              f"({rep['started']} -> {rep['finished']})", flush=True)


if __name__ == "__main__":
    main()
