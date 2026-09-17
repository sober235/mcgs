#!/usr/bin/env python3
"""make_noisy_kspace.py -- the noise study's measurements: the clean jiangyizhou k-space plus complex Gaussian noise.

Spec: /data0/congcong/code/GS/mcgs_noise_wt/docs/superpowers/specs/2026-09-16-jyz-af16-noise-robustness-design.md,
section 3. Per contrast:
  1. ground truth as t1t2_common.Pair builds it (sum_c conj(S_c) ifft3c(y_c), scale = P99.9 of |GT|), here in double
     precision on the CPU;
  2. sigma0_img = sqrt(var Re + var Im) of GT/scale over air (|GT| > 0.05 P99.9, closed 3x, dilated 12x, complement,
     8-voxel border excluded); sigma0_k = sigma0_img / sqrt(f_acq), f_acq = acquired fraction of the PE plane;
  3. n ~ CN(0, sigma^2) with sigma = MULT * sigma0_k (gauge units), added only on the acquired support (the samples
     that are non-zero in the clean file); PCG64 from SeedSequence(SEED), one spawned child per contrast in the order
     T1, FLAIR, drawn coil by coil;
  4. data/meas_<c>.h5 (key kspace, complex64, the source chunking), re-read from disk for the checks:
       support_exact      meas - clean is exactly 0 outside the support
       mean_z             every coil's mean (Re and Im) within 4 standard errors of 0 (52 tests; 4 SE keeps the
                          chance of a false alarm below 0.5 %, 3 SE would raise one 13 % of the time)
       std_rel            every coil's std of Re and Im equals sigma_raw / sqrt(2) within 0.5 %
       air_noise_rel      air noise of the SENSE-combined noisy full image = sqrt(sigma0_img^2 + sigma^2 f_acq), 2 %
       gt_vs_house        max |GT/scale - house recon/gt_<c>_complex.npy| <= 1e-4 (the house volume is single
                          precision; a wrong map or a flipped array differs by 0.1 or more)
       scale_vs_house     |scale - house scale_p999| / house <= 1e-5 (same reason)
       nested_vs_1x      (0.5 sigma0 level, spec addendum 2026-09-17) same seed and draws as the 1 sigma0 level, so
                          RATIO (meas_this - clean) = meas_ref - clean before single-precision rounding (RATIO = 2);
                          checked sample by sample against a rounding bound 4e-7 (|clean| + 10 sigma_raw_ref), max ratio <= 1
  5. recon/fullnoisy_<c>_complex.npy: the SENSE-combined noisy full image in gauge units (for the noise figure).
Refuses to overwrite. A failed check renames the outputs to *.failed and exits 1.
usage: make_noisy_kspace.py [--contrasts T1,FLAIR] [--threads 16]
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy import ndimage

R = Path(__file__).resolve().parents[1]
HOUSE = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou")
SPEC = "/data0/congcong/code/GS/mcgs_noise_wt/docs/superpowers/specs/2026-09-16-jyz-af16-noise-robustness-design.md"
SEED, MULT = 20260916, 0.5
TWO_X = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_noise1x_v3_jiangyizhou")   # nested check: the 1 sigma0 level (same draws)
RATIO = 2.0   # its multiplier / this one
ORDER = ("T1", "FLAIR")

ap = argparse.ArgumentParser()
ap.add_argument("--contrasts", default="T1,FLAIR")
ap.add_argument("--threads", type=int, default=16)
a = ap.parse_args()
torch.set_num_threads(a.threads)


def log(*x):
    print(*x, flush=True)


def ifft3c(x):
    d = (-3, -2, -1)
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(x, dim=d), dim=d, norm="ortho"), dim=d)


def sense_adj(y, S):
    """sum_c conj(S_c) ifft3c(y_c), coil by coil, double precision"""
    acc = torch.zeros(y.shape[1:], dtype=torch.complex128)
    for i in range(y.shape[0]):
        acc += S[i].conj() * ifft3c(torch.from_numpy(y[i]).to(torch.complex128))
    return acc


def md5(p, bs=1 << 24):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def air_mask(mag):
    """mag in gauge units (P99.9 = 1)"""
    head = ndimage.binary_closing(mag > 0.05, iterations=3)
    air = ~ndimage.binary_dilation(head, iterations=12)
    keep = np.zeros_like(air)
    keep[8:-8, 8:-8, 8:-8] = True
    return air & keep


def noise_std(v, air):
    return float(np.sqrt(v.real[air].var() + v.imag[air].var()))


out_json = R / "logs/make_noisy_kspace.json"
report = json.load(open(out_json)) if out_json.exists() else {"spec": SPEC, "seed": SEED, "multiplier": MULT,
                                                              "contrasts": {}}
children = dict(zip(ORDER, np.random.SeedSequence(SEED).spawn(len(ORDER))))
failed = False
for c in [x for x in a.contrasts.split(",") if x]:
    dst, img_out = R / f"data/meas_{c}.h5", R / f"recon/fullnoisy_{c}_complex.npy"
    for p in (dst, img_out):
        if p.exists():
            sys.exit(f"refusing to overwrite {p}")
    t0 = time.time()
    src = R / f"data/ksp_{c}.h5"
    with h5py.File(src, "r") as f:
        chunks = f["kspace"].chunks
        y = f["kspace"][:]                                      # (26, 256, 256, 176) complex64
    with h5py.File(R / f"data/csm_espirit_{c}.h5", "r") as f:
        S = torch.from_numpy(f["map"][:]).to(torch.complex128)
    log(f"[{c}] loaded {y.shape} {y.dtype} chunks={chunks} ({time.time() - t0:.0f} s)")

    gt = sense_adj(y, S)
    scale = float(np.quantile(gt.abs().numpy().reshape(-1), 0.999))
    gtn = (gt / scale).numpy()
    del gt
    air = air_mask(np.abs(gtn))
    sigma0_img = noise_std(gtn, air)
    support = np.abs(y).max(axis=0) > 0                        # (X, Y, Z): any coil non-zero
    plane = support.any(axis=0)
    if not np.array_equal(support, np.broadcast_to(plane, support.shape)):
        sys.exit(f"[{c}] the acquired support is not constant along the readout")
    f_acq = float(plane.mean())
    sigma0_k = sigma0_img / np.sqrt(f_acq)
    sigma = MULT * sigma0_k
    sigma_raw = sigma * scale
    log(f"[{c}] scale {scale:.6e}  f_acq {f_acq:.6f} ({int(plane.sum())} lines)  sigma0_img {sigma0_img:.6f}  "
        f"sigma0_k {sigma0_k:.6f}  sigma {sigma:.6f} (raw {sigma_raw:.6e})")

    rng = np.random.Generator(np.random.PCG64(children[c]))
    iy, iz = np.nonzero(plane)
    meas = np.empty_like(y)
    for i in range(y.shape[0]):
        n = (rng.standard_normal((y.shape[1], iy.size)) + 1j * rng.standard_normal((y.shape[1], iy.size))) \
            * (sigma_raw / np.sqrt(2.0))
        yi = y[i].astype(np.complex128)
        yi[:, iy, iz] += n
        meas[i] = yi.astype(np.complex64)
    tmp = dst.with_name(dst.name + ".partial")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("kspace", data=meas, chunks=chunks)
        f.attrs.update(dict(source=str(src.resolve()), seed=SEED, child=ORDER.index(c), contrast=c, multiplier=MULT,
                            sigma_gauge=sigma, sigma_raw=sigma_raw, scale=scale, spec=SPEC))
    tmp.rename(dst)
    del meas
    log(f"[{c}] wrote {dst} ({time.time() - t0:.0f} s)")

    with h5py.File(dst, "r") as f:                             # every check reads what is on disk
        mz = f["kspace"][:]
    checks = {}
    outside = ~support
    exact = all(np.array_equal(mz[i][outside], y[i][outside]) for i in range(y.shape[0]))
    checks["support_exact"] = dict(value=bool(exact), limit=True, ok=bool(exact))
    worst_z, worst_std = 0.0, 0.0
    for i in range(y.shape[0]):
        d = mz[i][:, iy, iz].astype(np.complex128) - y[i][:, iy, iz].astype(np.complex128)
        se = float(sigma_raw / np.sqrt(2.0) / np.sqrt(d.size))
        worst_z = max(worst_z, float(abs(d.real.mean())) / se, float(abs(d.imag.mean())) / se)
        for part in (d.real, d.imag):
            worst_std = max(worst_std, abs(float(part.std()) / float(sigma_raw / np.sqrt(2.0)) - 1.0))
    checks["mean_z"] = dict(value=worst_z, limit=4.0, ok=worst_z <= 4.0)
    checks["std_rel"] = dict(value=worst_std, limit=0.005, ok=worst_std <= 0.005)
    noisy = (sense_adj(mz, S) / scale).numpy()
    air_noisy = noise_std(noisy, air)
    expected = float(np.sqrt(sigma0_img ** 2 + sigma ** 2 * f_acq))
    rel = abs(air_noisy / expected - 1.0)
    checks["air_noise_rel"] = dict(value=rel, limit=0.02, ok=rel <= 0.02)
    house_gt = np.load(HOUSE / f"recon/gt_{c}_complex.npy")
    gdiff = float(np.abs(gtn - house_gt).max())
    checks["gt_vs_house"] = dict(value=gdiff, limit=1e-4, ok=gdiff <= 1e-4)
    house_scale = float(json.load(open(HOUSE / f"logs/classical_{c}.json"))["scale_p999"])
    srel = abs(scale - house_scale) / house_scale
    checks["scale_vs_house"] = dict(value=srel, limit=1e-5, ok=srel <= 1e-5)
    # same seed and draws as the 1 sigma0 level: n_ref = RATIO n_this before single-precision rounding
    with h5py.File(TWO_X / f"data/meas_{c}.h5", "r") as f:
        m2 = f["kspace"][:]
    sigma_raw_2x = RATIO * float(sigma_raw)
    worst = 0.0
    for i in range(y.shape[0]):
        yc = y[i][:, iy, iz].astype(np.complex128)
        d1 = mz[i][:, iy, iz].astype(np.complex128) - yc
        d2 = m2[i][:, iy, iz].astype(np.complex128) - yc
        bound = 4e-7 * (np.abs(yc) + 10.0 * sigma_raw_2x)
        worst = max(worst, float((np.abs(RATIO * d1 - d2) / bound).max()))
    checks["nested_vs_1x"] = dict(value=worst, limit=1.0, ok=worst <= 1.0)
    del m2
    np.save(img_out, noisy.astype(np.complex64))
    ok = all(v["ok"] for v in checks.values())
    for k, v in checks.items():
        log(f"[{c}]   {'OK ' if v['ok'] else 'BAD'} {k:15s} {v['value']} (limit {v['limit']})")
    report["contrasts"][c] = dict(
        shape=list(y.shape), dtype=str(y.dtype), chunks=list(chunks), support_lines=int(plane.sum()), f_acq=f_acq,
        scale=scale, house_scale=house_scale, sigma0_img=sigma0_img, sigma0_k=float(sigma0_k), sigma=float(sigma),
        sigma_raw=float(sigma_raw), air_noise_noisy=air_noisy, air_noise_expected=expected, air_voxels=int(air.sum()),
        checks=checks, ok=ok, md5_src=md5(src.resolve()), md5_meas=md5(dst), seconds=time.time() - t0)
    json.dump(report, open(out_json, "w"), indent=1)
    if not ok:
        failed = True
        for p in (dst, img_out):
            p.rename(p.with_name(p.name + ".failed"))
        log(f"[{c}] CHECKS FAILED: outputs renamed to *.failed")
    del mz, noisy, y, S
log("MAKE_NOISY_KSPACE " + ("FAILED" if failed else "DONE"))
sys.exit(1 if failed else 0)
