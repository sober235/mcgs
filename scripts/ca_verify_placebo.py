#!/usr/bin/env python3
"""ca_verify_placebo.py -- check a Stage-3 placebo pair built by ca_flip_flair.py against its source.

Everything the placebo's validity rests on, reproducible from the repo instead of from a log:

  1. support   -- is the flipped k-space support the original's reflection about the DC sample, and
                  is it identical to the original (the condition under which `known_zero` is matched)?
  2. SENSE     -- is the SENSE combination of the flipped pair exactly np.flip of the original's,
                  in COMPLEX (a magnitude-only check would miss a phase error in the CSM flip)?
  3. round-off -- how much junk does the FFT round trip leave in the shuttered samples, i.e. how much
                  the zero re-imposition in flip_pair has to remove?
  4. anatomy   -- NCC to T1 before and after, whole-volume and head-masked; the flip has to break
                  registration, not merely perturb it.
  5. known_zero-- estimate_known_zero on the original and the flipped data, and Z equality. This is
                  the estimator the trainer actually runs (joint_trainer.py:195).

Run: env PYTHONNOUSERSITE=1 PYTHONPATH= python scripts/ca_verify_placebo.py [--axis 0|1]
"""
import argparse
import dataclasses
import os
import sys

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "paired_gs"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from ca_flip_flair import fft3c, ifft3c  # noqa: E402

SRC = "/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/data"
OUT = os.path.join(ROOT, "results_arc/placebo")


def sense(kpath, cpath):
    """SENSE combination over the full k-space, accumulated coil by coil in complex128."""
    with h5py.File(kpath, "r") as fk, h5py.File(cpath, "r") as fc:
        k, s = fk["kspace"], fc["map"]
        g = np.zeros(k.shape[1:], np.complex128)
        for c in range(k.shape[0]):
            g += np.conj(np.asarray(s[c])) * ifft3c(np.asarray(k[c]))
    return g


def support(kpath):
    with h5py.File(kpath, "r") as f:
        d = f["kspace"]
        a = np.zeros(d.shape[1:], np.float32)
        for c in range(d.shape[0]):
            np.maximum(a, np.abs(np.asarray(d[c])).astype(np.float32), out=a)
    return a > 0


def ncc(u, v, m=None):
    u, v = (u[m], v[m]) if m is not None else (u.ravel(), v.ravel())
    u, v = u.astype(np.float64) - u.mean(), v.astype(np.float64) - v.mean()
    return float((u * v).sum() / np.sqrt((u * u).sum() * (v * v).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", type=int, default=0, choices=[0, 1],
                    help="the axis ca_flip_flair.py was run with: 0 = readout (primary), 1 = ky")
    args = ap.parse_args()
    sfx = "" if args.axis == 0 else "_ky"
    kf, cf = f"{OUT}/ksp_FLAIR_flip{sfx}.h5", f"{OUT}/csm_espirit_FLAIR_flip{sfx}.h5"
    rev0 = lambda a: np.roll(np.flip(a, axis=args.axis), 1, axis=args.axis)   # noqa: E731
    print(f"placebo axis {args.axis} ({'readout x' if args.axis == 0 else 'ky'})  {kf}")

    print("\n[1] k-space support", flush=True)
    s0, s1 = support(f"{SRC}/ksp_FLAIR.h5"), support(kf)
    print(f"    nonzero samples orig / flip : {int(s0.sum())} / {int(s1.sum())}")
    print(f"    flip == rev0(orig) exactly  : {bool(np.array_equal(s1, rev0(s0)))}")
    print(f"    flip == orig exactly        : {bool(np.array_equal(s1, s0))}"
          f"   differing samples {int((s1 != s0).sum())}")
    pe0, pe1 = s0[0], s1[0]                                # readout-separable: one x plane suffices
    print(f"    readout-separable orig/flip : {bool(np.all(s0 == pe0[None]))} / {bool(np.all(s1 == pe1[None]))}"
          f"   PE lines {int(pe0.sum())} / {int(pe1.sum())}, differing {int((pe0 != pe1).sum())}")

    print("\n[2] SENSE ground truth, COMPLEX", flush=True)
    g0 = sense(f"{SRC}/ksp_FLAIR.h5", f"{SRC}/csm_espirit_FLAIR.h5")
    g1 = sense(kf, cf)
    want = np.flip(g0, axis=args.axis)
    den = float(np.abs(g0).max())
    print(f"    max|GT_flip - flip(GT)| / max|GT| : {np.abs(g1 - want).max() / den:.6e}")
    print(f"    rel L2 |GT_flip - flip(GT)|       : {np.linalg.norm(g1 - want) / np.linalg.norm(want):.6e}")
    print(f"    max|GT_flip - GT|      / max|GT|  : {np.abs(g1 - g0).max() / den:.6f}   (the flip is not a no-op)")

    print("\n[3] round-off the zero re-imposition removes (coil 0)", flush=True)
    with h5py.File(f"{SRC}/ksp_FLAIR.h5", "r") as f:
        k0 = np.asarray(f["kspace"][0])[..., None]
    raw = fft3c(np.flip(ifft3c(k0), axis=args.axis))       # before flip_pair re-imposes the zeros
    zero = rev0(k0 == 0)
    print(f"    max |k| left in the shuttered samples : {np.abs(raw[zero]).max():.3e} absolute, "
          f"{np.abs(raw[zero]).max() / np.abs(raw).max():.3e} relative to max|k|")
    del raw, k0, zero

    print("\n[4] registration to T1", flush=True)
    t1 = np.abs(sense(f"{SRC}/ksp_T1.h5", f"{SRC}/csm_espirit_T1.h5"))
    a0, a1 = np.abs(g0), np.abs(g1)
    head = t1 > 0.05 * np.quantile(t1, 0.999)              # the scorer's head rule on T1
    print(f"    NCC(|T1|,|FLAIR|)      whole {ncc(t1, a0):+.4f}   head-masked {ncc(t1, a0, head):+.4f}")
    print(f"    NCC(|T1|,|FLAIR_flip|) whole {ncc(t1, a1):+.4f}   head-masked {ncc(t1, a1, head):+.4f}")
    del t1, a0, a1, g0, g1, want, head

    print("\n[5] known_zero estimator (the one joint_trainer.py runs)", flush=True)
    import torch
    from data.known_zero import estimate_known_zero
    with h5py.File(f"{SRC}/mask_af16_3d.h5", "r") as h:
        m = torch.from_numpy((h["mask"][()] > 0).astype(np.float32))
    Zs = {}
    for tag, path in (("orig", f"{SRC}/ksp_FLAIR.h5"), ("flip", kf)):
        with h5py.File(path, "r") as f:
            y = torch.from_numpy(f["kspace"][()])
        Z, info = estimate_known_zero(y * m, m, 0)         # readout_axis 0, mask applied as the loader does
        d = dataclasses.asdict(info)
        print(f"    {tag:4s}: cuts={d['cuts']} ellipse={d['ellipse']} frac_z={d['frac_z']:.6f} "
              f"zero_lines={d['n_zero_lines']}/{d['n_sampled_lines']} explained={d['n_zero_explained']} "
              f"nonzero_in_z={d['n_nonzero_in_z']} dc={d['dc']}")
        Zs[tag] = Z.numpy()
        del y, Z
    print(f"    Z identical: {bool(np.array_equal(Zs['orig'], Zs['flip']))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
