#!/usr/bin/env python3
"""make_mask_af16_vdpois.py -- the house-convention (ESPIRiT utils vdPoisMex) counterpart of make_mask_af16.py.

Why this exists (audit 2026-09-29): the registered CE AF16 mask came from `bart poisson -v`, whose density outside
the 24x24 ACS is flat (~0.06 from normalised radius 0.18 to the edge, one cliff at r 0.10-0.18). The jyz T1+FLAIR
AF16 line used vdPoisMex, whose density decays smoothly (0.53 at r 0.12, 0.33 at 0.18, 0.24 at 0.23, 0.14 at 0.33).
The two "AF16"s are therefore different experiments. This script builds the vdPoisMex mask on the CE (232, 176)
phase-encode plane so the four-arm comparison (ZF, CS-SENSE, GS-single, MC-GS) can be rerun under the house density.

Same point budget as the registered mask: exactly 2552 = 40832 / 16 points (AF 16.00 under the 2026-09-14 whole-plane
definition), a strictly fully sampled 24x24 ACS (vdPoisMex's `calib` only raises the centre to ~0.87, so the block is
forced to 1 here, as the registered mask has it), no elliptical shutter (ellipse = 0, corners sampled, as jyz).

vdPoisMex cannot be seeded, so the raw MATLAB draws are inputs, not something this script regenerates:
    addpath('/data0/congcong/code/ESPIRiT/utils');  % accely 4, calib 24, ellipse 0, pp 1, as jyz prep_02_mask.m
    mask = vdPoisMex(232, 176, 232, 176, accelx, 4, 24, 0, 1); save('vd_ax<accelx>_rep1.mat', 'mask', '-v7');
for accelx in a grid (3.6 ... 4.8). The script forces the ACS on every draw, keeps the draw whose count after that is
the smallest count >= 2552, trims the excess from the outside in (largest normalised radius first, ACS never touched)
and records every candidate in mask_af16.json so the choice is auditable. The chosen raw draw is copied next to the
outputs (`vdpoismex_raw.mat`) so the mask is reproducible from the repository without MATLAB.

  python scripts/make_mask_af16_vdpois.py --raw_dir <dir with vd_ax*.mat>   # writes $RETRO_TOPIC/sim/mask/* and figures/mask_af16.png
"""
import argparse
import glob
import json
import os
import re
import shutil
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois"))
NX, NY, NZ, ACS = 256, 232, 176, 24
TARGET = NY * NZ // 16                                 # 2552
ACCELY, CALIB, ELLIPSE, PP = 4, 24, 0, 1               # jyz prep_02_mask.m settings


def radius():
    u = (np.arange(NY) - NY // 2) / (NY / 2)
    v = (np.arange(NZ) - NZ // 2) / (NZ / 2)
    return np.sqrt(u[:, None] ** 2 + v[None, :] ** 2)


def acs_slice():
    cy, cz = NY // 2, NZ // 2
    return slice(cy - ACS // 2, cy + ACS // 2), slice(cz - ACS // 2, cz + ACS // 2)


def force_acs(m):
    m = m.copy(); sy, sz = acs_slice(); m[sy, sz] = True
    return m


def trim_outside_in(m, target):
    """Drop the sampled points with the largest normalised radius until exactly `target` remain; the ACS is never touched."""
    m = m.copy(); r = radius(); sy, sz = acs_slice()
    protect = np.zeros_like(m); protect[sy, sz] = True
    excess = int(m.sum()) - target
    assert excess >= 0, (int(m.sum()), target)
    if excess:
        cand = np.flatnonzero(m & ~protect)
        order = cand[np.argsort(-r.ravel()[cand], kind="stable")]
        m.ravel()[order[:excess]] = False
    assert int(m.sum()) == target
    return m


def density_profile(m, nb=20):
    r = radius(); edges = np.linspace(0, 1.0, nb + 1); out = []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (r >= a) & (r < b); out.append(float(m[s].mean()) if s.any() else float("nan"))
    return ((edges[:-1] + edges[1:]) / 2).tolist(), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="directory with the MATLAB draws vd_ax<accelx>_rep<n>.mat")
    ap.add_argument("--ref_h5", default="/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16/sim/mask/mask_af16_3d.h5",
                    help="registered mask h5, copied for dtype/layout so the trainer reads the new file identically")
    a = ap.parse_args()
    cands = []
    for fn in sorted(glob.glob(str(Path(a.raw_dir) / "vd_ax*_rep*.mat"))):
        mm = re.search(r"vd_ax([0-9.]+)_rep(\d+)\.mat$", fn)
        raw = np.abs(loadmat(fn)["mask"]) > 0.5
        assert raw.shape == (NY, NZ), (fn, raw.shape)
        forced = force_acs(raw)
        cands.append({"file": fn, "accelx": float(mm.group(1)), "rep": int(mm.group(2)), "nnz_raw": int(raw.sum()),
                      "acs_frac_raw": float(raw[acs_slice()].mean()), "nnz_after_acs": int(forced.sum())})
    ok = [c for c in cands if c["nnz_after_acs"] >= TARGET]
    assert ok, f"no draw reaches {TARGET} points after forcing the ACS: {cands}"
    chosen = min(ok, key=lambda c: (c["nnz_after_acs"], c["accelx"]))
    raw = np.abs(loadmat(chosen["file"])["mask"]) > 0.5
    m = trim_outside_in(force_acs(raw), TARGET)
    sy, sz = acs_slice()
    assert m[sy, sz].all() and int(m.sum()) == TARGET
    (R / "sim/mask").mkdir(parents=True, exist_ok=True); (R / "figures").mkdir(parents=True, exist_ok=True)
    shutil.copy(chosen["file"], R / "sim/mask/vdpoismex_raw.mat")
    np.save(R / "sim/mask/mask_af16_pe.npy", m.astype(np.uint8))
    with h5py.File(a.ref_h5, "r") as f:
        ref_dtype = f["mask"].dtype; ref_shape = f["mask"].shape
    assert ref_shape == (NX, NY, NZ), ref_shape
    m3 = np.broadcast_to(m[None], (NX, NY, NZ)).astype(ref_dtype)
    with h5py.File(R / "sim/mask/mask_af16_3d.h5", "w") as f:
        f.create_dataset("mask", data=m3)
    centres, prof = density_profile(m)
    meta = {"generator": "ESPIRiT utils vdPoisMex (MATLAB, unseedable; raw draw kept as vdpoismex_raw.mat)",
            "call": f"vdPoisMex({NY}, {NZ}, {NY}, {NZ}, accelx, {ACCELY}, {CALIB}, {ELLIPSE}, {PP})",
            "accelx": chosen["accelx"], "accely": ACCELY, "calib": CALIB, "ellipse": ELLIPSE, "pp": PP,
            "plane": [NY, NZ], "target": TARGET, "af": float(m.size / m.sum()), "acs": ACS, "acs_forced_full": True,
            "nnz_raw": chosen["nnz_raw"], "acs_frac_raw": chosen["acs_frac_raw"], "nnz_after_acs": chosen["nnz_after_acs"],
            "trimmed": chosen["nnz_after_acs"] - TARGET, "candidates": cands,
            "density_profile": {"radius_centres": centres, "sampled_fraction": prof}}
    json.dump(meta, open(R / "sim/mask/mask_af16.json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].imshow(m.T, cmap="gray", origin="lower", interpolation="nearest"); ax[0].set_xlabel("ky (A-P)"); ax[0].set_ylabel("kz (L-R)")
    ax[0].set_title(f"PE plane {NY}x{NZ}: {TARGET} points, AF {m.size / m.sum():.2f}, ACS {ACS}x{ACS} (vdPoisMex)")
    ax[1].step(centres, prof, where="mid"); ax[1].set_ylim(0, 1.05); ax[1].set_xlabel("normalised radius"); ax[1].set_ylabel("sampled fraction")
    ax[1].set_title("variable density profile"); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(R / "figures/mask_af16.png", dpi=150)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("candidates", "density_profile")}, indent=1))
    print("density:", " ".join(f"{c:.2f}:{p:.2f}" for c, p in zip(centres, prof)))


if __name__ == "__main__":
    main()
