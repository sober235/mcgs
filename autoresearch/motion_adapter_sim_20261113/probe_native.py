#!/usr/bin/env python3
"""READ-ONLY probe in the TRAINER's frame (the topic's ksp_T1.h5 / ksp_T2.h5 with its ESPIRiT maps), CPU.

1. Orientation of axis 0: head cross-section area along the axis. The end where the area runs out inside the FOV is
   the vertex; the end cut by the FOV edge is the neck.
2. Brain mask by lesions.brain_mask (kept: the superior part), drawn on three planes for a visual check.
3. Native T1 -> T2 offset: similarity (NGF, NMI) of T1 against the T2 shifted by s voxels along one axis, inside the
   brain mask (method of results/motion_adapter_hyw_align/scripts/hyw_align_probe.py). The peak s* is the shift that
   aligns T2 with T1, so the T2 anatomy sits at -s* relative to T1: canonical -> T2 displacement = -s*.

Writes probe_native.json and probe_native_brain.png to MCGS_ARTIFACT_DIR and ROOT/probe/; nothing else.
"""
import json
import os
from pathlib import Path

import numpy as np
import scipy.fft as sfft
from scipy import ndimage as ndi

import lesions
import simulate
from common import HYW, ROOT

GRID = np.round(np.arange(-1.5, 1.5001, 0.1), 2)
log = lambda *a: print(*a, flush=True)


def fshift(v, s, axis):
    n = v.shape[axis]
    ph = np.exp(-2j * np.pi * np.fft.fftfreq(n) * s)
    shp = [1] * v.ndim
    shp[axis] = n
    return np.real(sfft.ifft(sfft.fft(v, axis=axis, workers=16) * ph.reshape(shp), axis=axis, workers=16))


def ent(q):
    q = q[q > 0]
    return -np.sum(q * np.log(q))


def curves(a, b, m, axis, bins=64):
    qa, qb = np.quantile(a[m], 0.999), np.quantile(b[m], 0.999)
    x = np.clip(a[m] / qa, 0, 1)
    ga = [g[m] for g in np.gradient(a)]
    na = sum(g ** 2 for g in ga)
    na = na + (1e-2 * np.median(np.sqrt(na))) ** 2
    nmi, ngf = [], []
    for s in GRID:
        bs = fshift(b, s, axis)
        y = np.clip(bs[m] / qb, 0, 1)
        h, _, _ = np.histogram2d(x, y, bins=bins, range=[[0, 1], [0, 1]])
        p = h / h.sum()
        gb = [g[m] for g in np.gradient(bs)]
        nb = sum(g ** 2 for g in gb)
        nb = nb + (1e-2 * np.median(np.sqrt(nb))) ** 2
        dot = sum(u * w for u, w in zip(ga, gb))
        nmi.append(float((ent(p.sum(1)) + ent(p.sum(0))) / ent(p)))
        ngf.append(float(np.mean(dot ** 2 / (na * nb))))
    return nmi, ngf


def peak(y):
    i = int(np.argmax(y))
    if 0 < i < len(y) - 1:
        y0, y1, y2 = y[i - 1], y[i], y[i + 1]
        den = y0 - 2 * y1 + y2
        return round(float(GRID[i] + (0.5 * (y0 - y2) / den if den else 0.0) * (GRID[1] - GRID[0])), 3)
    return round(float(GRID[i]), 3)


def main():
    mags = {}
    for c in ("T1", "T2"):
        ksp = simulate._read(HYW / f"ksp_{c}.h5", ("kspace", "ksp"))
        csm = simulate._read(HYW / f"csm_espirit_{c}.h5", ("map", "csm"))
        mags[c] = np.abs(simulate.combine(ksp, csm, 16)).astype(np.float64)
        del ksp, csm
    t1, t2 = mags["T1"], mags["T2"]
    head = lesions.head_mask(t1)
    area = head.sum((1, 2))
    nz = np.flatnonzero(area > 200)
    res = dict(shape=list(t1.shape), head_area_axis0_every16=area[::16].tolist(),
               head_first_last_index=[int(nz[0]), int(nz[-1])],
               axis0="index 0 inferior (neck cut by the FOV), vertex at the high end"
               if area[:8].mean() > area[-8:].mean() else "index 0 superior")
    brain = lesions.brain_mask(t1, erode=simulate.FULL["brain_erode"], cut_below=simulate.FULL["brain_cut"])
    idx = np.argwhere(brain)
    res.update(brain_voxels=int(brain.sum()), brain_min=idx.min(0).tolist(), brain_max=idx.max(0).tolist(),
               brain_centroid=idx.mean(0).round(1).tolist())
    lo, hi = np.maximum(idx.min(0) - 6, 0), np.minimum(idx.max(0) + 7, brain.shape)
    cr = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    res["peaks_shift_that_aligns_T2_to_T1"], res["canonical_to_T2_mm"] = {}, {}
    for ax in range(3):
        nmi, ngf = curves(t1[cr], t2[cr], brain[cr], ax)
        res["peaks_shift_that_aligns_T2_to_T1"][f"axis{ax}"] = dict(nmi=peak(nmi), ngf=peak(ngf))
        res["canonical_to_T2_mm"][f"axis{ax}"] = dict(nmi=-peak(nmi), ngf=-peak(ngf))
        log(f"axis {ax}: aligning shift NGF {peak(ngf)} NMI {peak(nmi)}")
    log(json.dumps(res, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = [int(round(x)) for x in idx.mean(0)]
    fig, axs = plt.subplots(2, 3, figsize=(13, 9), facecolor="black")
    for row, (vol, name) in enumerate(((t1, "T1"), (t2, "T2"))):
        vmax = float(np.quantile(vol, 0.999))
        planes = ((vol[c[0]], brain[c[0]], f"transverse, axis-0 index {c[0]}", False),
                  (vol[:, c[1]], brain[:, c[1]], f"coronal, axis-1 index {c[1]}", True),
                  (vol[:, :, c[2]], brain[:, :, c[2]], f"sagittal, axis-2 index {c[2]}", True))
        for a, (img, m, title, flip) in zip(axs[row], planes):
            if flip:
                img, m = img[::-1], m[::-1]                 # vertex up
            a.imshow(img, cmap="gray", vmin=0, vmax=vmax)
            a.contour(m.astype(float), levels=[0.5], colors=["#ffd400"], linewidths=0.8)
            a.set_title(f"{name}: {title}", color="white", fontsize=10)
            a.axis("off")
    fig.suptitle("brain mask (yellow) used for the simulation limits, lesion sites and motion errors", color="white")
    fig.tight_layout()
    out = ROOT / "probe"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "probe_native_brain.png", dpi=110, facecolor="black")
    (out / "probe_native.json").write_text(json.dumps(res, indent=1))
    art = os.environ.get("MCGS_ARTIFACT_DIR")
    if art:
        for f in ("probe_native_brain.png", "probe_native.json"):
            (Path(art) / f).write_bytes((out / f).read_bytes())


if __name__ == "__main__":
    main()
