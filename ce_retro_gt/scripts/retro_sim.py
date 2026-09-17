#!/usr/bin/env python3
"""retro_sim.py -- build a retrospective pre/post-contrast k-space pair with ground truth (2026-09-16).

The prospective UIH pair has no fully sampled reference, so nothing there can be scored against truth. UCSF-PDGM
gives co-registered T1 and T1c at 1 mm isotropic with a tumour segmentation that marks both an enhancing rim
(label 4) and a necrotic core that does not enhance (label 1) -- exactly the two things the prospective pair can
only ask qualitatively. This turns one case into the same kind of acquisition as the UIH pair:

  object      T1 / T1c in white-matter units (median of tumour-free white matter = 1), so enhancement is measured
              in the same units as the prospective work
  phase       a smooth synthetic phase per scan, so the arms cannot exploit real-valued images
  coils       --coils smooth complex sensitivities, sum-of-squares 1 inside the head, the same maps for both scans
  sampling    readout axis 0 fully sampled; the phase-encode plane gets the UIH recipe -- fully sampled --acs x --acs
              centre, elliptical shutter, uniform random density elsewhere -- at --af, and one mask for both scans
  noise       complex Gaussian in k-space, scaled to --snr in white matter of the fully sampled SENSE image

Writes, per case, the layout the reconstruction scripts already read (ksp_full.npy, csm.npy, mask2d.npy) plus
gt.npz (both ground-truth complex volumes, the masks and the labels) and meta.json.

  python scripts/retro_sim.py --case UCSF-PDGM-0004 [--coils 12] [--af 5.33] [--snr 30] [--out sim]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

T = Path(__file__).resolve().parents[1]
PDGM = Path("/data2/congcong/data/FM_data/UCSF-PDGM_lh/PKG - UCSF-PDGM Version 5/UCSF-PDGM-v5")
ap = argparse.ArgumentParser()
ap.add_argument("--case", default="UCSF-PDGM-0004")
ap.add_argument("--coils", type=int, default=12)
ap.add_argument("--af", type=float, default=5.33, help="phase-encode points / sampled points, the UIH definition")
ap.add_argument("--acs", type=int, default=24)
ap.add_argument("--snr", type=float, default=30.0, help="white-matter SNR of the fully sampled SENSE image")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="sim")
ap.add_argument("--no_gs_h5", action="store_true", help="skip the .h5 copies the GS trainer reads (3.2 GB per case)")
a = ap.parse_args()
OUT = T / a.out / a.case
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(a.seed)


def fft3c(x):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=(-3, -2, -1)), axes=(-3, -2, -1), norm="ortho"),
                           axes=(-3, -2, -1))


def ifft3c(x):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x, axes=(-3, -2, -1)), axes=(-3, -2, -1), norm="ortho"),
                           axes=(-3, -2, -1))


def smooth_field(shape, seed, n=5):
    """a smooth random field on [-1, 1], built from an n^3 grid of normals."""
    g = np.random.default_rng(seed).normal(size=(n, n, n))
    f = ndi.zoom(g, [s / n for s in shape], order=3)[:shape[0], :shape[1], :shape[2]]
    f = np.pad(f, [(0, s - f.shape[i]) for i, s in enumerate(shape)], mode="edge")
    return f / np.abs(f).max()


# ---- object ---------------------------------------------------------------------------------------------------
import nibabel as nib  # noqa: E402

d = PDGM / f"{a.case}_nifti"
pre = np.asarray(nib.load(d / f"{a.case}_T1.nii.gz").dataobj).astype(np.float32)
post = np.asarray(nib.load(d / f"{a.case}_T1c.nii.gz").dataobj).astype(np.float32)
tum = np.asarray(nib.load(d / f"{a.case}_tumor_segmentation.nii.gz").dataobj).astype(np.uint8)
brain = np.asarray(nib.load(d / f"{a.case}_brain_parenchyma_segmentation.nii.gz").dataobj) > 0.5
healthy = brain & (tum == 0)
wm = healthy & (pre > np.percentile(pre[healthy], 60))          # white matter is the bright tissue on T1
pre, post = pre / np.median(pre[wm]), post / np.median(post[wm])
G = pre.shape
print(f"{a.case}: grid {G}, white matter {int(wm.sum())} vox, tumour labels "
      f"{ {int(v): int((tum == v).sum()) for v in np.unique(tum) if v} }")

xs = {}
for name, vol, sd in (("pre", pre, 11), ("post", post, 12)):
    xs[name] = (vol * np.exp(1j * 1.2 * smooth_field(G, sd))).astype(np.complex64)   # smooth phase, about +-1.2 rad

# ---- coils: loops on a cylinder around the head, so the maps actually differ from each other -------------------
zz, yy, xx = np.meshgrid(*[np.arange(s) - s / 2 for s in G], indexing="ij")
R = 0.62 * max(G[1], G[2])                                       # loop radius, a little outside the head
rings = max(1, a.coils // 6)
maps = []
for c in range(a.coils):
    ring, k = divmod(c, max(1, a.coils // rings))
    th = 2 * np.pi * k / max(1, a.coils // rings) + 0.4 * ring
    p = np.array([(ring - (rings - 1) / 2) * G[0] / max(rings, 2), R * np.cos(th), R * np.sin(th)])
    d2 = (zz - p[0]) ** 2 + (yy - p[1]) ** 2 + (xx - p[2]) ** 2
    amp = 1.0 / (d2 + (0.25 * R) ** 2)                           # near-field fall-off of a small loop
    ph = np.arctan2(xx - p[2], yy - p[1]) + 0.004 * np.sqrt(d2)  # a phase that also varies in space
    maps.append((amp * np.exp(1j * ph)).astype(np.complex64))
S = np.stack(maps)
sos = np.sqrt((np.abs(S) ** 2).sum(0))
S = (S / np.maximum(sos, 1e-12)).astype(np.complex64)            # sum of squares 1 everywhere
cc = np.abs(np.corrcoef(S[:, brain].real))                       # how different the maps are from each other
print(f"coils {a.coils} on {rings} ring(s), |S| range {np.abs(S).min():.3f}-{np.abs(S).max():.3f}, "
      f"mean |correlation| between maps {(cc.sum() - a.coils) / (a.coils * (a.coils - 1)):.3f}")

# ---- sampling: the UIH recipe on the phase-encode plane --------------------------------------------------------
n1, n2 = G[1], G[2]
u = (np.arange(n1) - n1 // 2) / (n1 / 2)
v = (np.arange(n2) - n2 // 2) / (n2 / 2)
rho = np.sqrt(u[:, None] ** 2 + v[None, :] ** 2)
acs = np.zeros((n1, n2), bool)
acs[n1 // 2 - a.acs // 2:n1 // 2 + a.acs // 2, n2 // 2 - a.acs // 2:n2 // 2 + a.acs // 2] = True
inside = rho <= 1.0
target = int(round(n1 * n2 / a.af))                              # AF = whole plane / sampled points
p = (target - acs.sum()) / max((inside & ~acs).sum(), 1)
mask2d = acs | (inside & ~acs & (rng.random((n1, n2)) < p))
af = n1 * n2 / mask2d.sum()
print(f"mask {mask2d.sum()} of {n1 * n2} phase-encode points, AF {af:.2f} (density {p:.3f} inside the shutter)")
M = np.broadcast_to(mask2d[None, :, :], G).astype(np.float32)

# ---- k-space with noise ---------------------------------------------------------------------------------------
sigma = 1.0 / a.snr
meta = dict(case=a.case, grid=list(G), coils=a.coils, af=float(af), acs=a.acs, snr_target=a.snr,
            sigma=float(sigma), seed=a.seed, pe_points=int(mask2d.sum()), pe_total=int(n1 * n2),
            note="object in white-matter units; readout axis 0 fully sampled; one mask for both scans")
for name in ("pre", "post"):
    w = OUT / f"work_{name}"
    w.mkdir(exist_ok=True)
    y = fft3c(S * xs[name][None])
    y = y + sigma * (rng.normal(size=y.shape) + 1j * rng.normal(size=y.shape)).astype(np.complex64) / np.sqrt(2)
    y = (y * M[None]).astype(np.complex64)
    np.save(w / "ksp_full.npy", y)
    np.save(w / "csm.npy", S)
    np.save(w / "mask2d.npy", M)
    zf = (np.conj(S) * ifft3c(y)).sum(0)                          # adjoint, for the record
    full = (np.conj(S) * ifft3c(fft3c(S * xs[name][None]) + sigma * (rng.normal(size=y.shape) + 1j * rng.normal(size=y.shape)).astype(np.complex64) / np.sqrt(2))).sum(0)
    # SNR = mean |signal| over white matter / the complex noise standard deviation of the same voxels
    meta[f"{name}_snr_full"] = float(np.abs(xs[name][wm]).mean() / np.std((full - xs[name])[wm]))
    meta[f"{name}_zf_kres"] = float(np.linalg.norm((fft3c(S * zf[None]) * M[None] - y).ravel()) / np.linalg.norm(y.ravel()))
    print(f"  {name}: wrote {w}, fully sampled SENSE SNR in WM {meta[f'{name}_snr_full']:.1f}")
if not a.no_gs_h5:
    # the GS trainer reads .mat/.h5 in the CE layout: (coils, readout, PE2, PE1), complex as a (real, imag) compound,
    # and a mask of the same 4-D shape
    import h5py
    h5py.get_config().complex_names = ("real", "imag")
    for name in ("pre", "post"):
        w = OUT / f"work_{name}"
        g = OUT / f"gs_{name}"
        g.mkdir(exist_ok=True)
        y = np.load(w / "ksp_full.npy", mmap_mode="r")
        for key, arr in (("ksp", np.ascontiguousarray(np.transpose(np.asarray(y), (0, 1, 3, 2)))),
                         ("csm", np.ascontiguousarray(np.transpose(S, (0, 1, 3, 2)))),
                         ("mask", np.ascontiguousarray(np.transpose(np.broadcast_to(M[None], S.shape), (0, 1, 3, 2)).astype(np.uint8)))):
            with h5py.File(g / f"{key}.h5", "w") as f:
                f.create_dataset(key, data=arr)
            del arr
        print(f"  {name}: wrote {g} (GS inputs)")
np.savez_compressed(OUT / "gt.npz", pre=xs["pre"], post=xs["post"], tumor=tum, brain=brain, wm=wm, mask2d=mask2d)
json.dump(meta, open(OUT / "meta.json", "w"), indent=1)
print(json.dumps(meta, indent=1))
