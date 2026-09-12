#!/usr/bin/env python
"""Figures and the vendor check for one stage-1 contrast.

    ./py fig_mtp_grappa.py --contrast FA16_TE3.25

Writes figures/<contrast>_kspace.png (log-magnitude coil-RSS k-space before and after GRAPPA, PE x SPE plane
at the readout centre and RO x PE plane at the SPE centre, plus the three masks), figures/<contrast>_image.png
(zero-filled RSS, GRAPPA RSS, vendor magnitude; transverse, coronal, sagittal) and logs/vendor_check_<contrast>.json
(correlations against the vendor volume after the known axis mapping). Figures are mirrored to ~/figs/peds_tsc_age16.
"""
import argparse, json, os, shutil

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TOPIC = "/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_3D"
FIGS = "/home/congcongliu/figs/peds_tsc_age16"
VOXEL = (0.7, 0.7, 1.35)


def to_vendor_frame(vol: np.ndarray, pe_offset: int) -> np.ndarray:
    """(RO, PE, SPE) -> vendor (Z=80, rows=RO, cols=PE 300): central 80 partitions, SPE reversed, PE cropped."""
    v = vol[:, pe_offset:pe_offset + 300, 12:92][:, :, ::-1]
    return np.transpose(v, (2, 0, 1))


def corr(a, b):
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-30))


def flatten(vol, sigma_mm=15.0):
    """Divide by a heavily smoothed copy: removes the coil-sensitivity shading that an RSS image keeps and the
    vendor's normalised image does not, so the correlation compares anatomy rather than brightness profiles."""
    from scipy.ndimage import gaussian_filter
    sig = (sigma_mm / VOXEL[2], sigma_mm / VOXEL[0], sigma_mm / VOXEL[1])
    low = gaussian_filter(vol.astype(np.float32), sig)
    return vol / (low + 0.05 * low.max())


def vendor_check(rss_g, rss_z, vendor):
    best = None
    for off in range(0, 5):
        r = corr(to_vendor_frame(rss_g, off), vendor)
        if best is None or r > best[1]:
            best = (off, r)
    off = best[0]
    g = to_vendor_frame(rss_g, off); z = to_vendor_frame(rss_z, off)
    per_slice = [corr(g[i], vendor[i]) for i in range(vendor.shape[0])]
    gf, zf, vf = flatten(g), flatten(z), flatten(vendor)
    grad = lambda v: np.sqrt(sum(np.gradient(v, axis=a) ** 2 for a in range(3)))
    return dict(pe_offset=off, corr_grappa=best[1], corr_zerofilled=corr(z, vendor),
                corr_grappa_per_slice_median=float(np.median(per_slice)), corr_grappa_per_slice_min=float(np.min(per_slice)),
                corr_flat_grappa=corr(gf, vf), corr_flat_zerofilled=corr(zf, vf),
                corr_gradient_grappa=corr(grad(gf), grad(vf)), corr_gradient_zerofilled=corr(zf if False else grad(zf), grad(vf)))


def kspace_figure(kmag_before, kmag_after, masks, name, path):
    X, Y, Z = kmag_before.shape
    x0, z0 = X // 2, Z // 2
    lo = np.log10(kmag_after.max() * 1e-6)
    hi = np.log10(kmag_after.max())
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    for row, (km, lab) in enumerate(((kmag_before, "acquired lines only"), (kmag_after, "after GRAPPA"))):
        im = ax[row, 0].imshow(np.log10(km[x0] + 1e-30).T, cmap="gray", vmin=lo, vmax=hi, origin="lower", aspect="auto")
        ax[row, 0].set_title(f"{lab}: PE x SPE plane at RO centre (log10 coil-RSS |k|)"); ax[row, 0].set_xlabel("PE index"); ax[row, 0].set_ylabel("SPE index")
        ax[row, 1].imshow(np.log10(km[:, :, z0] + 1e-30).T, cmap="gray", vmin=lo, vmax=hi, origin="lower", aspect="auto")
        ax[row, 1].set_title(f"{lab}: RO x PE plane at SPE centre"); ax[row, 1].set_xlabel("RO index"); ax[row, 1].set_ylabel("PE index")
    rgb = np.zeros(masks["m_acq"].shape + (3,), np.float32)
    rgb[masks["m_acq"]] = (1, 1, 1); rgb[masks["m_grappa"]] = (1.0, 0.55, 0.1); rgb[masks["m_zero"]] = (0.12, 0.12, 0.2)
    ax[0, 2].imshow(np.transpose(rgb, (1, 0, 2)), origin="lower", aspect="auto"); ax[0, 2].set_title("mask: white acquired, orange GRAPPA, dark left empty (partial Fourier, corners)")
    ax[0, 2].set_xlabel("PE index"); ax[0, 2].set_ylabel("SPE index")
    prof_b = kmag_before[x0].sum(1); prof_a = kmag_after[x0].sum(1)
    ax[1, 2].semilogy(prof_b + 1e-30, label="acquired only"); ax[1, 2].semilogy(prof_a + 1e-30, label="after GRAPPA", alpha=0.8)
    ax[1, 2].set_title("|k| summed over SPE along PE (RO centre)"); ax[1, 2].set_xlabel("PE index"); ax[1, 2].legend()
    n_acq, n_g, n_z = int(masks["m_acq"].sum()), int(masks["m_grappa"].sum()), int(masks["m_zero"].sum())
    fig.suptitle(f"{name}: MTP Cartesian 3D k-space, {n_acq} acquired + {n_g} GRAPPA-filled + {n_z} empty of {Y}x{Z} (PE x SPE)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def image_figure(rss_g, rss_z, vendor, off, name, path):
    g = to_vendor_frame(rss_g, off); z = to_vendor_frame(rss_z, off); v = vendor
    cols = [("zero-filled RSS", z), ("GRAPPA RSS", g), ("vendor (uMR)", v)]
    Zn, R, Cn = g.shape
    iz, ir, ic = Zn // 2, R // 2, Cn // 2
    fig, ax = plt.subplots(3, 3, figsize=(13.5, 13))
    for j, (lab, vol) in enumerate(cols):
        vmax = np.percentile(vol, 99.7)
        ax[0, j].imshow(vol[iz], cmap="gray", vmin=0, vmax=vmax); ax[0, j].set_title(f"{lab}, transverse {iz}/{Zn}")
        ax[1, j].imshow(vol[:, ir, :], cmap="gray", vmin=0, vmax=vmax, aspect=VOXEL[2] / VOXEL[1]); ax[1, j].set_title(f"coronal (row {ir})")
        ax[2, j].imshow(vol[:, :, ic], cmap="gray", vmin=0, vmax=vmax, aspect=VOXEL[2] / VOXEL[0]); ax[2, j].set_title(f"sagittal (col {ic})")
        for i in range(3): ax[i, j].axis("off")
    fig.suptitle(f"{name}: prospective 2x (PPA) + partial Fourier, 0.7 x 0.7 x 1.35 mm, 32 channels")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--contrast", required=True); args = ap.parse_args()
    name = args.contrast
    d = os.path.join(TOPIC, "data", name)
    masks = dict(np.load(os.path.join(d, "masks.npz")))
    rss_g = np.load(os.path.join(d, "rss_grappa.npy")); rss_z = np.load(os.path.join(d, "rss_zf.npy"))
    with h5py.File(os.path.join(d, "kspace_grappa.h5"), "r") as f:
        k = f["kspace"]
        kmag = np.zeros(k.shape[1:], np.float64)
        for c in range(k.shape[0]):
            kmag += np.abs(k[c]).astype(np.float64) ** 2
    kmag_after = np.sqrt(kmag).astype(np.float32)
    kmag_before = kmag_after * masks["m_acq"][None].astype(np.float32)
    vendor = np.load(os.path.join(TOPIC, "data", "vendor", f"M_{name}.npy")).astype(np.float32)
    chk = vendor_check(rss_g, rss_z, vendor)
    os.makedirs(os.path.join(TOPIC, "figures"), exist_ok=True); os.makedirs(os.path.join(TOPIC, "logs"), exist_ok=True); os.makedirs(FIGS, exist_ok=True)
    json.dump(chk, open(os.path.join(TOPIC, "logs", f"vendor_check_{name}.json"), "w"), indent=1)
    print(name, json.dumps(chk))
    p1 = os.path.join(TOPIC, "figures", f"{name}_kspace.png"); kspace_figure(kmag_before, kmag_after, masks, name, p1)
    p2 = os.path.join(TOPIC, "figures", f"{name}_image.png"); image_figure(rss_g, rss_z, vendor, chk["pe_offset"], name, p2)
    for p in (p1, p2):
        shutil.copy2(p, os.path.join(FIGS, os.path.basename(p)))
    print("figures:", p1, p2)


if __name__ == "__main__":
    main()
