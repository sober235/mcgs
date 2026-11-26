"""T1 / PD / T2 at the same layers and the ratio maps T1/PD, T2/PD: UIH 3D knee, subject S01 (read-only, CPU).

Images: centred inverse 3D FFT, RSS over coils; T1 and T2 moved onto PD by the integer shift that maximises the
cross-correlation of gradient magnitudes; each contrast divided by its median inside the mask (PD above 10 % of
its 99.5th percentile, eroded by 2 voxels). Ratio maps use 3x3x3 box-smoothed images and are drawn inside the
un-eroded mask only, on a log2 scale centred at 1; the coil shading is common to the three scans and cancels.
Row 1: sagittal layer (readout index 88). Row 2: section through the readout axis (n3 index 96, coronal).
Arrays as stored, no reorientation. Subjects numbered S01..S16 in sorted path order; no names printed.
"""
import glob
import hashlib
import json
import os
import shutil
import sys

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import fft as F, ndimage as nd  # noqa: E402

ROOT = "/data2/congcong/data/3D/3D_data20261112/kspace_csm/knee"
# usage: fig_contrast_compare.py [SUBJECT [X_LAYER N3_SECTION]]; S01 keeps its original layers (88, 96),
# other subjects default to the centroid of the knee mask along each axis.
SUBJECT = int(sys.argv[1]) if len(sys.argv) > 1 else 1
X_LAYER, N3_SECTION = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else ((88, 96) if SUBJECT == 1 else (None, None))
SID = "S%02d" % SUBJECT
STEM = "uih_knee_%s_t1_pd_t2_contrast_compare" % SID
OUT = os.environ.get("MCGS_ARTIFACT_DIR", ".")
SERVER_COPY = os.path.expanduser("~/codex_artifacts/mcgs_uih_knee_contrast_compare_20261125"
                                 + ("" if SUBJECT == 1 else "_" + SID))
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
                     "font.size": 6, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def rss(tag):
    subs = sorted({os.path.dirname(p) for p in glob.glob(ROOT + "/**/*.mat", recursive=True)})
    p = glob.glob(subs[SUBJECT - 1] + "/*_" + tag + "_mx3d*.mat")[0]
    with h5py.File(p, "r") as f:
        k = f["kspace"][()]
    ax = (0, 2, 3)
    x = F.fftshift(F.ifftn(F.ifftshift(k, axes=ax), axes=ax, norm="ortho", workers=4), axes=ax)
    return np.sqrt((np.abs(x) ** 2).sum(axis=1)).astype(np.float32)


def shift_to(a, b):
    ga = np.sqrt(sum(g ** 2 for g in np.gradient(a)))
    gb = np.sqrt(sum(g ** 2 for g in np.gradient(b)))
    ga -= ga.mean()
    gb -= gb.mean()
    xc = np.real(F.ifftn(F.fftn(ga) * np.conj(F.fftn(gb))))
    pk = np.unravel_index(np.argmax(xc), xc.shape)
    s = [int(p if p <= n // 2 else p - n) for p, n in zip(pk, xc.shape)]
    return np.roll(a, [-v for v in s], axis=(0, 1, 2)), s


def main():
    global X_LAYER, N3_SECTION
    im = {t: rss(t) for t in ("t1", "pd", "t2")}
    assert len({v.shape for v in im.values()}) == 1, {t: v.shape for t, v in im.items()}
    shifts = {"pd": [0, 0, 0]}
    for t in ("t1", "t2"):
        im[t], shifts[t] = shift_to(im[t], im["pd"])
    mask0 = im["pd"] > 0.1 * np.percentile(im["pd"], 99.5)
    mask = nd.binary_erosion(mask0, iterations=2)
    if X_LAYER is None:
        cen = nd.center_of_mass(mask)
        X_LAYER, N3_SECTION = int(round(cen[1])), int(round(cen[2]))
    sm = {t: nd.uniform_filter(im[t], 3) for t in im}
    med = {t: float(np.median(sm[t][mask])) for t in im}
    nv = {t: im[t] / med[t] for t in im}
    ns = {t: sm[t] / med[t] for t in im}
    vmax = float(np.percentile(np.concatenate([nv[t][mask] for t in im]), 99.5))
    ratio = {t: np.where(mask0, np.log2(np.maximum(ns[t], 1e-6) / np.maximum(ns["pd"], 1e-6)), np.nan)
             for t in ("t1", "t2")}

    def cut(v, row):
        return v[:, X_LAYER, :] if row == 0 else v[:, :, N3_SECTION]

    n0, n2, n3 = im["pd"].shape
    W = 183 / 25.4
    L, GC, R, G, HB, TOP, BOT = 0.25, 0.47, 0.45, 0.03, 0.14, 0.01, 0.02
    pw = (W - L - GC - R - 3 * G) / 5
    ph = pw * n0 / n3
    H = TOP + HB + 2 * ph + G + BOT
    fig = plt.figure(figsize=(W, H))
    xs = [L, L + pw + G, L + 2 * (pw + G), L + 3 * pw + 2 * G + GC, L + 4 * pw + 3 * G + GC]
    titles = ["T1", "PD", "T2", "T1 / PD", "T2 / PD"]
    cm = plt.get_cmap("RdBu_r").copy()
    cm.set_bad("white")
    hg = hr = None
    for row in range(2):
        y = TOP + HB + row * (ph + G)
        for ci in range(5):
            ax = fig.add_axes([xs[ci] / W, 1 - (y + ph) / H, pw / W, ph / H])
            if ci < 3:
                hg = ax.imshow(cut(nv[("t1", "pd", "t2")[ci]], row), cmap="gray", vmin=0, vmax=vmax,
                               interpolation="nearest", aspect="auto")
            else:
                hr = ax.imshow(cut(ratio[("t1", "t2")[ci - 3]], row), cmap=cm, vmin=-1.5, vmax=1.5,
                               interpolation="nearest", aspect="auto")
            ax.set_axis_off()
            if row == 0:
                fig.text((xs[ci] + pw / 2) / W, 1 - (TOP + HB * 0.5) / H, titles[ci], ha="center", va="center",
                         fontsize=6.5)
        fig.text(0.01 / W, 1 - y / H, "ab"[row], fontsize=8, fontweight="bold", ha="left", va="top")
        fig.text(0.17 / W, 1 - (y + ph / 2) / H, ("Sagittal layer", "Coronal section")[row], rotation=90,
                 ha="center", va="center", fontsize=6.5)
    y0, hh = 1 - (TOP + HB + 2 * ph + G) / H, (2 * ph + G) / H
    for x_in, h, lab, ticks in ((xs[2] + pw + 0.05, hg, "Intensity / tissue median", None),
                                (xs[4] + pw + 0.05, hr, "Intensity ratio", [-1, 0, 1])):
        cax = fig.add_axes([x_in / W, y0, 0.055 / W, hh])
        cb = fig.colorbar(h, cax=cax)
        if ticks:
            cb.set_ticks(ticks)
            cb.set_ticklabels(["0.5", "1", "2"])
        cb.outline.set_linewidth(0.4)
        cb.ax.tick_params(labelsize=5.5, width=0.4, length=1.5, pad=1)
        cb.set_label(lab, fontsize=6, labelpad=2)

    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, STEM + ".png"), dpi=600)
    fig.savefig(os.path.join(OUT, STEM + ".pdf"))
    plt.close(fig)
    legend = (
        "T1, PD and T2 of subject " + SID + " (UIH 3D knee set) at the same positions, and the ratio maps T1/PD and T2/PD. "
        "a, sagittal layer at readout index %d. b, section through the readout axis at n3 index %d (coronal). "
        "Images: centred inverse 3D FFT, root-sum-of-squares over 12 coils, T1 and T2 moved onto PD by integer shifts "
        "%s and %s voxels, each contrast divided by its median inside the knee mask; one grey scale [0, %.2f] for the "
        "three contrasts. Ratio maps: 3x3x3 box-smoothed images, drawn inside the mask, log2 colour scale from 0.35 "
        "to 2.8 centred at 1 (white); red = brighter than PD, blue = darker than PD. The coil shading is common to "
        "the scans and cancels in the ratio. Arrays as stored, no reorientation.\n"
    ) % (X_LAYER, N3_SECTION, shifts["t1"], shifts["t2"], vmax)
    with open(os.path.join(OUT, STEM + "_legend.txt"), "w") as f:
        f.write(legend)
    with open(os.path.join(OUT, STEM + "_params.json"), "w") as f:
        json.dump({"subject": SID, "x_layer": X_LAYER, "n3_section": N3_SECTION, "shifts_to_pd": shifts,
                   "median_in_mask": med, "grey_vmax": vmax, "ratio_log2_range": [-1.5, 1.5],
                   "mask_voxels": int(mask.sum())}, f, indent=1)
    names = [STEM + s for s in (".png", ".pdf", "_legend.txt", "_params.json")]
    sums = {n: sha256(os.path.join(OUT, n)) for n in names}
    with open(os.path.join(OUT, "SHA256SUMS"), "w") as f:
        for n in names:
            f.write("%s  %s\n" % (sums[n], n))
    dest, i = SERVER_COPY, 2
    while os.path.exists(dest):
        dest = "%s_r%d" % (SERVER_COPY, i)
        i += 1
    os.makedirs(dest)
    for n in names + ["SHA256SUMS"]:
        shutil.copy2(os.path.join(OUT, n), os.path.join(dest, n))
    ok = all(sha256(os.path.join(dest, n)) == sums[n] for n in names)
    print("server copy:", dest, "verified" if ok else "CHECKSUM MISMATCH")
    print("shifts", shifts, "| grey vmax", round(vmax, 2))
    for t in ("t1", "t2"):
        v = ratio[t][mask]
        print(t, "/pd ratio percentiles 5/25/50/75/95:", [round(float(2 ** q), 2) for q in np.percentile(v, [5, 25, 50, 75, 95])])


if __name__ == "__main__":
    main()
