"""3D k-space and image layers along the readout direction: UIH 3D knee, subject S01, T1 and PD.

Read-only on the data, CPU only. Data: /data2/congcong/data/3D/3D_data20261112/kspace_csm/knee/ (MATLAB v7.3,
key kspace, h5py shape (n0, coils, n2, n3)); subjects are numbered S01..S16 in sorted path order and no names are
printed or written. The readout is array axis n2 (checked again here with the criterion of audit_sampling.py).

Figure, 4 rows x 6 columns, 183 mm wide:
  a, b  k-space of T1 / PD, root-sum-of-squares (RSS) over coils, log10(|k| / max) on one common scale.
        Column 1: section through the readout axis (axis n0 x readout) at the centre of phase-encode axis n3;
        red lines mark the layers. Columns 2-6: phase-encode planes (n0 x n3) at readout offsets KX_OFFSETS.
  c, d  images of T1 / PD: centred inverse 3D FFT, RSS over coils, each contrast divided by its 99.5th percentile.
        Column 1: section through the readout axis at the n3 index of the object's centre; red lines mark the
        layers. Columns 2-6: planes (n0 x n3) at five readout positions spread over the object (X_FRACTIONS).
Arrays are shown as stored (rows = axis n0 from index 0 at the top), with no anatomical reorientation.

Writes <stem>.png (600 dpi), <stem>.pdf, <stem>_legend.txt, <stem>_params.json and SHA256SUMS to
$MCGS_ARTIFACT_DIR and the same files to a server copy under ~/codex_artifacts/.
"""
import glob
import hashlib
import json
import os
import shutil

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = "/data2/congcong/data/3D/3D_data20261112/kspace_csm/knee"
SUBJECT = 1
CONTRASTS = ("t1", "pd")
KX_OFFSETS = (-64, -16, 0, 16, 64)          # readout offsets from the k-space centre (176 readout points)
X_FRACTIONS = (0.15, 0.325, 0.5, 0.675, 0.85)  # image layers, as fractions of the object's extent along the readout
STEM = "uih_knee_S01_t1_pd_readout_layers"
OUT = os.environ.get("MCGS_ARTIFACT_DIR", ".")
SERVER_COPY = os.path.expanduser("~/codex_artifacts/mcgs_uih_knee_readout_layers_20261125")
RED = "#E64B35"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
    "font.size": 6, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.5,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def load(tag):
    subs = sorted({os.path.dirname(p) for p in glob.glob(ROOT + "/**/*.mat", recursive=True)})
    ps = glob.glob(subs[SUBJECT - 1] + "/*_" + tag + "_mx3d*.mat")
    assert len(ps) == 1, (tag, len(ps))
    with h5py.File(ps[0], "r") as f:
        k = f["kspace"][()]
    return k, sha256(ps[0])


def ifft3c(k):
    axes = (0, 2, 3)
    try:
        from scipy import fft as F
        kw = {"workers": 4}
    except ImportError:
        F, kw = np.fft, {}
    return F.fftshift(F.ifftn(F.ifftshift(k, axes=axes), axes=axes, norm="ortho", **kw), axes=axes)


def prepare(tag):
    k, sha = load(tag)
    n0, nc, n2, n3 = k.shape
    A = np.sqrt((np.abs(k) ** 2).sum(axis=1)).astype(np.float32)          # (n0, n2, n3)
    P = A.astype(np.float64) ** 2
    c = [int(np.argmax(P.sum(axis=tuple(j for j in range(3) if j != a)))) for a in range(3)]
    shares = []
    for a in range(3):
        L = P.shape[a]
        idx = np.where(np.abs(np.arange(L) - c[a]) > 0.25 * L)[0]
        Q = np.take(P, idx, axis=a).mean(axis=a)
        shares.append(float((Q < 0.01 * np.median(Q)).mean()))
    assert int(np.argmax(shares)) == 1, ("readout is not axis n2", shares)
    far = np.abs(np.arange(n2) - c[1]) > 0.25 * n2
    Q = P[:, far, :].mean(axis=1)
    measured = Q >= 0.01 * np.median(Q)                                     # PE plane (n0, n3)
    outer = A[:, far, :]
    noise_med = float(np.median(outer[np.broadcast_to(measured[:, None, :], outer.shape)]))
    x = ifft3c(k)
    del k
    img = np.sqrt((np.abs(x) ** 2).sum(axis=1)).astype(np.float32)          # (n0, n2, n3)
    del x
    return {"A": A, "I": img, "c": c, "sha256": sha, "shape_h5": [n0, nc, n2, n3], "noise_med": noise_med,
            "amax": float(A.max()), "p995": float(np.percentile(img, 99.5)),
            "measured_share_pe": float(measured.mean()), "readout_empty_shares": shares}


def main():
    D = {tag: prepare(tag) for tag in CONTRASTS}
    n0, _, n2, n3 = D["t1"]["shape_h5"]
    assert all(D[t]["shape_h5"] == D["t1"]["shape_h5"] for t in CONTRASTS) and n2 == n3

    # image layer positions from the T1 object extent along the readout; section through the object centre
    fg = D["t1"]["I"] / D["t1"]["p995"] > 0.1
    frac = fg.mean(axis=(0, 2))
    xr = np.where(frac > 0.02)[0]
    x0, x1 = int(xr.min()), int(xr.max())
    xs = [int(round(x0 + f * (x1 - x0))) for f in X_FRACTIONS]
    w3 = fg.sum(axis=(0, 1)).astype(np.float64)
    j3 = int(round((w3 * np.arange(n3)).sum() / w3.sum()))
    vmin = min(np.log10(D[t]["noise_med"] / D[t]["amax"]) for t in CONTRASTS) - 1.0
    vmin = float(np.floor(vmin * 2.0) / 2.0)

    mm = 1 / 25.4
    W = 183 * mm
    L, R, G0, G = 0.31, 0.45, 0.08, 0.03
    pw = (W - L - R - G0 - 4 * G) / 6
    ph = pw * n0 / n3
    HB, GR, GB, BB, TOP = 0.13, 0.03, 0.06, 0.15, 0.01
    H = TOP + 2 * (HB + 2 * ph + GR) + GB + BB
    fig = plt.figure(figsize=(W, H))

    def fx(x_in):
        return x_in / W

    def fy(y_from_top_in):
        return 1 - y_from_top_in / H

    def ax_at(x_in, y_top_in):
        return fig.add_axes([fx(x_in), fy(y_top_in + ph), pw / W, ph / H])

    col_x = [L] + [L + pw + G0 + i * (pw + G) for i in range(5)]
    minus = "−"
    k_titles = ["Readout section"] + ["kx = " + (minus + str(-o) if o < 0 else ("+" + str(o) if o > 0 else "0"))
                                      for o in KX_OFFSETS]
    x_titles = ["Readout section"] + ["x = %d" % x for x in xs]
    rows = [("a", "t1", "k"), ("b", "pd", "k"), ("c", "t1", "i"), ("d", "pd", "i")]
    handles = {}
    for ri, (letter, tag, kind) in enumerate(rows):
        blk = 0 if kind == "k" else 1
        blk_top = TOP + blk * (HB + 2 * ph + GR + GB)
        y_top = blk_top + HB + (ri % 2) * (ph + GR)
        d = D[tag]
        if kind == "k":
            def logk(a2, amax=d["amax"]):
                return np.log10(np.maximum(a2, 1e-30) / amax)
            panels = [logk(d["A"][:, :, d["c"][2]])] + [logk(d["A"][:, d["c"][1] + o, :]) for o in KX_OFFSETS]
            marks = [d["c"][1] + o for o in KX_OFFSETS]
            kw = {"cmap": "gray", "vmin": vmin, "vmax": 0.0}
        else:
            In = d["I"] / d["p995"]
            panels = [In[:, :, j3]] + [In[:, x, :] for x in xs]
            marks = xs
            kw = {"cmap": "gray", "vmin": 0.0, "vmax": 1.0}
        for ci, im in enumerate(panels):
            ax = ax_at(col_x[ci], y_top)
            h = ax.imshow(im, interpolation="nearest", aspect="auto", **kw)
            ax.set_axis_off()
            if ci == 0:
                for m in marks:
                    ax.axvline(m, color=RED, lw=0.5)
        handles[blk] = h
        fig.text(fx(0.01), fy(y_top), letter, fontsize=8, fontweight="bold", ha="left", va="top")
        fig.text(fx(0.19), fy(y_top + ph / 2), "%s %s" % (tag.upper(), "k-space" if kind == "k" else "image"),
                 rotation=90, ha="center", va="center", fontsize=6.5)
    for blk, titles in enumerate([k_titles, x_titles]):
        blk_top = TOP + blk * (HB + 2 * ph + GR + GB)
        for ci, t in enumerate(titles):
            fig.text(fx(col_x[ci] + pw / 2), fy(blk_top + HB * 0.5), t, ha="center", va="center", fontsize=6)
        cax = fig.add_axes([fx(col_x[-1] + pw + 0.06), fy(blk_top + HB + 2 * ph + GR), 0.055 / W, (2 * ph + GR) / H])
        cb = fig.colorbar(handles[blk], cax=cax)
        cb.outline.set_linewidth(0.4)
        cb.ax.tick_params(labelsize=5.5, width=0.4, length=1.5, pad=1)
        cb.set_label("log10(|k| / max)" if blk == 0 else "Intensity / 99.5th percentile", fontsize=6, labelpad=2)
    yb = fy(H - BB * 0.5)
    fig.text(fx(col_x[0] + pw / 2), yb, "horizontal: readout", ha="center", va="center", fontsize=6)
    fig.text(fx((col_x[1] + col_x[5] + pw) / 2), yb,
             "horizontal: phase-encode axis n3 (176);  vertical in all panels: phase-encode axis n0 (210)",
             ha="center", va="center", fontsize=6)

    os.makedirs(OUT, exist_ok=True)
    png = os.path.join(OUT, STEM + ".png")
    pdf = os.path.join(OUT, STEM + ".pdf")
    fig.savefig(png, dpi=600)
    fig.savefig(pdf)
    plt.close(fig)

    kx_txt = ", ".join(t.split(" = ")[1] for t in k_titles[1:])
    legend = (
        "a,b, k-space of T1 (a) and PD (b) of subject S01 (UIH 3D knee set, 12 coils), root-sum-of-squares over coils, "
        "shown as log10(|k|/max) on one scale [%.1f, 0]. Left: section through the readout axis at the centre of "
        "phase-encode axis n3; red lines mark the layers on the right, at readout offsets kx = %s from the k-space "
        "centre (176 readout points). Right: phase-encode planes at those offsets. The black corners are positions "
        "that were not acquired (elliptical scan); they are the same in every layer.\n"
        "c,d, images of T1 (c) and PD (d): centred inverse 3D FFT, root-sum-of-squares over coils, each contrast "
        "divided by its own 99.5th percentile (display 0-1, no other processing). Left: section through the readout "
        "axis at phase-encode-axis-n3 index %d (object centre); red lines mark the layers on the right, at readout "
        "positions x = %s of 176 (15-85 %% of the object's extent, %d-%d).\n"
        "All panels: arrays as stored in the files, rows = phase-encode axis n0 (210, index 0 at the top), no "
        "anatomical reorientation, no interpolation.\n"
    ) % (vmin, kx_txt, j3, ", ".join(str(x) for x in xs), x0, x1)
    with open(os.path.join(OUT, STEM + "_legend.txt"), "w") as f:
        f.write(legend)
    params = {
        "subject": "S%02d" % SUBJECT, "contrasts": list(CONTRASTS), "readout_axis_h5": 2,
        "kx_offsets": list(KX_OFFSETS), "x_fractions": list(X_FRACTIONS), "x_positions": xs,
        "object_extent_readout": [x0, x1], "section_index_n3_image": j3, "kspace_log_vmin": vmin,
        "per_contrast": {t: {"source_sha256": D[t]["sha256"], "shape_h5": D[t]["shape_h5"], "kspace_centre": D[t]["c"],
                             "amax": D[t]["amax"], "noise_median": D[t]["noise_med"], "image_p995": D[t]["p995"],
                             "measured_share_pe_plane": D[t]["measured_share_pe"],
                             "readout_empty_shares_by_axis": D[t]["readout_empty_shares"]} for t in CONTRASTS},
    }
    with open(os.path.join(OUT, STEM + "_params.json"), "w") as f:
        json.dump(params, f, indent=1)

    names = [STEM + s for s in (".png", ".pdf", "_legend.txt", "_params.json")]
    sums = {n: sha256(os.path.join(OUT, n)) for n in names}
    with open(os.path.join(OUT, "SHA256SUMS"), "w") as f:
        for n in names:
            f.write("%s  %s\n" % (sums[n], n))
    dest = SERVER_COPY
    i = 2
    while os.path.exists(dest):
        dest = "%s_r%d" % (SERVER_COPY, i)
        i += 1
    os.makedirs(dest)
    for n in names + ["SHA256SUMS"]:
        shutil.copy2(os.path.join(OUT, n), os.path.join(dest, n))
    ok = all(sha256(os.path.join(dest, n)) == sums[n] for n in names)
    print("server copy:", dest, "verified" if ok else "CHECKSUM MISMATCH")
    for n in names:
        print(sums[n], n, os.path.getsize(os.path.join(OUT, n)))
    print("kx offsets", list(KX_OFFSETS), "| x positions", xs, "| section n3 index", j3, "| log vmin", vmin)
    for t in CONTRASTS:
        print(t, "centre", D[t]["c"], "measured share", round(D[t]["measured_share_pe"], 3),
              "noise/max log10", round(float(np.log10(D[t]["noise_med"] / D[t]["amax"])), 2))


if __name__ == "__main__":
    main()
