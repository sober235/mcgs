#!/usr/bin/env python3
"""fig_knee.py -- UIH knee T1 + T2, AF12: reference, zero-filled, GS-single, MC-GS and the two error maps (CPU).

Reads the volumes saved by run_knee.py (trainer output: magnitude after hard data consistency, reference from
the prepared k-space with the same coil maps). Rows: T1 sagittal, T1 coronal, T2 sagittal, T2 coronal; the
layers go through the centroid of the knee mask (reference above 5 % of its 99.9th percentile). Every image is
divided by the 99.9th percentile of its contrast's reference; grey scale [0, 1], error maps |image - reference|
on [0, 0.1], inferno. Arrays as stored (X = readout = left-right, Y, Z), rows of every panel = Y, index 0 at
the top; no reorientation, no interpolation.

    gpu.py run --cpu -- @python autoresearch/uih_knee_t1t2_20261125/fig_knee.py 6
"""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import ndimage as nd  # noqa: E402

SI = int(sys.argv[1])
TAG = "knee_s%02d" % SI
OUT = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_uih_knee_t1t2_20261125")
RES = OUT / "trainer/results_arc"
ART = Path(os.environ.get("MCGS_ARTIFACT_DIR", "."))
STEM = f"{TAG}_t1t2_af12_recon"
EMAX = 0.1
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
                     "font.size": 6, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    j = np.load(RES / f"{TAG}_full_volumes.npz")
    s1 = np.load(RES / f"{TAG}_single_T1_volumes.npz")
    s2 = np.load(RES / f"{TAG}_single_T2_volumes.npz")
    D = {"T1": dict(gt=j["gtT1"], zf=j["zfT1"], mc=j["recT1"], gs=s1["rec"], gt_single=s1["gt"]),
         "T2": dict(gt=j["gtFL"], zf=j["zfFL"], mc=j["recFL"], gs=s2["rec"], gt_single=s2["gt"])}
    info = {"subject": "S%02d" % SI, "error_max": EMAX, "contrasts": {}}
    for c, d in D.items():
        ref_diff = float(np.abs(d["gt"] - d["gt_single"]).max() / d["gt"].max())
        assert ref_diff < 1e-4, f"{c}: joint and single runs have different references ({ref_diff})"
        sc = float(np.percentile(d["gt"], 99.9))
        for k in ("gt", "zf", "mc", "gs"):
            d[k] = d[k] / sc
        d["mask"] = d["gt"] > 0.05
        info["contrasts"][c] = {"scale_p999": sc, "reference_max_rel_diff_joint_vs_single": ref_diff,
                                "mask_frac": float(d["mask"].mean())}
    cen = nd.center_of_mass(D["T1"]["mask"])
    xs, zs = int(round(cen[0])), int(round(cen[2]))
    info["layers"] = {"sagittal_x": xs, "coronal_z": zs}

    def cut(v, view):
        return v[xs] if view == 0 else v[:, :, zs].T          # (Y, Z) or (Y, X)

    nx, ny, nz = D["T1"]["gt"].shape
    cols = [("Reference", "gt"), ("Zero-filled", "zf"), ("GS-single", "gs"), ("MC-GS", "mc"),
            ("GS-single error", "gs"), ("MC-GS error", "mc")]
    W = 183 / 25.4
    L, R, G, HB, TOP, BOT = 0.30, 0.42, 0.03, 0.14, 0.01, 0.02
    pw = (W - L - R - 5 * G) / 6
    hs = [pw * ny / nz, pw * ny / nx]                          # sagittal, coronal panel heights (same width)
    rows = [("T1", 0), ("T1", 1), ("T2", 0), ("T2", 1)]
    H = TOP + HB + sum(hs[v] for _, v in rows) + 3 * G + BOT
    fig = plt.figure(figsize=(W, H))
    y = TOP + HB
    hi = he = None
    for ri, (c, v) in enumerate(rows):
        ph = hs[v]
        for ci, (title, key) in enumerate(cols):
            ax = fig.add_axes([(L + ci * (pw + G)) / W, 1 - (y + ph) / H, pw / W, ph / H])
            if ci < 4:
                hi = ax.imshow(cut(D[c][key], v), cmap="gray", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
            else:
                he = ax.imshow(cut(np.abs(D[c][key] - D[c]["gt"]), v), cmap="inferno", vmin=0, vmax=EMAX,
                               interpolation="nearest", aspect="auto")
            ax.set_axis_off()
            if ri == 0:
                fig.text((L + ci * (pw + G) + pw / 2) / W, 1 - (TOP + HB * 0.5) / H, title, ha="center", va="center",
                         fontsize=6.5)
        if v == 0:
            fig.text(0.01 / W, 1 - y / H, "ab"[ri // 2], fontsize=8, fontweight="bold", ha="left", va="top")
        fig.text(0.20 / W, 1 - (y + ph / 2) / H, f"{c}, {('sagittal', 'coronal')[v]}", rotation=90, ha="center",
                 va="center", fontsize=6.5)
        y += ph + G
    y0, hh = BOT / H, (H - TOP - HB - BOT) / H
    cax = fig.add_axes([(L + 6 * pw + 5 * G + 0.05) / W, y0 + hh * 0.55, 0.055 / W, hh * 0.45])
    cb = fig.colorbar(he, cax=cax)
    cb.set_label("Absolute error", fontsize=6, labelpad=2)
    cax2 = fig.add_axes([(L + 6 * pw + 5 * G + 0.05) / W, y0, 0.055 / W, hh * 0.45])
    cb2 = fig.colorbar(hi, cax=cax2)
    cb2.set_label("Intensity / reference 99.9th percentile", fontsize=6, labelpad=2)
    for b in (cb, cb2):
        b.outline.set_linewidth(0.4)
        b.ax.tick_params(labelsize=5.5, width=0.4, length=1.5, pad=1)

    fig.savefig(ART / f"{STEM}.png", dpi=600)
    fig.savefig(ART / f"{STEM}.pdf")
    plt.close(fig)
    legend = (
        f"UIH 3D knee, subject S{SI:02d}, T1 (a) and T2 (b), retrospective 12-fold undersampling (variable-density "
        f"Poisson, 24 x 24 centre block, one mask for both contrasts). Columns: reference, zero-filled, GS-single (each "
        f"contrast trained alone), MC-GS (joint model), and the absolute error of the two reconstructions against the "
        f"reference. Sagittal layer at readout index {xs} of {nx}, coronal section at n3 index {zs} of {nz}. Images "
        f"divided by the 99.9th percentile of the contrast's reference, grey scale [0, 1]; error maps [0, {EMAX}], "
        f"inferno. The reference is the image of all acquired k-space positions with the same coil maps; for T2 it is "
        f"the partial-Fourier zero-filled image, not a fully sampled one. One subject, one seed (42), brain recipe "
        f"unchanged. Arrays as stored, no reorientation.\n")
    (ART / f"{STEM}_legend.txt").write_text(legend)
    json.dump(info, open(ART / f"{STEM}_params.json", "w"), indent=1)
    names = [STEM + s for s in (".png", ".pdf", "_legend.txt", "_params.json")]
    sums = {n: sha256(ART / n) for n in names}
    (ART / "SHA256SUMS").write_text("".join(f"{sums[n]}  {n}\n" for n in names))
    dest, i = OUT / "figures", 2
    while dest.exists():
        dest = OUT / f"figures_r{i}"
        i += 1
    dest.mkdir()
    for n in names + ["SHA256SUMS"]:
        shutil.copy2(ART / n, dest / n)
    ok = all(sha256(dest / n) == sums[n] for n in names)
    print("server copy:", dest, "verified" if ok else "CHECKSUM MISMATCH")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
