#!/usr/bin/env python3
"""fig_knee_11methods.py -- UIH knee T1 + T2, AF12: the reference and all 11 methods in one figure (CPU).

Methods: zero-filled, CG-SENSE, CS-SENSE, joint CS-SENSE, SPIRiT, ZS-SSDU, 3D-DIP, 3D-INR, MC-INR (run_knee_arms.py),
GS-single and MC-GS (run_knee.py). Per contrast three rows: the sagittal layer through the centroid of the knee
mask, a magnified box of it, and the absolute error against the reference. All 11 methods are scored here by one
function (diag_artefact.headonly.score_headonly on magnitudes, against the trainer's reference); PSNR / SSIM are
printed under each method, the best in bold. The comparators' own reference is checked to agree with the trainer's.
Images divided by the 99.9th percentile of the reference: grey [0, 1], error [0, 0.1] inferno. Arrays as stored.

    gpu.py run --cpu -- @python autoresearch/uih_knee_t1t2_20261125/fig_knee_11methods.py 6
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
from matplotlib.patches import Rectangle  # noqa: E402
from scipy import ndimage as nd  # noqa: E402

ROOT = Path("/data0/congcong/code/GS/Paired_ARC_GS_general_framework")
sys.path.insert(0, str(ROOT))
from diag_artefact.headonly import score_headonly  # noqa: E402

SI = int(sys.argv[1])
TAG = "knee_s%02d" % SI
OUT = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_uih_knee_t1t2_20261125")
RES = OUT / "trainer/results_arc"
REC = OUT / f"arms/topics/{TAG}_af12/recon"
ART = Path(os.environ.get("MCGS_ARTIFACT_DIR", "."))
STEM = f"{TAG}_t1t2_af12_11methods"
EMAX, BOX = 0.1, 56
# comparator file stems; the cloned scripts call the knee T2 "FLAIR"
ARMS = [("Zero-filled", "zf", "zf"), ("CG-SENSE", "cgsense", "cgsense"),
        ("CS-SENSE", "cssense_o200_lam0.0007", "cssense_o200_lam0.0007"),
        ("Joint CS-SENSE", "jcssense_o200_lam0.001", "jcssense_o200_lam0.001"),
        ("SPIRiT", "spirit_ks5_lam0.002", "spirit_ks5_lam0.001"),
        ("ZS-SSDU", "zsssdu", "zsssdu"), ("3D-DIP", "dip", "dip"), ("3D-INR", "inr", "inr"), ("MC-INR", "mcinr", "mcinr")]
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
                     "font.size": 5.5, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    j = np.load(RES / f"{TAG}_full_volumes.npz")
    info = {"subject": "S%02d" % SI, "error_max": EMAX, "box": BOX, "scores": {}, "reference_check": {}}
    D = {}
    for c, key, fl in (("T1", "T1", "T1"), ("T2", "FL", "FLAIR")):
        gt = j[f"gt{key}"].astype(np.float64)
        sc = np.percentile(gt, 99.9)
        gt = gt / sc
        vols = []
        for name, s1, s2 in ARMS:
            p = REC / f"{s1 if c == 'T1' else s2}_{fl}_complex.npy"
            vols.append((name, np.abs(np.load(p)) if p.exists() else None))
        s = np.load(RES / f"{TAG}_single_{c}_volumes.npz")
        vols += [("GS-single", s["rec"] / sc), ("MC-GS", j[f"rec{key}"] / sc)]
        pg = REC / f"gt_{fl}_complex.npy"
        if pg.exists():                                     # comparator gauge = |GT| / its own 99.9th percentile
            g2 = np.abs(np.load(pg))
            d = float(np.abs(g2 - gt).max())
            info["reference_check"][c] = d
            assert d < 2e-3, f"{c}: comparator reference differs from the trainer reference (max abs {d})"
        info["scores"][c] = {}
        for name, v in vols:
            if v is not None:
                ho = score_headonly(np.asarray(v, np.float32), gt.astype(np.float32))
                info["scores"][c][name] = {"psnr": float(ho["head_psnr"]), "ssim": float(ho["head_ssim"])}
        D[c] = dict(gt=gt, vols=vols)
    mask = D["T1"]["gt"] > 0.05
    xs = int(round(nd.center_of_mass(mask)[0]))
    cy, cz = (int(round(v)) for v in nd.center_of_mass(mask[xs]))
    nx, ny, nz = mask.shape
    y0, z0 = min(max(cy - BOX // 2, 0), ny - BOX), min(max(cz - BOX // 2, 0), nz - BOX)
    info["layer"] = {"sagittal_x": xs, "box_y0": y0, "box_z0": z0}

    ncol = 12
    W = 183 / 25.4
    L, R, G, HT, GAP = 0.16, 0.36, 0.02, 0.30, 0.10
    pw = (W - L - R - (ncol - 1) * G) / ncol
    h1, h2 = pw * ny / nz, pw
    blk = HT + 2 * h1 + h2 + 2 * G
    H = 2 * blk + GAP + 0.02
    fig = plt.figure(figsize=(W, H))
    him = her = None

    def axes(ci, top, h):
        ax = fig.add_axes([(L + ci * (pw + G)) / W, 1 - (top + h) / H, pw / W, h / H])
        ax.set_axis_off()
        return ax

    for bi, c in enumerate(("T1", "T2")):
        top = bi * (blk + GAP)
        gt = D[c]["gt"]
        sco = info["scores"][c]
        best = {k: max(sco, key=lambda n: sco[n][k]) for k in ("psnr", "ssim")} if sco else {}
        fig.text(0.01 / W, 1 - (top + 0.02) / H, "ab"[bi], fontsize=8, fontweight="bold", ha="left", va="top")
        fig.text(0.09 / W, 1 - (top + HT + h1 / 2) / H, c, rotation=90, ha="center", va="center", fontsize=6.5)
        for ci, (name, v) in enumerate([("Reference", gt)] + D[c]["vols"]):
            xc = (L + ci * (pw + G) + pw / 2) / W
            fig.text(xc, 1 - (top + 0.13) / H, name, ha="center", va="bottom", fontsize=5.3)
            if name in sco:
                for k, yy, fmt in (("psnr", 0.20, "%.2f dB"), ("ssim", 0.27, "%.4f")):
                    fig.text(xc, 1 - (top + yy) / H, fmt % sco[name][k], ha="center", va="bottom", fontsize=4.8,
                             fontweight="bold" if best[k] == name else "normal")
            if v is None:
                continue
            sl = np.asarray(v[xs])
            ax = axes(ci, top + HT, h1)
            him = ax.imshow(sl, cmap="gray", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
            ax.add_patch(Rectangle((z0 - 0.5, y0 - 0.5), BOX, BOX, fill=False, ec="#E64B35", lw=0.4))
            ax = axes(ci, top + HT + h1 + G, h2)
            ax.imshow(sl[y0:y0 + BOX, z0:z0 + BOX], cmap="gray", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
            if ci:
                ax = axes(ci, top + HT + h1 + h2 + 2 * G, h1)
                her = ax.imshow(np.abs(sl - gt[xs]), cmap="inferno", vmin=0, vmax=EMAX, interpolation="nearest", aspect="auto")
        xb = (L + ncol * pw + (ncol - 1) * G + 0.04) / W
        cax = fig.add_axes([xb, 1 - (top + HT + h1) / H, 0.04 / W, h1 / H])
        cb1 = fig.colorbar(him, cax=cax)
        cax = fig.add_axes([xb, 1 - (top + blk) / H, 0.04 / W, h1 / H])
        cb2 = fig.colorbar(her, cax=cax)
        cb2.set_label("Absolute error", fontsize=5, labelpad=1)
        for b in (cb1, cb2):
            b.outline.set_linewidth(0.3)
            b.ax.tick_params(labelsize=4.5, width=0.3, length=1.2, pad=1)

    fig.savefig(ART / f"{STEM}.png", dpi=600)
    fig.savefig(ART / f"{STEM}.pdf")
    plt.close(fig)
    missing = sorted({n for c in D for n, v in D[c]["vols"] if v is None})
    legend = (
        f"UIH 3D knee, subject S{SI:02d}, T1 (a) and T2 (b), retrospective 12-fold undersampling (variable-density "
        f"Poisson, 24 x 24 centre block, one mask for both contrasts): reference and 11 methods. Rows per contrast: "
        f"sagittal layer at readout index {xs} of {nx}; the {BOX}-voxel box (red) magnified; absolute error against the "
        f"reference, [0, {EMAX}], inferno. Images divided by the 99.9th percentile of the reference, grey scale [0, 1]. "
        f"Numbers: PSNR and SSIM inside the tissue mask over the whole volume, all methods scored by one function "
        f"against the same reference; best in bold. Comparator settings are those frozen on the brain T1 + FLAIR "
        f"tuning subject (T2 uses the FLAIR values); MC-GS and GS-single use the brain T1 + T2 recipe; nothing was "
        f"tuned on the knee. The T2 reference is the partial-Fourier zero-filled image. One subject, one seed."
        + (f" Not run: {', '.join(missing)}." if missing else "") + "\n")
    (ART / f"{STEM}_legend.txt").write_text(legend)
    json.dump(info, open(ART / f"{STEM}_params.json", "w"), indent=1)
    names = [STEM + s for s in (".png", ".pdf", "_legend.txt", "_params.json")]
    sums = {n: sha256(ART / n) for n in names}
    (ART / "SHA256SUMS_11methods").write_text("".join(f"{sums[n]}  {n}\n" for n in names))
    dest, i = OUT / "figures_11methods", 2
    while dest.exists():
        dest = OUT / f"figures_11methods_r{i}"
        i += 1
    dest.mkdir()
    for n in names + ["SHA256SUMS_11methods"]:
        shutil.copy2(ART / n, dest / n)
    print("server copy:", dest, "verified" if all(sha256(dest / n) == sums[n] for n in names) else "CHECKSUM MISMATCH")
    print("missing:", missing)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
