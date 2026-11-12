#!/usr/bin/env python
"""fig_capacity_slice.py -- Gaussian-capacity sensitivity shown on ONE transverse slice (2026-11-11).

User 2026-11-11: "Not considering the total volume, only one layer is selected to support the above
conclusion." Every number drawn here is a metric of that one slice; nothing whole-head is plotted.

Slice. Default --slice 181: the transverse slice nearest to the figure-of-record slice (180) on which
the whole sentence holds on PSNR and SSIM for both contrasts (0.9M < 1.8M < 3.6M; at 7.2M T1 down and
FLAIR up). Slice 180 itself fails one item (T1 SSIM is not lower at 0.9M). slice_scan.py gives the
context that must go with the figure: the full pattern holds on 78 of 175 transverse slices; the
7.2M trade-off direction alone on 123 of 175, and its reverse on none.

Slice metrics are those of the figure of record (fig_three_planes_compact.py): head mask
|GT| > 0.05 * P99.9 of the volume; PSNR at the volume's max|GT|; SSIM = mean over the head pixels of the
2-D SSIM map (7 x 7 uniform window, data range max|GT|); NMSE over the head pixels. On one slice NMSE is
a monotone transform of that slice's PSNR, so the NMSE panel repeats the PSNR panel in another unit.

Display conventions and the zoom-box rule are those of fig_capacity_full.py (this file was derived
from it). Error scale: [0, 0.1] x max|GT|, the house scale (user 2026-11-11; the first render used 0.05). The saved volumes are still checked against logs/hp_summary.csv (whole-head PSNR to
0.01 dB) before anything is drawn. The script stops if the chosen slice does not show the pattern.
One run per setting (seed 42). Writes to MCGS_ARTIFACT_DIR only. CPU only.
"""
import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter
from scipy.ndimage import binary_dilation, binary_fill_holes, binary_opening, uniform_filter
from skimage.metrics import structural_similarity

ap = argparse.ArgumentParser()
ap.add_argument("--slice", type=int, default=181, help="transverse slice (axis 0) every metric is computed on")
args = ap.parse_args()

T = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_hparam_v3_jiangyizhou")
ARC = Path("/data0/congcong/code/GS/mcgs_hparam_wt/results_arc")
SRC = T / "logs/hp_summary.csv"
OUT = Path(os.environ["MCGS_ARTIFACT_DIR"])

SLICES = (180, 124, 76)            # transverse / coronal / sagittal of the figure of record
BOXW = 48
TRA_ROT = 1
ZC = "#FFC23D"                     # house zoom-box colour

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 6.5,
    "axes.titlesize": 7,
    "axes.labelsize": 6.5,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "axes.edgecolor": "#4D4D4D",
    "xtick.color": "#4D4D4D",
    "ytick.color": "#4D4D4D",
    "text.color": "#272727",
    "axes.labelcolor": "#272727",
})

C = {"T1": "#3775BA", "FLAIR": "#E28E2C"}
MARK = {"T1": "o", "FLAIR": "s"}
INK, MID, LIGHT = "#272727", "#767676", "#BDBDBD"
LW, MS, RING = 1.2, 4.2, 0.8
MINUS = lambda s: s.replace("-", "−")

# ---------------------------------------------------------------- table
raw = SRC.read_bytes()
print("source", SRC, "sha256", hashlib.sha256(raw).hexdigest())
rows = {r["exp"]: r for r in csv.DictReader(raw.decode().splitlines())}
val = lambda e, k: float(rows[e][k])
CAP = ["hp_n09", "hp_n18", "hp_n36", "hp_n72"]
REF = "hp_n36"
assert all(rows[e]["status"] == "done" for e in CAP)
assert [int(val(e, "n_shared")) for e in CAP] == [900000, 1800000, 3600000, 7200000]
millions = [val(e, "n_shared") / 1e6 for e in CAP]
x = np.arange(len(CAP))

d_psnr = {c: np.array([val(e, f"{c}_head_psnr") - val(REF, f"{c}_head_psnr") for e in CAP]) for c in C}
d_ssim = {c: np.array([(val(e, f"{c}_head_ssim") - val(REF, f"{c}_head_ssim")) / val(REF, f"{c}_head_ssim") * 100
                       for e in CAP]) for c in C}
nm_gain = {c: np.array([(val(REF, f"{c}_head_nmse") - val(e, f"{c}_head_nmse")) / val(REF, f"{c}_head_nmse") * 100
                        for e in CAP]) for c in C}            # positive = lower NMSE than 3.6M
for c in C:
    via = np.array([-10 * np.log10(val(e, f"{c}_head_nmse") / val(REF, f"{c}_head_nmse")) for e in CAP])
    assert np.max(np.abs(via - d_psnr[c])) < 1e-3, (c, via, d_psnr[c])
minutes = np.array([val(e, "elapsed_h") * 60 for e in CAP])
vram = np.array([val(e, "vram_gb") for e in CAP])
count = np.array([val(e, "n_shared") for e in CAP])
for arr in (minutes, vram, count):
    assert np.all(np.isfinite(arr)) and not np.any(arr <= 0)
assert d_psnr["T1"][0] < -0.02 and d_psnr["FLAIR"][0] < -0.02, "0.9M is not below 3.6M on both contrasts"
assert d_psnr["T1"][3] < 0 < d_psnr["FLAIR"][3], "7.2M is not a T1-down / FLAIR-up trade-off"

# ---------------------------------------------------------------- volumes
KEYS = {"T1": ("recT1", "gtT1"), "FLAIR": ("recFL", "gtFL")}
DISK = np.hypot(*np.mgrid[-6:7, -6:7]) <= 6


def head_region(m):
    return binary_dilation(binary_fill_holes(binary_opening(binary_fill_holes(m), structure=DISK)), iterations=2)


def D(a):                                                  # transverse display transform of the house script
    return np.rot90(a.T, TRA_ROT)


i0, j0, k0 = SLICES
i0 = args.slice
hw = BOXW // 2
vol = {c: {} for c in C}
gt, gmax, head = {}, {}, {}
for e in CAP:
    p = ARC / f"{e}_volumes.npz"
    z = np.load(p)
    for c, (rk, gk) in KEYS.items():
        g = z[gk].astype(np.float32)
        if c not in gt:
            gt[c], gmax[c] = g, float(g.max())
            head[c] = g > 0.05 * float(np.quantile(g, 0.999))
        else:
            assert np.array_equal(g, gt[c]), f"GT differs between runs ({e}, {c})"
        r = z[rk].astype(np.float32)
        err = (r.astype(np.float64) - g.astype(np.float64))[head[c]]
        ps = 20 * np.log10(gmax[c]) - 10 * np.log10(float((err * err).mean()))
        ref = val(e, f"{c}_head_psnr")
        print(f"check {e} {c}: head PSNR from volume {ps:.3f} dB, csv {ref:.3f} dB, diff {ps - ref:+.4f}")
        assert abs(ps - ref) < 0.01, f"{e} {c}: saved volume does not reproduce the table ({ps} vs {ref})"
        vol[c][e] = r[i0]
    z.close()

planes = {}
for c in C:
    g2, m2 = gt[c][i0], head[c][i0]
    box = np.zeros(g2.shape, bool)
    box[j0 - hw:j0 + hw, k0 - hw:k0 + hw] = True
    rr, cc = np.nonzero(D(box))
    assert rr.max() - rr.min() + 1 == BOXW and cc.max() - cc.min() + 1 == BOXW
    planes[c] = dict(g2=g2, vmax=float(np.quantile(g2, 0.999)), emask=head_region(m2),
                     zy=int(rr.min()), zx=int(cc.min()), hm=D(head_region(m2)))
# one display crop for every panel: the head bounding box of both contrasts plus a margin
hm = planes["T1"]["hm"] | planes["FLAIR"]["hm"]
rr, cc = np.nonzero(hm)
PAD = 5
r0, r1 = max(rr.min() - PAD, 0), min(rr.max() + PAD + 1, hm.shape[0])
c0, c1 = max(cc.min() - PAD, 0), min(cc.max() + PAD + 1, hm.shape[1])
CH, CW = r1 - r0, c1 - c0
print(f"display crop rows {r0}:{r1} cols {c0}:{c1} of {hm.shape}; box at row {planes['T1']['zy']} col {planes['T1']['zx']}")


# ---- metrics of this slice only; from here on d_psnr / d_ssim / nm_gain are slice quantities
SM = {c: {} for c in C}
for c in C:
    g = planes[c]["g2"].astype(np.float64)
    m = head[c][i0]
    for e in CAP:
        r = vol[c][e].astype(np.float64)
        d = r[m] - g[m]
        _, smap = structural_similarity(g, r, data_range=gmax[c], full=True, win_size=7)
        SM[c][e] = dict(psnr=20 * np.log10(gmax[c]) - 10 * np.log10(float((d * d).mean())),
                        ssim=float(smap[m].mean()), nmse=float((d * d).sum() / (g[m] ** 2).sum()))
    print(f"slice {i0} {c}: " + "; ".join(
        f"{e} PSNR {SM[c][e]['psnr']:.3f} SSIM {SM[c][e]['ssim']:.5f} NMSE {SM[c][e]['nmse']:.5f}" for e in CAP))
d_psnr = {c: np.array([SM[c][e]["psnr"] - SM[c][REF]["psnr"] for e in CAP]) for c in C}
d_ssim = {c: np.array([(SM[c][e]["ssim"] - SM[c][REF]["ssim"]) / SM[c][REF]["ssim"] * 100 for e in CAP]) for c in C}
nm_gain = {c: np.array([(SM[c][REF]["nmse"] - SM[c][e]["nmse"]) / SM[c][REF]["nmse"] * 100 for e in CAP]) for c in C}
for c in C:
    for k, dd in (("PSNR", d_psnr[c]), ("SSIM", d_ssim[c])):
        assert dd[0] < dd[1] < dd[2], f"slice {i0} {c} {k}: not monotone from 0.9M to 3.6M ({dd})"
for k, dd in (("PSNR", d_psnr), ("SSIM", d_ssim)):
    assert dd["T1"][3] < 0 < dd["FLAIR"][3], f"slice {i0} {k}: 7.2M is not T1 down / FLAIR up"
print(f"slice {i0}: pattern holds on PSNR and SSIM for both contrasts")

HOUSE_BOX = (planes["T1"]["zy"], planes["T1"]["zx"])
assert HOUSE_BOX == (planes["FLAIR"]["zy"], planes["FLAIR"]["zx"])


def pick_sensitive_box():
    """48 x 48 window of the displayed slice where 0.9M loses most against 3.6M, both contrasts summed"""
    gain = np.zeros(hm.shape)
    for c in C:
        g = planes[c]["g2"].astype(np.float64)
        m = head[c][i0]
        e09 = ((vol[c]["hp_n09"].astype(np.float64) - g) / gmax[c]) ** 2 * m
        e36 = ((vol[c]["hp_n36"].astype(np.float64) - g) / gmax[c]) ** 2 * m
        gain += uniform_filter(D(e09 - e36), size=BOXW, mode="constant")
    frac = uniform_filter(hm.astype(np.float64), size=BOXW, mode="constant")
    ok = frac >= 0.98
    ok[:hw] = ok[-hw:] = False
    ok[:, :hw] = ok[:, -hw:] = False
    cy, cx = np.unravel_index(np.argmax(np.where(ok, gain, -np.inf)), gain.shape)
    return int(cy - hw), int(cx - hw)


def box_psnr(c, e, zy, zx):
    p = planes[c]
    cut = (slice(zy, zy + BOXW), slice(zx, zx + BOXW))
    g = D(p["g2"])[cut].astype(np.float64)
    m = D(head[c][i0])[cut]
    d = D(vol[c][e])[cut].astype(np.float64)[m] - g[m]
    return 20 * np.log10(gmax[c]) - 10 * np.log10(float((d * d).mean()))


# ---------------------------------------------------------------- drawing
def series(ax, xs, ys, c):
    ax.plot(xs, ys, color=C[c], lw=LW, solid_capstyle="round", zorder=3)
    ax.plot(xs, ys, ls="none", marker=MARK[c], ms=MS, mfc=C[c], mec="white", mew=RING, zorder=4)


def ring_best(ax, ys, c):
    b = int(np.argmax(ys))
    ax.plot(x[b], ys[b], ls="none", marker="o", ms=8.6, mfc="none", mec=C[c], mew=0.9, zorder=6)


def auto_ylim(ax, dd):
    ys = np.concatenate([dd[c] for c in C])
    span = float(ys.max() - ys.min())
    ax.set_ylim(ys.min() - 0.10 * span, ys.max() + 0.16 * span)


def capacity_ticks(ax):
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m:g}" for m in millions])
    ax.get_xticklabels()[2].set_fontweight("bold")
    ax.spines["bottom"].set_bounds(0, 3)
    ax.set_xlabel("Shared Gaussians (millions)")


def render(err_vmax, stem, zy, zx, nmse=False):
    FW = 7.2                                               # 183 mm double column
    left, right, ggap, gap = 0.36, 0.05, 0.16, 0.025
    pw = (FW - left - right - ggap - 8 * gap) / 10
    hs = pw * CH / CW                                      # slice panel height (square pixels, as the house figure)
    top, title_h, grp_h = 0.34, 0.24, 0.16
    img_h = hs + gap + pw + gap + pw
    curves_gap, curves_h, bottom = 0.52, 1.95, 0.36
    FH = top + grp_h + title_h + img_h + curves_gap + curves_h + bottom
    fig = plt.figure(figsize=(FW, FH))

    def axes_in(x0, y_top, w, h):                          # inches, y measured from the top edge
        return fig.add_axes([x0 / FW, 1 - (y_top + h) / FH, w / FW, h / FH])

    def img_axes(x0, y_top, w, h):
        a = axes_in(x0, y_top, w, h)
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
        return a

    y_img = top + grp_h + title_h
    cols = [("GT", None)] + [(f"{m:g}M", e) for m, e in zip(millions, CAP)]
    last_err = None
    for gi, c in enumerate(("T1", "FLAIR")):
        p = planes[c]
        gx = left + gi * (5 * pw + 4 * gap + ggap)
        fig.text((gx + (5 * pw + 4 * gap) / 2) / FW, 1 - (top + 0.02) / FH, c, ha="center", va="top",
                 fontsize=7.5, fontweight="bold", color=C[c])
        fig.add_artist(Line2D([gx / FW, (gx + 5 * pw + 4 * gap) / FW], [1 - (top + grp_h - 0.02) / FH] * 2,
                              color=C[c], lw=0.8))
        cut = (slice(zy, zy + BOXW), slice(zx, zx + BOXW))
        for j, (lab, e) in enumerate(cols):
            x0 = gx + j * (pw + gap)
            img = D(p["g2"] if e is None else vol[c][e])
            a = img_axes(x0, y_img, pw, hs)
            a.imshow(img[r0:r1, c0:c1], cmap="gray", vmin=0, vmax=p["vmax"], origin="lower", aspect="auto",
                     interpolation="nearest")
            a.add_patch(Rectangle((zx - c0 - 0.5, zy - r0 - 0.5), BOXW, BOXW, fill=False, ec=ZC, lw=0.7))
            sub = "" if e is None else f"\n{SM[c][e]['psnr']:.2f} dB"
            a.set_title(lab + sub, fontsize=6, pad=2.5, linespacing=1.15,
                        fontweight="bold" if e == REF else "normal")
            b = img_axes(x0, y_img + hs + gap, pw, pw)
            b.imshow(img[cut], cmap="gray", vmin=0, vmax=p["vmax"], origin="lower", aspect="auto",
                     interpolation="nearest")
            for s in b.spines.values():
                s.set_visible(True); s.set_edgecolor(ZC); s.set_linewidth(0.7)
            if e is None:
                continue
            emap = D(np.abs(vol[c][e] - p["g2"]) / gmax[c] * p["emask"])
            d = img_axes(x0, y_img + hs + gap + pw + gap, pw, pw)
            last_err = d.imshow(emap[cut], cmap="jet", vmin=0.0, vmax=err_vmax, origin="lower", aspect="auto",
                                interpolation="nearest")
            d.text(0.04, 0.05, f"{box_psnr(c, e, zy, zx):.2f} dB", transform=d.transAxes, fontsize=5,
                   color="white", ha="left", va="bottom", bbox=dict(fc="black", alpha=0.55, pad=0.8, lw=0))
        # the GT column has no error map: its cell carries the error scale
        cb = axes_in(gx + 0.06 * pw, y_img + hs + gap + pw + gap + 0.36 * pw, 0.88 * pw, 0.11 * pw)
        bar = fig.colorbar(last_err, cax=cb, orientation="horizontal")
        bar.set_ticks([0, err_vmax])
        bar.set_ticklabels(["0", f"{err_vmax:g}"])
        bar.ax.tick_params(labelsize=5, length=1.5, width=0.4, pad=1)
        bar.outline.set_linewidth(0.4)
        tl = bar.ax.get_xticklabels()
        tl[0].set_ha("left"); tl[-1].set_ha("right")
        cb.set_title("|error| / max GT", fontsize=5, pad=2)
    for lab, yc in ((f"Slice {i0}", y_img + hs / 2), ("Zoom", y_img + hs + gap + pw / 2),
                    ("Error", y_img + hs + gap + pw + gap + pw / 2)):
        fig.text((left - 0.07) / FW, 1 - yc / FH, lab, rotation=90, ha="center", va="center", fontsize=6.5)
    fig.text(0.02 / FW, 1 - (top + grp_h + 0.02) / FH, "a", fontsize=8, fontweight="bold", color="black",
             ha="left", va="top")

    # ---- curves
    y_c = y_img + img_h + curves_gap
    if nmse:
        wb, wc, wn, wd = 1.52, 1.08, 1.08, 1.66
        xb = 0.44
        xc = xb + wb + 0.44
        xn = xc + wc + 0.46
        xd = xn + wn + 0.44
    else:
        wb, wc, wd = 2.10, 1.30, 2.10
        xb = 0.50
        xc = xb + wb + 0.52
        xd = xc + wc + 0.58
    assert xd + wd <= FW - 0.05
    ax_b, ax_c, ax_d = axes_in(xb, y_c, wb, curves_h), axes_in(xc, y_c, wc, curves_h), axes_in(xd, y_c, wd, curves_h)
    ax_n = axes_in(xn, y_c, wn, curves_h) if nmse else None
    unit = "" if nmse else " dB"                           # narrower panel: the unit is on the axis

    ax = ax_b
    ax.axhline(0, color=LIGHT, lw=0.7, zorder=1)
    for c in C:
        series(ax, x, d_psnr[c], c)
        ring_best(ax, d_psnr[c], c)
    ax.plot(2, 0, ls="none", marker="o", ms=MS, mfc="white", mec=INK, mew=0.9, zorder=5)
    capacity_ticks(ax)
    ax.set_xlim(-0.3, 4.6)
    auto_ylim(ax, d_psnr)
    ax.set_ylabel("Δ PSNR (dB)")
    ax.set_title("Slice PSNR", loc="left", pad=3)
    dT, dF = d_psnr["T1"], d_psnr["FLAIR"]
    ax.text(0.16, dF[0], "Under-capacity\n" + MINUS(f"FLAIR {dF[0]:+.2f} dB, T1 {dT[0]:+.2f} dB"),
            fontsize=6, color=MID, ha="left", va="center", linespacing=1.3)
    ax.text(3.22, dF[3] + 0.012, MINUS(f"FLAIR {dF[3]:+.2f}{unit}"), fontsize=6, ha="left", va="center")
    ax.text(3.22, dT[3] - 0.012, MINUS(f"T1 {dT[3]:+.2f}{unit}"), fontsize=6, ha="left", va="center")

    ax = ax_c
    ax.axhline(0, color=LIGHT, lw=0.7, zorder=1)
    for c in C:
        series(ax, x, d_ssim[c], c)
        ring_best(ax, d_ssim[c], c)
    ax.plot(2, 0, ls="none", marker="o", ms=MS, mfc="white", mec=INK, mew=0.9, zorder=5)
    capacity_ticks(ax)
    ax.set_xlim(-0.3, 3.3)
    auto_ylim(ax, d_ssim)
    ax.set_ylabel("Δ SSIM (%)")
    ax.set_title("Slice SSIM", loc="left", pad=3)

    if nmse:
        ax = ax_n
        ax.axhline(0, color=LIGHT, lw=0.7, zorder=1)
        for c in C:
            series(ax, x, nm_gain[c], c)
            ring_best(ax, nm_gain[c], c)
        ax.plot(2, 0, ls="none", marker="o", ms=MS, mfc="white", mec=INK, mew=0.9, zorder=5)
        capacity_ticks(ax)
        ax.set_xlim(-0.3, 3.3)
        auto_ylim(ax, nm_gain)
        ax.set_ylabel("NMSE reduction (%)")
        ax.set_title("Slice NMSE", loc="left", pad=3)

    ax = ax_d
    ax.set_yscale("log", base=2)
    ax.plot(x, count / count[0], color=LIGHT, lw=LW, zorder=2)
    ax.plot(x, vram / vram[0], color=MID, lw=LW, zorder=3)
    ax.plot(x, vram / vram[0], ls="none", marker="D", ms=MS - 0.6, mfc=MID, mec="white", mew=RING, zorder=4)
    ax.plot(x, minutes / minutes[0], color=INK, lw=LW, zorder=3)
    ax.plot(x, minutes / minutes[0], ls="none", marker="^", ms=MS + 0.4, mfc=INK, mec="white", mew=RING, zorder=4)
    ax.text(3.2, count[-1] / count[0], f"×{count[-1] / count[0]:.0f} Gaussians", fontsize=6, ha="left",
            va="center", color=MID)
    ax.text(3.2, minutes[-1] / minutes[0], f"×{minutes[-1] / minutes[0]:.2f} time\n"
            f"({minutes[0]:.0f} → {minutes[-1]:.0f} min)", fontsize=6, ha="left", va="bottom", linespacing=1.25)
    ax.text(3.2, vram[-1] / vram[0], f"×{vram[-1] / vram[0]:.2f} peak" + ("\n" if nmse else " ") + "GPU memory\n"
            f"({vram[0]:.1f} → {vram[-1]:.1f} GB)", fontsize=6, ha="left", va="top", linespacing=1.25, color=MID,
            clip_on=False)
    ax.set_ylim(0.58, 11)
    ax.yaxis.set_major_locator(FixedLocator([1, 2, 4, 8]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"×{v:g}"))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    capacity_ticks(ax)
    ax.set_xlim(-0.3, 7.4 if nmse else 6.9)
    ax.set_ylabel("Relative to 0.9M")
    ax.set_title("Compute cost", loc="left", pad=3)

    for ax, s in (((ax_b, "b"), (ax_c, "c"), (ax_n, "d"), (ax_d, "e")) if nmse else
                  ((ax_b, "b"), (ax_c, "c"), (ax_d, "d"))):
        bb = ax.get_position()
        fig.text(bb.x0 - 0.36 / FW, bb.y1 + 0.05 / FH, s, fontsize=8, fontweight="bold", color="black",
                 ha="left", va="bottom")

    handles = [
        Line2D([], [], color=C["T1"], lw=LW, marker="o", ms=MS, mfc=C["T1"], mec="white", mew=RING, label="T1"),
        Line2D([], [], color=C["FLAIR"], lw=LW, marker="s", ms=MS, mfc=C["FLAIR"], mec="white", mew=RING,
               label="FLAIR"),
        Line2D([], [], ls="none", marker="o", ms=8.6, mfc="none", mec=MID, mew=0.9,
               label="Best setting for that contrast"),
        Line2D([], [], ls="none", marker="o", ms=MS, mfc="white", mec=INK, mew=0.9,
               label="Setting used in this work (3.6M)"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.25 / FW, 1 - 0.02 / FH), ncol=4,
               handlelength=1.8, columnspacing=1.8, handletextpad=0.5, borderaxespad=0.0)

    fig.savefig(f"{stem}.pdf", dpi=600)
    fig.savefig(f"{stem}.png", dpi=600)
    plt.close(fig)
    print("wrote", stem, f"({FW:.2f} x {FH:.2f} in = {FW * 25.4:.0f} x {FH * 25.4:.0f} mm; error scale [0, {err_vmax:g}])")


SENS_BOX = pick_sensitive_box()
print(f"zoom box: display row {SENS_BOX[0]} col {SENS_BOX[1]}; 48-box PSNR " + "; ".join(
    f"{c} " + " / ".join(f"{box_psnr(c, e, *SENS_BOX):.2f}" for e in CAP) for c in C))
STEM = OUT / f"hp_capacity_slice{i0}"
render(0.1, str(STEM), *SENS_BOX, nmse=True)

with open(f"{STEM}_source_data.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["panel", "experiment", "contrast", "x_million_shared", "value", "value_unit"])
    for c in C:
        for n_, (e, m) in enumerate(zip(CAP, millions)):
            w.writerow(["a,b", e, c, m, f"{SM[c][e]['psnr']:.4f}", f"PSNR dB, transverse slice {i0}, head pixels"])
            w.writerow(["c", e, c, m, f"{SM[c][e]['ssim']:.6f}", f"SSIM, transverse slice {i0}, head pixels"])
            w.writerow(["d", e, c, m, f"{SM[c][e]['nmse']:.6f}", f"NMSE, transverse slice {i0}, head pixels"])
            w.writerow(["b", e, c, m, f"{d_psnr[c][n_]:.4f}", "dB vs 3.6M"])
            w.writerow(["c", e, c, m, f"{d_ssim[c][n_]:.4f}", "% SSIM change vs 3.6M"])
            w.writerow(["d", e, c, m, f"{nm_gain[c][n_]:.3f}", "% NMSE reduction vs 3.6M"])
            w.writerow(["a", e, c, m, f"{box_psnr(c, e, *SENS_BOX):.3f}",
                        f"48-box PSNR dB, display row {SENS_BOX[0]} col {SENS_BOX[1]}"])
    for e, m, t, v, n in zip(CAP, millions, minutes, vram, count):
        w.writerow(["e", e, "joint", m, f"{t:.2f}", "wall-clock minutes"])
        w.writerow(["e", e, "joint", m, f"{v:.2f}", "peak GPU memory GB"])
        w.writerow(["e", e, "joint", m, f"{n:.0f}", "shared Gaussians"])
print("done")
