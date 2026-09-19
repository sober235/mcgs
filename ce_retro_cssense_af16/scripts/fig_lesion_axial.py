#!/usr/bin/env python3
"""fig_lesion_axial.py -- the axial lesion figure: pre image / pre error / post image / post error, twelve columns
(truth plus the eleven arms), a zoom inset on every panel including the error panels.

Rows
  1  pre-contrast slice, grey P99.9 window divided by --brightness (display only; every number is computed on the
     unscaled data), lesion core outlined magenta, in-panel number = head-masked slice PSNR
  2  |rec - GT| / max|GT| of the same slice, jet [0, 0.1], head-masked, in-panel numbers = shell RMSE and
     edge-geometry RMSE of that slice
  3  4  the same two rows for the post-contrast scan

Every panel carries the same zoom box -- the lesion's boundary shell grown by 4 voxels -- magnified into the corner
given by --inset_corner. The error panels get it too, at the same place and scale, so the eye compares like with
like across all four rows.

The quantitative lesion story is NOT in this figure: contrast, CNR, edge width and invented enhancement are 3-D
numbers over the whole lesion and live in fig_lesion_metrics.py / logs/lesion_preservation.json. Keeping the
per-slice numbers here and the 3-D numbers there avoids printing two different gauges in one panel.

  env PYTHONNOUSERSITE=1 PYTHONPATH= <GSTMI python> scripts/fig_lesion_axial.py --index 201

Writes figures_seg/ce_retro_af16_lesion_axial_<idx>.png (+ .pdf with --pdf), copied to ~/figs/ce_retro_seg/.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from arms import ORDER, LABEL, arm_files

R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"))
PROS = Path(os.environ.get("CE_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast"))
ROI_NPZ = PROS / "lesion_candidates/enhance_rois_cssense.npz"
FIGDIR = R / "figures_seg"
FIGDIR_SERVED = Path("/home/congcongliu/figs/ce_retro_seg")

SCANS = ("pre", "post")
AXIS = 0                       # this grid is (S-I, A-P, L-R): axis 0 is the axial plane
K_SHELL = 2
METRIC_COLOR = "#FFC23D"
LESION_COLOR = "#FF33FF"


def gmag(v):
    gx = np.zeros_like(v); gy = np.zeros_like(v); gz = np.zeros_like(v)
    gx[:-1] = v[1:] - v[:-1]
    gy[:, :-1] = v[:, 1:] - v[:, :-1]
    gz[:, :, :-1] = v[:, :, 1:] - v[:, :, :-1]
    return np.sqrt(gx * gx + gy * gy + gz * gz + 1e-12)


def Dfun(x, axis, rot=1):
    """axial slices are transposed and rotated so the eyes point up, the topic's --tra_rot 1 convention"""
    return np.rot90(x.T, rot) if axis == 0 else x


def transformed_box(mask2d, axis, pad):
    rr0, cc0 = np.nonzero(mask2d)
    r0, r1 = max(int(rr0.min()) - pad, 0), min(int(rr0.max()) + pad + 1, mask2d.shape[0])
    c0, c1 = max(int(cc0.min()) - pad, 0), min(int(cc0.max()) + pad + 1, mask2d.shape[1])
    box = np.zeros_like(mask2d, dtype=bool)
    box[r0:r1, c0:c1] = True
    dbox = Dfun(box, axis)
    ry, rx = np.nonzero(dbox)
    return int(ry.min()), int(ry.max()) + 1, int(rx.min()), int(rx.max()) + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=int, default=201, help="axial slice (the topic's lesion slice is 201)")
    ap.add_argument("--pad", type=int, default=15, help="crop padding around the dilated lesion core")
    ap.add_argument("--whole", action="store_true",
                    help="show the whole axial slice instead of the lesion crop; the zoom box and its inset are "
                         "unchanged, so the lesion is still read at full magnification in the corner")
    ap.add_argument("--inset", type=int, default=4, help="zoom box = the boundary shell grown by this many voxels")
    ap.add_argument("--inset_corner", default="tr", choices=["bl", "br", "tl", "tr"])
    ap.add_argument("--contour", default="all", choices=["all", "err", "none"],
                    help="where to draw the magenta lesion outline. 'all' is the inherited style; 'err' keeps it on "
                         "the truth column and the error panels only, because on an arm's image panel the line "
                         "covers the very boundary the reader is being asked to judge, and the orange box already "
                         "says where the lesion is; 'none' leaves the box to do that job alone")
    ap.add_argument("--arms", default="",
                    help="comma-separated subset of the eleven arm keys, in the order they should appear "
                         "(default: all of them, in the topic's column order). 'best of' bolding is then computed "
                         "over the arms actually shown, not over the eleven, and the caption says so")
    ap.add_argument("--brightness", type=float, default=1.5)
    ap.add_argument("--err_vmax", type=float, default=0.1)
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--pdf_dpi", type=int, default=600)
    a = ap.parse_args()
    keys = ORDER if not a.arms else [k.strip() for k in a.arms.split(",") if k.strip()]
    unknown = [k for k in keys if k not in ORDER]
    if unknown:
        raise SystemExit(f"unknown arm key(s) {unknown}; choose from {ORDER}")

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    core = np.load(ROI_NPZ, allow_pickle=True)["core"]
    shell = ndi.binary_dilation(core, iterations=K_SHELL) & ~ndi.binary_erosion(core, iterations=K_SHELL)
    idx = a.index
    sl = [slice(None)] * 3; sl[AXIS] = idx; sl = tuple(sl)
    if core[sl].sum() == 0:
        raise SystemExit(f"axial slice {idx} holds no lesion core")

    summary = json.load(open(R / "logs/summary.json"))
    gt, head, gmax, vols, gg, ga = {}, {}, {}, {}, {}, {}
    for c in SCANS:
        g = np.abs(np.load(R / f"recon/gt_{c}_complex.npy")).astype(np.float64)
        gt[c], gmax[c] = g, float(g.max())
        head[c] = g > 0.05 * float(np.quantile(g, 0.999))
        gg[c] = gmag(g)
        vols[c], ga[c] = {}, {}
        for key, _, path in arm_files(c, summary):
            if key not in keys:
                continue
            v = np.abs(np.load(path)).astype(np.float64)
            vols[c][key] = v
            ga[c][key] = gmag(v)

    if a.whole:
        fh_, fw_ = Dfun(gt[SCANS[0]][sl], AXIS).shape
        my0, my1, mx0, mx1 = 0, fh_, 0, fw_
    else:
        my0, my1, mx0, mx1 = transformed_box(ndi.binary_dilation(core, iterations=K_SHELL)[sl], AXIS, pad=a.pad)
    iy0, iy1, ix0, ix1 = transformed_box(shell[sl], AXIS, pad=a.inset)
    iy0, iy1, ix0, ix1 = iy0 - my0, iy1 - my0, ix0 - mx0, ix1 - mx0

    def crop(f2d):
        return Dfun(f2d, AXIS)[my0:my1, mx0:mx1]

    core2d = crop(core[sl].astype(float))
    H, W = my1 - my0, mx1 - mx0

    # per-slice numbers on the FULL slice, so they do not change with --whole or the crop padding.
    # PSNR / SSIM / NMSE follow diag_artefact/headonly.py's frozen head-only definitions, applied to one slice:
    # head = |GT| > 0.05 * P99.9(|GT|) of the volume, peak = max|GT| of the volume, SSIM averaged over the head
    # voxels of the slice's 2-D map (a 7x7 window here against the volume metric's 7^3, so slice and volume SSIM
    # are not interchangeable -- the ranking is what carries over).
    from skimage.metrics import structural_similarity
    meas = {}
    for c in SCANS:
        h, sh = head[c][sl], shell[sl]
        g2d = gt[c][sl]
        g_head, gsh = g2d[h], g2d[sh]
        peak_db = 20 * np.log10(gmax[c])
        e_gt = float((g_head ** 2).sum())
        psnr, ssim, nmse, rmse, gerr = {}, {}, {}, {}, {}
        for k in keys:
            v2d = vols[c][k][sl]
            e = v2d[h] - g_head
            mse = float((e * e).mean())
            psnr[k] = float(peak_db - 10 * np.log10(mse)) if mse > 0 else float("inf")
            nmse[k] = float((e * e).sum() / e_gt)
            _, smap = structural_similarity(v2d, g2d, data_range=gmax[c], full=True)
            ssim[k] = float(smap[h].mean())
            d = v2d[sh] - gsh
            rmse[k] = float(np.sqrt((d * d).mean()))
            gd = ga[c][k][sl][sh] - gg[c][sl][sh]
            gerr[k] = float(np.sqrt((gd * gd).mean()))
        meas[c] = dict(psnr=psnr, ssim=ssim, nmse=nmse, rmse=rmse, gerr=gerr)

    ncol = 1 + len(keys)
    pw = 1.9
    ph = pw * H / W
    gap = 0.05
    top_in, cb_room = 1.05, 1.0        # room for a two-line caption above the column titles
    rows_spec = [(c, kind) for c in SCANS for kind in ("img", "err")]
    fw = ncol * pw + (ncol - 1) * gap + cb_room
    fh = top_in + len(rows_spec) * ph + (len(rows_spec) - 1) * gap + 0.30
    fig = plt.figure(figsize=(fw, fh))

    def axes_at(x0, y0, w, h):
        ax = fig.add_axes([x0 / fw, 1 - (y0 + h) / fh, w / fw, h / fh])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        return ax

    def label(ax, text, bold=False, line=0, fs=9.5, color=METRIC_COLOR, bottom=False):
        """bottom=True keeps a label clear of the zoom inset, which always sits in the top-right corner"""
        dy = line * fs * 1.3 / 72 / ph
        y0, va = (0.03 + dy, "bottom") if bottom else (0.97 - dy, "top")
        t = ax.text(0.03, y0, text, transform=ax.transAxes, ha="left", va=va, color=color,
                    fontsize=fs, fontweight="bold" if bold else "normal")
        t.set_path_effects([pe.withStroke(linewidth=2.2, foreground="black")])

    def put_zoom(x, y, img2d, **imkw):
        """the same box, the same corner, on every panel of every row -- error panels included"""
        side = 0.42 * pw
        dx = pw - side - 0.02 if a.inset_corner in ("tr", "br") else 0.02
        dy = 0.02 if a.inset_corner in ("tl", "tr") else ph - side - 0.02
        za = fig.add_axes([(x + dx) / fw, 1 - (y + dy + side) / fh, side / fw, side / fh])
        za.set_xticks([]); za.set_yticks([])
        for sp in za.spines.values():
            sp.set_visible(True); sp.set_edgecolor(METRIC_COLOR); sp.set_linewidth(1.4)
        za.imshow(img2d[iy0:iy1, ix0:ix1], origin="lower", aspect="auto", **imkw)

    im_err = None
    for r_i, (c, kind) in enumerate(rows_spec):
        y = top_in + r_i * (ph + gap)
        g2 = crop(gt[c][sl])
        vmax_c = float(np.quantile(g2, 0.999)) / a.brightness
        h2 = crop(head[c][sl].astype(float)) > 0.5
        for j in range(ncol):
            x = j * (pw + gap)
            ax = axes_at(x, y, pw, ph)
            if j == 0:
                if kind == "img":
                    ax.imshow(g2, cmap="gray", vmin=0, vmax=vmax_c, origin="lower", aspect="auto")
                    if a.contour != "none":
                        ax.contour(core2d, levels=[0.5], colors=LESION_COLOR, linewidths=1.1)
                    ax.add_patch(Rectangle((ix0, iy0), ix1 - ix0, iy1 - iy0, fill=False, ec=METRIC_COLOR, lw=1.6))
                    put_zoom(x, y, g2, cmap="gray", vmin=0, vmax=vmax_c)
                    label(ax, f"GT ({c})", bold=True)
                else:
                    ax.imshow(core2d, cmap="gray", vmin=0, vmax=1, origin="lower", aspect="auto")
                    put_zoom(x, y, core2d, cmap="gray", vmin=0, vmax=1)
                    label(ax, "lesion core", bold=True, color="white", bottom=True)
                if r_i == 0:
                    ax.set_title("GT", fontsize=12, pad=4)
                continue
            key = keys[j - 1]
            v2 = crop(vols[c][key][sl])
            if kind == "img":
                ax.imshow(v2, cmap="gray", vmin=0, vmax=vmax_c, origin="lower", aspect="auto")
                if a.contour == "all":
                    ax.contour(core2d, levels=[0.5], colors=LESION_COLOR, linewidths=1.1)
                ax.add_patch(Rectangle((ix0, iy0), ix1 - ix0, iy1 - iy0, fill=False, ec=METRIC_COLOR, lw=1.6))
                put_zoom(x, y, v2, cmap="gray", vmin=0, vmax=vmax_c)
                ps, ss, nm = meas[c]["psnr"], meas[c]["ssim"], meas[c]["nmse"]
                label(ax, f"{ps[key]:.2f} dB", bold=key == max(ps, key=ps.get), line=0)
                label(ax, f"SSIM {ss[key]:.4f}", bold=key == max(ss, key=ss.get), line=1)
                label(ax, f"NMSE {nm[key]:.4f}", bold=key == min(nm, key=nm.get), line=2)
                if r_i == 0:
                    ax.set_title(LABEL[key], fontsize=12, pad=4)
            else:
                e2 = np.abs(v2 - g2) / gmax[c] * h2
                im_err = ax.imshow(e2, cmap="jet", vmin=0.0, vmax=a.err_vmax, origin="lower", aspect="auto")
                if a.contour != "none":
                    ax.contour(core2d, levels=[0.5], colors=LESION_COLOR, linewidths=1.1)
                put_zoom(x, y, e2, cmap="jet", vmin=0.0, vmax=a.err_vmax)
                rm, ge = meas[c]["rmse"], meas[c]["gerr"]
                label(ax, f"edge {ge[key]:.4f}", bold=key == min(ge, key=ge.get), line=0, bottom=True)
                label(ax, f"shell {rm[key]:.4f}", bold=key == min(rm, key=rm.get), line=1, bottom=True)

    cax = fig.add_axes([(ncol * pw + (ncol - 1) * gap + 0.15) / fw,
                        1 - (top_in + len(rows_spec) * ph + (len(rows_spec) - 1) * gap) / fh,
                        0.16 / fw, (len(rows_spec) * ph + (len(rows_spec) - 1) * gap) / fh])
    cb = fig.colorbar(im_err, cax=cax); cb.ax.tick_params(labelsize=9)
    cb.set_label("|err| / max|GT|", fontsize=9)

    fig.suptitle(f"CE retro AF 16.00 -- axial slice {idx}; rows: pre image / pre error / post image / post error.  "
                 f"Truth = CS-SENSE reconstruction of the AF 5.33 acquisition",
                 fontsize=11.5, fontweight="bold", y=1 - 0.06 / fh)
    fig.text(0.5, 1 - 0.34 / fh,
             ("" if a.contour == "none" else "magenta = lesion core; ") +
             "orange-yellow box = the zoom inset in the top-right of every panel, error panels and the truth "
             "included; errors are head-masked, jet [0, %.2f].\n"
             "Image panels: head-masked slice PSNR / SSIM / NMSE.  Error panels: edge / shell = the RMSE of this "
             "slice's gradient field and of its boundary band.  Bold = best of the %d arms shown."
             % (a.err_vmax, len(keys)),
             ha="center", va="top", fontsize=8.5)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR_SERVED.mkdir(parents=True, exist_ok=True)
    name = (f"ce_retro_af16_lesion_axial_{idx}" + ("_whole" if a.whole else "")
            + (f"_b{a.brightness:g}" if a.brightness != ap.get_default("brightness") else "")
            + (f"_c{a.contour}" if a.contour != ap.get_default("contour") else "")
            + (f"_a{len(keys)}" if len(keys) != len(ORDER) else "")
            + (f"_{a.dpi}dpi" if a.dpi != ap.get_default("dpi") else ""))
    png = FIGDIR / f"{name}.png"
    fig.savefig(png, dpi=a.dpi)
    if a.pdf:
        fig.savefig(FIGDIR / f"{name}.pdf", dpi=a.pdf_dpi)
    plt.close(fig)
    shutil.copy2(png, FIGDIR_SERVED / png.name)
    print(f"wrote {png}\n      http://localhost:8765/ce_retro_seg/{png.name}")
    for c in SCANS:
        ps = meas[c]["psnr"]
        print(f"[{c}] slice PSNR best {LABEL[max(ps, key=ps.get)]} {max(ps.values()):.2f} dB; "
              f"MC-GS {ps['gs_joint']:.2f}, GS-single {ps['gs_single']:.2f}")


if __name__ == "__main__":
    main()
