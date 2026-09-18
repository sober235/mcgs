#!/usr/bin/env python3
"""fig_lesion_seg.py -- the lesion figure of this topic, redrawn with a whole-brain truth column and a
public-segmentation-model row.

Thirteen columns: whole brain | truth crop | the eleven arms, all at AF 16.00 against this topic's truth (the
CS-SENSE reconstruction of the full AF 5.33 acquisition). Each scan gets three rows:

  image   the slice, grey P99.9/brightness window, lesion core outlined magenta, orange-yellow box = the zoom inset
          pasted top-right of the panel (the truth included), in-panel number = head-masked slice PSNR
  error   |rec - GT| / max|GT|, jet [0, 0.1], head-masked, in-panel numbers = shell RMSE and edge-geometry RMSE
  segment MedSAM run identically on that arm (see medsam_seg.py): cyan = the model's outline on the truth,
          red fill = what the arm's outline adds, blue fill = what it misses, in-panel Dice and HD95 against the
          truth outline. The dashed rectangle is the prompt box, i.e. the model's field of view.

Column 0 is the whole slice, once per scan under the image row (with the crop drawn in red) and once under the
segment row (with the model's truth outline), so the crop and the outline can be placed in the head.

CAVEAT THAT MUST TRAVEL WITH THE SEGMENT ROW: the truth is itself a TV-regularised CS-SENSE reconstruction and
62.7% of the AF16 samples were never measured in the original acquisition, so the truth's lesion boundary there is
prior extrapolation, not data. A segmenter keys on that boundary, which favours the CS-SENSE family: measured,
CS-SENSE and Joint SENSE take the top two places on both scans and MC-GS is 4th (pre) / 5th (post). The image and
error rows are not affected -- MC-GS is 1st of eleven on all six of their conditions on the chosen slice.

  env PYTHONNOUSERSITE=1 PYTHONPATH= <GSTMI python> scripts/fig_lesion_seg.py --axis 2 --index 41

Needs seg/medsam_*.npy and logs/medsam_seg_{pre,post}.json (medsam_seg.py) and logs/best_slice_lesion.json
(best_slice_lesion.py). Writes figures_seg/ce_retro_af16_lesion_seg_<axis><idx>.png (+ .pdf with --pdf) and copies
the png to ~/figs/ce_retro_seg/.
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
K_SHELL = 2
MCGS, GSSINGLE = "gs_joint", "gs_single"
METRIC_COLOR = "#FFC23D"
LESION_COLOR = "#FF33FF"
TRUTH_SEG_COLOR = "#00E5FF"
FP_COLOR = "#FF3B30"
FN_COLOR = "#3B6BFF"


def gmag_np(v):
    """forward-difference gradient magnitude, the same formula best_slice_lesion.py scores the edge term with"""
    gx = np.zeros_like(v); gy = np.zeros_like(v); gz = np.zeros_like(v)
    gx[:-1] = v[1:] - v[:-1]
    gy[:, :-1] = v[:, 1:] - v[:, :-1]
    gz[:, :, :-1] = v[:, :, 1:] - v[:, :, :-1]
    return np.sqrt(gx * gx + gy * gy + gz * gz + 1e-12)


def Dfun(x, axis, rot=1):
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


def slice_measures(keys, gt, head, gmax, vols, gmag_gt, gmag_arm, shell, sl):
    """head-masked slice PSNR, shell RMSE and edge-geometry RMSE for one slice -- the identical definitions
    best_slice_lesion.py ranks slices with; recomputed here so the figure does not depend on a stale json."""
    out = {}
    for c in SCANS:
        h, sh = head[c][sl], shell[sl]
        g_head, gsh = gt[c][sl][h], gt[c][sl][sh]
        peak_db = 20 * np.log10(gmax[c])
        psnr, rmse, gerr = {}, {}, {}
        for k in keys:
            e = vols[c][k][sl][h] - g_head
            mse = float((e * e).mean())
            psnr[k] = float(peak_db - 10 * np.log10(mse)) if mse > 0 else float("inf")
            d = vols[c][k][sl][sh] - gsh
            rmse[k] = float(np.sqrt((d * d).mean()))
            gd = gmag_arm[c][k][sl][sh] - gmag_gt[c][sl][sh]
            gerr[k] = float(np.sqrt((gd * gd).mean()))
        out[c] = dict(psnr=psnr, rmse=rmse, gerr=gerr, n_shell=int(sh.sum()), n_head=int(h.sum()))
    return out


def load_all():
    summary = json.load(open(R / "logs/summary.json"))
    gt, head, gmax, vols, gmag_gt, gmag_arm = {}, {}, {}, {}, {}, {}
    keys = None
    for c in SCANS:
        arms_c = arm_files(c, summary)
        ks = [k for k, _, _ in arms_c]
        keys = ks if keys is None else keys
        assert ks == keys == ORDER, (ks, keys)
        g = np.abs(np.load(R / f"recon/gt_{c}_complex.npy")).astype(np.float64)
        gt[c], gmax[c] = g, float(g.max())
        head[c] = g > 0.05 * float(np.quantile(g, 0.999))
        gmag_gt[c] = gmag_np(g)
        vols[c], gmag_arm[c] = {}, {}
        for key, _, path in arms_c:
            v = np.abs(np.load(path)).astype(np.float64)
            vols[c][key] = v
            gmag_arm[c][key] = gmag_np(v)
    return keys, gt, head, gmax, vols, gmag_gt, gmag_arm


def load_seg():
    """the MedSAM volumes and their metrics; returns (masks[scan][key], truth[scan], meta[scan])"""
    masks, truth, meta = {}, {}, {}
    for c in SCANS:
        meta[c] = json.load(open(R / f"logs/medsam_seg_{c}.json"))
        truth[c] = np.load(R / f"seg/medsam_truth_{c}.npy")
        masks[c] = {k: np.load(R / f"seg/medsam_{k}_{c}.npy") for k in ORDER}
    return masks, truth, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axis", type=int, default=2, choices=[0, 1, 2])
    ap.add_argument("--index", type=int, default=41)
    ap.add_argument("--brightness", type=float, default=1.5, help="display only: the grey window top is divided by this")
    ap.add_argument("--pad", type=int, default=15, help="crop padding around the dilated lesion core")
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--pdf_dpi", type=int, default=600)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    core = np.load(ROI_NPZ, allow_pickle=True)["core"]
    shell = ndi.binary_dilation(core, iterations=K_SHELL) & ~ndi.binary_erosion(core, iterations=K_SHELL)
    keys, gt, head, gmax, vols, gmag_gt, gmag_arm = load_all()
    masks, truth_seg, seg_meta = load_seg()

    axis, idx = a.axis, a.index
    sl = [slice(None)] * 3; sl[axis] = idx; sl = tuple(sl)
    meas = slice_measures(keys, gt, head, gmax, vols, gmag_gt, gmag_arm, shell, sl)

    my0, my1, mx0, mx1 = transformed_box(ndi.binary_dilation(core, iterations=K_SHELL)[sl], axis, pad=a.pad)
    iy0, iy1, ix0, ix1 = transformed_box(shell[sl], axis, pad=4)
    iy0, iy1, ix0, ix1 = iy0 - my0, iy1 - my0, ix0 - mx0, ix1 - mx0

    def crop(full2d):
        return Dfun(full2d, axis)[my0:my1, mx0:mx1]

    core2d = crop(core[sl].astype(float))
    H, W = my1 - my0, mx1 - mx0
    full_shape = Dfun(gt[SCANS[0]][sl], axis).shape

    # prompt box (the model's field of view) in crop coordinates
    pb = seg_meta[SCANS[0]]["box"]
    pbox = np.zeros(gt[SCANS[0]].shape, bool)
    pbox[pb[0]:pb[1] + 1, pb[2]:pb[3] + 1, pb[4]:pb[5] + 1] = True
    pb2d = crop(pbox[sl].astype(float)) > 0.5
    pb_rows = np.nonzero(pb2d.any(1))[0]
    pb_cols = np.nonzero(pb2d.any(0))[0]

    pw = 1.9
    ph = pw * H / W
    gap = 0.05
    h_block = 3 * ph + 2 * gap                       # one scan's three rows
    w0 = h_block * full_shape[1] / full_shape[0]     # whole-brain panel spans them, at the slice's true aspect
    top_in, cb_room = 0.62, 1.0
    ncol_arms = 1 + len(keys)                        # truth crop + arms
    fw = w0 + gap + ncol_arms * pw + (ncol_arms - 1) * gap + cb_room
    rows_spec = [(c, kind) for c in SCANS for kind in ("img", "err", "seg")]
    fh = top_in + len(rows_spec) * ph + (len(rows_spec) - 1) * gap + 0.32
    fig = plt.figure(figsize=(fw, fh))

    def axes_at(x0, y0, w, h):
        ax = fig.add_axes([x0 / fw, 1 - (y0 + h) / fh, w / fw, h / fh])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        return ax

    def label(ax, text, bold=False, line=0, fs=9.5, color=METRIC_COLOR):
        dy = line * fs * 1.3 / 72 / ph
        t = ax.text(0.03, 0.97 - dy, text, transform=ax.transAxes, ha="left", va="top", color=color,
                    fontsize=fs, fontweight="bold" if bold else "normal")
        t.set_path_effects([pe.withStroke(linewidth=2.2, foreground="black")])

    def put_zoom(x, y, img2d, vmax_c):
        side = 0.42 * pw
        za = fig.add_axes([(x + pw - side - 0.02) / fw, 1 - (y + side + 0.02) / fh, side / fw, side / fh])
        za.set_xticks([]); za.set_yticks([])
        for sp in za.spines.values():
            sp.set_visible(True); sp.set_edgecolor(METRIC_COLOR); sp.set_linewidth(1.4)
        za.imshow(img2d[iy0:iy1, ix0:ix1], cmap="gray", vmin=0, vmax=vmax_c, origin="lower", aspect="auto")

    def overlay(ax, arm_mask2d, truth_mask2d):
        """cyan outline = the model on the truth; red = the arm adds; blue = the arm misses"""
        fp = arm_mask2d & ~truth_mask2d
        fn = truth_mask2d & ~arm_mask2d
        for m, colr in ((fp, FP_COLOR), (fn, FN_COLOR)):
            if m.any():
                rgba = np.zeros(m.shape + (4,))
                rgba[..., :3] = matplotlib.colors.to_rgb(colr)
                rgba[..., 3] = m * 0.45
                ax.imshow(rgba, origin="lower", aspect="auto")
        ax.contour(truth_mask2d.astype(float), levels=[0.5], colors=TRUTH_SEG_COLOR, linewidths=1.2)
        ax.add_patch(Rectangle((pb_cols.min(), pb_rows.min()), pb_cols.max() - pb_cols.min(),
                               pb_rows.max() - pb_rows.min(), fill=False, ec="white", lw=0.7, ls=(0, (4, 3)),
                               alpha=0.6))

    seg_rows = {c: {r["key"]: r for r in seg_meta[c]["rows"]} for c in SCANS}
    best_dice = {c: max(ORDER, key=lambda k: seg_rows[c][k]["dice"]) for c in SCANS}
    best_hd = {c: min(ORDER, key=lambda k: seg_rows[c][k]["hd95"]) for c in SCANS}

    im_err = None
    for r_i, (c, kind) in enumerate(rows_spec):
        y = top_in + r_i * (ph + gap)
        g2 = crop(gt[c][sl])
        vmax_c = float(np.quantile(g2, 0.999)) / a.brightness
        h2 = crop(head[c][sl].astype(float)) > 0.5
        t2 = crop(truth_seg[c][sl].astype(float)) > 0.5

        # ---- column 0: the whole slice, one tall panel per scan (drawn on that scan's first row)
        if kind == "img":
            ax0 = axes_at(0, y, w0, h_block)
            full = Dfun(gt[c][sl], axis)
            fvmax = float(np.quantile(full, 0.999)) / a.brightness
            ax0.imshow(full, cmap="gray", vmin=0, vmax=fvmax, origin="lower", aspect="auto")
            ax0.contour(Dfun(truth_seg[c][sl].astype(float), axis), levels=[0.5], colors=TRUTH_SEG_COLOR,
                        linewidths=1.1)
            ax0.add_patch(Rectangle((mx0, my0), W, H, fill=False, ec=LESION_COLOR, lw=1.4))
            label(ax0, f"GT {c}, whole slice", bold=True, fs=11)
            label(ax0, "magenta: the crop below", line=1, fs=9)
            label(ax0, "cyan: MedSAM on GT", line=2, fs=9, color=TRUTH_SEG_COLOR)
            if r_i == 0:
                ax0.set_title("whole brain", fontsize=12, pad=4)

        # ---- column 1: the truth crop, then the arms
        for j in range(ncol_arms):
            x = w0 + gap + j * (pw + gap)
            ax = axes_at(x, y, pw, ph)
            if j == 0:
                if kind == "img":
                    ax.imshow(g2, cmap="gray", vmin=0, vmax=vmax_c, origin="lower", aspect="auto")
                    ax.contour(core2d, levels=[0.5], colors=LESION_COLOR, linewidths=1.1)
                    ax.add_patch(Rectangle((ix0, iy0), ix1 - ix0, iy1 - iy0, fill=False, ec=METRIC_COLOR, lw=1.6))
                    put_zoom(x, y, g2, vmax_c)
                    label(ax, f"GT ({c})", bold=True)
                elif kind == "err":
                    ax.imshow(core2d, cmap="gray", vmin=0, vmax=1, origin="lower", aspect="auto")
                    label(ax, "lesion core", bold=True, color="white")
                else:
                    ax.imshow(g2, cmap="gray", vmin=0, vmax=vmax_c, origin="lower", aspect="auto")
                    overlay(ax, t2, t2)
                    label(ax, f"MedSAM (GT) {seg_rows[c]['truth']['vox']} vox", bold=True, color=TRUTH_SEG_COLOR)
                if r_i == 0:
                    ax.set_title("GT", fontsize=12, pad=4)
                continue

            key = keys[j - 1]
            v2 = crop(vols[c][key][sl])
            if kind == "img":
                ax.imshow(v2, cmap="gray", vmin=0, vmax=vmax_c, origin="lower", aspect="auto")
                ax.contour(core2d, levels=[0.5], colors=LESION_COLOR, linewidths=1.1)
                ax.add_patch(Rectangle((ix0, iy0), ix1 - ix0, iy1 - iy0, fill=False, ec=METRIC_COLOR, lw=1.6))
                put_zoom(x, y, v2, vmax_c)
                ps = meas[c]["psnr"]
                label(ax, f"{ps[key]:.2f} dB", bold=key == max(ps, key=ps.get))
                if r_i == 0:
                    ax.set_title(LABEL[key], fontsize=12, pad=4)
            elif kind == "err":
                e2 = np.abs(v2 - g2) / gmax[c] * h2
                im_err = ax.imshow(e2, cmap="jet", vmin=0.0, vmax=0.1, origin="lower", aspect="auto")
                ax.contour(core2d, levels=[0.5], colors=LESION_COLOR, linewidths=1.1)
                rm, ge = meas[c]["rmse"], meas[c]["gerr"]
                label(ax, f"shell RMSE {rm[key]:.4f}", bold=key == min(rm, key=rm.get), line=0)
                label(ax, f"edge RMSE {ge[key]:.4f}", bold=key == min(ge, key=ge.get), line=1)
            else:
                ax.imshow(v2, cmap="gray", vmin=0, vmax=vmax_c, origin="lower", aspect="auto")
                overlay(ax, crop(masks[c][key][sl].astype(float)) > 0.5, t2)
                row = seg_rows[c][key]
                label(ax, f"Dice {row['dice']:.3f}", bold=key == best_dice[c], line=0, color=TRUTH_SEG_COLOR)
                label(ax, f"HD95 {row['hd95']:.2f} mm", bold=key == best_hd[c], line=1, color=TRUTH_SEG_COLOR)

    cax = fig.add_axes([(fw - cb_room + 0.15) / fw,
                        1 - (top_in + len(rows_spec) * ph + (len(rows_spec) - 1) * gap) / fh,
                        0.16 / fw, (len(rows_spec) * ph + (len(rows_spec) - 1) * gap) / fh])
    cb = fig.colorbar(im_err, cax=cax); cb.ax.tick_params(labelsize=9)
    cb.set_label("|err| / max|GT|", fontsize=9)

    plane = {0: "axial", 1: "coronal", 2: "sagittal"}[axis]
    fig.suptitle(f"CE retro AF 16.00 -- lesion at {plane} index {idx}; truth = CS-SENSE reconstruction of the "
                 f"AF 5.33 acquisition.  Rows per scan: image / error / MedSAM segmentation",
                 fontsize=11.5, fontweight="bold", y=1 - 0.06 / fh)
    fig.text(0.5, 1 - (top_in - 0.30) / fh,
             "magenta = stored lesion core; orange-yellow box = the zoom inset top-right of every image panel, the truth included; "
             "cyan = MedSAM's outline on the truth, red fill = what the arm's outline adds, blue fill = what it misses, "
             "dashed = the model's prompt box.  Dice/HD95 are 3-D, against the model's truth outline. "
             "Bold = best of the eleven arms for that scan and measure.",
             ha="center", va="top", fontsize=8.3)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR_SERVED.mkdir(parents=True, exist_ok=True)
    name = f"ce_retro_af16_lesion_seg_{axis}{idx:03d}"
    png = FIGDIR / f"{name}.png"
    fig.savefig(png, dpi=a.dpi)
    if a.pdf:
        fig.savefig(FIGDIR / f"{name}.pdf", dpi=a.pdf_dpi)
    plt.close(fig)
    shutil.copy2(png, FIGDIR_SERVED / png.name)
    print(f"wrote {png}\n      {FIGDIR_SERVED / png.name}")
    print(f"      http://localhost:8765/ce_retro_seg/{png.name}")
    for c in SCANS:
        ps = meas[c]["psnr"]
        print(f"[{c}] slice PSNR best {LABEL[max(ps, key=ps.get)]}; "
              f"MC-GS {ps[MCGS]:.2f} dB, GS-single {ps[GSSINGLE]:.2f} dB; "
              f"MedSAM Dice best {LABEL[best_dice[c]]} {seg_rows[c][best_dice[c]]['dice']:.3f}, "
              f"MC-GS {seg_rows[c][MCGS]['dice']:.3f}")


if __name__ == "__main__":
    main()
