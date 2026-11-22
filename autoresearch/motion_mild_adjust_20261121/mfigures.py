#!/usr/bin/env python3
"""Figures of the second version (runs on amax, CPU). No clock value appears on any figure.

    mfigures.py --feas     the feasibility check: z score of every echo train, AUC along the training, T2 PSNR
    mfigures.py --all      the result figures, from <root>/eval/summary.json and the runs' files
    mfigures.py --record --conds DTa DTc [--tag DTa:B2=d2 DTc:B123=d3] [--name NAME]
                           only the image figure, from the runs that exist (an interim look while the matrix runs)

The image figure has the format of the different-mask record figure, article look (user, 2026-11-22: "consistent with
the styles of different masks"; diffmask_af16_masks_20261111/fig_masks_record_format.py --role --err_cmap inferno
--arrow): one transverse slice, a row pair per condition and contrast (image row; under it the magnified box of the
image | the same box of the error map), columns GT, ZF, the models, the full method last; head-masked slice PSNR /
SSIM / NMSE in every panel; one grey bar and one error bar; red arrows in the magnified image boxes. Here the three
sampling masks become the motion conditions, and the mask cell marks the lines acquired during the motion.

Colours: blue = echo-train weights (M2), orange = the full arm (M1 + M2 + M3), aqua = registration pose alone (M1),
grey = the unchanged model; a truth upper bound is the outlined bar of its method's colour. The pairs that sit side by
side were checked with the chart skill's palette validator (colour-vision separation and normal-vision floor pass).
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from mcommon import CONTAM_MM, DATA2, ORDER_NAME, RES2, ROOT2, THRESH2
from mcommon import c1

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = ROOT2 / "figures"
C_CLEAN, C_BAD, C_LINE = "#4c72b0", "#c44e52", "#555555"
BLUE, ORANGE, AQUA, GREY, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a86", "#1a1a19", "#5f5e5a", "#e4e3df"
ARM_LABEL = {"B0": "B0  MC-GS unchanged", "B1": "B1  registration pose (M1)", "B2": "B2  echo-train weights (M2)",
             "B123": "B123  M1 + M2 + M3", "Ow": "Ow  truth weights (upper bound of M2)",
             "Op": "Op  truth states, poses and weights (upper bound)"}
# (arm, colour, filled): a truth upper bound is drawn as the outline of its method's colour
REC_ARMS = (("B2", BLUE, True), ("Ow", BLUE, False), ("B123", ORANGE, True), ("Op", ORANGE, False))


def _style(a, grid="y"):
    a.set_facecolor("white")
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a.spines[s].set_color(MUTED)
        a.spines[s].set_linewidth(0.8)
    a.tick_params(colors=MUTED, labelsize=9, length=3)
    a.grid(axis=grid, color=GRID, lw=0.8)
    a.set_axisbelow(True)


PDF_DPI = None                      # set by --all: every figure is also written as a PDF at this resolution


def _save(fig, out_png, dpi, face="white"):
    out_png = Path(out_png)
    fig.savefig(out_png, dpi=dpi, facecolor=face)
    if PDF_DPI:
        fig.savefig(out_png.with_suffix(".pdf"), dpi=PDF_DPI, facecolor=face)
    plt.close(fig)
    return out_png


# ------------------------------------------------------------------------------------------- recovered loss
def recovered_data(summary):
    """{"il": [(cond, {arm: share}), ...], "kz": [...]} for the bar figure. A condition whose loss is below the
    minimum has {"_loss": loss} instead (no share is defined there)."""
    out = {}
    for order, conds in (("il", ("Ta", "Tb", "Tc", "E2Tb")), ("kz", ("Ta", "Tc"))):
        rows = []
        for cond in conds:
            rec = summary["recovered"].get(f"{order}:{cond}") or {}
            vals = {a: rec[a]["value"] for a, _, _ in REC_ARMS if rec.get(a) and rec[a]["value"] is not None}
            loss = next((r["loss"] for r in rec.values() if r), None)
            rows.append((cond, vals if vals or loss is None else {"_loss": loss}))
        out[order] = rows
    return out


def recovered_figure(summary, out_png, dpi=200):
    data = recovered_data(summary)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True, gridspec_kw=dict(width_ratios=[4, 2], wspace=0.06))
    width = 0.19
    for a, order, name in zip(axes, ("il", "kz"), ("interleaved order (main)", "linear order (control, reported only)")):
        _style(a)
        for i, (cond, vals) in enumerate(data[order]):
            if "_loss" in vals:
                a.text(i, 0.04, f"loss {vals['_loss']:.2f} dB\n< {THRESH2['min_loss_db']} dB:\nnot significant", ha="center",
                       va="bottom", fontsize=8, color=MUTED)
                continue
            for j, (arm, col, filled) in enumerate(REC_ARMS):
                if arm not in vals:
                    continue
                x, v = i + (j - 1.5) * (width + 0.02), vals[arm]
                a.bar(x, v, width=width, color=col if filled else "white", edgecolor=col, linewidth=1.4)
                a.text(x, v + (0.02 if v >= 0 else -0.02), f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top",
                       fontsize=8, color=INK)
        a.axhline(0.0, color=MUTED, lw=0.8)
        a.set_xticks(range(len(data[order])))
        a.set_xticklabels([c for c, _ in data[order]], fontsize=10, color=INK)
        a.set_xlim(-0.6, len(data[order]) - 0.4)
        a.set_title(name, fontsize=10, color=INK, loc="left")
    axes[0].axhline(THRESH2["s3_recover"], color=INK, lw=1.0, ls="--")
    axes[0].set_ylabel("share of the T2 loss taken back\n(0 = unchanged MC-GS, 1 = level without motion)", fontsize=9,
                       color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=col if filled else "white", edgecolor=col, linewidth=1.4)
               for _, col, filled in REC_ARMS]
    handles.append(plt.Line2D([0], [0], color=INK, lw=1.0, ls="--"))
    fig.legend(handles, [ARM_LABEL[a] for a, _, _ in REC_ARMS]
               + [f"criterion S3 for B123, interleaved order: at least {THRESH2['s3_recover']}"],
               loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Mild motion inside the T2W scan: how much of the lost T2 PSNR each arm takes back", fontsize=11,
                 color=INK, x=0.06, ha="left")
    fig.subplots_adjust(left=0.09, right=0.985, top=0.86, bottom=0.24)
    return _save(fig, out_png, dpi)


# ------------------------------------------------------------------------------------------- echo trains
def trains_figure(truth, mm, title, out_png, dpi=200):
    """Three panels over the echo trains in acquisition order: the true mean brain displacement, the statistic z of
    the last computation, the final weight. True and found state boundaries as vertical lines."""
    disp = np.asarray(truth["train_brain_disp_mm"], dtype=np.float64)
    cont = np.asarray(truth["contaminated"], dtype=bool)
    n = np.arange(disp.size)
    z = mm["m2"].get("z")
    w = np.asarray(mm["m2"]["w"], dtype=np.float64)
    st = mm["m2"].get("settings") or {}
    true_b = [int(i) + 1 for i in np.flatnonzero(np.diff(np.asarray(truth.get("state_of_train", [0] * disp.size))))]
    found_b = list((mm.get("m3") or {}).get("boundaries") or [])
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 6.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.4, 1], hspace=0.12))
    for a in axes:
        _style(a)
        for b in true_b:
            a.axvline(b - 0.5, color=MUTED, lw=1.0, ls="--")
        for b in found_b:
            a.axvline(b - 0.5, color=INK, lw=1.4)
    a = axes[0]
    a.bar(n, disp, width=0.9, color=GREY)
    a.set_ylabel("true mean brain\ndisplacement, mm", fontsize=9, color=INK)
    a = axes[1]
    if z is not None:
        z = np.asarray(z, dtype=np.float64)
        a.plot(n[~cont], z[~cont], "o", ms=3.5, color=BLUE, label=f"clean echo trains ({int((~cont).sum())})")
        a.plot(n[cont], z[cont], "o", ms=3.5, color=ORANGE, label=f"trains with motion > {truth.get('contam_mm', 0.2)} mm ({int(cont.sum())})")
        for key in ("z0", "z1"):
            if st.get(key) is not None:
                a.axhline(float(st[key]), color=MUTED, lw=0.8, ls=":")
                a.text(n[-1] + 0.5, float(st[key]), f"{key} = {st[key]:g} ", ha="right", va="bottom", fontsize=8, color=MUTED)
        a.legend(fontsize=8, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.1)
    else:
        a.text(0.5, 0.5, "no statistic recorded", transform=a.transAxes, ha="center", va="center", fontsize=9, color=MUTED)
    a.set_ylabel("statistic z\n(last computation)", fontsize=9, color=INK)
    a = axes[2]
    a.step(n, w, where="mid", color=BLUE, lw=2.0)
    a.set_ylim(-0.08, 1.12)
    a.set_ylabel("final weight", fontsize=9, color=INK)
    a.set_xlabel("echo train, in acquisition order", fontsize=9, color=INK)
    a.set_xlim(-1, disp.size)
    notes = []
    if true_b:
        notes.append("dashed line: true lasting shift")
    if found_b:
        notes.append("solid line: state boundary found by M3")
    fig.suptitle(title + ("   (" + "; ".join(notes) + ")" if notes else ""), fontsize=10.5, color=INK, x=0.09, ha="left")
    fig.subplots_adjust(left=0.09, right=0.985, top=0.93, bottom=0.08)
    return _save(fig, out_png, dpi)


# ------------------------------------------------------------------------------------------- images
METRIC_COLOR = "#FFC23D"
ZC, ZLW = METRIC_COLOR, 2.2
ARROW_COLOR = "#FF1A1A"
METRICS = ("psnr", "ssim", "nmse")
# columns after GT; the full method is the last one. A column takes the first of its arms that a condition has.
REC_COLS = ("ZF", "GS-single", "MC-GS", "MC-GS + M2", "True motion", "MC-GS + M1-M3")
REC_TRUTH = ("True motion",)
REC_ARMS_OF = {"GS-single": ("I-T2",), "MC-GS": ("B0",), "MC-GS + M2": ("B2", "B12"), "True motion": ("Op", "Ow"),
               "MC-GS + M1-M3": ("B123",)}
REC_TAG = {"B12": "with M1 (arm B12)", "Ow": "true weights (arm Ow)", "Op": "true states, poses, weights (arm Op)"}
# row labels (they must fit the 2.6 in panel): Ta one brief movement, Tb two brief movements and a small lasting
# one, Tc a lasting shift, E2Tb the inter-scan misalignment plus Tb
COND_NAME = {"C0": "No motion (C0)", "Ta": "Brief motion (Ta)", "Tb": "Two motions (Tb)", "Tc": "Lasting shift (Tc)",
             "E2Tb": "Inter-scan + Tb", "TbL": "Tb with lesions", "DTa": "Brief motion (DTa)",
             "DTc": "Lasting shift (DTc)"}


def head_region(m):
    """Region of the error maps: the filled head without thin protrusions and a 2-pixel rim (house rule)."""
    from scipy.ndimage import binary_dilation, binary_fill_holes, binary_opening
    disk = np.hypot(*np.mgrid[-6:7, -6:7]) <= 6
    return binary_dilation(binary_fill_holes(binary_opening(binary_fill_holes(m), structure=disk)), iterations=2)


def box_origin(score, box):
    """(y0, x0) of a box x box window centred on the largest value of the 2-D score, moved back inside the slice."""
    y, x = np.unravel_index(int(np.argmax(score)), score.shape)
    return (int(min(max(y - box // 2, 0), score.shape[0] - box)), int(min(max(x - box // 2, 0), score.shape[1] - box)))


def record_numbers(row, k, truth_cols=()):
    """Panel numbers of slice k (axis 0) of one row: head = |GT| > 0.05 P99.9 of the volume, PSNR at max|GT| of the
    volume, 7x7 SSIM map averaged over the head pixels, head NMSE. best = per metric the best column among those that
    do not use the true motion."""
    from figures import plane_metrics                       # first version: the house panel metrics
    gt = row["gt"]
    gmax = float(gt.max())
    head = gt[k] > 0.05 * float(np.quantile(gt, 0.999))
    num = {lab: plane_metrics(v[k], gt[k], head, gmax) for lab, v in row["cells"].items()}
    own = [lab for lab in num if lab not in truth_cols]
    best = {m: (min if m == "nmse" else max)(own, key=lambda lab: num[lab][m]) for m in METRICS} if own else {}
    return dict(num=num, best=best, gmax=gmax, head=head)


def arrow_point(g2, errs, others, mc, box_yx, box):
    """(x, y, gap) inside the box: where the error of the other columns (median, 7x7 local mean) exceeds the error of
    column mc most, among the tissue boundaries of the box (steepest fifth of the GT gradient). None when the row has
    no column mc or none of the others."""
    from scipy.ndimage import gaussian_filter, uniform_filter
    others = [o for o in others if o in errs]
    if mc not in errs or not others:
        return None
    y0, x0 = box_yx
    sm = lambda e: uniform_filter(np.asarray(e, dtype=np.float64), size=7, mode="nearest")
    sc = (np.median([sm(errs[o]) for o in others], axis=0) - sm(errs[mc]))[y0:y0 + box, x0:x0 + box].copy()
    m = 7
    sc[:m] = sc[-m:] = -np.inf
    sc[:, :m] = sc[:, -m:] = -np.inf
    gy, gx = np.gradient(gaussian_filter(np.asarray(g2, dtype=np.float64), 1.0))
    gmag = np.hypot(gx, gy)[y0:y0 + box, x0:x0 + box]
    sc[gmag < np.percentile(gmag, 80)] = -np.inf
    if not np.isfinite(sc).any():
        return None
    yy, xx = np.unravel_index(int(np.argmax(sc)), sc.shape)
    return int(xx), int(yy), float(sc[yy, xx])


def _wrap(parts, width_in, fs):
    """Join caption parts with ';   ' and break the text between words so that no line is wider than the figure."""
    meas = plt.figure(figsize=(1, 1))
    rend = meas.canvas.get_renderer()
    wide = lambda text: meas.text(0, 0, text, fontsize=fs).get_window_extent(rend).width > 0.97 * width_in * 100.0
    lines, cur = [], ""
    for i, part in enumerate(parts):
        for j, word in enumerate(part.split()):
            sep = "" if not cur else (";   " if j == 0 else " ")
            if cur and wide(cur + sep + word):
                lines.append(cur + (";" if j == 0 else ""))
                cur = word
            else:
                cur = cur + sep + word
    plt.close(meas)
    return lines + [cur]


def view_box(heads, margin):
    """(y0, y1, x0, x1): the bounding box of the 2-D head masks with a margin, inside the slice."""
    m = np.any(np.asarray(heads, dtype=bool), axis=0)
    yy, xx = np.nonzero(m)
    return (int(max(yy.min() - margin, 0)), int(min(yy.max() + 1 + margin, m.shape[0])),
            int(max(xx.min() - margin, 0)), int(min(xx.max() + 1 + margin, m.shape[1])))


def mask_codes(mask, train_of, contaminated):
    """(ky, kz) plane: 0 not sampled, 1 sampled, 2 sampled by an echo train that ran while the brain was displaced.
    mask = the sampled lines; train_of = echo train of every line of the support (-1 outside), None for a scan
    without motion."""
    codes = (np.asarray(mask) > 0).astype(np.int8)
    if train_of is not None:
        cont = np.asarray(contaminated, dtype=bool)
        t = np.asarray(train_of)
        codes[(codes > 0) & (t >= 0) & cont[np.clip(t, 0, cont.size - 1)]] = 2
    return codes


def record_figure(rows, cols, k, box_yx, box, title, caption, out_png, dpi=200, truth_cols=(), err_vmax=0.1,
                  brightness=1.25, title_fs=22.0):
    """The record-format figure (geometry of the different-mask record figure). rows: dicts with name, contrast, gt
    (3-D), cells {column: 3-D volume}, optional tags {column: text}, blank {column: text of an empty cell}, kmask
    (2-D: 0 not sampled, 1 sampled, 2 sampled during the motion). The slice is drawn as stored (anterior up). Writes
    the PNG and, beside it, a JSON record of every panel number."""
    import matplotlib.patheffects as pe
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.patches import Rectangle
    y0b, x0b = box_yx
    mc = cols[-1]
    info = []
    for r in rows:
        n = record_numbers(r, k, truth_cols)
        g2 = r["gt"][k].astype(np.float64)
        n["g2"], n["vmax"], n["emask"] = g2, float(np.quantile(g2, 0.999)) / brightness, head_region(n["head"])
        n["err"] = {lab: np.abs(v[k].astype(np.float64) - g2) / n["gmax"] * n["emask"] for lab, v in r["cells"].items()}
        n["arrow"] = arrow_point(g2 / n["gmax"], n["err"], [c for c in cols[:-1] if c != "ZF" and c not in truth_cols],
                                 mc, box_yx, box)
        info.append(n)
    vy0, vy1, vx0, vx1 = view_box([n["emask"] for n in info], 10)
    vy0, vy1, vx0, vx1 = min(vy0, y0b), max(vy1, y0b + box), min(vx0, x0b), max(vx1, x0b + box)
    view = lambda img: img[vy0:vy1, vx0:vx1]
    ncol = len(cols) + 1
    pw, gap, zgap = 2.6, 0.025, 0.02
    cbtrim, cbpad, cbw, cbroom = 0.10, 0.06, 0.13, 0.95
    half = (pw - zgap) / 2
    H, W = vy1 - vy0, vx1 - vx0
    h, eh = pw * H / W, half
    fw = ncol * pw + (ncol - 1) * gap + cbpad + cbw + cbroom
    cap_lines = _wrap(caption, fw, 11)
    cap_y = 0.10 + 13 * 1.35 / 72 + 0.06
    top_in = cap_y + len(cap_lines) * 11 * 1.32 / 72 + 0.10 + (title_fs * 1.2 + 4) / 72
    fh = top_in + len(rows) * (h + eh) + (2 * len(rows) - 1) * gap + 0.16
    fig = plt.figure(figsize=(fw, fh))

    def axes_at(x0, y0, w, hh, face=None):
        a = fig.add_axes([x0 / fw, 1 - (y0 + hh) / fh, w / fw, hh / fh])
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
        if face:
            a.set_facecolor(face)
        return a

    def label(a, text, bold=False, line=0, fs=12.0, hh=1.0, color="white", bottom=False):
        t = a.text(0.025, 0.025 if bottom else 0.975 - line * fs * 1.28 / 72 / hh, text, transform=a.transAxes, ha="left",
                   va="bottom" if bottom else "top", color=color, fontsize=fs, fontweight="bold" if bold else "normal")
        t.set_path_effects([pe.withStroke(linewidth=2.5, foreground="black")])

    def zframe(a):
        for sp in a.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(ZC)
            sp.set_linewidth(ZLW)

    def arrow_at(a, pt):
        x, y = pt[0], pt[1]
        v, length = np.array([box / 2 - x, box / 2 - y], dtype=np.float64), 13.0 * box / 48
        nv = float(np.hypot(*v))
        v = v / nv * length if nv > 4 else np.array([length, length]) / 1.4
        a.annotate("", xy=(x, y), xytext=(x + v[0], y + v[1]),
                   arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR, lw=1.8, mutation_scale=13, shrinkA=0, shrinkB=1))

    crop = lambda img: img[y0b:y0b + box, x0b:x0b + box]
    imkw = dict(origin="upper", aspect="auto")
    mask_cmap = ListedColormap(["black", "white", ARROW_COLOR])
    y, im_e, first = top_in, None, None
    for r, n in zip(rows, info):
        gkw = dict(cmap="gray", vmin=0, vmax=n["vmax"], **imkw)
        for j in range(ncol):
            x = j * (pw + gap)
            a = axes_at(x, y, pw, h, "black")
            lab = None if j == 0 else cols[j - 1]
            if r is rows[0]:
                a.set_title("GT" if j == 0 else lab.upper(), fontsize=title_fs, pad=4)
            if j and lab not in r["cells"]:
                axes_at(x, y + h + gap, pw, eh, "black")
                a.text(0.5, 0.5, (r.get("blank") or {}).get(lab, "not run on this condition"), transform=a.transAxes,
                       ha="center", va="center", color="#8a8a86", fontsize=11)
                continue
            img = n["g2"] if j == 0 else r["cells"][lab][k]
            a.imshow(view(img), **gkw)
            a.add_patch(Rectangle((x0b - vx0 - 0.5, y0b - vy0 - 0.5), box, box, fill=False, ec=ZC, lw=ZLW))
            bl = axes_at(x, y + h + gap, half, eh)
            bl.imshow(crop(img), **gkw)
            zframe(bl)
            if n["arrow"]:
                arrow_at(bl, n["arrow"])
            if j == 0:
                label(a, r["name"], bold=True, fs=13, hh=h)
                label(a, r["contrast"], bold=True, line=1, fs=13, hh=h)
                axes_at(x + half + zgap, y + h + gap, half, eh, "black")
                km = r.get("kmask")
                if km is not None:                          # the sampling mask at its true aspect, on black
                    mw = min(half, eh * km.shape[1] / km.shape[0])
                    mh = mw * km.shape[0] / km.shape[1]
                    bm = axes_at(x + half + zgap + (half - mw) / 2, y + h + gap + (eh - mh) / 2, mw, mh)
                    bm.imshow(km, cmap=mask_cmap, vmin=0, vmax=2, interpolation="nearest", **imkw)
                continue
            num = n["num"][lab]
            for i, (txt, met) in enumerate(((f"{num['psnr']:.2f} dB", "psnr"), (f"SSIM {num['ssim']:.4f}", "ssim"),
                                             (f"NMSE {num['nmse']:.5f}", "nmse"))):
                label(a, txt, bold=n["best"].get(met) == lab, line=i, fs=10.5, hh=h, color=METRIC_COLOR)
            if (r.get("tags") or {}).get(lab):
                label(a, r["tags"][lab], fs=9.5, bottom=True)
            br = axes_at(x + half + zgap, y + h + gap, half, eh)
            im_e = br.imshow(crop(n["err"][lab]), cmap="inferno", vmin=0.0, vmax=err_vmax, **imkw)
            zframe(br)
        if first is None:
            first = y
        y += h + eh + 2 * gap
    x_cb = (ncol * pw + (ncol - 1) * gap + cbpad) / fw      # one grey bar (each row's own window), one error bar
    bars = [(ScalarMappable(norm=Normalize(0, 1), cmap="gray"), first, first + h - cbtrim, np.linspace(0, 1, 5),
             "intensity / row window")]
    if im_e is not None:
        bars.append((im_e, first + h + gap + 0.04, first + h + gap + eh, np.arange(0, 5) * err_vmax / 4, "|error| / max GT"))
    for handle, t0, t1, ticks, text in bars:
        cb = fig.colorbar(handle, cax=fig.add_axes([x_cb, 1 - t1 / fh, cbw / fw, (t1 - t0) / fh]), ticks=ticks)
        cb.ax.tick_params(labelsize=9.5)
        cb.set_label(text, fontsize=10)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1 - 0.10 / fh, va="top")
    fig.text(0.5, 1 - cap_y / fh, "\n".join(cap_lines), ha="center", va="top", fontsize=11, linespacing=1.32)
    out_png = _save(fig, out_png, dpi)
    out_png.with_suffix(".json").write_text(json.dumps(dict(
        slice=int(k), box=dict(y0=int(y0b), x0=int(x0b), edge=int(box)), view=[vy0, vy1, vx0, vx1], columns=["GT", *cols], truth_columns=list(truth_cols),
        error_display=dict(cmap="inferno", vmax=err_vmax), window="slice P99.9 / %g" % brightness,
        rows=[dict(name=r["name"], contrast=r["contrast"], numbers=n["num"], best=n["best"], tags=r.get("tags") or {},
                   arrow=None if n["arrow"] is None else dict(crop_xy=list(n["arrow"][:2]), gap=n["arrow"][2]))
              for r, n in zip(rows, info)]), indent=1))
    return out_png


def record_caption(box, k, others="the models other than ZF, the truth column and the last column"):
    return ["head-masked slice PSNR / SSIM (7x7 window) / NMSE in each panel, the best of each row among the columns that "
            "do not use the true motion in bold",
            "error |recon - GT| / max(GT), inferno [0, 0.1], inside the head only",
            "row under each image row: left = the magnified box of the image, right = the same box of the error map; "
            "GT column: GT box | (ky, kz) sampling mask, red = T2W lines acquired while the brain was displaced by more than "
            f"{CONTAM_MM} mm",
            f"the {box}x{box} box, the same in every row, is outlined on every slice (transverse {k}, cropped to the head); grey bar: intensity as "
            "a fraction of each row's own window (slice P99.9 / 1.25)",
            f"red arrow in the magnified image box (same place along a row): where the error of {others} (median, 7x7 "
            "local mean) exceeds the last column's most, among the tissue boundaries of the box (steepest fifth of the "
            "GT gradient)",
            "T2W of a model with echo-train weights: only the kept lines are inserted, as in the tables"]


def zero_filled_volumes(order, conds, gts):
    """{(cond, contrast): zero-filled magnitude on the scale of the runs' GT}. The scale is max(GT) / max(fully
    sampled reconstruction of the reference k-space); the largest difference between the two is printed."""
    import h5py
    import reg
    import simulate
    with h5py.File(c1.MASKS["af12"], "r") as hf:
        mask3 = hf["mask"][:].astype(np.float32)
    w = simulate.FULL["fft_workers"]
    csm = {c: simulate._read(c1.HYW / f"csm_espirit_{c}.h5", ("map", "csm")) for c in ("T1", "T2")}
    out, cache = {}, {}
    for cond in conds:
        for c in ("T1", "T2"):
            if (cond, c) not in gts:
                continue
            gt = gts[(cond, c)]
            path = reg.t1_path(order, cond) if c == "T1" else reg.t2_path(order, cond)
            ref = path
            if c == "T2":
                tp = Path(json.loads((_cdir_of(order, cond) / "motion_truth.json").read_text()).get("gt_ksp_path") or path)
                ref = tp if tp.exists() else path
            if ref not in cache:
                full = np.abs(simulate.combine(simulate._read(ref, ("kspace", "ksp")), csm[c], w))
                cache[ref] = float(full.max()), full
            fmax, full = cache[ref]
            s = float(gt.max()) / fmax
            print(f"[zf] {cond} {c}: scale {s:.6g}; max |scaled full reconstruction - GT| / max GT = "
                  f"{float(np.abs(full * s - gt).max()) / float(gt.max()):.2e}", flush=True)
            if (path, c) not in cache:
                cache[(path, c)] = reg._zero_filled(path, mask3, csm[c], w)
            out[(cond, c)] = (cache[(path, c)] * s).astype(np.float32)
    return out


def _cdir_of(order, cond):
    import mevaluate as ev
    return ev._cdir(order, cond)


def sampling_codes(order, cond, contrast):
    """mask_codes of one condition and contrast, from the AF12 mask, the assumed echo-train order and the truth."""
    import h5py
    with h5py.File(c1.MASKS["af12"], "r") as hf:
        mask = hf["mask"][0] > 0                             # readout (axis 0) fully sampled: one (ky, kz) plane
    d = _cdir_of(order, cond)
    if contrast != "T2" or not (d / "schedule.npz").exists():
        return mask_codes(mask, None, None)
    with np.load(d / "schedule.npz") as z:
        train_of = z["train_of"]
    return mask_codes(mask, train_of, json.loads((d / "motion_truth.json").read_text())["contaminated"])


def record_rows(order, conds, tags=None, zf=True):
    """Rows of the record figure from the runs that exist: per condition a T1W and a T2W row."""
    import mevaluate as ev
    from mcommon import exp_name
    per, gts = {}, {}
    for cond in conds:
        for col, arms in REC_ARMS_OF.items():
            for arm in arms:
                v = ev._vols2(exp_name("af12", order, cond, arm, tag=(tags or {}).get((cond, arm), "")), arm)
                if v is None:
                    continue
                for c, (rec, gt) in v.items():
                    ref = gts.setdefault((cond, c), gt)
                    assert np.allclose(gt, ref, atol=1e-5 * float(ref.max())), f"{cond} {arm} {c}: a different GT"
                    per.setdefault((cond, c), ({}, {}))
                    per[(cond, c)][0][col] = rec
                    if arm in REC_TAG:
                        per[(cond, c)][1][col] = REC_TAG[arm]
                break
    zfv = zero_filled_volumes(order, conds, gts) if zf else {}
    rows = []
    for cond in conds:
        for c in ("T1", "T2"):
            if (cond, c) not in per:
                continue
            cells, tagd = per[(cond, c)]
            if (cond, c) in zfv:
                cells = {"ZF": zfv[(cond, c)], **cells}
            rows.append(dict(name=COND_NAME.get(cond, cond), cond=cond, contrast=f"{c}W", gt=gts[(cond, c)], cells=cells,
                             tags=tagd, blank={"GS-single": "T2W only"} if c == "T1" else {},
                             kmask=sampling_codes(order, cond, c)))
    return rows


def record_from_runs(order, conds, out_png, tags=None, box=64, title=None):
    """Draws the record figure of the given conditions. Slice = the transverse slice where the unchanged model has
    the largest T2W error in the brain, summed over the conditions; box = centred on the largest 15x15 local mean of
    that error on the slice."""
    from scipy.ndimage import uniform_filter
    rows = record_rows(order, conds, tags)
    t2 = [r for r in rows if r["contrast"] == "T2W" and "MC-GS" in r["cells"]]
    if not t2:
        return None
    brain = np.load(_cdir_of(order, t2[0]["cond"]) / "brain.npy").astype(bool)
    k, e = worst_slice([(r["gt"], r["cells"]["MC-GS"]) for r in t2], brain)
    yx = box_origin(uniform_filter(e, size=15, mode="constant"), box)
    title = title or (f"Mild motion inside the T2W scan: T1W + T2W, acceleration factor 12, {ORDER_NAME[order]} echo-train "
                      f"order   --   Transverse {k}")
    return record_figure(rows, REC_COLS, k, yx, box, title, record_caption(box, k), out_png, truth_cols=REC_TRUTH)


def worst_slice(pairs, brain):
    """(k, e): k = index along axis 0 (transverse) of the slice where the reconstructions of pairs [(gt, rec), ...]
    have the largest squared error inside the brain, summed over the pairs; e = that error map of slice k."""
    e = sum(((np.asarray(rec, dtype=np.float64) - gt) ** 2) * brain for gt, rec in pairs)
    k = int(e.sum(axis=(1, 2)).argmax())
    return k, e[k]


# ------------------------------------------------------------------------------------------- inter-scan
def align_figure(summary, out_png, dpi=200):
    """E2: registration error of the arms with M1 (left) and the normalised change of PSNR (right)."""
    c = summary["criteria"]["S2"]["arms"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.28))
    a = axes[0]
    _style(a)
    rows = (("rigid part against the true rigid part", "tre_rigid", "value", THRESH2["s2_tre_rigid"]),
            ("rigid + control grid against the true field", "tre_full", "value", THRESH2["s2_tre_nonrigid"]),
            ("rigid part alone against the true field", "tre_full", "rigid_alone", None))
    width = 0.3
    for i, (_, item, key, thr) in enumerate(rows):
        for j, (arm, col) in enumerate((("B1", AQUA), ("B123", ORANGE))):
            v = c[arm][item].get(key)
            if v is None:
                continue
            x = i + (j - 0.5) * (width + 0.03)
            a.bar(x, v, width=width, color=col)
            a.text(x, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=INK)
        if thr is not None:
            a.plot([i - 0.42, i + 0.42], [thr, thr], color=INK, lw=1.0, ls="--")
            a.annotate(f"criterion: at most {thr}", (i + 0.42, thr), xytext=(0, 4), textcoords="offset points",
                       ha="right", va="bottom", fontsize=8, color=INK)
    a.set_xticks(range(len(rows)))
    a.set_xticklabels([r[0].replace(" against ", "\nagainst ") for r in rows], fontsize=8.5, color=INK)
    a.set_ylabel("mean error in the brain, mm\n(differential basis)", fontsize=9, color=INK)
    a.set_title("Inter-scan transform found on E2", fontsize=10, color=INK, loc="left")
    a = axes[1]
    _style(a)
    arms = (("B0", GREY), ("B1", AQUA), ("B123", ORANGE))
    for i, (con, key, thr) in enumerate((("T1", "delta_T1", THRESH2["s2_dT1"]), ("T2", "delta_T2", THRESH2["s2_dT2"]))):
        for j, (arm, col) in enumerate(arms):
            v = c[arm][key]["value"]
            if v is None:
                continue
            x = i + (j - 1) * 0.26
            a.bar(x, v, width=0.23, color=col)
            a.text(x, v + (0.004 if v >= 0 else -0.004), f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top",
                   fontsize=8, color=INK)
        a.plot([i - 0.45, i + 0.45], [thr, thr], color=INK, lw=1.0, ls="--")
        a.annotate(f"criterion: not below {thr}", (i + 0.45, thr), xytext=(0, -4), textcoords="offset points",
                   ha="right", va="top", fontsize=8, color=INK)
    a.axhline(0.0, color=MUTED, lw=0.8)
    a.set_xticks([0, 1])
    a.set_xticklabels(["T1W", "T2W\n(minus the independent model's change)"], fontsize=8.5, color=INK)
    lo = min([THRESH2["s2_dT2"]] + [c[arm][k]["value"] for arm, _ in arms for k in ("delta_T1", "delta_T2")
                                    if c[arm][k]["value"] is not None])
    a.set_ylim(bottom=lo - 0.025)
    a.set_ylabel("PSNR on E2 minus PSNR without motion, dB", fontsize=9, color=INK)
    a.set_title("Normalised change of image quality", fontsize=10, color=INK, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=col) for _, col in arms]
    fig.legend(handles, [ARM_LABEL[a_] for a_, _ in arms], loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.90, bottom=0.24)
    return _save(fig, out_png, dpi)


def feas_figure(feas, mm, truth, out_png, dpi=200):
    """Three panels: (a) z score of every echo train of the unchanged model on DTa at the last computation, in
    acquisition order; (b) AUC of that score for the contaminated trains along the training; (c) T2 head PSNR of
    the three runs. feas = mevaluate.feasibility(...)."""
    z = np.asarray(mm["m2"]["z"], dtype=np.float64)
    cont = np.asarray(truth["contaminated"], dtype=bool)
    n = np.arange(z.size)
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4), gridspec_kw=dict(width_ratios=[2.2, 1.0, 1.0]))
    a = ax[0]
    a.bar(n[~cont], z[~cont], color=C_CLEAN, width=0.9, label=f"clean echo trains ({int((~cont).sum())})")
    a.bar(n[cont], z[cont], color=C_BAD, width=0.9, label=f"trains acquired during the motion ({int(cont.sum())})")
    st = mm["m2"].get("settings") or {}
    for key, ls in (("z0", "--"), ("z1", ":")):
        if key in st:
            a.axhline(float(st[key]), color=C_LINE, ls=ls, lw=1.0)
            a.text(z.size - 1, float(st[key]), f" {key} = {st[key]:g}", va="bottom", ha="right", fontsize=9, color=C_LINE)
    a.set_xlabel("echo train (acquisition order)")
    a.set_ylabel("robust z score of the residual statistic")
    a.set_title("(a) Per-echo-train residual of MC-GS unchanged", fontsize=11, loc="left")
    a.legend(frameon=False, fontsize=9, loc="upper left")
    a = ax[1]
    hist = feas.get("auc_history") or []
    if hist:
        a.plot([h["it"] for h in hist], [h["auc"] for h in hist], "-o", ms=3, color=C_CLEAN)
    a.axhline(THRESH2["feas_auc"], color=C_LINE, ls="--", lw=1.0)
    a.text(0.98, THRESH2["feas_auc"], f"required {THRESH2['feas_auc']:.2f}", transform=a.get_yaxis_transform(),
           ha="right", va="bottom", fontsize=9, color=C_LINE)
    a.set_ylim(0.4, 1.02)
    a.set_xlabel("iteration")
    a.set_ylabel("AUC (contaminated vs clean trains)")
    auc = feas.get("auc")
    a.set_title("(b) Separation along the training" + ("" if auc is None else f"; final {auc:.3f}"), fontsize=11, loc="left")
    a = ax[2]
    bars = [("no motion\nMC-GS", feas.get("b0_c0"), C_CLEAN), ("motion\nMC-GS", feas.get("b0"), C_BAD),
            ("motion\ntrue weights", feas.get("ow"), "#55a868")]
    vals = [v for _, v, _ in bars if v is not None]
    for i, (name, v, col) in enumerate(bars):
        if v is None:
            continue
        a.bar(i, v, color=col, width=0.6)
        a.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    a.set_xticks(range(len(bars)))
    a.set_xticklabels([b[0] for b in bars], fontsize=9)
    if vals:
        a.set_ylim(min(vals) - 0.6, max(vals) + 0.4)
    a.set_ylabel("T2 head PSNR (dB)")
    rec, loss = feas.get("recover"), feas.get("loss")
    note = "" if loss is None else f"loss {loss:.3f} dB" + ("" if rec is None else f", recovered {100 * rec:.0f} %")
    a.set_title(f"(c) {note}" if note else "(c) T2 head PSNR", fontsize=11, loc="left")
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Feasibility check (development condition DTa, interleaved order): {feas.get('decision')}",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_png = Path(out_png)
    fig.savefig(out_png, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_png


def all_figures(summary):
    """The result figures from the summary and the runs' files. Returns the PNG paths."""
    import mevaluate as ev
    from mcommon import exp_name
    done = [recovered_figure(summary, OUT / "fig_recovered.png"), align_figure(summary, OUT / "fig_align_E2.png")]
    tab = {(r["order"], r["cond"], r["arm"]): r for r in summary["table"]}
    for order, conds in (("il", ("Ta", "Tb", "Tc", "E2Tb")), ("kz", ("Ta", "Tc"))):
        for cond in conds:
            exp = exp_name("af12", order, cond, "B123")
            mm = ev._load_mm(exp)
            if mm is None:
                continue
            done.append(trains_figure(ev._truth(order, cond), mm, f"{ORDER_NAME[order]} order, condition {cond}, arm B123",
                                      OUT / f"fig_trains_{order}_{cond}.png"))
    p = record_from_runs("il", ("Ta", "Tb", "Tc", "E2Tb"), OUT / "fig_images_record.png")
    if p is not None:
        done.append(p)
    return done


def main():
    global PDF_DPI
    ap = argparse.ArgumentParser()
    ap.add_argument("--feas", action="store_true")
    ap.add_argument("--all", action="store_true", help="result figures from <root>/eval/summary.json")
    ap.add_argument("--record", action="store_true", help="only the image figure (record format), from the runs that exist")
    ap.add_argument("--order", default="il", choices=sorted(ORDER_NAME))
    ap.add_argument("--conds", nargs="+", default=["Ta", "Tb", "Tc", "E2Tb"])
    ap.add_argument("--tag", nargs="*", default=[], metavar="COND:ARM=TAG", help="runs that carry a tag (development rounds)")
    ap.add_argument("--name", default="fig_images_record", help="file stem of --record")
    args = ap.parse_args()
    import mevaluate
    OUT.mkdir(parents=True, exist_ok=True)
    art = os.environ.get("MCGS_ARTIFACT_DIR")
    if args.feas:
        f = mevaluate.feasibility(mevaluate.table(mevaluate.load_ledger()))
        mm = json.loads((RES2 / "mm_af12_il_DTa_B0_mm.json").read_text())
        truth = json.loads((DATA2 / "il/DTa/motion_truth.json").read_text())
        p = feas_figure(f, mm, truth, OUT / "feas_il_DTa.png")
        (OUT / "feas_il_DTa.json").write_text(json.dumps(f, indent=1))
        print(f"wrote {p.name}", flush=True)
        if art:
            shutil.copy(p, Path(art) / p.name)
            shutil.copy(OUT / "feas_il_DTa.json", Path(art) / "feas_il_DTa.json")
    elif args.all:
        PDF_DPI = 600
        for p in all_figures(json.loads((ROOT2 / "eval" / "summary.json").read_text())):
            print(f"wrote {p.name} and {p.with_suffix('.pdf').name}", flush=True)
            if art:
                shutil.copy(p, Path(art) / p.name)
    elif args.record:
        tags = {tuple(t.split("=")[0].split(":")): t.split("=")[1] for t in args.tag}
        p = record_from_runs(args.order, args.conds, OUT / f"{args.name}.png", tags)
        if p is None:
            ap.error("no finished MC-GS (B0) run on these conditions")
        print(f"wrote {p.name} and {p.with_suffix('.json').name}", flush=True)
        if art:
            shutil.copy(p, Path(art) / p.name)
            shutil.copy(p.with_suffix(".json"), Path(art) / p.with_suffix(".json").name)
    else:
        ap.error("nothing to do")


if __name__ == "__main__":
    main()
