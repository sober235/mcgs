#!/usr/bin/env python3
"""Figures of the second version (runs on amax, CPU). No clock value appears on any figure.

    mfigures.py --feas     the feasibility check: z score of every echo train, AUC along the training, T2 PSNR
    mfigures.py --all      the result figures, from <root>/eval/summary.json and the runs' files

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

from mcommon import DATA2, ORDER_NAME, RES2, ROOT2, THRESH2

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
def error_scale(gt, recs, k):
    """Upper end of the shared error colour scale: the largest 99.5th percentile of |rec - gt| on slice k."""
    return float(max(np.percentile(np.abs(np.asarray(r[k], dtype=np.float64) - gt[k]), 99.5) for r in recs))


def images_figure(gt, recs, k, title, out_png, dpi=200):
    """Transverse slice k (axis 0): reference and reconstructions in the top row, absolute error with one colour scale
    below. recs: [(name, volume, head PSNR of the whole volume in dB)]."""
    vmax = float(np.percentile(gt[k], 99.9))
    emax = error_scale(gt, [r for _, r, _ in recs], k)
    ncol = len(recs) + 1
    h, wd = gt[k].shape
    fig, axes = plt.subplots(2, ncol, figsize=(2.9 * ncol * wd / max(h, wd) + 0.9, 2.9 * 2 * h / max(h, wd) + 0.9),
                             gridspec_kw=dict(wspace=0.02, hspace=0.02))
    fig.patch.set_facecolor("black")
    for a in axes.ravel():
        a.set_axis_off()
        a.set_facecolor("black")
    axes[0, 0].imshow(gt[k], cmap="gray", vmin=0, vmax=vmax)
    axes[0, 0].set_title("Reference (fully sampled)", fontsize=9, color="white")
    im = None
    for j, (name, rec, psnr) in enumerate(recs, start=1):
        axes[0, j].imshow(rec[k], cmap="gray", vmin=0, vmax=vmax)
        axes[0, j].set_title(f"{name}   {psnr:.2f} dB", fontsize=9, color="white")
        im = axes[1, j].imshow(np.abs(np.asarray(rec[k], dtype=np.float64) - gt[k]), cmap="inferno", vmin=0, vmax=emax)
    axes[1, 0].text(0.5, 0.5, f"absolute error\n(one scale, 0 to {emax / vmax:.2f}\nof the display maximum)", color="white",
                    fontsize=8, ha="center", va="center", transform=axes[1, 0].transAxes)
    fig.suptitle(title, fontsize=10, color="white", x=0.02, ha="left")
    fig.subplots_adjust(left=0.01, right=0.93, top=0.90, bottom=0.01)
    cax = fig.add_axes([0.94, 0.03, 0.012, 0.40])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(colors="white", labelsize=7)
    return _save(fig, out_png, dpi, face="black")


def worst_slice(gt, rec, brain):
    """Index along axis 0 (transverse) of the slice where rec has the largest squared error inside the brain."""
    e = ((np.asarray(rec, dtype=np.float64) - gt) ** 2) * brain
    return int(e.sum(axis=(1, 2)).argmax())


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
    for cond in ("Ta", "Tb", "Tc", "E2Tb"):
        vols = {a: ev._vols2(exp_name("af12", "il", cond, a), a) for a in ("B0", "B123", "Ow")}
        if any(v is None for v in vols.values()):
            continue
        gt = vols["B0"]["T2"][1]
        brain = np.load(ev._cdir("il", cond) / "brain.npy").astype(bool)
        k = worst_slice(gt, vols["B0"]["T2"][0], brain)
        recs = [(a, vols[a]["T2"][0], tab[("il", cond, a)]["T2"]["head_psnr_eval"]) for a in ("B0", "B123", "Ow")]
        done.append(images_figure(gt, recs, k, f"T2W, interleaved order, condition {cond}: transverse slice {k} "
                                               f"(largest B0 error in the brain); head PSNR of the volume",
                                  OUT / f"fig_images_{cond}.png"))
    return done


def main():
    global PDF_DPI
    ap = argparse.ArgumentParser()
    ap.add_argument("--feas", action="store_true")
    ap.add_argument("--all", action="store_true", help="result figures from <root>/eval/summary.json")
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
    else:
        ap.error("nothing to do")


if __name__ == "__main__":
    main()
