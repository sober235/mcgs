#!/usr/bin/env python3
"""best_slice_lesion.py -- the slice (any axis) where MC-GS is simultaneously the best reconstruction (head-masked
slice PSNR, the study's gauge) AND the most faithful to the truth's lesion boundary, on BOTH scans.

Boundary region: a shell around the lesion core, `dilate(core, K_SHELL) AND NOT erode(core, K_SHELL)`, K_SHELL=2
(~2 mm on this 1 mm isotropic grid). A slice is only scored if its shell cross-section has >= MIN_SHELL voxels.
MIN_SHELL=50 was picked by looking at the real per-axis, per-slice shell counts: on every axis they fall to a
handful (1-47) at the very tip of the lesion and jump to >=64 one slice later, so 50 excludes every tip slice
without cutting into the body of the lesion (axis0 keeps index 191-215, axis1 105-141, axis2 34-60 -- 25+37+27=89
slices scored, both scans). This mirrors find_best_slice.py's own MIN_HEAD=2000 (an absolute floor "so a handful of
voxels does not make the number meaningless"), not a fraction of a per-axis maximum, because the lesion is small
and irregular and a fraction-of-peak rule would need a separate, less legible justification per axis.

The shell is NOT intersected with the head mask used for the PSNR metric below: the lesion core sits close to that
mask's cutoff (median core intensity is only ~1.1x the 0.05*P99.9 threshold on the `post` scan -- see the printed
head-membership check at start-up), so doing that would silently drop lesion voxels from the one region this
script exists to look at.

Two independent boundary measures, both against the truth, both restricted to the shell voxels of the slice, both
on magnitude images (same convention as find_best_slice.py / best_slice_mcgs.py):
  * shell RMSE (and NMSE): sqrt(mean (|rec|-|GT|)^2) over the shell -- plain intensity agreement, no assumption
    about exactly where the "true" edge sits inside the shell.
  * edge-geometry error: RMSE, over the same shell voxels, of the gradient-magnitude field, |grad|rec|| - |grad|GT||.
    |grad| here (`gmag_np`) is the forward-difference formula of t1t2_common._gmag, transcribed rather than
    imported so this stays a plain-numpy slice scorer with no torch/h5py/trainer import chain -- the same reason
    find_best_slice.py and best_slice_mcgs.py do not import t1t2_common either.
    Chosen over the other two options the brief offered: a 50%-crossing displacement needs a boundary *normal*,
    which is ill-posed on an irregular, lobulated 2-D contour at concavities; a level-set Dice needs an extra,
    somewhat arbitrary intensity threshold. The gradient-field RMSE needs neither.
    Why a blurrier arm cannot win it for free: at a real intensity step, blurring *lowers* |grad| there relative to
    the truth's own |grad|, which *raises* -- not lowers -- the RMS disagreement with the truth's gradient field.
    Blur can only help this measure by accident (by matching how much the truth itself was already smoothed by its
    own TV prior), never by construction.

Reconstruction metric per slice: head-masked slice PSNR, the identical definition find_best_slice.py /
best_slice_mcgs.py use (head = |GT| > 0.05*P99.9(|GT|) of the volume, peak = max|GT| of the volume, both computed
once per scan on the full volume, not per slice; MIN_HEAD=2000, same floor find_best_slice.py uses).

Selection: among slices with a scored shell on both scans, MC-GS (gs_joint) must rank 1st of all eleven arms
(GS-single included -- it is a comparison arm here, not excluded from the ranking) on
{slice PSNR, shell RMSE, edge-geometry RMSE} x {pre, post}: 6 conditions. For each condition the margin is
MC-GS's value minus the best of the other ten, signed so positive = MC-GS ahead; the same signed quantity, divided
by the runner-up's |value|, gives a unit-free relative margin that stays meaningful whether MC-GS is ahead or
behind. `combined` = the min of those 6 relative margins (the weakest link). Sorting every scored slice by
`combined` descending puts every slice that passes all 6 conditions (combined > 0) ahead of every slice that fails
at least one (combined <= 0), and within each group ranks by the tightest/closest condition -- which is exactly
"ranked by the smallest of its margins" for winners and "the closest cases, with what MC-GS loses on" for
near-misses, from one consistent rule instead of two ad hoc ones.

Writes logs/best_slice_lesion.json (every scored slice, full audit table) and, for the top row of that ranking
(the winner if one exists, else the closest near-miss), figures/ce_retro_af16_lesion_slice_<axis><idx>.png (also
copied to ~/figs/ce_retro_af16/).

  env PYTHONNOUSERSITE=1 PYTHONPATH= <PY> scripts/best_slice_lesion.py
"""
import json
import os
import shutil
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from arms import ORDER, LABEL, arm_files

R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"))
ROI_NPZ = Path(os.environ.get(
    "RETRO_ROI", "/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast/lesion_candidates/enhance_rois_cssense.npz"))
FIGDIR_SERVED = Path("/home/congcongliu/figs/ce_retro_af16")

SCANS = ("pre", "post")
K_SHELL = 2
MIN_SHELL = 50
MIN_HEAD = 2000
MCGS, GSSINGLE = "gs_joint", "gs_single"
MEASURES = [("psnr", True), ("rmse", False), ("gerr", False)]   # (name, higher_is_better)


def gmag_np(v):
    """forward-difference gradient magnitude; transcribed verbatim from t1t2_common._gmag (torch -> numpy) so this
    script does not need to import t1t2_common (and its h5py/torch/trainer chain) just for one helper."""
    gx = np.zeros_like(v); gy = np.zeros_like(v); gz = np.zeros_like(v)
    gx[:-1] = v[1:] - v[:-1]
    gy[:, :-1] = v[:, 1:] - v[:, :-1]
    gz[:, :, :-1] = v[:, :, 1:] - v[:, :, :-1]
    return np.sqrt(gx * gx + gy * gy + gz * gz + 1e-12)


def load_all():
    summary = json.load(open(R / "logs/summary.json"))
    keys = None
    gt, head, gmax, vols, gmag_gt, gmag_arm = {}, {}, {}, {}, {}, {}
    for c in SCANS:
        arms_c = arm_files(c, summary)
        ks = [k for k, _, _ in arms_c]
        keys = ks if keys is None else keys
        assert ks == keys == ORDER, (ks, keys)
        g = np.abs(np.load(R / f"recon/gt_{c}_complex.npy")).astype(np.float64)
        gt[c] = g
        gmax[c] = float(g.max())
        head[c] = g > 0.05 * float(np.quantile(g, 0.999))
        gmag_gt[c] = gmag_np(g)
        vols[c], gmag_arm[c] = {}, {}
        for key, _, path in arms_c:
            v = np.abs(np.load(path)).astype(np.float64)
            vols[c][key] = v
            gmag_arm[c][key] = gmag_np(v)
    return keys, gt, head, gmax, vols, gmag_gt, gmag_arm


def score_slices(keys, gt, head, gmax, vols, gmag_gt, gmag_arm, shell):
    rows = []
    for axis in range(3):
        n = shell.shape[axis]
        for i in range(n):
            sl = [slice(None)] * 3; sl[axis] = i; sl = tuple(sl)
            sh = shell[sl]
            nsh = int(sh.sum())
            if nsh < MIN_SHELL:
                continue
            row = {"axis": axis, "index": i, "n_shell": nsh, "scans": {}}
            skip = False
            for c in SCANS:
                h = head[c][sl]
                nh = int(h.sum())
                if nh < MIN_HEAD:
                    skip = True
                g_head = gt[c][sl][h]
                gsh = gt[c][sl][sh]
                peak_db = 20 * np.log10(gmax[c])
                psnr, rmse, nmse_b, gerr = {}, {}, {}, {}
                for k in keys:
                    vh = vols[c][k][sl][h]
                    e = vh - g_head
                    mse = float((e * e).mean())
                    psnr[k] = float(peak_db - 10 * np.log10(mse)) if mse > 0 else float("inf")
                    vsh = vols[c][k][sl][sh]
                    d_sh = vsh - gsh
                    rmse[k] = float(np.sqrt((d_sh * d_sh).mean()))
                    denom = float((gsh * gsh).sum())
                    nmse_b[k] = float((d_sh * d_sh).sum() / denom) if denom > 0 else float("nan")
                    gd = gmag_arm[c][k][sl][sh] - gmag_gt[c][sl][sh]
                    gerr[k] = float(np.sqrt((gd * gd).mean()))
                row["scans"][c] = dict(n_head=nh, psnr=psnr, rmse=rmse, nmse=nmse_b, gerr=gerr)
            if skip:
                continue
            rows.append(row)
    return rows


def rank_and_margin(vals, higher):
    """vals: dict arm -> value. Returns (rank of MCGS [1-based], runner_up_key, raw_margin, rel_margin).
    raw_margin = MCGS - best_other (higher-is-better) or best_other - MCGS (lower-is-better); positive = MCGS ahead."""
    others = {k: v for k, v in vals.items() if k != MCGS}
    order = sorted(vals, key=(lambda k: -vals[k]) if higher else (lambda k: vals[k]))
    rank = order.index(MCGS) + 1
    best_other_key = max(others, key=others.get) if higher else min(others, key=others.get)
    best_other_val = others[best_other_key]
    raw = (vals[MCGS] - best_other_val) if higher else (best_other_val - vals[MCGS])
    rel = raw / (abs(best_other_val) + 1e-12)
    gs_val = vals[GSSINGLE]
    gs_margin = (vals[MCGS] - gs_val) if higher else (gs_val - vals[MCGS])
    return rank, best_other_key, raw, rel, gs_margin


def annotate(rows):
    for row in rows:
        n_win, rel_list, cond = 0, [], {}
        for c in SCANS:
            sc = row["scans"][c]
            for name, higher in MEASURES:
                rank, runner_up, raw, rel, gs_margin = rank_and_margin(sc[name], higher)
                cond[f"{c}_{name}"] = dict(rank=rank, runner_up=runner_up, margin_best_other=raw,
                                            rel_margin=rel, margin_gs_single=gs_margin,
                                            mcgs=sc[name][MCGS], best_other=sc[name][runner_up])
                n_win += int(rank == 1)
                rel_list.append(rel)
        row["conditions"] = cond
        row["n_win"] = n_win
        row["combined_rel_margin"] = min(rel_list)
        row["pass_all"] = n_win == len(rel_list)
    rows.sort(key=lambda r: -r["combined_rel_margin"])
    return rows


def print_summary(rows):
    passing = [r for r in rows if r["pass_all"]]
    print(f"\nscored {len(rows)} slices (>= {MIN_SHELL} shell voxels, >= {MIN_HEAD} head voxels both scans)")
    print(f"slices where MC-GS ranks 1st of 11 on all 6 conditions (PSNR+shellRMSE+edgeRMSE x pre/post): {len(passing)}")
    print(f"\ntop {min(10, len(rows))} by combined (weakest-link) relative margin:")
    header = f"{'axis':4s}{'idx':5s}{'n_shell':8s}{'n_win':6s}{'combined':>10s}  " + "  ".join(
        f"{c}_{name}" for c in SCANS for name, _ in MEASURES)
    print(header)
    for r in rows[:10]:
        cells = []
        for c in SCANS:
            for name, _ in MEASURES:
                cd = r["conditions"][f"{c}_{name}"]
                mark = "OK" if cd["rank"] == 1 else f"#{cd['rank']}"
                cells.append(f"{mark:>3s}({cd['margin_best_other']:+.4f})")
        print(f"{r['axis']:<4d}{r['index']:<5d}{r['n_shell']:<8d}{r['n_win']:<6d}{r['combined_rel_margin']:>+10.4f}  " + "  ".join(cells))
    best = rows[0]
    print(f"\nBEST ROW: axis {best['axis']} index {best['index']}  n_win={best['n_win']}/6  pass_all={best['pass_all']}")
    for c in SCANS:
        for name, _ in MEASURES:
            cd = best["conditions"][f"{c}_{name}"]
            status = "MC-GS #1" if cd["rank"] == 1 else f"MC-GS rank {cd['rank']} (loses to {cd['runner_up']})"
            print(f"  {c:4s} {name:5s}: {status}, MC-GS={cd['mcgs']:.5f} best_other({cd['runner_up']})={cd['best_other']:.5f} "
                  f"margin={cd['margin_best_other']:+.5f} margin_vs_GS-single={cd['margin_gs_single']:+.5f}")


# ---- figure -----------------------------------------------------------------------------------
METRIC_COLOR = "#FFC23D"
LESION_COLOR = "#FF33FF"


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


def make_figure(row, keys, gt, head, gmax, vols, core, shell):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    axis, idx = row["axis"], row["index"]
    sl = [slice(None)] * 3; sl[axis] = idx; sl = tuple(sl)
    my0, my1, mx0, mx1 = transformed_box(ndi.binary_dilation(core, iterations=K_SHELL)[sl], axis, pad=15)
    iy0, iy1, ix0, ix1 = transformed_box(shell[sl], axis, pad=4)
    iy0, iy1, ix0, ix1 = iy0 - my0, iy1 - my0, ix0 - mx0, ix1 - mx0

    def crop(full2d):
        return Dfun(full2d, axis)[my0:my1, mx0:mx1]

    core2d = crop(core[sl].astype(float))
    ncol = 1 + len(keys)
    row_types = [(c, "img") for c in SCANS for _ in [0]]
    # build interleaved [pre_img, pre_err, post_img, post_err]
    panel_rows = []
    for c in SCANS:
        panel_rows.append((c, "img"))
        panel_rows.append((c, "err"))
    H = my1 - my0; W = mx1 - mx0
    pw = 1.9
    ph = pw * H / W
    gap = 0.05
    top_in, cb_room = 0.55, 1.0
    fw = ncol * pw + (ncol - 1) * gap + cb_room
    fh = top_in + len(panel_rows) * ph + (len(panel_rows) - 1) * gap + 0.3
    fig = plt.figure(figsize=(fw, fh))

    def axes_at(x0, y0, w, h):
        a = fig.add_axes([x0 / fw, 1 - (y0 + h) / fh, w / fw, h / fh])
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
        return a

    def label(a, text, bold=False, line=0, fs=9.5, color=METRIC_COLOR):
        dy = line * fs * 1.3 / 72 / ph
        t = a.text(0.03, 0.97 - dy, text, transform=a.transAxes, ha="left", va="top", color=color,
                   fontsize=fs, fontweight="bold" if bold else "normal")
        t.set_path_effects([pe.withStroke(linewidth=2.2, foreground="black")])

    def put_zoom(x, y, img2d, vmax_c):
        """the second zoom level: the shell's own box, pasted top-right of the panel. Drawn for the truth column too,
        otherwise the eye has nothing to compare the arms' insets against."""
        zoom_side = 0.42 * pw
        za = fig.add_axes([(x + pw - zoom_side - 0.02) / fw, 1 - (y + zoom_side + 0.02) / fh,
                           zoom_side / fw, zoom_side / fh])
        za.set_xticks([]); za.set_yticks([])
        for sp in za.spines.values():
            sp.set_visible(True); sp.set_edgecolor(METRIC_COLOR); sp.set_linewidth(1.4)
        za.imshow(img2d[iy0:iy1, ix0:ix1], cmap="gray", vmin=0, vmax=vmax_c, origin="lower", aspect="auto")
        return za

    im_err = None
    for r_i, (c, kind) in enumerate(panel_rows):
        y = top_in + r_i * (ph + gap)
        g2 = crop(gt[c][sl])
        vmax_c = float(np.quantile(g2, 0.999))
        h2 = crop(head[c][sl].astype(float)) > 0.5
        for j in range(ncol):
            x = j * (pw + gap)
            a = axes_at(x, y, pw, ph)
            if j == 0:
                if kind == "img":
                    a.imshow(g2, cmap="gray", vmin=0, vmax=vmax_c, origin="lower", aspect="auto")
                    a.contour(core2d, levels=[0.5], colors=LESION_COLOR, linewidths=1.1)
                    a.add_patch(Rectangle((ix0, iy0), ix1 - ix0, iy1 - iy0, fill=False, ec=METRIC_COLOR, lw=1.6))
                    put_zoom(x, y, g2, vmax_c)
                    label(a, f"GT ({c})", bold=True)
                    if r_i == 0:
                        a.set_title("GT", fontsize=12, pad=4)
                else:
                    a.imshow(core2d, cmap="gray", vmin=0, vmax=1, origin="lower", aspect="auto")
                    label(a, "lesion core", bold=True, color="white")
                continue
            key = keys[j - 1]
            v2 = crop(vols[c][key][sl])
            cond_psnr = row["conditions"][f"{c}_psnr"]
            cond_rmse = row["conditions"][f"{c}_rmse"]
            cond_ge = row["conditions"][f"{c}_gerr"]
            if kind == "img":
                a.imshow(v2, cmap="gray", vmin=0, vmax=vmax_c, origin="lower", aspect="auto")
                a.contour(core2d, levels=[0.5], colors=LESION_COLOR, linewidths=1.1)
                a.add_patch(Rectangle((ix0, iy0), ix1 - ix0, iy1 - iy0, fill=False, ec=METRIC_COLOR, lw=1.6))
                put_zoom(x, y, v2, vmax_c)
                pm = cond_psnr["mcgs"] if key == MCGS else (row["scans"][c]["psnr"][key])
                mcgs_val = row["scans"][c]["psnr"][MCGS]
                is_best_psnr = key == max(row["scans"][c]["psnr"], key=row["scans"][c]["psnr"].get)
                label(a, f"{pm:.2f} dB", bold=is_best_psnr, line=0)
                if r_i == 0:
                    a.set_title(LABEL[key], fontsize=12, pad=4)
            else:
                e2 = np.abs(v2 - g2) / gmax[c]
                e2 = e2 * h2
                im_err = a.imshow(e2, cmap="jet", vmin=0.0, vmax=0.1, origin="lower", aspect="auto")
                a.contour(core2d, levels=[0.5], colors=LESION_COLOR, linewidths=1.1)
                rm = row["scans"][c]["rmse"][key]
                ge = row["scans"][c]["gerr"][key]
                is_best_rmse = key == min(row["scans"][c]["rmse"], key=row["scans"][c]["rmse"].get)
                is_best_ge = key == min(row["scans"][c]["gerr"], key=row["scans"][c]["gerr"].get)
                label(a, f"shell RMSE {rm:.4f}", bold=is_best_rmse, line=0)
                label(a, f"edge RMSE {ge:.4f}", bold=is_best_ge, line=1)
    cax = fig.add_axes([(ncol * pw + (ncol - 1) * gap + 0.15) / fw, 1 - (top_in + len(panel_rows) * ph + (len(panel_rows) - 1) * gap) / fh,
                         0.16 / fw, (len(panel_rows) * ph + (len(panel_rows) - 1) * gap) / fh])
    cb = fig.colorbar(im_err, cax=cax); cb.ax.tick_params(labelsize=9)
    cb.set_label("|err| / max|GT|", fontsize=9)
    status = "PASSES all 6 criteria" if row["pass_all"] else f"near-miss: {row['n_win']}/6 conditions"
    fig.suptitle(f"CE retro AF16 -- lesion boundary, axis {axis} index {idx} ({status})  "
                 f"image rows: grey P99.9 window per scan; error rows: |rec-GT|/max|GT|, jet [0, 0.1], head-masked",
                 fontsize=11, fontweight="bold", y=1 - 0.06 / fh)
    fig.text(0.5, 1 - (top_in - 0.28) / fh,
              "lesion core outlined in magenta on every panel; orange-yellow box = the zoom inset shown top-right of every image panel, the truth included; "
              "bold in-panel number = best of the 11 arms for that scan/measure",
              ha="center", va="top", fontsize=8.5)
    name = f"ce_retro_af16_lesion_slice_{axis}{idx:03d}.png"
    FIGDIR_SERVED.mkdir(parents=True, exist_ok=True)
    for d in (R / "figures", FIGDIR_SERVED):
        fig.savefig(d / name, dpi=140)
    plt.close(fig)
    return R / "figures" / name, FIGDIR_SERVED / name


def main():
    keys, gt, head, gmax, vols, gmag_gt, gmag_arm = load_all()
    with np.load(ROI_NPZ) as d:
        core = d["core"]
    assert core.shape == gt["pre"].shape
    dil = ndi.binary_dilation(core, iterations=K_SHELL)
    ero = ndi.binary_erosion(core, iterations=K_SHELL)
    shell = dil & ~ero
    print(f"lesion core {int(core.sum())} voxels; shell = dilate({K_SHELL}) \\ erode({K_SHELL}): {int(shell.sum())} voxels")
    for c in SCANS:
        thr = 0.05 * float(np.quantile(gt[c], 0.999))
        print(f"  {c}: head cutoff {thr:.5f}; core median {float(np.median(gt[c][core])):.5f}"
              f" ({float(np.median(gt[c][core])) / thr:.2f}x cutoff); shell not intersected with head for boundary measures")

    rows = score_slices(keys, gt, head, gmax, vols, gmag_gt, gmag_arm, shell)
    rows = annotate(rows)
    print_summary(rows)

    best = rows[0]
    png_topic, png_served = make_figure(best, keys, gt, head, gmax, vols, core, shell)
    print(f"\nwrote figure {png_topic}\ncopied to {png_served}")

    out = {
        "config": dict(k_shell=K_SHELL, min_shell=MIN_SHELL, min_head=MIN_HEAD, mcgs_key=MCGS, gs_single_key=GSSINGLE,
                        arms_order=ORDER,
                        criterion="MC-GS ranks 1st of 11 arms on head-masked slice PSNR, shell RMSE and shell "
                                  "edge-gradient RMSE, on both pre and post (6 conditions); rows sorted by "
                                  "combined_rel_margin = min over the 6 conditions of (MCGS-best_other)/|best_other|, "
                                  "signed so positive=ahead -- this ranks passing slices by their tightest margin and "
                                  "near-miss slices by how close their worst condition came to passing, in one rule."),
        "n_scored": len(rows),
        "n_pass_all": sum(1 for r in rows if r["pass_all"]),
        "winner_axis": best["axis"], "winner_index": best["index"], "winner_pass_all": best["pass_all"],
        "rows": rows,
    }
    out_path = R / "logs/best_slice_lesion.json"
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
