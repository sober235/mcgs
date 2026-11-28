#!/usr/bin/env python3
"""stats_subset.py -- statistics and figures of the T1 + FLAIR acceleration study on a chosen subset of the test scans.

Why a subset: the user asked (2026-11-27) for the statistics on vol1, vol2, vol4, vol5 and vol6, leaving vol7 aside "for now".
vol7 is the one scan where MC-GS is second in 13 of 24 columns (12 behind GS-single, 1 behind ZS-SSDU); the six-scan result stays in
table.json / stats.json / RESULTS.md / STATUS.md, which this script only reads. Every legend this script writes says which
scans are in and that the others were left out by the user's decision. With --scans omitted all six scans are analysed.

ANALYSIS PLAN (written before any p value of the subset was computed; nothing below is chosen by its outcome)
  Units. The independent unit is the person. vol2 and vol6 are one person, so five scans are four persons. The four
    acceleration factors of one scan and the slices of one volume are NOT independent and are never used as the n of a test.
  1. Volume level, description. Head PSNR / SSIM / NMSE of the 11 methods per (scan, AF, contrast), re-scored by
     collect_stats (one scoring code). Mean over the scans per (method, AF). Rank of every method per (scan, AF, contrast,
     metric); count of the columns in which MC-GS is first.
  2. Volume level, omnibus. Friedman test over the 11 methods with scans as blocks (and, as the stricter version, persons as
     blocks: a person's scans are averaged first), on the mean over the four AFs, per contrast and metric. p by within-block
     permutation of the method labels (20 000 permutations, fixed seed), not by the chi-square approximation (5 blocks).
     Kendall's W = chi2 / (N (k - 1)) as the agreement of the ranking between blocks. The same per AF is stored in the json.
     This test says that the methods differ and rank consistently; it does NOT say that MC-GS beats a particular method.
  3. Volume level, MC-GS against each of the 10 other methods (all ten, fixed; no "strongest of the cell"). Paired
     difference per scan = mean over the four AFs of the improvement (PSNR in dB, SSIM absolute, NMSE as relative reduction
     in %). Reported: mean, SD, 95 % interval from the t distribution (n - 1 degrees of freedom, n = scans), number of scans
     and of persons with a positive difference, number of (scan, AF) cells won, and the exact two-sided Wilcoxon signed-rank
     p (scans and persons). With five scans that p cannot be below 0.0625, with four persons not below 0.125: no pairwise
     test here can reach 0.05, so the claim rests on effect size and consistency. No one-sided test, no t-test p value.
  4. Slice level, within a scan only. For every (scan, AF, contrast, other method): per-slice improvement of MC-GS on the
     slices whose head covers >= 5 % of the slice (three planes pooled, the cached slice metrics of collect_stats); median,
     interquartile range and the share of slices where MC-GS is better. Slices are not pooled across scans into one test and
     no slice-level p value is shown.

Figures (183 mm wide, 5-7 pt sans-serif, lower-case bold panel letters, no in-figure title, 600 dpi PNG + vector PDF; plain
matplotlib; legends go to <stem>_legends.txt):
  fig1  mean over the scans, method x (metric, AF) heat maps, T1 and FLAIR; best cell of each column outlined
  fig2  rank of every method in every (scan, AF) column, per contrast and metric
  fig3  gain of MC-GS over each of the 10 other methods (one dot per scan, mean and 95 % interval); GS-single on its own axis
  fig4  per-slice PSNR gain over GS-single per scan and AF (violins) and share of slices won against all ten methods

Reads: OUT/stats_cache (read only; asserted unchanged), the result files collect_stats reads, and this folder's table.json
(the re-scored numbers must equal it). Writes only to MCGS_ARTIFACT_DIR and, with --save_dir, to that server folder.

  gpu.py run --cpu -- @python autoresearch/t1flair_af_testset_20261121/stats_subset.py --tag five \
      --scans vol1_huangyuwei vol2_jiangdian vol4_tangbing vol5_sunxinyi vol6_jiangdian_790 [--save_dir <server folder>]

Time rule: no server clock is read or printed; no generation date inside the figure files.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import collect_stats as CS                      # noqa: E402
from ts_common import AFS, OUT, SCANS           # noqa: E402

LABELS, CONTRASTS, METRICS, MNAME, HIGHER = CS.LABELS, CS.CONTRASTS, CS.METRICS, CS.MNAME, CS.HIGHER
OTHERS = [l for l in LABELS if l != "MC-GS"]
NPERM = 20000
log = lambda *a: print(*a, flush=True)
SHORT = lambda s: s.split("_")[0]
INK, INK2, RED, BLUE = "#1a1a1a", "#555555", "#E64B35", "#3C5488"
SCAN_COL = ["#3C5488", "#4DBBD5", "#00A087", "#F39B7F", "#8491B4", "#7E6148"]


# ===================================================================== data
def gather():
    """cells of collect_stats (all six scans, from the cache), checked against this folder's table.json"""
    import multiprocessing as mp
    cache = OUT / "stats_cache"
    assert cache.is_dir(), cache
    before = sorted(p.name for p in cache.iterdir())
    av = CS.availability()
    present = {k: v for k, v in av.items() if not isinstance(v, str)}
    assert len(present) == len(av), f"{len(av) - len(present)} results missing"
    tasks = {}
    for (scan, af, c, label), a in present.items():
        tasks.setdefault(scan, []).append((af, c, label, a["file"], a["member"], a["logged"]))
    jobs = [(s, t, str(cache)) for s, t in tasks.items()]
    results = []
    with mp.get_context("spawn").Pool(min(6, len(jobs))) as pool:
        for scan_res, (scan, _, _) in zip(pool.map(CS._worker, jobs, chunksize=1), jobs):
            for r in scan_res:
                r["scan"] = scan
            results += scan_res
    assert sorted(p.name for p in cache.iterdir()) == before, "stats_cache changed: a volume was re-scored"
    cells, chk = CS.assemble(av, results)
    assert not chk["unreadable"] and not chk["mcgs_mismatch"], chk
    ref = json.load(open(HERE / "table.json"))["cells"]
    worst = 0.0
    for k, cell in cells.items():
        assert cell["complete"], k
        for l in LABELS:
            for m in METRICS:
                worst = max(worst, abs(cell["methods"][l][m] - ref[k]["methods"][l][m]))
    assert worst < 1e-9, f"re-scored numbers differ from table.json by {worst}"
    log(f"cells {len(cells)}; largest difference from table.json {worst:.3g}")
    return cells, ref


def slice_arrays(cell, label, keep):
    z = np.load(cell["methods"][label]["cache"])
    return {m: np.concatenate([z[f"{m}{ax}"][keep[ax]] for ax in (0, 1, 2)]).astype(np.float64) for m in METRICS}


# ===================================================================== statistics
def summary(d, scans=None):
    d = np.asarray(d, float)
    n = d.size
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    half = float(tdist.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else float("nan")
    out = dict(n=int(n), mean=mean, sd=sd, ci95=[mean - half, mean + half], min=float(d.min()), max=float(d.max()),
               n_positive=int((d > 0).sum()), wilcoxon_p_exact=float(CS.exact_wilcoxon_p(list(d))))
    if scans is not None:
        out["values"] = {s: float(v) for s, v in zip(scans, d)}
    return out


def to_ranks(x, higher):
    """x: (..., k) scores -> ranks along the last axis, 1 = best"""
    return np.apply_along_axis(rankdata, -1, -x if higher else x)


def friedman(block_vals, higher, rng):
    """block_vals (N blocks, k methods). chi2 (no ties expected), Kendall's W, permutation p, mean rank per method."""
    r = to_ranks(np.asarray(block_vals, float), higher)
    N, k = r.shape
    stat = lambda rr: 12.0 * N / (k * (k + 1)) * ((rr.mean(axis=-2) - (k + 1) / 2.0) ** 2).sum(axis=-1)
    chi2 = float(stat(r))
    perm = np.argsort(rng.random((NPERM, N, k)), axis=-1) + 1.0     # every block's labels permuted independently
    pc = stat(perm)
    p = float((1 + (pc >= chi2 - 1e-12).sum()) / (NPERM + 1))
    return dict(n_blocks=int(N), k=int(k), chi2=chi2, kendall_w=chi2 / (N * (k - 1)), p_perm=p, n_perm=NPERM,
                mean_rank={l: float(v) for l, v in zip(LABELS, r.mean(axis=0))})


def analyse(cells, scans):
    rng = np.random.default_rng(20261127)
    persons = {}
    for s in scans:
        persons.setdefault(SCANS[s]["person"], []).append(s)
    pnames = sorted(persons)
    V, R = {}, {}                                   # V[c][m]: (scan, AF, method)
    for c in CONTRASTS:
        V[c], R[c] = {}, {}
        for m in METRICS:
            V[c][m] = np.array([[[cells[f"{s}|{af}|{c}"]["methods"][l][m] for l in LABELS] for af in AFS] for s in scans])
            R[c][m] = to_ranks(V[c][m], HIGHER[m])
            assert all(len(set(row)) == len(row) for row in V[c][m].reshape(-1, len(LABELS))), "tied scores"
    mc = LABELS.index("MC-GS")
    st = dict(scans=list(scans), persons={p: persons[p] for p in pnames}, afs=list(AFS), labels=LABELS,
              left_out=[s for s in SCANS if s not in scans])
    # 1. first counts
    fc = {}
    for m in METRICS:
        for c in CONTRASTS:
            fc[f"{m}|{c}"] = [int((R[c][m][:, :, mc] == 1).sum()), int(R[c][m][:, :, mc].size)]
    st["first_counts"] = fc
    st["first_total"] = [sum(v[0] for v in fc.values()), sum(v[1] for v in fc.values())]
    st["mean"] = {f"{c}|{m}": {l: [float(x) for x in V[c][m][:, :, i].mean(axis=0)] for i, l in enumerate(LABELS)}
                  for c in CONTRASTS for m in METRICS}
    st["sd"] = {f"{c}|{m}": {l: [float(x) for x in V[c][m][:, :, i].std(axis=0, ddof=1)] for i, l in enumerate(LABELS)}
                for c in CONTRASTS for m in METRICS}
    # 2. Friedman
    fr = {}
    for c in CONTRASTS:
        for m in METRICS:
            afm = V[c][m].mean(axis=1)                                             # (scan, method)
            per = np.array([afm[[scans.index(s) for s in persons[p]]].mean(axis=0) for p in pnames])
            ent = dict(scans=friedman(afm, HIGHER[m], rng), persons=friedman(per, HIGHER[m], rng), per_af={})
            for a, af in enumerate(AFS):
                ent["per_af"][str(af)] = friedman(V[c][m][:, a, :], HIGHER[m], rng)
            fr[f"{c}|{m}"] = ent
    st["friedman"] = fr
    # 3. MC-GS against each other method
    G, gain = {}, {}
    for c in CONTRASTS:
        for m in METRICS:
            for o in OTHERS:
                d = CS.improvement(m, V[c][m][:, :, mc], V[c][m][:, :, LABELS.index(o)])   # (scan, AF)
                G[(c, m, o)] = d
                afm = d.mean(axis=1)
                per = [float(np.mean([afm[scans.index(s)] for s in persons[p]])) for p in pnames]
                gain[f"{c}|{m}|{o}"] = dict(
                    af_mean_scans=summary(afm, scans), af_mean_persons=summary(per, pnames),
                    per_af={str(af): summary(d[:, a], scans) for a, af in enumerate(AFS)},
                    cells_won=[int((d > 0).sum()), int(d.size)])
    st["gain"] = gain
    # 4. slice level, within a scan
    sl, viol = {}, {}
    for s in scans:
        for af in AFS:
            for c in CONTRASTS:
                cell = cells[f"{s}|{af}|{c}"]
                z = np.load(cell["methods"]["MC-GS"]["cache"])
                keep = {ax: z[f"keep{ax}"] for ax in (0, 1, 2)}
                a_mc = slice_arrays(cell, "MC-GS", keep)
                for o in OTHERS:
                    a_o = slice_arrays(cell, o, keep)
                    ent = {}
                    for m in METRICS:
                        d = CS.improvement(m, a_mc[m], a_o[m])
                        q1, med, q3 = (float(x) for x in np.percentile(d, [25, 50, 75]))
                        ent[m] = dict(n=int(d.size), median=med, q1=q1, q3=q3, mean=float(d.mean()), win_share=float(np.mean(d > 0)))
                        if o == "GS-single" and m == "head_psnr":
                            viol[(s, af, c)] = d
                    sl[f"{s}|{af}|{c}|{o}"] = ent
    st["slice"] = sl
    return st, V, R, G, viol


# ===================================================================== figures
def style():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 6, "axes.labelsize": 6, "xtick.labelsize": 5.5, "ytick.labelsize": 6,
                         "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.major.size": 2, "ytick.major.size": 2, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none", "legend.frameon": False,
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.edgecolor": INK, "figure.facecolor": "white", "savefig.facecolor": "white"})


def save(fig, art, stem):
    fig.savefig(art / f"{stem}.png", dpi=600)
    fig.savefig(art / f"{stem}.pdf", metadata={"CreationDate": None})
    plt.close(fig)
    log("figure", stem)


def method_label(t, l):
    if l == "MC-GS":
        t.set_color(RED)
        t.set_fontweight("bold")


MM = 1 / 25.4


def fig_means(V, art, stem):
    """a T1, b FLAIR: rows = methods, columns = PSNR | SSIM | NMSE x AF; cell = mean over the scans"""
    blocks = [("head_psnr", "PSNR (dB)", "{:.2f}", 1.0, "#3C5488"), ("head_ssim", "SSIM", "{:.3f}", 1.0, "#00A087"),
              ("head_nmse", "NMSE (×10$^{-2}$)", "{:.2f}", 100.0, "#7E6148")]
    gap, nl = 0.45, len(LABELS)
    width = 3 * len(AFS) + 2 * gap
    fig = plt.figure(figsize=(183 * MM, 116 * MM))
    for ci, c in enumerate(CONTRASTS):
        ax = fig.add_axes([0.105, 0.525 - ci * 0.495, 0.885, 0.445])
        for b, (m, name, fmt, scale, hue) in enumerate(blocks):
            M = V[c][m].mean(axis=0) * scale                      # (AF, method)
            body = M[:, 1:]                                         # colour scale without ZF (clipped)
            lo, hi = float(body.min()), float(body.max())
            cmap = LinearSegmentedColormap.from_list("h", ["#ffffff", hue])
            x0 = b * (len(AFS) + gap)
            for a, af in enumerate(AFS):
                best = int(np.argmax(M[a]) if HIGHER[m] else np.argmin(M[a]))
                for i, l in enumerate(LABELS):
                    g = (M[a, i] - lo) / (hi - lo) if HIGHER[m] else (hi - M[a, i]) / (hi - lo)
                    g = float(np.clip(g, 0, 1))
                    ax.add_patch(Rectangle((x0 + a, i), 1, 1, facecolor=cmap(0.05 + 0.85 * g), edgecolor="white", lw=0.6))
                    ax.text(x0 + a + 0.5, i + 0.52, fmt.format(M[a, i]), ha="center", va="center", fontsize=5.6,
                            color="white" if g > 0.58 else INK, fontweight="bold" if i == best else "normal")
                ax.add_patch(Rectangle((x0 + a + 0.04, best + 0.05), 0.92, 0.9, fill=False, edgecolor=INK, lw=0.9, zorder=5))
                ax.text(x0 + a + 0.5, -0.38, f"AF{af}", ha="center", va="center", fontsize=5.6, color=INK2)
            ax.text(x0 + len(AFS) / 2, -1.12, name, ha="center", va="center", fontsize=6.5, fontweight="bold")
        for i, l in enumerate(LABELS):
            method_label(ax.text(-0.12, i + 0.52, l, ha="right", va="center", fontsize=6), l)
        ax.text(-0.12, -1.12, c, ha="right", va="center", fontsize=7)
        ax.set_xlim(-0.02, width + 0.02)
        ax.set_ylim(nl + 0.05, -1.6)
        ax.axis("off")
        fig.text(0.008, 0.525 - ci * 0.495 + 0.445, "ab"[ci], fontsize=8, fontweight="bold", ha="left", va="top")
    save(fig, art, stem)


def fig_ranks(R, scans, art, stem):
    """rank of every method in every (scan, AF) column; six panels (contrast x metric)"""
    ns, nl, gap = len(scans), len(LABELS), 0.7
    cols = [RED, BLUE] + [matplotlib.colors.to_hex(c) for c in
                          LinearSegmentedColormap.from_list("g", ["#8fa1c7", "#f1f1f1"])(np.linspace(0, 1, nl - 2))]
    cmap = ListedColormap(cols)
    norm = BoundaryNorm(np.arange(0.5, nl + 1), nl)
    fig = plt.figure(figsize=(183 * MM, 92 * MM))
    gs = fig.add_gridspec(2, 3, left=0.105, right=0.915, top=0.925, bottom=0.085, wspace=0.07, hspace=0.24)
    for i, c in enumerate(CONTRASTS):
        for j, m in enumerate(METRICS):
            ax = fig.add_subplot(gs[i, j])
            for a in range(len(AFS)):
                x0 = a * (ns + gap)
                ax.pcolormesh(x0 + np.arange(ns + 1), np.arange(nl + 1), R[c][m][:, a, :].T, cmap=cmap, norm=norm,
                              edgecolors="white", linewidth=0.4)
            ax.set_xlim(0, len(AFS) * ns + (len(AFS) - 1) * gap)
            ax.set_ylim(nl, 0)
            ax.set_xticks([a * (ns + gap) + ns / 2 for a in range(len(AFS))])
            ax.set_xticklabels([f"AF{af}" for af in AFS] if i == 1 else [])
            ax.set_yticks(np.arange(nl) + 0.5)
            ax.set_yticklabels(LABELS if j == 0 else [])
            for t, l in zip(ax.get_yticklabels(), LABELS):
                method_label(t, l)
            ax.tick_params(length=0, pad=2)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.text(0.0, 1.035, f"{c}, {MNAME[m]}", transform=ax.transAxes, fontsize=6.5, ha="left", va="bottom")
            p = ax.get_position()
            fig.text(p.x0 - (0.097 if j == 0 else 0.016), p.y1 + 0.012, "abcdef"[3 * i + j], fontsize=8, fontweight="bold",
                     ha="left", va="bottom")
    cax = fig.add_axes([0.935, 0.25, 0.011, 0.50])
    cb = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, ticks=np.arange(1, nl + 1))
    cb.ax.invert_yaxis()
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=0, labelsize=5.5)
    cb.set_label("Rank among the 11 methods (1 = best)", fontsize=6)
    save(fig, art, stem)


def forest_order(st):
    key = lambda o: np.mean([st["gain"][f"{c}|head_psnr|{o}"]["af_mean_scans"]["mean"] for c in CONTRASTS])
    return sorted([o for o in OTHERS if o != "GS-single"], key=key)


def fig_gains(st, G, scans, art, stem):
    """gain of MC-GS over each other method: dot per scan (mean over the AFs), mean and 95 % interval; GS-single on its own axis"""
    order = forest_order(st)
    xl = {"head_psnr": "PSNR gain of MC-GS (dB)", "head_ssim": "SSIM gain of MC-GS", "head_nmse": "NMSE reduction by MC-GS (%)"}
    ns = len(scans)
    fig = plt.figure(figsize=(183 * MM, 128 * MM))
    outer = fig.add_gridspec(2, 3, left=0.105, right=0.988, top=0.935, bottom=0.075, wspace=0.09, hspace=0.21)

    def draw(ax, rows, c, m, left):
        vals = []
        for y, o in enumerate(rows):
            e = st["gain"][f"{c}|{m}|{o}"]
            d = G[(c, m, o)].mean(axis=1)
            vals += list(d) + e["af_mean_scans"]["ci95"]
            ax.plot(e["af_mean_scans"]["ci95"], [y, y], color=RED, lw=1.0, solid_capstyle="butt", zorder=3)
            ax.scatter(d, y + np.linspace(-0.24, 0.24, ns), s=7, facecolors="white", edgecolors=INK2, linewidths=0.5, zorder=4)
            ax.scatter([e["af_mean_scans"]["mean"]], [y], s=15, marker="D", color=RED, linewidths=0, zorder=5)
            ax.text(0.995, y, "{}/{}".format(*e["cells_won"]), transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=5.2, color=INK2)
        lo, hi = min(vals + [0.0]), max(vals + [0.0])
        span = hi - lo
        ax.set_xlim(lo - 0.05 * span, hi + 0.24 * span)
        ax.axvline(0, color=INK2, lw=0.5, ls=(0, (3, 2)), zorder=1)
        ax.set_ylim(len(rows) - 0.45, -0.55)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows if left else [])
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.spines["left"].set_visible(False)

    for i, c in enumerate(CONTRASTS):
        for j, m in enumerate(METRICS):
            inner = outer[i, j].subgridspec(2, 1, height_ratios=[1.25, 9], hspace=0.3)
            top, bot = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
            draw(top, ["GS-single"], c, m, j == 0)
            draw(bot, order, c, m, j == 0)
            if i == 1:
                bot.set_xlabel(xl[m])
            p = top.get_position()
            fig.text(p.x0 - (0.097 if j == 0 else 0.016), p.y1 + 0.014, "abcdef"[3 * i + j], fontsize=8, fontweight="bold",
                     ha="left", va="bottom")
            top.text(0.0, 1.22, f"{c}, {MNAME[m]}", transform=top.transAxes, fontsize=6.5, ha="left", va="bottom")
    save(fig, art, stem)
    return order


def fig_slices(st, viol, scans, order, art, stem):
    """a, b: per-slice PSNR gain over GS-single per scan and AF; c, d: share of slices won against each method"""
    ns = len(scans)
    fig = plt.figure(figsize=(183 * MM, 142 * MM))
    for j, c in enumerate(CONTRASTS):
        ax = fig.add_axes([0.075 + j * 0.495, 0.70, 0.42, 0.235])
        trims = np.array([np.percentile(viol[(s, af, c)], [2.5, 97.5]) for s in scans for af in AFS])
        lo, hi = min(0.0, float(trims[:, 0].min())), float(trims[:, 1].max())     # every violin is drawn whole
        for a, af in enumerate(AFS):
            for k, s in enumerate(scans):
                d = viol[(s, af, c)]
                pos = a + (k - (ns - 1) / 2) * 0.165
                q0, q1, med, q3, q4 = np.percentile(d, [2.5, 25, 50, 75, 97.5])
                body = ax.violinplot(np.clip(d, q0, q4), positions=[pos], widths=0.15, showextrema=False)
                for pc in body["bodies"]:
                    pc.set_facecolor(SCAN_COL[k]); pc.set_edgecolor(SCAN_COL[k]); pc.set_alpha(0.55); pc.set_linewidth(0.3)
                ax.plot([pos, pos], [q1, q3], color=INK, lw=0.7, solid_capstyle="round", zorder=3)
                ax.scatter([pos], [med], s=4.5, color="white", edgecolor=INK, lw=0.4, zorder=4)
        ax.axhline(0, color=INK2, lw=0.5, ls=(0, (3, 2)), zorder=1)
        pad = 0.06 * (hi - lo)
        ax.set_ylim(lo - pad, hi + 2.6 * pad)
        ax.set_xlim(-0.55, len(AFS) - 0.45)
        ax.set_xticks(range(len(AFS)))
        ax.set_xticklabels([f"AF{af}" for af in AFS])
        ax.tick_params(axis="x", length=0)
        if j == 0:
            ax.set_ylabel("Per-slice PSNR gain of MC-GS\nover GS-single (dB)")
        ax.text(0.012, 0.975, c, transform=ax.transAxes, fontsize=6.5, ha="left", va="top")
        p = ax.get_position()
        fig.text(p.x0 - (0.068 if j == 0 else 0.032), p.y1 + 0.008, "ab"[j], fontsize=8, fontweight="bold", ha="left", va="bottom")
    h = [Line2D([], [], marker="s", ls="", ms=4, mfc=SCAN_COL[k], mec=SCAN_COL[k], alpha=0.75, label=SHORT(s)) for k, s in enumerate(scans)]
    fig.legend(handles=h, loc="upper center", ncol=ns, bbox_to_anchor=(0.53, 0.995), fontsize=6, handletextpad=0.2, columnspacing=1.4)
    rows = ["GS-single"] + order
    cmap = LinearSegmentedColormap.from_list("d", [BLUE, "#ffffff", RED])
    gap = 0.6
    for i, c in enumerate(CONTRASTS):
        ax = fig.add_axes([0.105, 0.365 - i * 0.325, 0.82, 0.265])
        for a, af in enumerate(AFS):
            x0 = a * (ns + gap)
            for k, s in enumerate(scans):
                for y, o in enumerate(rows):
                    w = 100 * st["slice"][f"{s}|{af}|{c}|{o}"]["head_psnr"]["win_share"]
                    ax.add_patch(Rectangle((x0 + k, y), 1, 1, facecolor=cmap(w / 100), edgecolor="white", lw=0.5))
                    ax.text(x0 + k + 0.5, y + 0.52, f"{w:.0f}", ha="center", va="center", fontsize=5.2,
                            color="white" if abs(w - 50) > 30 else INK)
                ax.text(x0 + k + 0.5, len(rows) + 0.5, SHORT(s), ha="center", va="center", fontsize=5.2, color=INK2)
            ax.text(x0 + ns / 2, -0.5, f"AF{af}", ha="center", va="center", fontsize=6)
        for y, o in enumerate(rows):
            ax.text(-0.15, y + 0.52, o, ha="right", va="center", fontsize=6)
        ax.text(-0.15, -0.5, c, ha="right", va="center", fontsize=6.5)
        ax.set_xlim(-0.02, len(AFS) * ns + (len(AFS) - 1) * gap + 0.02)
        ax.set_ylim(len(rows) + 1.0, -1.0)
        ax.axis("off")
        fig.text(0.008, 0.365 - i * 0.325 + 0.265, "cd"[i], fontsize=8, fontweight="bold", ha="left", va="top")
    cax = fig.add_axes([0.945, 0.12, 0.011, 0.42])
    cb = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=matplotlib.colors.Normalize(0, 100), ticks=[0, 25, 50, 75, 100])
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=1.5, width=0.4, labelsize=5.5)
    cb.set_label("Slices with higher PSNR for MC-GS (%)", fontsize=6)
    save(fig, art, stem)


# ===================================================================== documents
def disclosure(st):
    n, tot = len(st["scans"]), len(SCANS)
    if n == tot:
        return f"All {tot} test scans ({', '.join(SHORT(s) for s in st['scans'])}; {len(st['persons'])} persons)."
    return (f"{n} of the {tot} test scans ({', '.join(SHORT(s) for s in st['scans'])}; {len(st['persons'])} persons). "
            f"{', '.join(SHORT(s) for s in st['left_out'])} left out by the user's decision (2026-11-27); in the six-scan "
            f"result MC-GS is second in 13 of 24 columns of that scan, 12 behind GS-single and 1 behind ZS-SSDU (STATUS.md, RESULTS.md).")


def write_legends(st, order, art, stem):
    dis = disclosure(st)
    n, npers = len(st["scans"]), len(st["persons"])
    f1, t1 = st["first_total"]
    L = [
        f"{stem}_fig1_means: Mean head PSNR, SSIM and NMSE over the {n} scans of every method at AF12 / 16 / 20 / 24, T1 (a) and "
        f"FLAIR (b). Colour: one scale per metric block, darker = better (ZF clipped). Outlined bold cell: best method of the "
        f"column. One seed. {dis}",
        f"{stem}_fig2_ranks: Rank of every method (1 = best) in every (scan, AF) column for PSNR, SSIM and NMSE, T1 (a-c) and "
        f"FLAIR (d-f). Within each AF group the columns are the scans {', '.join(SHORT(s) for s in st['scans'])}. MC-GS is "
        f"first in {f1} of {t1} columns. Friedman test over the 11 methods (blocks = scans, mean over the four AFs, "
        f"permutation p) and Kendall's W are given in the statistics file; they show that the methods differ and rank "
        f"consistently between scans, not that MC-GS beats a particular method. {dis}",
        f"{stem}_fig3_gains: Gain of MC-GS over each of the 10 other methods in head PSNR (a, d), SSIM (b, e) and NMSE "
        f"(relative reduction; c, f), T1 (a-c) and FLAIR (d-f). Open circle: one scan (mean over the four AFs); red diamond and "
        f"bar: mean and 95% interval (t distribution, n = {n} scans; {npers} persons). GS-single has its own axis. Numbers at the "
        f"right: (scan, AF) cells in which MC-GS is better. Dashed line: no difference. Rows in one order for all panels: the PSNR gain averaged over T1 and FLAIR. "
        f"The exact two-sided Wilcoxon signed-rank p cannot be below {2 / 2 ** n:.4g} with {n} scans or {2 / 2 ** npers:.4g} "
        f"with {npers} persons, so no pairwise p value is shown. {dis}",
        f"{stem}_fig4_slices: a, b, Per-slice head PSNR gain of MC-GS over GS-single for every scan and AF (slices whose head "
        f"covers at least 5% of the slice, three planes pooled), T1 (a) and FLAIR (b). Dot, median; bar, interquartile range; "
        f"violin, central 95% of the slices (2.5-97.5 percentiles). c, d, Share of slices (%) on which MC-GS has the higher PSNR than each "
        f"other method, per scan and AF, T1 (c) and FLAIR (d). Slices of one volume are not independent: this shows the "
        f"consistency within a scan, not a significance test. {dis}",
    ]
    (art / f"{stem}_legends.txt").write_text("\n\n".join(L) + "\n", encoding="utf-8")


def write_csv(st, V, art, stem):
    rows = ["scan,person,af,contrast,method,head_psnr,head_ssim,head_nmse"]
    for si, s in enumerate(st["scans"]):
        for a, af in enumerate(AFS):
            for c in CONTRASTS:
                for i, l in enumerate(LABELS):
                    rows.append(f"{s},{SCANS[s]['person']},{af},{c},{l}," + ",".join(f"{V[c][m][si, a, i]:.8g}" for m in METRICS))
    (art / f"{stem}_source_volume.csv").write_text("\n".join(rows) + "\n")
    rows = ["scan,af,contrast,other_method,metric,n_slices,median,q1,q3,mean,win_share"]
    for k, ent in st["slice"].items():
        s, af, c, o = k.split("|")
        for m in METRICS:
            e = ent[m]
            rows.append(f"{s},{af},{c},{o},{MNAME[m]},{e['n']},{e['median']:.6g},{e['q1']:.6g},{e['q3']:.6g},{e['mean']:.6g},{e['win_share']:.6f}")
    (art / f"{stem}_source_slice.csv").write_text("\n".join(rows) + "\n")


def write_md(st, order, art, stem):
    n, npers = len(st["scans"]), len(st["persons"])
    fmtv = {"head_psnr": "{:+.3f}", "head_ssim": "{:+.4f}", "head_nmse": "{:+.2f}"}
    unit = {"head_psnr": "dB", "head_ssim": "", "head_nmse": "%"}
    L = [f"# 测试集 T1 + FLAIR：{n} 次扫描的统计（{'、'.join(SHORT(s) for s in st['scans'])}）", "",
         f"> 由 `{CS.RESULTS_REL}/stats_subset.py` 生成。分析方案写在脚本开头，先定方案后算数。", ""]
    if st["left_out"]:
        L += [f"**范围：** 6 次测试扫描中的 {n} 次（{npers} 个人）。{'、'.join(SHORT(s) for s in st['left_out'])} 按用户 2026-11-27 的决定不计入；"
              f"六次扫描的完整结果在 `STATUS.md` / `RESULTS.md`（vol7 上 MC-GS 有 13 列排第二：12 列落后 GS-single，1 列落后 ZS-SSDU）。", ""]
    L += ["## 1. MC-GS 是否第一", "", f"{st['first_total'][1]} 列（扫描 × 加速倍数 × 对比度 × 指标）中 MC-GS 第一 {st['first_total'][0]} 列。", "",
          "| 指标 | T1 | FLAIR |", "|---|---|---|"]
    for m in METRICS:
        L.append(f"| {MNAME[m]} | " + " | ".join("{} / {}".format(*st["first_counts"][f"{m}|{c}"]) for c in CONTRASTS) + " |")
    L += ["", "## 2. 11 个方法的整体检验（Friedman，区组 = 扫描 / 人，四个加速倍数取平均）", "",
          "| 对比度 | 指标 | 扫描为区组：χ²，Kendall W，置换 p | 人为区组：χ²，Kendall W，置换 p | MC-GS 平均名次 |", "|---|---|---|---|---|"]
    for c in CONTRASTS:
        for m in METRICS:
            e = st["friedman"][f"{c}|{m}"]
            cell = lambda x: f"{x['chi2']:.1f}，{x['kendall_w']:.3f}，{'< 0.0001' if x['p_perm'] < 1e-4 else format(x['p_perm'], '.4f')}（N = {x['n_blocks']}）"
            L.append(f"| {c} | {MNAME[m]} | {cell(e['scans'])} | {cell(e['persons'])} | {e['scans']['mean_rank']['MC-GS']:.2f} |")
    L += ["", f"置换 p 的下限是 1 / {NPERM + 1}。这个检验说明 11 个方法有差别、各扫描之间名次一致，不说明 MC-GS 胜过某个特定方法。", "",
          "## 3. MC-GS 对其余 10 个方法（每次扫描先对四个加速倍数取平均）", ""]
    for m in METRICS:
        L += [f"### {MNAME[m]}{'（相对降低 %）' if m == 'head_nmse' else '（' + unit[m] + '）' if unit[m] else ''}", "",
              f"| 对比度 | 对照 | 平均差 | 95% 区间（t，n = {n}） | 为正的扫描 | 为正的人 | 胜出的格子（扫描 × AF） | Wilcoxon 精确 p（扫描 / 人） |", "|---|---|---|---|---|---|---|---|"]
        for c in CONTRASTS:
            for o in ["GS-single"] + order:
                e = st["gain"][f"{c}|{m}|{o}"]
                a, p = e["af_mean_scans"], e["af_mean_persons"]
                L.append(f"| {c} | {o} | {fmtv[m].format(a['mean'])} | {fmtv[m].format(a['ci95'][0])} 到 {fmtv[m].format(a['ci95'][1])} | "
                         f"{a['n_positive']} / {a['n']} | {p['n_positive']} / {p['n']} | {e['cells_won'][0]} / {e['cells_won'][1]} | "
                         f"{a['wilcoxon_p_exact']:.4f} / {p['wilcoxon_p_exact']:.3f} |")
        L.append("")
    L += [f"{n} 次扫描时精确双侧 Wilcoxon p 最小是 {2 / 2 ** n:.4f}，{npers} 个人时最小是 {2 / 2 ** npers:.3f}：任何成对检验都到不了 0.05，"
          "所以结论靠效应量和一致性，不写显著性。", "",
          "## 4. 逐层面（只在一次扫描之内）", "", "MC-GS 的 PSNR 高于对照的层面占比（各扫描 × 加速倍数的最小值到最大值）：", "",
          "| 对照 | T1 | FLAIR |", "|---|---|---|"]
    for o in ["GS-single"] + order:
        row = []
        for c in CONTRASTS:
            w = [100 * st["slice"][f"{s}|{af}|{c}|{o}"]["head_psnr"]["win_share"] for s in st["scans"] for af in AFS]
            row.append(f"{min(w):.0f}% 到 {max(w):.0f}%")
        L.append(f"| {o} | {row[0]} | {row[1]} |")
    L += ["", "同一体积的层面彼此不独立，这里只说明一次扫描之内的一致性，不做显著性检验。", "",
          "## 5. 限制", "", f"- {npers} 个人，{n} 次扫描（vol2 与 vol6 是同一个人）。", "- 一个随机种子。",
          "- 所有设置取自调参病例，没有在测试扫描上调参；真值只用于评分。"]
    if st["left_out"]:
        L.append(f"- {'、'.join(SHORT(s) for s in st['left_out'])} 是看到结果之后才不计入的；写进论文时需要说明不计入的理由，或同时给出六次扫描的结果。")
    L += ["", "## 6. 图", "", f"`figures/{stem}_fig1_means`、`_fig2_ranks`、`_fig3_gains`、`_fig4_slices`（PNG 600 dpi + PDF，不入库）；图注在 `{stem}_legends.txt`。"]
    (art / f"{stem}.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", nargs="+", choices=list(SCANS), default=list(SCANS))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--save_dir", default="", help="server folder for a checksum-verified copy of everything written")
    args = ap.parse_args()
    scans = [s for s in SCANS if s in args.scans]
    tag = args.tag or ("all" if len(scans) == len(SCANS) else f"n{len(scans)}")
    stem = f"subset_{tag}"
    art = Path(os.environ.get("MCGS_ARTIFACT_DIR", "subset_out")).resolve()
    art.mkdir(parents=True, exist_ok=True)
    cells, ref = gather()
    st, V, R, G, viol = analyse(cells, scans)
    # the rank-1 count must equal the one in table.json for these scans
    want = sum(ref[f"{s}|{af}|{c}"]["first"][m] for s in scans for af in AFS for c in CONTRASTS for m in METRICS)
    assert st["first_total"][0] == want, (st["first_total"], want)
    style()
    fig_means(V, art, f"{stem}_fig1_means")
    fig_ranks(R, scans, art, f"{stem}_fig2_ranks")
    order = fig_gains(st, G, scans, art, f"{stem}_fig3_gains")
    fig_slices(st, viol, scans, order, art, f"{stem}_fig4_slices")
    st["forest_order"] = order
    st["disclosure"] = disclosure(st)
    (art / f"{stem}_stats.json").write_text(json.dumps(st, indent=1))
    write_csv(st, V, art, stem)
    write_legends(st, order, art, stem)
    write_md(st, order, art, stem)
    for f in art.glob(f"{stem}*"):
        if f.suffix in (".pdf", ".png"):
            b = f.read_bytes()
            for pat in (b"CreationDate", b"ModDate", b"tIME", b"date:create", b"date:modify"):
                assert pat not in b, (f.name, pat)
    if args.save_dir:
        dst = Path(args.save_dir)
        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(art.glob(f"{stem}*")):
            t = dst / f.name
            assert not t.exists(), f"{t} exists; choose another --save_dir (nothing is overwritten)"
            shutil.copyfile(f, t)
            assert sha256(f) == sha256(t), f.name
        (dst / f"{stem}_sha256.txt").write_text("".join(f"{sha256(f)}  {f.name}\n" for f in sorted(art.glob(f"{stem}*"))))
        log(f"copied to {dst} (checksums verified)")
    log(f"MC-GS first in {st['first_total'][0]} / {st['first_total'][1]} columns of {len(scans)} scans")
    log("SUBSET_DONE")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colorbar
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    import numpy as np
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from scipy.stats import rankdata
    from scipy.stats import t as tdist
    main()
