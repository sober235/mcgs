#!/usr/bin/env python3
"""analyze_af_trend.py -- does every head-only metric get worse at each acceleration step (AF 12 -> 16 -> 20 -> 24), for every arm?

Extension of the AF 12 / 16 / 20 review (T1FLAIR_AF12_16_20_trend_v3_jiangyizhou, 2026-09-14) to the AF 24 topic (2026-09-15).
Input : logs/rescore_af<AF>_<C>.json from scripts/rescore_topic.py (one scorer, same GT, four topics).
Output: logs/af_trend_tables.md (per contrast x metric: the four values and the three step changes), logs/af_trend_verdict.json,
        figures/jyz_t1flair_af12_16_20_24_metric_steps.png (mirrored to the served figs folder), and on stdout the re-score check,
        the compact summary, the MC-GS margin over the strongest competing arm at every AF, and the same-recipe (P6) cross-check.
AF labels: whole-plane AF = PE-plane lines / sampled lines that carry that contrast's data (user decision 2026-09-14).
An arm missing at some AF (MC-INR was not run at AF 24) is judged on the steps it has, and marked.
"""
import json, os, shutil
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

T = "/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_16_20_24_trend_v3_jiangyizhou"
OUTFIG = f"{T}/figures/jyz_t1flair_af12_16_20_24_metric_steps.png"
SERVED = "/home/congcongliu/figs/t1flair_v3_af_trend"
AFS = ("12", "16", "20", "24")
STEPS = list(zip(AFS[:-1], AFS[1:]))
WP = {"T1": {"12": 11.71, "16": 15.19, "20": 18.56, "24": 24.00}, "FLAIR": {"12": 12.86, "16": 16.49, "20": 19.96, "24": 25.51}}
ORDER = ["MC-GS", "GS-single", "L1-ESPIRiT", "CS-SENSE (converged TV)", "Joint SENSE", "CS-SENSE (35 it)",
         "ZS-SSDU", "CG-SENSE", "MC-INR", "3D-DIP", "3D-INR", "ZF"]
OURS = ("MC-GS", "GS-single")
MET = [("head_psnr", "head PSNR (dB)", +1, "{:.3f}"), ("head_ssim", "head SSIM", +1, "{:.4f}"), ("head_hfen", "head HFEN", -1, "{:.4f}")]

rows = []
for af in AFS:
    for c in ("T1", "FLAIR"):
        p = f"{T}/logs/rescore_af{af}_{c}.json"
        rows += json.load(open(p)) if os.path.exists(p) else []
V = {(r["af"], r["contrast"], r["label"]): r for r in rows if "head_psnr" in r}
missing = [r for r in rows if "head_psnr" not in r]

# 1. rescore vs recorded
worst = {"head_psnr": 0.0, "head_ssim": 0.0, "head_hfen": 0.0}; offenders = []
for k, r in V.items():
    for m, tol in (("head_psnr", 1e-3), ("head_ssim", 1e-4), ("head_hfen", 1e-4)):
        d = abs(r[m] - r["rec_" + m]); worst[m] = max(worst[m], d)
        if d > tol: offenders.append((k, m, r[m], r["rec_" + m], r["file"]))
print("rows rescored:", len(V), " missing:", [(r['af'], r['contrast'], r['label'], r['missing']) for r in missing])
print("worst |rescore - recorded|:", {m: f"{v:.2e}" for m, v in worst.items()})
for o in offenders: print("  MISMATCH", o)

# 2. tables + verdicts (a step counts only where both ends exist)
verdict, lines = {}, []
for c in ("T1", "FLAIR"):
    for m, name, sgn, fmt in MET:
        lines.append(f"\n### {c} {name}   (whole-plane AF " + " -> ".join(str(WP[c][af]) for af in AFS) + ")")
        lines.append("| method | " + " | ".join(f"AF{af}" for af in AFS) + " | " + " | ".join(f"step {a}->{b}" for a, b in STEPS)
                     + " | worse at every step |")
        lines.append("|---" * (1 + len(AFS) + len(STEPS) + 1) + "|")
        for lab in ORDER:
            v = [V[(af, c, lab)][m] if (af, c, lab) in V else None for af in AFS]
            q = [sgn * (v[j] - v[i]) if v[i] is not None and v[j] is not None else None
                 for i, j in ((AFS.index(a), AFS.index(b)) for a, b in STEPS)]
            ev = [x for x in q if x is not None]
            ok = bool(ev) and all(x < 0 for x in ev)
            partial = any(x is None for x in v)
            verdict[(c, m, lab)] = (ok, q, v, partial)
            cells = [fmt.format(x) if x is not None else "--" for x in v]
            steps = [f"{sgn * x:+.4f}" if x is not None else "--" for x in q]      # raw value change
            lines.append(f"| {lab} | " + " | ".join(cells) + " | " + " | ".join(steps) + " | "
                         + ("yes" if ok else "NO") + (" (steps with data only)" if partial else "") + " |")
txt = "\n".join(lines); print(txt)
open(f"{T}/logs/af_trend_tables.md", "w").write(txt)
json.dump({f"{c}|{m}|{lab}": dict(worse_every_step=ok, step_quality=q, values=v, partial=pa)
           for (c, m, lab), (ok, q, v, pa) in verdict.items()}, open(f"{T}/logs/af_trend_verdict.json", "w"), indent=1)

# 3. figure: per-step quality change, negative = worse (categorical slots 1-3, validated; marker shape as second encoding)
BLUE, ORANGE, AQUA, SURF, INK, INK2, MUTED, GRID, BASE = "#2a78d6", "#eb6834", "#1baf7a", "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = [(BLUE, "o"), (ORANGE, "s"), (AQUA, "^")]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
                     "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
fig, axes = plt.subplots(2, 3, figsize=(15, 10.4), sharey=True)
y = np.arange(len(ORDER)); OFF = (-0.24, 0.0, 0.24)
for i, c in enumerate(("T1", "FLAIR")):
    for j, (m, name, sgn, fmt) in enumerate(MET):
        ax = axes[i, j]
        ax.axvline(0, color=BASE, lw=1, zorder=1)
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
        allq = []
        for s, ((a, b), (col, mk), dy) in enumerate(zip(STEPS, SERIES, OFF)):
            qs = [verdict[(c, m, lab)][1][s] for lab in ORDER]
            qn = [np.nan if x is None else x for x in qs]; allq += [x for x in qs if x is not None]
            ax.plot(qn, y + dy, mk, ms=7, color=col, mec=SURF, mew=1.2, zorder=3, label=f"AF{a} → AF{b}")
            for k in range(len(ORDER)):                         # label only the steps where the metric did NOT get worse
                if qs[k] is not None and qs[k] >= 0:
                    ax.annotate(f"{qs[k]:+.3f}" if m == "head_psnr" else f"{qs[k]:+.4f}", (qs[k], y[k] + dy), xytext=(6, 0),
                                textcoords="offset points", va="center", fontsize=8.5, color=INK)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ttl = {"head_psnr": "head PSNR change (dB)", "head_ssim": "head SSIM change", "head_hfen": "head HFEN change, sign flipped"}[m]
        ax.set_title(f"{c} · {ttl}", fontsize=11, color=INK, loc="left")
        ax.set_xlabel("← worse      better →", color=MUTED, fontsize=9)
        lo, hi = min(allq), max(allq); pad = 0.08 * (hi - lo)
        ax.set_xlim(min(lo - pad, -pad), max(hi + pad * 3, pad * 3))
for i in range(2):
    axes[i, 0].set_yticks(y); axes[i, 0].set_yticklabels([lab + (" (no AF24)" if verdict[("T1", "head_psnr", lab)][3] else "") for lab in ORDER])
axes[0, 0].invert_yaxis()
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.005, 0.935), ncol=3, frameon=False, fontsize=10)
fig.suptitle("jiangyizhou T1 + FLAIR: change of each head-only metric per acceleration step (one scorer, same GT)",
             x=0.01, ha="left", fontsize=13, color=INK, y=0.985)
fig.text(0.01, 0.948, "whole-plane AF  T1 " + " → ".join(f"{WP['T1'][af]:.2f}" for af in AFS) + "   ·   FLAIR "
         + " → ".join(f"{WP['FLAIR'][af]:.2f}" for af in AFS) + "   ·   left of 0 = the metric got worse at that step; labelled marks did not",
         fontsize=9.5, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.905))
fig.savefig(OUTFIG, dpi=150); os.makedirs(SERVED, exist_ok=True); shutil.copy2(OUTFIG, SERVED)
print("wrote", OUTFIG)

# 4. compact summary: per method, per contrast
print("\n=== compact summary (Y = worse at every step; head PSNR values; dB per doubling of whole-plane AF, per step) ===")
for c in ("T1", "FLAIR"):
    rs = [np.log2(WP[c][b] / WP[c][a]) for a, b in STEPS]
    print(f"\n{c}: | method | " + " | ".join(f"AF{af}" for af in AFS) + " | 12->24 | dB/doubling per step | PSNR SSIM HFEN wholePSNR |")
    for lab in ORDER:
        ok, q, v, partial = verdict[(c, "head_psnr", lab)]
        flags = "".join("Y" if verdict[(c, m, lab)][0] else "N" for m, *_ in MET)
        wv = [V[(af, c, lab)].get("rec_psnr") if (af, c, lab) in V else None for af in AFS]
        wok = [wv[j] < wv[i] for i, j in ((AFS.index(a), AFS.index(b)) for a, b in STEPS) if wv[i] is not None and wv[j] is not None]
        wflag = "Y" if wok and all(wok) else "N"
        dpd = ", ".join(f"{(sgn_q / r):+.2f}" if sgn_q is not None else "--" for sgn_q, r in zip(q, rs))
        tot = f"{v[-1] - v[0]:+.3f}" if v[0] is not None and v[-1] is not None else "--"
        print(f"  | {lab} | " + " | ".join(f"{x:.3f}" if x is not None else "--" for x in v) + f" | {tot} | {dpd} | "
              f"{flags[0]} {flags[1]} {flags[2]} {wflag} |" + (" (partial)" if partial else ""))

# 5. the MC-GS margin over the strongest competing arm (MC-GS and GS-single excluded), head PSNR, at every AF
print("\n=== MC-GS margin over the strongest competing arm, head PSNR (one scorer) ===")
margins = {}
for c in ("T1", "FLAIR"):
    for af in AFS:
        comp = [(V[(af, c, lab)]["head_psnr"], lab) for lab in ORDER if lab not in OURS and (af, c, lab) in V]
        best, who = max(comp)
        mc, gs = V[(af, c, "MC-GS")]["head_psnr"], V[(af, c, "GS-single")]["head_psnr"]
        margins[f"{c}|{af}"] = dict(margin=mc - best, strongest=who, gs_single_over_strongest=gs - best, coupling=mc - gs)
        print(f"  {c:5s} AF{af} ({WP[c][af]:5.2f}x)  MC-GS {mc:.3f}  strongest competing {best:.3f} ({who})  margin {mc - best:+.3f}"
              f"   GS-single {gs - best:+.3f}   coupling {mc - gs:+.3f}")
json.dump(margins, open(f"{T}/logs/af_trend_margins.json", "w"), indent=1)

# 6. same-recipe cross-check for MC-GS: P6 exists on the AF 12 and AF 16 masks only
print("\n=== same-recipe MC-GS cross-check (P6, each run's own score json) ===")
for tag, p in (("AF12 P6", "/data0/congcong/code/GS/mcgs_af12_wt/results_arc/ar0_P6_parent_full4k_score.json"),
               ("AF16 P6", "/data0/congcong/code/GS/mcgs_af16_wt/results_arc/ar14_n36_v2tv_full4k_score.json")):
    d = json.load(open(p))
    print(f"  {tag}: " + " | ".join(f"{d[ch + '_name']} PSNR {d[ch]['head_psnr']:.3f} SSIM {d[ch]['head_ssim']:.4f} HFEN {d[ch]['head_hfen']:.4f}"
                                  for ch in ("plain", "contrast")))
