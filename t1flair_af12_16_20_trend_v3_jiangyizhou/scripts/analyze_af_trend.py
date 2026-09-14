#!/usr/bin/env python3
"""analyze_af_trend.py -- does every head-only metric get worse at each acceleration step, for every arm?

Input : logs/rescore_af<AF>_<C>.json from scripts/rescore_topic.py (one scorer, same GT, three topics).
Output: logs/af_trend_tables.md (per contrast x metric, AF12/16/20 values and both step changes),
        logs/af_trend_verdict.json, figures/jyz_t1flair_af12_16_20_metric_steps.png (mirrored to the served figs folder),
        and on stdout the re-score check, the compact summary and the same-recipe (P6) MC-GS cross-check.
AF labels: whole-plane AF = PE-plane lines / sampled lines that carry that contrast's data (user decision 2026-09-14).
"""
import json, os, shutil
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

T = "/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_16_20_trend_v3_jiangyizhou"
OUTFIG = f"{T}/figures/jyz_t1flair_af12_16_20_metric_steps.png"
SERVED = "/home/congcongliu/figs/t1flair_v3_af_trend"
AFS = ("12", "16", "20")
WP = {"T1": {"12": 11.71, "16": 15.19, "20": 18.56}, "FLAIR": {"12": 12.86, "16": 16.49, "20": 19.96}}   # whole-plane AF
ORDER = ["MC-GS", "GS-single", "L1-ESPIRiT", "CS-SENSE (converged TV)", "Joint SENSE", "CS-SENSE (35 it)",
         "ZS-SSDU", "CG-SENSE", "MC-INR", "3D-DIP", "3D-INR", "ZF"]
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

# 2. tables + monotonic verdicts
verdict = {}
lines = []
for c in ("T1", "FLAIR"):
    for m, name, sgn, fmt in MET:
        lines.append(f"\n### {c} {name}   (whole-plane AF {WP[c]['12']} -> {WP[c]['16']} -> {WP[c]['20']})")
        lines.append("| method | AF12 | AF16 | AF20 | step 12->16 | step 16->20 | worse at both steps |")
        lines.append("|---|---|---|---|---|---|---|")
        for lab in ORDER:
            if not all((af, c, lab) in V for af in AFS): lines.append(f"| {lab} | (incomplete) |"); continue
            v = [V[(af, c, lab)][m] for af in AFS]
            q1, q2 = sgn * (v[1] - v[0]), sgn * (v[2] - v[1])          # quality change: negative = worse
            ok = q1 < 0 and q2 < 0
            verdict[(c, m, lab)] = (ok, q1, q2, v)
            lines.append(f"| {lab} | {fmt.format(v[0])} | {fmt.format(v[1])} | {fmt.format(v[2])} | {v[1]-v[0]:+.4f} | {v[2]-v[1]:+.4f} | {'yes' if ok else 'NO'} |")
txt = "\n".join(lines); print(txt)
open(f"{T}/logs/af_trend_tables.md", "w").write(txt)
json.dump({f"{c}|{m}|{lab}": dict(monotonic=ok, step1_quality=q1, step2_quality=q2, values=v) for (c, m, lab), (ok, q1, q2, v) in verdict.items()},
          open(f"{T}/logs/af_trend_verdict.json", "w"), indent=1)

# 3. figure: per-step quality change, negative = worse
BLUE, ORANGE, SURF, INK, INK2, MUTED, GRID, BASE = "#2a78d6", "#eb6834", "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
                     "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
fig, axes = plt.subplots(2, 3, figsize=(15, 9.6), sharey=True)
y = np.arange(len(ORDER))
for i, c in enumerate(("T1", "FLAIR")):
    for j, (m, name, sgn, fmt) in enumerate(MET):
        ax = axes[i, j]
        q1 = [verdict[(c, m, lab)][1] if (c, m, lab) in verdict else np.nan for lab in ORDER]
        q2 = [verdict[(c, m, lab)][2] if (c, m, lab) in verdict else np.nan for lab in ORDER]
        ax.axvline(0, color=BASE, lw=1, zorder=1)
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
        ax.plot(q1, y - 0.16, "o", ms=7, color=BLUE, mec=SURF, mew=1.2, zorder=3, label=f"AF12 → AF16")
        ax.plot(q2, y + 0.16, "o", ms=7, color=ORANGE, mec=SURF, mew=1.2, zorder=3, label=f"AF16 → AF20")
        for k, lab in enumerate(ORDER):                       # label only the steps where the metric did NOT get worse
            for qq, dy in ((q1[k], -0.16), (q2[k], 0.16)):
                if np.isfinite(qq) and qq >= 0:
                    ax.annotate(f"{qq:+.3f}" if m == "head_psnr" else f"{qq:+.4f}", (qq, y[k] + dy), xytext=(6, 0),
                                textcoords="offset points", va="center", fontsize=8.5, color=INK)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ttl = {"head_psnr": "head PSNR change (dB)", "head_ssim": "head SSIM change", "head_hfen": "head HFEN change, sign flipped"}[m]
        ax.set_title(f"{c} · {ttl}", fontsize=11, color=INK, loc="left")
        ax.set_xlabel("← worse      better →", color=MUTED, fontsize=9)
        lo = np.nanmin(q1 + q2); hi = np.nanmax(q1 + q2); pad = 0.08 * (hi - lo)
        ax.set_xlim(min(lo - pad, -pad), max(hi + pad * 3, pad * 3))
ax0 = axes[0, 0]; ax0.set_yticks(y); ax0.set_yticklabels(ORDER); ax0.invert_yaxis()
for i in range(2): axes[i, 0].set_yticklabels(ORDER)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper right", bbox_to_anchor=(0.985, 0.975), ncol=2, frameon=False, fontsize=10)
fig.suptitle("jiangyizhou T1 + FLAIR: change of each head-only metric per acceleration step (one scorer, same GT)",
             x=0.01, ha="left", fontsize=13, color=INK, y=0.985)
fig.text(0.01, 0.945, "whole-plane AF  T1 11.71 → 15.19 → 18.56   ·   FLAIR 12.86 → 16.49 → 19.96   ·   "
         "left of 0 = the metric got worse at that step; labelled dots did not", fontsize=9.5, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUTFIG, dpi=150); os.makedirs(SERVED, exist_ok=True); shutil.copy2(OUTFIG, SERVED)
print("wrote", OUTFIG)

# 4. compact summary for the reply: per method, per contrast
print("\n=== compact summary (Y = worse at both steps; head PSNR values; dB per doubling of whole-plane AF) ===")
for c in ("T1", "FLAIR"):
    r1, r2 = np.log2(WP[c]["16"] / WP[c]["12"]), np.log2(WP[c]["20"] / WP[c]["16"])
    print(f"\n{c}: | method | AF12 | AF16 | AF20 | 12->20 | dB/doubling step1, step2 | PSNR SSIM HFEN wholePSNR |")
    for lab in ORDER:
        if (c, "head_psnr", lab) not in verdict: print(f"  {lab}: incomplete"); continue
        v = verdict[(c, "head_psnr", lab)][3]
        flags = "".join("Y" if verdict[(c, m, lab)][0] else "N" for m, *_ in MET)
        wv = [V[(af, c, lab)].get("rec_psnr") for af in AFS]
        wflag = "Y" if all(x is not None for x in wv) and wv[1] < wv[0] and wv[2] < wv[1] else "N"
        print(f"  | {lab} | {v[0]:.3f} | {v[1]:.3f} | {v[2]:.3f} | {v[2]-v[0]:+.3f} | {(v[1]-v[0])/r1:+.2f}, {(v[2]-v[1])/r2:+.2f} | "
              f"{flags[0]} {flags[1]} {flags[2]} {wflag} | whole {wv[0]:.3f}/{wv[1]:.3f}/{wv[2]:.3f}")

# 5. same-recipe cross-check for MC-GS: P6 exists on the AF 12 and AF 16 masks (no P6 run at AF 20)
print("\n=== same-recipe MC-GS cross-check (P6, each run's own score json) ===")
for tag, p in (("AF12 P6", "/data0/congcong/code/GS/mcgs_af12_wt/results_arc/ar0_P6_parent_full4k_score.json"),
               ("AF16 P6", "/data0/congcong/code/GS/mcgs_af16_wt/results_arc/ar14_n36_v2tv_full4k_score.json")):
    d = json.load(open(p))
    print(f"  {tag}: " + " | ".join(f"{d[ch + '_name']} PSNR {d[ch]['head_psnr']:.3f} SSIM {d[ch]['head_ssim']:.4f} HFEN {d[ch]['head_hfen']:.4f}"
                                  for ch in ("plain", "contrast")))
