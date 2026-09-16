#!/usr/bin/env python3
"""compare_masks.py -- the diff-mask study's headline table: every arm on the three mask patterns, one gauge.

The three topics share the subject, the k-space, the ESPIRiT maps, the ground truth, the scoring code and the frozen
AF16 phase-3 MC-GS recipe; they differ only in the sampling PATTERN, at a matched point budget (2964 / 2966 PE points,
T1 15.20x / 15.19x). So a column difference here is the pattern's effect and nothing else.

    vd-Poisson   T1FLAIR_AF16_v3_jiangyizhou          the house mask, not re-run for this study
    Gaussian     T1FLAIR_AF16_gauss_v3_jiangyizhou    2-D Gaussian variable-density random
    radial       T1FLAIR_AF16_radial_v3_jiangyizhou   Cartesian pseudo-radial spokes

Usage: compare_masks.py [--metric head_psnr] [--out <file>]   (read-only on all three topics)
"""
import argparse
import json
from pathlib import Path

RES = Path("/data0/congcong/code/GS/Results_20260908")
TOPICS = [("vd-Poisson", RES / "T1FLAIR_AF16_v3_jiangyizhou"),
          ("Gaussian", RES / "T1FLAIR_AF16_gauss_v3_jiangyizhou"),
          ("radial", RES / "T1FLAIR_AF16_radial_v3_jiangyizhou")]
# logs/summary.json records each arm under its KEY, not its label, so the comparison matches on keys. The phase-3
# GS arms are gs_joint_p3 / gs_single_p3 in all three topics -- the house topic ALSO carries older gs_joint /
# gs_joint_ar / gs_joint_p2 rows from its own history, and matching the bare "gs_joint" there would silently compare
# this study's phase-3 arm against a 2026-09-10 one (it read 32.890 instead of 35.706 before this was fixed).
# L1-ESPIRiT is excluded from the listing (user 2026-09-15).
ARMS = [("gs_joint_p3", "MC-GS"), ("gs_single_p3", "GS-single"), ("zsssdu", "ZS-SSDU"),
        ("cssense_o200", "CS-SENSE (TV, converged)"), ("jcssense", "joint CS-SENSE (JTV)"),
        ("cssense", "CS-SENSE"), ("spirit", "L1-SPIRiT"), ("cgsense", "CG-SENSE"),
        ("mcinr", "MC-INR"), ("dip", "3D-DIP"), ("inr", "3D-INR"), ("zf", "ZF")]
LOWER_IS_BETTER = {"head_hfen", "nmse", "hfen"}

ap = argparse.ArgumentParser()
ap.add_argument("--metric", default="head_psnr")
ap.add_argument("--gs_lead", action="store_true", help="also print MC-GS minus GS-single, the study's own coupling gain")
ap.add_argument("--out", default=None)
a = ap.parse_args()

rows = {}
labels = {}
for name, T in TOPICS:
    js = json.load(open(T / "logs/summary.json"))
    extra = T / "logs/extra_gs_arms.json"
    lab = {r["key"]: r["label"] for r in json.load(open(extra))} if extra.exists() else {}
    labels[name] = lab
    for r in js:
        rows[(name, r["contrast"], r["method"])] = r


def get(name, c, key, disp):
    """summary.json is keyed by the arm KEY; the label is only applied when a table is printed"""
    return rows.get((name, c, key))


out = [f"# Diff-mask study: every arm on three sampling patterns, one gauge ({a.metric})",
       "",
       "Same subject, same k-space, same coil maps, same ground truth, same scoring code, same frozen AF16 phase-3",
       "MC-GS recipe; matched point budget (2964 points, T1 15.20x; the house mask 2966, 15.19x). Only the sampling",
       "PATTERN differs, so every difference across a row is the pattern's effect.", ""]
best_of = {}
for c in ("T1", "FLAIR"):
    out += [f"**{c}** ({a.metric}; higher is better unless the metric says otherwise)", "",
            "| arm | vd-Poisson (house) | Gaussian | radial | Gaussian − vd-Poisson | radial − vd-Poisson |",
            "|---|---|---|---|---|---|"]
    col = {n: [] for n, _ in TOPICS}
    for key, disp in ARMS:
        vals = {}
        for name, _ in TOPICS:
            r = get(name, c, key, disp)
            vals[name] = None if r is None else float(r[a.metric])
            if vals[name] is not None:
                col[name].append((vals[name], disp))
        f = lambda v: "--" if v is None else f"{v:.3f}"
        d = lambda x, y: "--" if (x is None or y is None) else f"{x - y:+.2f}"
        out.append(f"| {disp} | {f(vals['vd-Poisson'])} | {f(vals['Gaussian'])} | {f(vals['radial'])} | "
                   f"{d(vals['Gaussian'], vals['vd-Poisson'])} | {d(vals['radial'], vals['vd-Poisson'])} |")
    out.append("")
    for name, _ in TOPICS:
        if not col[name]:
            continue
        pick = min if a.metric in LOWER_IS_BETTER else max
        v, arm = pick(col[name])
        best_of[(c, name)] = (v, arm)
    out += ["Best arm per pattern: " + ";  ".join(
        f"**{n}** {best_of[(c, n)][1]} {best_of[(c, n)][0]:.3f}" for n, _ in TOPICS if (c, n) in best_of), ""]

# does MC-GS keep its lead over the strongest comparison arm on every pattern?
out += ["## MC-GS against the strongest comparison arm, per pattern", "",
        "| contrast | pattern | MC-GS | strongest comparison arm | margin |", "|---|---|---|---|---|"]
for c in ("T1", "FLAIR"):
    for name, _ in TOPICS:
        mc = get(name, c, "gs_joint_p3", "MC-GS")
        if mc is None:
            continue
        others = [(float(r[a.metric]), disp) for key, disp in ARMS if key != "gs_joint_p3"
                  for r in [get(name, c, key, disp)] if r is not None]
        if not others:
            continue
        pick = min if a.metric in LOWER_IS_BETTER else max
        v, arm = pick(others)
        m = float(mc[a.metric]) - v
        out.append(f"| {c} | {name} | {float(mc[a.metric]):.3f} | {arm} {v:.3f} | {m:+.3f} |")
out.append("")

if a.gs_lead:
    out += ["## MC-GS minus GS-single (the coupling gain), per pattern", "",
            "| contrast | pattern | MC-GS | GS-single | gain |", "|---|---|---|---|---|"]
    for c in ("T1", "FLAIR"):
        for name, _ in TOPICS:
            j, s_ = get(name, c, "gs_joint_p3", "MC-GS"), get(name, c, "gs_single_p3", "GS-single")
            if j is None or s_ is None:
                continue
            out.append(f"| {c} | {name} | {float(j[a.metric]):.3f} | {float(s_[a.metric]):.3f} | "
                       f"{float(j[a.metric]) - float(s_[a.metric]):+.3f} |")
    out.append("")

text = "\n".join(out)
print(text)
if a.out:
    Path(a.out).write_text(text + "\n")
    print(f"wrote {a.out}")
