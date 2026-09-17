#!/usr/bin/env python3
"""compare_noise_levels.py -- the noise study's tables across every level run so far, one gauge.

    no noise      T1FLAIR_AF16_v3_jiangyizhou             recorded, not re-run
    k sigma0      T1FLAIR_AF16_noise<k>x_v3_jiangyizhou   k = 0.25, 0.5, 0.75, 1, 2 (0p25, 0p5, 0p75, 1, 2); every noisy
                                                          level is the same draw scaled (spec addenda 2026-09-16/17)
    levels whose topic has no logs/summary.json yet are left out.

Same subject, reference k-space, coil maps, house vd-Poisson mask, ground truth, scorer and frozen AF16 phase-3 MC-GS
recipe at every level; the classical lambdas are re-selected per level. Sections: every arm per level with the change
from no noise; MC-GS against the strongest comparison arm (GS-single counts) per level; the rank of MC-GS on the three
head metrics per level; the lambda each swept family chose per level; the GS render-only diagnostic per noisy level.
usage: compare_noise_levels.py [--metric head_psnr] [--out FILE]      (read-only on every topic)
"""
import argparse
import json
from pathlib import Path

RES = Path("/data0/congcong/code/GS/Results_20260908")
LEVELS = [(name, T) for name, T in (("no noise", RES / "T1FLAIR_AF16_v3_jiangyizhou"),
                                    ("0.25 sigma0", RES / "T1FLAIR_AF16_noise0p25x_v3_jiangyizhou"),
                                    ("0.5 sigma0", RES / "T1FLAIR_AF16_noise0p5x_v3_jiangyizhou"),
                                    ("0.75 sigma0", RES / "T1FLAIR_AF16_noise0p75x_v3_jiangyizhou"),
                                    ("1 sigma0", RES / "T1FLAIR_AF16_noise1x_v3_jiangyizhou"),
                                    ("2 sigma0", RES / "T1FLAIR_AF16_noise2x_v3_jiangyizhou"))
          if (T / "logs/summary.json").exists()]
# summary.json is keyed by arm KEY; the house topic also carries older gs_joint / gs_joint_ar rows (see compare_masks.py)
ARMS = [("gs_joint_p3", "MC-GS"), ("gs_single_p3", "GS-single"), ("zsssdu", "ZS-SSDU"),
        ("cssense_o200", "CS-SENSE (TV, converged)"), ("jcssense", "Joint SENSE (JTV)"),
        ("cssense", "CS-SENSE (35 outer)"), ("spirit", "L1-SPIRiT"), ("cgsense", "CG-SENSE"),
        ("mcinr", "MC-INR"), ("dip", "3D-DIP"), ("inr", "3D-INR"), ("zf", "ZF")]
SWEPT = ("cssense_o200", "jcssense", "spirit")
LOWER = {"head_hfen", "nmse", "hfen"}
HEAD = ("head_psnr", "head_ssim", "head_hfen")

ap = argparse.ArgumentParser()
ap.add_argument("--metric", default="head_psnr")
ap.add_argument("--out", default=None)
a = ap.parse_args()
M = a.metric
better = min if M in LOWER else max
digits = 4 if ("ssim" in M or "hfen" in M) else 3
fmt = lambda v: "--" if v is None else f"{v:.{digits}f}"
sfmt = lambda v: "--" if v is None else f"{v:+.{digits}f}"

rows = {}
for name, T in LEVELS:
    p = T / "logs/summary.json"
    rows[name] = {(r["contrast"], r["method"]): r for r in json.load(open(p))} if p.exists() else {}


def val(name, c, k, m=M):
    r = rows[name].get((c, k))
    return None if r is None or r.get(m) is None else float(r[m])


names = [n for n, _ in LEVELS]
out = [f"# Noise study: every arm at every level of added measurement noise run so far ({M})", "",
       "Levels: " + ", ".join(names) + ". Same subject, reference k-space, coil maps, house vd-Poisson mask, ground truth,",
       "scorer and frozen AF16 phase-3 MC-GS recipe at every level; the classical lambdas are re-selected per level; every",
       "noisy level is the same realization, scaled.", ""]
for c in ("T1", "FLAIR"):
    noisy = names[1:]
    out += [f"**{c}**", "", "| arm | " + " | ".join(names) + " | " + " | ".join(f"{n} − none" for n in noisy) + " |",
            "|---|" + "---|" * (len(names) + len(noisy))]
    for key, lab in ARMS:
        v = [val(n, c, key) for n in names]
        d = [None if v[0] is None or x is None else x - v[0] for x in v[1:]]
        out.append(f"| {lab} | " + " | ".join(fmt(x) for x in v) + " | " + " | ".join(sfmt(x) for x in d) + " |")
    out.append("")

out += ["## MC-GS against the strongest comparison arm (GS-single counts)", "",
        "| contrast | level | MC-GS | strongest comparison arm | margin |", "|---|---|---|---|---|"]
for c in ("T1", "FLAIR"):
    for n in names:
        mc = val(n, c, "gs_joint_p3")
        others = [(val(n, c, k), lab) for k, lab in ARMS if k != "gs_joint_p3" and val(n, c, k) is not None]
        if mc is None or not others:
            continue
        v, lab = better(others)
        margin = mc - v if M not in LOWER else v - mc
        out.append(f"| {c} | {n} | {fmt(mc)} | {lab} {fmt(v)} | {margin:+.4f} |")
out.append("")

out += ["## Rank of MC-GS on the three head metrics (1 = best among the listed arms)", "",
        "| contrast | level | head PSNR | head SSIM | head HFEN |", "|---|---|---|---|---|"]
for c in ("T1", "FLAIR"):
    for n in names:
        cells = []
        for m in HEAD:
            vs = sorted(((val(n, c, k, m), k) for k, _ in ARMS if val(n, c, k, m) is not None),
                        key=lambda t: t[0], reverse=m not in LOWER)
            cells.append(next((f"{i + 1}/{len(vs)}" for i, (_, k) in enumerate(vs) if k == "gs_joint_p3"), "--"))
        out.append(f"| {c} | {n} | " + " | ".join(cells) + " |")
out.append("")

out += ["## Lambda chosen by each swept family, per level", "",
        "| contrast | family | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
for c in ("T1", "FLAIR"):
    for key in SWEPT:
        cells = []
        for n in names:
            r = rows[n].get((c, key))
            if not r or r.get("lam") is None:
                cells.append("--")
                continue
            lams = sorted(set(float(x) for x in r.get("lams") or []))
            edge = "" if not lams else (" (upper edge)" if float(r["lam"]) >= lams[-1] else
                                        " (lower edge)" if float(r["lam"]) <= lams[0] else "")
            ks = f", kernel {r['ksize']}" if r.get("ksize") else ""
            cells.append(f"{float(r['lam']):g}{ks}{edge}")
        out.append(f"| {c} | {key} | " + " | ".join(cells) + " |")
out.append("")

out += ["## Diagnostic: GS arms with and without hard data consistency (both through each topic's Pair)", "",
        "| contrast | level | arm | with DC | render only | render − with DC |", "|---|---|---|---|---|---|"]
for n, T in LEVELS[1:]:
    for c in ("T1", "FLAIR"):
        p = T / f"logs/gs_render_{c}.json"
        if not p.exists():
            continue
        by = {r["method"]: r for r in json.load(open(p))["rows"]}
        for key, lab in (("gs_joint_p3", "MC-GS"), ("gs_single_p3", "GS-single")):
            dc, rd = by.get(f"{key}_dc"), by.get(f"{key}_render")
            if dc and rd:
                out.append(f"| {c} | {n} | {lab} | {fmt(float(dc[M]))} | {fmt(float(rd[M]))} | "
                           f"{float(rd[M]) - float(dc[M]):+.4f} |")
out.append("")

text = "\n".join(out)
print(text)
if a.out:
    Path(a.out).write_text(text + "\n")
    print(f"wrote {a.out}")
