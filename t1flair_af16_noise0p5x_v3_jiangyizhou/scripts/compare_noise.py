#!/usr/bin/env python3
"""compare_noise.py -- the noise study's headline tables: every arm without and with 0.5 sigma0 of added noise.

    no noise   T1FLAIR_AF16_v3_jiangyizhou           the house topic, recorded, not re-run
    0.5 sigma0   T1FLAIR_AF16_noise0p5x_v3_jiangyizhou   this topic

Same subject, reference k-space, coil maps, mask, ground truth, scorer and frozen AF16 phase-3 MC-GS recipe; the
classical lambdas are re-selected on the noisy data. Only the measurements differ.
Sections: per-arm table; MC-GS against the strongest comparison arm (GS-single counts); the rank of MC-GS on the three
head metrics; the frozen-lambda view; where the re-selected lambdas sit on their grids; the render-only diagnostic.
usage: compare_noise.py [--metric head_psnr] [--out FILE]      (read-only on both topics)
"""
import argparse
import json
import re
from pathlib import Path

RES = Path("/data0/congcong/code/GS/Results_20260908")
CLEAN = RES / "T1FLAIR_AF16_v3_jiangyizhou"
NOISE = RES / "T1FLAIR_AF16_noise0p5x_v3_jiangyizhou"
# summary.json is keyed by arm KEY; the house topic also carries older gs_joint / gs_joint_ar rows (see compare_masks.py)
ARMS = [("gs_joint_p3", "MC-GS"), ("gs_single_p3", "GS-single"), ("zsssdu", "ZS-SSDU"),
        ("cssense_o200", "CS-SENSE (TV, converged)"), ("jcssense", "Joint SENSE (JTV)"),
        ("cssense", "CS-SENSE (35 outer)"), ("spirit", "L1-SPIRiT"), ("cgsense", "CG-SENSE"),
        ("mcinr", "MC-INR"), ("dip", "3D-DIP"), ("inr", "3D-INR"), ("zf", "ZF")]
LOWER = {"head_hfen", "nmse", "hfen"}
HEAD = ("head_psnr", "head_ssim", "head_hfen")
FAMILY = {"cssense_o200": "classical_{c}_o200_lam*.json", "jcssense": "jcssense_o200_lam*.json",
          "spirit": "spirit_{c}_ks{ks}_lam*.json"}

ap = argparse.ArgumentParser()
ap.add_argument("--metric", default="head_psnr")
ap.add_argument("--out", default=None)
a = ap.parse_args()
M = a.metric
better = min if M in LOWER else max
fmt = (lambda v: "--" if v is None else f"{v:.4f}") if "ssim" in M or "hfen" in M else \
      (lambda v: "--" if v is None else f"{v:.3f}")


def load(T):
    return {(r["contrast"], r["method"]): r for r in json.load(open(T / "logs/summary.json"))}


clean, noise = load(CLEAN), load(NOISE)


def val(rows, c, k, m=M):
    r = rows.get((c, k))
    return None if r is None or r.get(m) is None else float(r[m])


def lam_of(p):
    m = re.search(r"_lam([0-9.eE+-]+)\.json$", p.name)
    return None if m is None else float(m.group(1))


out = [f"# Noise study: every arm without and with 0.5 sigma0 of added measurement noise ({M})", "",
       "Same subject, reference k-space, coil maps, house vd-Poisson mask, ground truth, scorer and frozen AF16 phase-3",
       "MC-GS recipe. Only the measurements differ; the classical lambdas are re-selected on the noisy data.", ""]
for c in ("T1", "FLAIR"):
    out += [f"**{c}**", "", "| arm | no noise | 0.5 sigma0 | change |", "|---|---|---|---|"]
    for key, lab in ARMS:
        x, y = val(clean, c, key), val(noise, c, key)
        d = "--" if x is None or y is None else fmt(y - x) if y - x < 0 else "+" + fmt(y - x)
        lam = noise.get((c, key), {}).get("lam")
        out.append(f"| {lab}{'' if lam is None else f' (λ {float(lam):g})'} | {fmt(x)} | {fmt(y)} | {d} |")
    out.append("")

out += ["## MC-GS against the strongest comparison arm (GS-single counts)", "",
        "| contrast | data | MC-GS | strongest comparison arm | margin |", "|---|---|---|---|---|"]
for c in ("T1", "FLAIR"):
    for name, rows in (("no noise", clean), ("0.5 sigma0", noise)):
        mc = val(rows, c, "gs_joint_p3")
        others = [(val(rows, c, k), lab) for k, lab in ARMS if k != "gs_joint_p3" and val(rows, c, k) is not None]
        if mc is None or not others:
            continue
        v, lab = better(others)
        margin = mc - v if M not in LOWER else v - mc
        out.append(f"| {c} | {name} | {fmt(mc)} | {lab} {fmt(v)} | {margin:+.4f} |")
out.append("")

out += ["## Rank of MC-GS on the three head metrics (1 = best among the listed arms)", "",
        "| contrast | data | head PSNR | head SSIM | head HFEN |", "|---|---|---|---|---|"]
for c in ("T1", "FLAIR"):
    for name, rows in (("no noise", clean), ("0.5 sigma0", noise)):
        cells = []
        for m in HEAD:
            vs = sorted(((val(rows, c, k, m), k) for k, _ in ARMS if val(rows, c, k, m) is not None),
                        key=lambda t: t[0], reverse=m not in LOWER)
            cells.append(next((f"{i + 1}/{len(vs)}" for i, (_, k) in enumerate(vs) if k == "gs_joint_p3"), "--"))
        out.append(f"| {c} | {name} | " + " | ".join(cells) + " |")
out.append("")

out += ["## Frozen-lambda view: each swept family at its zero-noise lambda, on the noisy data", "",
        "| contrast | family | zero-noise λ | noisy data at that λ | noisy data at the re-selected λ |",
        "|---|---|---|---|---|"]
for c in ("T1", "FLAIR"):
    for key, pat in FAMILY.items():
        r0, r1 = clean.get((c, key)), noise.get((c, key))
        if not r0 or not r1 or r0.get("lam") is None:
            continue
        lam0 = float(r0["lam"])
        hit = None
        for p in sorted((NOISE / "logs").glob(pat.format(c=c, ks=r0.get("ksize", "*")))):
            if lam_of(p) is not None and abs(lam_of(p) - lam0) < 1e-12:
                hit = next((r for r in json.load(open(p)).get("rows", []) if r.get("contrast", c) == c), None)
        v0 = None if hit is None else float(hit[M])
        ks = f" (kernel {r0['ksize']})" if r0.get("ksize") else ""
        out.append(f"| {c} | {key} | {lam0:g}{ks} | {fmt(v0)} | {fmt(val(noise, c, key))} (λ {float(r1['lam']):g}) |")
out.append("")

out += ["## Where the re-selected lambdas sit on their grids (0.5 sigma0)", ""]
for c in ("T1", "FLAIR"):
    for key in FAMILY:
        r = noise.get((c, key))
        if not r or r.get("lam") is None or not r.get("lams"):
            continue
        lams = sorted(set(float(x) for x in r["lams"]))
        lam = float(r["lam"])
        where = "UPPER EDGE" if lam >= lams[-1] else "lower edge" if lam <= lams[0] else "interior"
        out.append(f"- {c} {key}: λ {lam:g} of [{', '.join(f'{x:g}' for x in lams)}] -> {where}")
out.append("")

rend = [NOISE / f"logs/gs_render_{c}.json" for c in ("T1", "FLAIR")]
if all(p.exists() for p in rend):
    out += ["## Diagnostic: GS arms with and without hard data consistency (both through this topic's Pair)", "",
            "| contrast | arm | with DC (as in the tables) | render only | render − with DC |", "|---|---|---|---|---|"]
    for p in rend:
        d = json.load(open(p))
        by = {r["method"]: r for r in d["rows"]}
        for key, lab in (("gs_joint_p3", "MC-GS"), ("gs_single_p3", "GS-single")):
            dc, rd = by.get(f"{key}_dc"), by.get(f"{key}_render")
            if dc and rd:
                out.append(f"| {d['contrast']} | {lab} | {fmt(float(dc[M]))} | {fmt(float(rd[M]))} | "
                           f"{float(rd[M]) - float(dc[M]):+.4f} |")
    out.append("")

text = "\n".join(out)
print(text)
if a.out:
    Path(a.out).write_text(text + "\n")
    print(f"wrote {a.out}")
