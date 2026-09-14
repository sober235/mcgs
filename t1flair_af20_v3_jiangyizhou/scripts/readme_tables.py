#!/usr/bin/env python3
"""readme_tables.py -- markdown tables for the topic README from logs/summary.json (+ the AF16 twin).

Usage: readme_tables.py <topic_dir> <af16_topic_dir>
Prints, per contrast, the eight arms sorted by head-only PSNR with the AF16 head-only PSNR of the
same arm alongside, then the joint-minus-single lines. Wall time is what the arm's own log reported.
"""
import json, sys
from pathlib import Path

T, T16 = Path(sys.argv[1]), Path(sys.argv[2])
LABEL = {"zf": "ZF", "cgsense": "CG-SENSE", "cssense": "CS-SENSE", "zsssdu": "ZS-SSDU",
         "dip": "3D-DIP", "inr": "3D-INR", "mcinr": "MC-INR (Niessen 2025)", "gs_single": "MC-GS single", "gs_joint": "MC-GS joint",
         "cssense_o200": "CS-SENSE (TV, converged)", "jcssense": "joint CS-SENSE (JTV)", "l1espirit": "L1-ESPIRiT"}
_x = T / "logs/extra_gs_arms.json"
if _x.exists():
    LABEL.update({a["key"]: a["label"] for a in json.load(open(_x))})
rows = json.load(open(T / "logs/summary.json"))
rows16 = json.load(open(T16 / "logs/summary.json")) if (T16 / "logs/summary.json").exists() else []


def find(rs, c, m):
    return next((r for r in rs if r["contrast"] == c and r["method"] in (m, LABEL.get(m, m))), None)


for c in ("T1", "FLAIR"):
    sub = sorted([r for r in rows if r["contrast"] == c], key=lambda r: -r["head_psnr"])
    best_h = max(r["head_psnr"] for r in sub); best_hs = max(r["head_ssim"] for r in sub)
    print(f"\n**{c}** (sorted by head-only PSNR; AF16 column = the same arm in `{T16.name}`)\n")
    print("| method | PSNR | SSIM | head PSNR | head SSIM | head HFEN | sharp | head PSNR @AF16 | AF20 − AF16 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in sub:
        m = r["method"]; a = find(rows16, c, m)
        hp = f"**{r['head_psnr']:.3f}**" if r["head_psnr"] == best_h else f"{r['head_psnr']:.3f}"
        hs = f"**{r['head_ssim']:.4f}**" if r["head_ssim"] == best_hs else f"{r['head_ssim']:.4f}"
        a16 = f"{a['head_psnr']:.3f}" if a else "--"; d16 = f"{r['head_psnr']-a['head_psnr']:+.2f}" if a else "--"
        lab = LABEL.get(m, m) + (f" (λ {r['lam']:g})" if r.get("lam") else "")
        print(f"| {lab} | {r['psnr']:.3f} | {r['ssim']:.4f} | {hp} | {hs} | {r['head_hfen']:.4f} | {r['sharp']:.3f} | {a16} | {d16} |")
print("\n**MC-GS joint minus MC-GS single**\n")
print("| channel | whole PSNR | head PSNR | head SSIM | (AF16: whole / head) |")
print("|---|---|---|---|---|")
for c in ("T1", "FLAIR"):
    j, s = find(rows, c, "gs_joint"), find(rows, c, "gs_single")
    j16, s16 = find(rows16, c, "gs_joint"), find(rows16, c, "gs_single")
    if j and s:
        old = f"{j16['psnr']-s16['psnr']:+.3f} / {j16['head_psnr']-s16['head_psnr']:+.3f}" if (j16 and s16) else "--"
        print(f"| {c} | {j['psnr']-s['psnr']:+.3f} | {j['head_psnr']-s['head_psnr']:+.3f} | {j['head_ssim']-s['head_ssim']:+.4f} | {old} |")
