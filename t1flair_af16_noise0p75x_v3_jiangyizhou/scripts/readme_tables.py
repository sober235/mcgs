#!/usr/bin/env python3
"""readme_tables.py -- markdown tables for the topic README from logs/summary.json (+ the comparison topic).

Usage: readme_tables.py <topic_dir> <comparison_topic_dir>
In the noise study the comparison topic is T1FLAIR_AF16_v3_jiangyizhou: the same arms on the clean
measurements, so the difference column is the effect of 0.75 sigma0 of added noise. Prints, per
contrast, the arms sorted by head-only PSNR with the comparison topic's head-only PSNR of the same arm
alongside, then the joint-minus-single lines.
"""
import json, sys
from pathlib import Path

T, T20 = Path(sys.argv[1]), Path(sys.argv[2])
REF = "no noise"                                     # the comparison column's short name in the headers
LABEL = {"zf": "ZF", "cgsense": "CG-SENSE", "cssense": "CS-SENSE", "zsssdu": "ZS-SSDU",
         "dip": "3D-DIP", "inr": "3D-INR", "mcinr": "MC-INR (Niessen 2025)", "gs_single": "MC-GS single", "gs_joint": "MC-GS joint",
         "cssense_o200": "CS-SENSE (TV, converged)", "jcssense": "joint CS-SENSE (JTV)", "l1espirit": "L1-ESPIRiT", "spirit": "L1-SPIRiT"}
_x = T / "logs/extra_gs_arms.json"
if _x.exists():
    LABEL.update({a["key"]: a["label"] for a in json.load(open(_x))})
rows = json.load(open(T / "logs/summary.json"))
# L1-ESPIRiT was dropped from the comparison on 2026-09-15 (user). Its volumes, logs and
# logs/summary.json entry are untouched -- it is only no longer listed. Sections written
# before that date still reference it; DROP is the single place that decides.
DROP = {"l1espirit"}
rows20 = json.load(open(T20 / "logs/summary.json")) if (T20 / "logs/summary.json").exists() else []


def find(rs, c, m):
    return next((r for r in rs if r["contrast"] == c and r["method"] in (m, LABEL.get(m, m))), None)


for c in ("T1", "FLAIR"):
    sub = sorted([r for r in rows if r["contrast"] == c and r["method"] not in DROP],
                 key=lambda r: -r["head_psnr"])
    best_h = max(r["head_psnr"] for r in sub); best_hs = max(r["head_ssim"] for r in sub)
    print(f"\n**{c}** (sorted by head-only PSNR; {REF} column = the same arm in `{T20.name}`)\n")
    print(f"| method | PSNR | SSIM | head PSNR | head SSIM | head HFEN | sharp | head PSNR @{REF} | this − {REF} |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in sub:
        m = r["method"]; a = find(rows20, c, m)
        hp = f"**{r['head_psnr']:.3f}**" if r["head_psnr"] == best_h else f"{r['head_psnr']:.3f}"
        hs = f"**{r['head_ssim']:.4f}**" if r["head_ssim"] == best_hs else f"{r['head_ssim']:.4f}"
        a20 = f"{a['head_psnr']:.3f}" if a else "--"; d20 = f"{r['head_psnr']-a['head_psnr']:+.2f}" if a else "--"
        lab = LABEL.get(m, m) + (f" (λ {r['lam']:g})" if r.get("lam") else "")
        print(f"| {lab} | {r['psnr']:.3f} | {r['ssim']:.4f} | {hp} | {hs} | {r['head_hfen']:.4f} | {r['sharp']:.3f} | {a20} | {d20} |")
print("\n**MC-GS joint minus MC-GS single**\n")
print(f"| channel | whole PSNR | head PSNR | head SSIM | ({REF}: whole / head) |")
print("|---|---|---|---|---|")
for c in ("T1", "FLAIR"):
    j, s = find(rows, c, "gs_joint_p3"), find(rows, c, "gs_single_p3")
    j20, s20 = find(rows20, c, "gs_joint_p3"), find(rows20, c, "gs_single_p3")
    if j and s:
        old = f"{j20['psnr']-s20['psnr']:+.3f} / {j20['head_psnr']-s20['head_psnr']:+.3f}" if (j20 and s20) else "--"
        print(f"| {c} | {j['psnr']-s['psnr']:+.3f} | {j['head_psnr']-s['head_psnr']:+.3f} | {j['head_ssim']-s['head_ssim']:+.4f} | {old} |")
