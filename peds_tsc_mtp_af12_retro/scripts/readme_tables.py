#!/usr/bin/env python3
"""readme_tables.py -- the two markdown tables of this topic's README.

Primary  (logs/rescore.json):  every arm on the frozen gauge, averaged over the head inside the
                               vendor's central 80 partitions -- the numbers the README quotes.
Secondary (logs/summary.json): each arm's own full-volume numbers, the cross-check.

Usage: readme_tables.py <topic_dir>
"""
import json
import sys
from pathlib import Path

T = Path(sys.argv[1])
CONTRASTS = ("T1w", "T2s")
WEIGHT = {"T1w": "T1-weighted (FA 16, TE 3.25 ms)", "T2s": "T2*-weighted (FA 4, TE 27.41 ms)"}
LABEL = {"cgsense": "CG-SENSE", "cssense_o200": "CS-SENSE (TV, converged)", "jcssense": "Joint SENSE (JTV)",
         "l1espirit": "L1-ESPIRiT", "zsssdu": "ZS-SSDU", "dip": "3D-DIP", "inr": "3D-INR",
         "gs_single_ar": "GS-single", "gs_joint_ar": "MC-GS"}
ORDER = list(LABEL)
AF = json.load(open(T / "logs/prep_mask.json"))
prim = json.load(open(T / "logs/rescore.json"))
rows_p = prim["rows"]
rows_s = {f"{r['method']}|{r['contrast']}": r for r in json.load(open(T / "logs/summary.json"))}


def fmt(v, p, best=False):
    if v is None:
        return "--"
    t = f"{v:.{p}f}"
    return f"**{t}**" if best else t


def table(c, rows, cols, title):
    sub = [(k, r) for k, r in rows.items() if r.get("contrast") == c and r.get("method") in ORDER]
    if not sub:
        return
    sub.sort(key=lambda kr: -kr[1]["head_psnr"])
    best = {}
    for name, _, hi, _ in cols:
        vals = [r.get(name) for _, r in sub if r.get(name) is not None]
        best[name] = (max(vals) if hi else min(vals)) if vals else None
    print(f"\n**{c}** — {WEIGHT[c]}. {title}\n")
    print("| method | " + " | ".join(h for _, h, _, _ in cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for k, r in sub:
        lab = LABEL.get(r["method"], r["method"])
        if r.get("lam"):
            lab += f" (λ {r['lam']:g})"
        cells = [fmt(r.get(n), p, r.get(n) == best[n]) for n, _, _, p in cols]
        print(f"| {lab} | " + " | ".join(cells) + " |")


PRIMARY = [("head_psnr", "head PSNR", True, 3), ("head_ssim", "head SSIM", True, 4),
           ("head_hfen", "head HFEN", False, 4), ("psnr", "PSNR", True, 3),
           ("nmse", "NMSE", False, 5), ("sharp", "sharp", True, 3)]
SECONDARY = [("head_psnr", "head PSNR", True, 3), ("head_ssim", "head SSIM", True, 4),
             ("head_hfen", "head HFEN", False, 4), ("psnr", "PSNR", True, 3), ("nmse", "NMSE", False, 5)]

print(f"### Primary table — {prim['region']}\n")
print(f"Retrospective AF {AF['AF_vs_full_grid']:.2f} of the {AF['plane_points']}-point phase-encode matrix "
      f"({AF['AF_vs_support']:.2f}x of the {AF['n_support']} acquired positions), one mask for both contrasts; "
      f"the partial-Fourier block and the elliptical corners ({AF['n_zero']} positions) are measured zeros for "
      f"every method alike. Best of each column in bold.")
for c in CONTRASTS:
    table(c, rows_p, PRIMARY, "head metrics inside the vendor's central 80 partitions.")

print("\n### Secondary table — full volume, each arm's own scorer\n")
print("The same gauge over all 104 partitions, taken from each arm's own JSON; the cross-check that the "
      "rescorer did not move anything (`scripts/rescore.py --full`).")
for c in CONTRASTS:
    table(c, rows_s, SECONDARY, "full volume.")

print("\n### MC-GS minus GS-single\n")
print("| channel | head PSNR | head SSIM | head HFEN | PSNR |")
print("|---|---|---|---|---|")
for c in CONTRASTS:
    j, s = rows_p.get(f"gs_joint_ar|{c}"), rows_p.get(f"gs_single_ar|{c}")
    if j and s:
        print(f"| {c} | {j['head_psnr']-s['head_psnr']:+.3f} | {j['head_ssim']-s['head_ssim']:+.4f} | "
              f"{j['head_hfen']-s['head_hfen']:+.4f} | {j['psnr']-s['psnr']:+.3f} |")
