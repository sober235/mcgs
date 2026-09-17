#!/usr/bin/env python3
"""readme_tables.py -- markdown tables for README.md from logs/summary.json and logs/enhancement_fidelity.json.
Per scan: the eleven arms sorted by head-only PSNR (bold = best per column), the MC-GS - GS-single line (the headline
margin, user rule 2026-09-13), then the enhancement-fidelity table (post - pre in white-matter units, err vs truth)."""
import json, os
from pathlib import Path
from arms import LABEL

R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"))
rows = json.load(open(R / "logs/summary.json"))
for c in ("pre", "post"):
    sub = sorted([r for r in rows if r["contrast"] == c], key=lambda r: -r["head_psnr"])
    best = {k: (max if k != "head_nmse" and k != "head_hfen" else min)(r[k] for r in sub) for k in ("head_psnr", "head_ssim", "head_nmse", "head_hfen")}
    print(f"\n**{c}** (sorted by head-only PSNR; truth = CS-SENSE of the prospective pair)\n")
    print("| method | head PSNR | head SSIM | head NMSE | head HFEN | complex PSNR | whole PSNR | λ / kernel |")
    print("|---|---|---|---|---|---|---|---|")
    for r in sub:
        f = lambda k, p: (f"**{r[k]:.{p}f}**" if r[k] == best[k] else f"{r[k]:.{p}f}")
        lam = (f"{r['lam']:g}" if r.get("lam") is not None else "") + (f" / ks{r['ksize']}" if r.get("ksize") else "")
        print(f"| {LABEL.get(r['method'], r['method'])} | {f('head_psnr',3)} | {f('head_ssim',4)} | {f('head_nmse',5)} | {f('head_hfen',4)} | {r['psnr_cplx']:.3f} | {r['psnr']:.3f} | {lam} |")
print("\n**MC-GS minus GS-single**\n\n| scan | head PSNR | head SSIM | head NMSE | head HFEN |\n|---|---|---|---|---|")
for c in ("pre", "post"):
    j = next(r for r in rows if r["contrast"] == c and r["method"] == "gs_joint"); s = next(r for r in rows if r["contrast"] == c and r["method"] == "gs_single")
    print(f"| {c} | {j['head_psnr']-s['head_psnr']:+.3f} | {j['head_ssim']-s['head_ssim']:+.4f} | {j['head_nmse']-s['head_nmse']:+.5f} | {j['head_hfen']-s['head_hfen']:+.4f} |")
ef = R / "logs/enhancement_fidelity.json"
if ef.exists():
    e = json.load(open(ef))["rows"]
    print("\n**Enhancement (post − pre, white-matter units); err = arm − truth**\n\n| arm | vessels | plexus | cortex | WM | lesion core | rim 0–3 mm | background(post) |\n|---|---|---|---|---|---|---|---|")
    for r in e:
        cell = lambda k: (f"{r[k]:+.3f}" if r["method"] == "truth" else f"{r[k]:+.3f} ({r['err_' + k]:+.3f})")
        print(f"| {r['label']} | {cell('vessels')} | {cell('plexus')} | {cell('cortex')} | {cell('white_matter')} | {cell('lesion_core')} | {cell('rim_0_3mm')} | {r['background_post']:.3f} |")
