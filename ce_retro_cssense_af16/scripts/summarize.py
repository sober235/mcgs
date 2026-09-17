#!/usr/bin/env python3
"""summarize.py -- one table of every arm on both scans, from the per-arm score jsons (all written by t1t2_common.Pair.score).
Swept families keep the lambda (and kernel) with the highest head-only PSNR per scan, the jyz protocol.
  python scripts/summarize.py   -> logs/summary.json, logs/summary.tsv, a printout per scan"""
import json
import os
from pathlib import Path

from arms import LABEL, ORDER

R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"))
COLS = ["psnr", "ssim", "nmse", "hfen", "head_psnr", "head_ssim", "head_nmse", "head_hfen", "psnr_cplx", "sharp", "grain", "grain_ratio"]
STEMS = ["classical", "zsssdu", "dip", "inr", "mcinr", "gs_single", "gs_joint"]
FAM = [("cssense_o200", "logs/classical_{c}_o200_lam*.json"), ("jcssense", "logs/jcssense_o200_lam*.json"), ("spirit", "logs/spirit_{c}_ks*_lam*.json")]
SCANS = ("pre", "post")


def summarize():
    rows = []
    for c in SCANS:
        for stem in STEMS:
            p = R / f"logs/{stem}_{c}.json"
            if p.exists():
                for r in json.load(open(p)).get("rows", []):
                    if r.get("contrast", c) == c:
                        rows.append({"contrast": c, "method": r["method"], **{k: r.get(k) for k in COLS}, "seconds": r.get("seconds")})
        for key, pat in FAM:
            cands = []
            for f in sorted(R.glob(pat.format(c=c))):
                for r in json.load(open(f)).get("rows", []):
                    if r.get("contrast", c) == c and r.get("head_psnr") is not None:
                        cands.append(r)
            if cands:
                b = max(cands, key=lambda r: r["head_psnr"])
                rows.append({"contrast": c, "method": key, **{k: b.get(k) for k in COLS}, "seconds": b.get("seconds"),
                             "lam": b.get("lam"), "n_lambda": len(cands), **({"ksize": b["ksize"]} if b.get("ksize") else {}),
                             "lams": sorted({float(r["lam"]) for r in cands if r.get("lam") is not None})})
    rows.sort(key=lambda r: (SCANS.index(r["contrast"]), ORDER.index(r["method"]) if r["method"] in ORDER else 99))
    hdr = ["contrast", "method"] + COLS
    lines = ["\t".join(hdr)]
    for r in rows:
        lines.append("\t".join([r["contrast"], LABEL.get(r["method"], r["method"])] + ["" if r.get(k) is None else f"{r[k]:.5f}" for k in COLS]))
    (R / "logs/summary.tsv").write_text("\n".join(lines) + "\n")
    json.dump(rows, open(R / "logs/summary.json", "w"), indent=2)
    return rows


def main():
    rows = summarize()
    for c in SCANS:
        print(f"\n=== {c}  (CS-SENSE truth, Poisson-disc AF16)")
        print(f"{'method':12s} {'headPSNR':>9s} {'headSSIM':>9s} {'headNMSE':>9s} {'headHFEN':>9s} {'PSNRcplx':>9s} {'PSNR':>8s} {'lam':>8s}")
        for r in [r for r in rows if r["contrast"] == c]:
            f = lambda k, w, p: (f"{r[k]:{w}.{p}f}" if isinstance(r.get(k), float) else " " * w)
            print(f"{LABEL.get(r['method'], r['method']):12s} {f('head_psnr',9,3)} {f('head_ssim',9,4)} {f('head_nmse',9,5)} {f('head_hfen',9,4)} {f('psnr_cplx',9,3)} {f('psnr',8,3)} {f('lam',8,5)}")
    print(f"\nwrote {R / 'logs/summary.json'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
