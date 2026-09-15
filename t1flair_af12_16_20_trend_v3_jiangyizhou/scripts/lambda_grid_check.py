#!/usr/bin/env python3
"""lambda_grid_check.py -- for the three swept classical families, the head-PSNR curve over lambda in each topic, the lambda
the topic table keeps, and whether that optimum is interior to its grid (an edge optimum can understate the arm)."""
import json, glob, re, os
ROOT = "/data0/congcong/code/GS/Results_20260908"
fam = {"CS-SENSE converged TV": ("classical_{c}_o200*.json", "cssense_o200"),
       "Joint SENSE": ("jcssense_o200*.json", "jcssense"),
       "L1-SPIRiT": ("spirit_{c}_ks*_lam*.json", "spirit")}        # replaces L1-ESPIRiT (dropped 2026-09-15);
                                                                    # the best over the two kernel sizes at each lambda
for af in ("12", "16", "20"):
    t = f"{ROOT}/T1FLAIR_AF{af}_v3_jiangyizhou/logs"
    summ = {(r["method"], r["contrast"]): r for r in json.load(open(f"{t}/summary.json"))}
    for name, (pat, meth) in fam.items():
        for c in ("T1", "FLAIR"):
            pts = {}
            for f in glob.glob(os.path.join(t, pat.format(c=c))):
                if "af105" in f or re.search(r"_d[bcd]_", f):      # other masks (AF 10.5 FLAIR study, AF 20 mask draws)
                    continue
                d = json.load(open(f)); rows = d.get("rows", []) if isinstance(d, dict) else d
                for r in rows:
                    if r.get("contrast", d.get("contrast") if isinstance(d, dict) else c) != c or "head_psnr" not in r:
                        continue
                    lam = r.get("lam")
                    if lam is None:
                        m = re.search(r"lam([0-9.]+)", r.get("method", "")) or re.search(r"_lam([0-9.]+)\.json$", f)
                        lam = float(m.group(1)) if m else None
                    if lam is not None:
                        pts[float(lam)] = max(pts.get(float(lam), -1e9), r["head_psnr"])
            ls = sorted(pts); best = max(ls, key=lambda L: pts[L]); edge = best in (ls[0], ls[-1])
            rec = summ.get((meth, c), {})
            print(f"AF{af} {c:5s} {name:22s} best {best:g} {'EDGE' if edge else 'interior'} | table: lam {rec.get('lam')} "
                  f"{rec.get('head_psnr', float('nan')):.3f} | " + "  ".join(f"{L:g}:{pts[L]:.3f}" for L in ls))
