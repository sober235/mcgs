#!/usr/bin/env python3
"""find_slice_nongs_gap.py -- transverse slices where MC-GS's gap to the best NON-GS column is largest on all three sampling
patterns (the gap a reader can see in a zoom: MC-GS against the classical and learned arms, not against its own single-contrast
sibling). Gauge = the stacked figure's head-masked slice PSNR. MC-GS must still be first over EVERY column, GS-single included.
Eligible slices: head area >= --min_area of the largest slice (find_slice_all_masks.py's rule). Writes logs/find_slice_nongs_gap.json.
"""
import argparse, json, re
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--min_area", type=float, default=0.5)
ap.add_argument("--gap", type=int, default=6, help="minimum spacing between reported slices")
ap.add_argument("--top", type=int, default=8)
ap.add_argument("--mcgs_key", default="gs_joint_p3")
ap.add_argument("--single_key", default="gs_single_p3")
args = ap.parse_args()
RES = Path("/data0/congcong/code/GS/Results_20260908")
TOPICS = [("vd-Poisson", RES / "T1FLAIR_AF16_v3_jiangyizhou"), ("Gaussian", RES / "T1FLAIR_AF16_gauss_v3_jiangyizhou"),
          ("Pseudo-radial", RES / "T1FLAIR_AF16_radial_v3_jiangyizhou")]
OUT = RES / "T1FLAIR_AF16_gauss_v3_jiangyizhou" / "logs" / "find_slice_nongs_gap.json"
NONGS = ["zf", "cgsense", "cssense_o200", "jcssense", "spirit", "zsssdu", "dip", "inr", "mcinr"]


def arms(R, c):
    rows = {r["method"]: r for r in json.load(open(R / "logs/summary.json")) if r["contrast"] == c}

    def lam(k, pre):
        v = rows[k]["lam"]
        return next(p for p in R.glob(f"recon/{pre}_lam*_{c}_complex.npy")
                    if abs(float(re.search(r"lam([0-9.]+)_", p.name).group(1)) - v) < 1e-12)

    r = rows["spirit"]
    return {"zf": R / f"recon/zf_{c}_complex.npy", "cgsense": R / f"recon/cgsense_{c}_complex.npy",
            "cssense_o200": lam("cssense_o200", "cssense_o200"), "jcssense": lam("jcssense", "jcssense_o200"),
            "spirit": R / f"recon/spirit_ks{r['ksize']}_lam{r['lam']:g}_{c}_complex.npy",
            "zsssdu": R / f"recon/zsssdu_{c}_complex.npy", "dip": R / f"recon/dip_{c}_complex.npy",
            "inr": R / f"recon/inr_{c}_complex.npy", "mcinr": R / f"recon/mcinr_{c}_complex.npy",
            args.single_key: R / f"recon/{args.single_key}_{c}_mag.npy",
            args.mcgs_key: R / f"recon/{args.mcgs_key}_{c}_mag.npy"}


res = {}
for c in ("T1", "FLAIR"):
    gt = np.abs(np.load(TOPICS[0][1] / f"recon/gt_{c}_complex.npy")).astype(np.float32)
    gmax = float(gt.max())
    head = gt > 0.05 * float(np.quantile(gt, 0.999))
    n = head.reshape(head.shape[0], -1).sum(1)
    elig = n >= args.min_area * n.max()
    for name, R in TOPICS:
        ps = {}
        for key, path in arms(R, c).items():
            v = np.abs(np.load(path, mmap_mode="r")).astype(np.float32)
            se = ((v - gt) ** 2 * head).reshape(gt.shape[0], -1).sum(1)
            ps[key] = 20 * np.log10(gmax) - 10 * np.log10(np.maximum(se / np.maximum(n, 1), 1e-30))
            del v
        res[(c, name)] = ps
        print("scored", c, name, flush=True)
S = gt.shape[0]
rows = []
for s in range(S):
    if not elig[s]:
        continue
    r = dict(s=s, rows=[])
    ok = True
    for c in ("T1", "FLAIR"):
        for name, _ in TOPICS:
            ps = res[(c, name)]
            mc = ps[args.mcgs_key][s]
            oth = {k: v[s] for k, v in ps.items() if k != args.mcgs_key}
            bo = max(oth, key=oth.get); bn = max(NONGS, key=lambda k: oth[k])
            ok &= mc > oth[bo]
            r["rows"].append(dict(contrast=c, pattern=name, mcgs=float(mc), best_all=bo, lead_all=float(mc - oth[bo]),
                                  best_nongs=bn, lead_nongs=float(mc - oth[bn])))
    r["first_everywhere"] = bool(ok)
    r["min_lead_all"] = min(x["lead_all"] for x in r["rows"])
    r["min_lead_nongs"] = min(x["lead_nongs"] for x in r["rows"])
    r["min_lead_nongs_T1"] = min(x["lead_nongs"] for x in r["rows"] if x["contrast"] == "T1")
    r["min_lead_nongs_FLAIR"] = min(x["lead_nongs"] for x in r["rows"] if x["contrast"] == "FLAIR")
    rows.append(r)


def report(title, sel, key):
    print(f"\n== {title}")
    kept = []
    for r in sorted(sel, key=lambda r: -r[key]):
        if all(abs(r["s"] - k["s"]) >= args.gap for k in kept):
            kept.append(r)
        if len(kept) >= args.top:
            break
    for r in kept:
        print(f"   slice {r['s']:3d}  nonGS min {r['min_lead_nongs']:+.3f}  (T1 {r['min_lead_nongs_T1']:+.3f}, "
              f"FLAIR {r['min_lead_nongs_FLAIR']:+.3f})  all min {r['min_lead_all']:+.3f}  | "
              + " ".join(f"{x['contrast'][0]}{x['pattern'][:4]} {x['lead_nongs']:+.2f}({x['best_nongs']})" for x in r["rows"]))
    return kept


first = [r for r in rows if r["first_everywhere"]]
print(f"eligible slices {len(rows)}, MC-GS first over every column in all six rows: {len(first)}")
report("两对比度合看，按对非 GS 的最小领先", first, "min_lead_nongs")
report("只看 T1", first, "min_lead_nongs_T1")
report("只看 FLAIR", first, "min_lead_nongs_FLAIR")
json.dump(dict(min_area=args.min_area, slices=rows), open(OUT, "w"), indent=1)
print("wrote", OUT)
