#!/usr/bin/env python3
"""p2_read.py -- rule-1 / rule-2 read of autoresearch rows against a parent row and a seed-floor row.

Every quantity comes from the rows' own score json (as autoresearch_jyz_af12_view.py does), margins against the
current bar (best non-MC-GS value per cell, best_competitors()). Six head cells per row: PSNR, SSIM, HFEN margins
per contrast (HFEN margin = bar - ours, higher is better). The floor of a cell is |parent - floor row|.

Verdict (autoresearch_jyz_af12_p2.md, rule 1):
  keep     min_margin up by more than its floor, no other cell down by more than its floor, sharpness of neither
           contrast more than 0.01 below the parent's
  reshape  min_margin up by more than its floor, but some other cell down beyond its floor (or sharpness gate)
  null     |d min_margin| within the floor
  loses    min_margin down by more than its floor

Usage: p2_read.py [--parent p2_P10_i1500] [--floor p2_P10_i1500_s43] [--prefix p2_] [--iters 1500] [--sharp_tol 0.01]
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from autoresearch_jyz_af12_row import best_competitors   # noqa: E402

CELLS = ("dT1", "dFL", "dsT1", "dsFL", "dhT1", "dhFL")


def cells(exp, B):
    sj = ROOT / "results_arc" / f"{exp}_score.json"
    if not sj.exists():
        return None
    d = json.loads(sj.read_text())
    ch = {}
    for k in ("plain", "contrast"):
        ch.setdefault(d.get(f"{k}_name", {"plain": "T1", "contrast": "FLAIR"}[k]), d[k])
    p, c = ch.get("T1"), ch.get("FLAIR")
    if p is None or c is None:
        return None
    r = {"iters": int(d["iters"]),
         "dT1": p["head_psnr"] - B["T1"]["head_psnr"], "dFL": c["head_psnr"] - B["FLAIR"]["head_psnr"],
         "dsT1": p["head_ssim"] - B["T1"]["head_ssim"], "dsFL": c["head_ssim"] - B["FLAIR"]["head_ssim"],
         "dhT1": B["T1"]["head_hfen"] - p["head_hfen"], "dhFL": B["FLAIR"]["head_hfen"] - c["head_hfen"],
         "shT1": p["sharp"], "shFL": c["sharp"], "T1": p["head_psnr"], "FL": c["head_psnr"]}
    r["min"] = min(r["dT1"], r["dFL"])
    r["bind"] = "dT1" if r["dT1"] <= r["dFL"] else "dFL"
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", default="p2_P10_i1500")
    ap.add_argument("--floor", default="p2_P10_i1500_s43")
    ap.add_argument("--prefix", default="p2_")
    ap.add_argument("--iters", type=int, default=None, help="only rows at this budget (default: the parent's)")
    ap.add_argument("--sharp_tol", type=float, default=0.01)
    a = ap.parse_args()
    B = best_competitors()
    P, F = cells(a.parent, B), cells(a.floor, B)
    if P is None:
        sys.exit(f"parent {a.parent}: no score json")
    iters = a.iters or P["iters"]
    fl = {k: (abs(P[k] - F[k]) if F else float("nan")) for k in CELLS + ("min",)}
    print(f"parent {a.parent} @{P['iters']}: T1 {P['T1']:.3f} FL {P['FL']:.3f}  min {P['min']:+.3f} ({P['bind']})  "
          + "  ".join(f"{k} {P[k]:+.4f}" for k in CELLS) + f"  sharp {P['shT1']:.3f}/{P['shFL']:.3f}")
    if F:
        print(f"floor  {a.floor}: " + "  ".join(f"{k} {fl[k]:.4f}" for k in ("min",) + CELLS))
    else:
        print(f"floor  {a.floor}: (no score json yet) -- verdicts below use floor 0")
        fl = {k: 0.0 for k in fl}
    hdr = f"{'exp':28s} {'it':>5s} {'T1':>7s} {'FL':>7s} {'d_min':>7s} " + " ".join(f"{'d'+k[1:]:>8s}" for k in CELLS) + f" {'shT1':>6s} {'shFL':>6s}  verdict"
    print(hdr)
    rows = [l.split("\t")[0] for l in (ROOT / "autoresearch_jyz_af12.tsv").read_text().splitlines()[1:]]
    seen = set()
    for exp in rows:
        if not exp.startswith(a.prefix) or exp in (a.parent, a.floor) or exp in seen:
            continue
        seen.add(exp)
        R = cells(exp, B)
        if R is None or R["iters"] != iters:
            continue
        d = {k: R[k] - P[k] for k in CELLS + ("min",)}
        others = [k for k in CELLS if k != P["bind"]]
        down = [k for k in others if d[k] < -fl[k]]
        sharp_ok = R["shT1"] >= P["shT1"] - a.sharp_tol and R["shFL"] >= P["shFL"] - a.sharp_tol
        if d["min"] > fl["min"]:
            v = "KEEP" if (not down and sharp_ok) else "reshape(" + ",".join(down + ([] if sharp_ok else ["sharp"])) + ")"
        elif d["min"] < -fl["min"]:
            v = "loses"
        else:
            v = "null"
        print(f"{exp:28s} {R['iters']:5d} {R['T1']:7.3f} {R['FL']:7.3f} {d['min']:+7.3f} "
              + " ".join(f"{d[k]:+8.4f}" for k in CELLS) + f" {R['shT1']:6.3f} {R['shFL']:6.3f}  {v}")


if __name__ == "__main__":
    main()
