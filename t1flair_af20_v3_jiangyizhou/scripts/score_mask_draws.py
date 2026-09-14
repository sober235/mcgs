#!/usr/bin/env python3
"""score_mask_draws.py -- the MC-GS margin over L1-ESPIRiT on each AF20 mask draw, and its spread.

Draw a is the registered mask (MC-GS af20_p3_joint_full4k, L1 from the main sweep); draws b, c, d come from
run_mask_draws_af20.sh. For every draw L1-ESPIRiT is read at the SAME three lambdas (T1 0.0007 / 0.001 /
0.0015, FLAIR 0.002 / 0.003 / 0.005) and its best is kept; a best on an edge of that grid is flagged.
Prints per-draw rows and mean / sample SD / range per contrast; writes logs/mask_draws_summary.json.
A draw whose runs have not finished is listed as pending and left out of the statistics.
"""
import json, statistics
from pathlib import Path

J = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF20_v3_jiangyizhou")
ARC = Path("/data0/congcong/code/GS/mcgs_af16_wt/results_arc")
GRID = {"T1": [0.0007, 0.001, 0.0015], "FLAIR": [0.002, 0.003, 0.005]}
CH = {"T1": "plain", "FLAIR": "contrast"}
AF16_MARGIN = {"T1": 0.497, "FLAIR": 0.425}
prep = json.load(open(J / "logs/prep_04b_export_draws.json"))


def mcgs(d):
    p = ARC / ("af20_p3_joint_full4k_score.json" if d == "a" else f"af20_p3_joint_d{d}_full4k_score.json")
    return json.load(open(p)) if p.exists() else None


def l1(d, c):
    best, n = None, 0
    for lam in GRID[c]:
        tag = f"lam{lam:g}" if d == "a" else f"d{d}_lam{lam:g}"
        p = J / f"logs/l1espirit_{c}_{tag}.json"
        if not p.exists():
            return None, n
        r = json.load(open(p))["rows"][0]
        n += 1
        if best is None or r["head_psnr"] > best["head_psnr"]:
            best = dict(r, lam=lam)
    return best, n


out, rows = {}, []
for d in "abcd":
    m = mcgs(d)
    for c in ("T1", "FLAIR"):
        b, n = l1(d, c)
        if m is None or b is None:
            rows.append((d, c, None)); continue
        g = m[CH[c]]
        edge = b["lam"] in (GRID[c][0], GRID[c][-1])
        r = dict(draw=d, contrast=c, samples=prep[d]["samples"], af=prep[d][f"{c}_AF_own"],
                 mcgs=g["head_psnr"], mcgs_ssim=g["head_ssim"], mcgs_sharp=g["sharp"],
                 l1=b["head_psnr"], l1_lam=b["lam"], l1_edge=edge, l1_sharp=b["sharp"],
                 margin=g["head_psnr"] - b["head_psnr"], sharp_gate=g["sharp"] >= b["sharp"])
        out.setdefault(c, []).append(r); rows.append((d, c, r))

print(f"{'draw':4s} {'c':5s} {'pts':>5s} {'AF':>6s} {'MC-GS':>7s} {'L1':>7s} {'lam':>7s} {'margin':>7s} {'sharp MC/L1':>12s}")
for d, c, r in rows:
    if r is None:
        print(f"{d:4s} {c:5s}  pending"); continue
    print(f"{d:4s} {c:5s} {r['samples']:5d} {r['af']:6.2f} {r['mcgs']:7.3f} {r['l1']:7.3f} {r['l1_lam']:7g}{'*' if r['l1_edge'] else ' '}"
          f"{r['margin']:+7.3f}  {r['mcgs_sharp']:.3f}/{r['l1_sharp']:.3f}{'' if r['sharp_gate'] else ' <'}")
print("(* = best L1 lambda on an edge of its three-point grid;  < = MC-GS sharpness below L1-ESPIRiT's)")
summ = {}
for c, rs in out.items():
    ms = [r["margin"] for r in rs]
    s = dict(n=len(ms), mean=statistics.mean(ms), sd=statistics.stdev(ms) if len(ms) > 1 else None,
             min=min(ms), max=max(ms), mcgs_sd=statistics.stdev([r["mcgs"] for r in rs]) if len(rs) > 1 else None,
             l1_sd=statistics.stdev([r["l1"] for r in rs]) if len(rs) > 1 else None,
             af16_margin=AF16_MARGIN[c], sharp_gate_passed=sum(r["sharp_gate"] for r in rs))
    summ[c] = s
    sd = f"{s['sd']:.3f}" if s["sd"] is not None else "--"
    print(f"{c:5s} margin over {s['n']} draws: mean {s['mean']:+.3f}  SD {sd}  range [{s['min']:+.3f}, {s['max']:+.3f}]"
          f"   (AF16, one draw: {AF16_MARGIN[c]:+.3f})   sharpness gate passed on {s['sharp_gate_passed']}/{s['n']}")
json.dump(dict(rows=[r for _, _, r in rows if r], summary=summ), open(J / "logs/mask_draws_summary.json", "w"), indent=2)
