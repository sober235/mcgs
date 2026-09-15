#!/usr/bin/env python3
"""score_mask_draws_af24.py -- the MC-GS margin over L1-ESPIRiT on each AF24 mask draw, its spread, and the AF20 -> AF24 shift.

Draw a is the registered mask (MC-GS af24_p3_joint_full4k, L1-ESPIRiT from the main sweep); draws b, c, d come from
run_all_af24.py --study draws. For every draw L1-ESPIRiT is read at the SAME three lambdas (T1 0.0007 / 0.001 / 0.0015,
FLAIR 0.0014 / 0.002 / 0.003) and its best is kept; a best on an edge of that grid is flagged. Every draw has exactly
1877 points (T1 24.00x); the FLAIR AF column is per draw (user's definition). The AF20 four-draw summary is read from
that topic's logs/mask_draws_summary.json. Prints per-draw rows, mean / sample SD / range per contrast and the shift of
the mean margin from AF20 in units of the larger of the two SDs; writes logs/mask_draws_af24_summary.json. A draw whose
runs have not finished is listed as pending and left out of the statistics (a port of the AF20 topic's score_mask_draws.py).
"""
import json, statistics
from pathlib import Path

J = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF24_v3_jiangyizhou")
A20 = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF20_v3_jiangyizhou")
ARC = Path("/data0/congcong/code/GS/mcgs_af16_wt/results_arc")
GRID = {"T1": ["0.0007", "0.001", "0.0015"], "FLAIR": ["0.0014", "0.002", "0.003"]}
CH = {"T1": "plain", "FLAIR": "contrast"}
prep = json.load(open(J / "logs/prep_03b_pick_export_draws_af24.json"))["draws"]
af20 = json.load(open(A20 / "logs/mask_draws_summary.json"))["summary"]


def mcgs(d):
    p = ARC / ("af24_p3_joint_full4k_score.json" if d == "a" else f"af24_p3_joint_d{d}_full4k_score.json")
    return json.load(open(p)) if p.exists() else None


def l1(d, c):
    best = None
    for lam in GRID[c]:
        tag = f"lam{lam}" if d == "a" else f"d{d}_lam{lam}"
        p = J / f"logs/l1espirit_{c}_{tag}.json"
        if not p.exists():
            return None
        r = json.load(open(p))["rows"][0]
        if best is None or r["head_psnr"] > best["head_psnr"]:
            best = dict(r, lam=lam)
    return best


out, rows = {}, []
for d in "abcd":
    m = mcgs(d)
    for c in ("T1", "FLAIR"):
        b = l1(d, c)
        if m is None or b is None:
            rows.append((d, c, None)); continue
        g = m[CH[c]]
        r = dict(draw=d, contrast=c, points=prep[d]["points"], af=prep[d][f"{c}_AF"],
                 mcgs=g["head_psnr"], mcgs_ssim=g["head_ssim"], mcgs_sharp=g["sharp"],
                 l1=b["head_psnr"], l1_lam=b["lam"], l1_edge=b["lam"] in (GRID[c][0], GRID[c][-1]), l1_sharp=b["sharp"],
                 margin=g["head_psnr"] - b["head_psnr"], sharp_gate=g["sharp"] >= b["sharp"])
        out.setdefault(c, []).append(r); rows.append((d, c, r))

print(f"{'draw':4s} {'c':5s} {'pts':>5s} {'AF':>6s} {'MC-GS':>7s} {'L1':>7s} {'lam':>7s} {'margin':>7s} {'sharp MC/L1':>12s}")
for d, c, r in rows:
    if r is None:
        print(f"{d:4s} {c:5s}  pending"); continue
    print(f"{d:4s} {c:5s} {r['points']:5d} {r['af']:6.2f} {r['mcgs']:7.3f} {r['l1']:7.3f} {r['l1_lam']:>7s}{'*' if r['l1_edge'] else ' '}"
          f"{r['margin']:+7.3f}  {r['mcgs_sharp']:.3f}/{r['l1_sharp']:.3f}{'' if r['sharp_gate'] else ' <'}")
print("(* = best L1 lambda on an edge of its three-point grid;  < = MC-GS sharpness below L1-ESPIRiT's)")
summ = {}
for c, rs in out.items():
    ms = [r["margin"] for r in rs]
    s = dict(n=len(ms), mean=statistics.mean(ms), sd=statistics.stdev(ms) if len(ms) > 1 else None,
             min=min(ms), max=max(ms), mcgs_sd=statistics.stdev([r["mcgs"] for r in rs]) if len(rs) > 1 else None,
             l1_sd=statistics.stdev([r["l1"] for r in rs]) if len(rs) > 1 else None,
             sharp_gate_passed=sum(r["sharp_gate"] for r in rs),
             af20_mean=af20[c]["mean"], af20_sd=af20[c]["sd"], af20_n=af20[c]["n"])
    s["shift_from_af20"] = s["mean"] - s["af20_mean"]
    ref_sd = max(x for x in (s["sd"], s["af20_sd"]) if x is not None)
    s["shift_in_sd"] = s["shift_from_af20"] / ref_sd if ref_sd else None
    summ[c] = s
    sd = f"{s['sd']:.3f}" if s["sd"] is not None else "--"
    print(f"{c:5s} margin over {s['n']} draws: mean {s['mean']:+.3f}  SD {sd}  range [{s['min']:+.3f}, {s['max']:+.3f}]"
          f"   AF20 ({s['af20_n']} draws): {s['af20_mean']:+.3f} ± {s['af20_sd']:.3f}   shift {s['shift_from_af20']:+.3f}"
          f" = {s['shift_in_sd']:+.1f} x the larger SD   sharpness gate passed on {s['sharp_gate_passed']}/{s['n']}")
json.dump(dict(rows=[r for _, _, r in rows if r], summary=summ), open(J / "logs/mask_draws_af24_summary.json", "w"), indent=2)
