#!/usr/bin/env python3
"""six_cells_now.py -- MC-GS on the six head cells of a noise topic, from whatever has finished (read-only, 2026-09-17).

For the favourable-range search (spec addendum 2026-09-17): the rows are built in memory exactly as summarize.py builds
them (classical / ZS-SSDU / DIP / INR / MC-INR json rows; each swept family at its best head PSNR; the GS runs straight
from their score json in the worktree), without registering anything or writing into the topic. Prints, per contrast
and head metric, MC-GS, the strongest other arm, the margin and the rank, and which arms are still missing.
usage: six_cells_now.py [--topic T1FLAIR_AF16_noise0p5x_v3_jiangyizhou]
"""
import argparse
import json
from pathlib import Path

RES = Path("/data0/congcong/code/GS/Results_20260908")
ARC = Path("/data0/congcong/code/GS/mcgs_noise_wt/results_arc")
ap = argparse.ArgumentParser()
ap.add_argument("--topic", default=Path(__file__).resolve().parents[1].name)
a = ap.parse_args()
R = RES / a.topic
prefix = {"noise2x": "af16n", "noise1x": "af16n1", "noise0p5x": "af16n05", "noise0p25x": "af16n025",
          "noise0p75x": "af16n075"}[a.topic.split("_")[2]]
HEAD = (("head_psnr", max), ("head_ssim", max), ("head_hfen", min))
ARMS = ("zf", "cgsense", "cssense", "cssense_o200", "jcssense", "spirit", "zsssdu", "dip", "inr", "mcinr",
        "gs_single_p3", "gs_joint_p3")

rows = {}
for c in ("T1", "FLAIR"):
    rows[c] = {}
    for stem in ("classical", "zsssdu", "dip", "inr", "mcinr"):
        p = R / f"logs/{stem}_{c}.json"
        if p.exists():
            for r in json.load(open(p)).get("rows", []):
                rows[c][r["method"]] = r
    for key, pat in (("cssense_o200", "logs/classical_{c}_o200_lam*.json"), ("jcssense", "logs/jcssense_o200_lam*.json"),
                     ("spirit", "logs/spirit_{c}_ks*_lam*.json")):
        cands = [r for f in sorted(R.glob(pat.format(c=c))) for r in json.load(open(f)).get("rows", [])
                 if r.get("contrast", c) == c and r.get("head_psnr") is not None]
        if cands:
            rows[c][key] = dict(max(cands, key=lambda r: r["head_psnr"]), n=len(cands))
    for key, exp, ch in (("gs_joint_p3", f"{prefix}_p3_joint_full4k", "plain" if c == "T1" else "contrast"),
                         ("gs_single_p3", f"{prefix}_p3_single{c}_full4k", "plain")):
        p = ARC / f"{exp}_score.json"
        if p.exists():
            rows[c][key] = json.load(open(p))[ch]

print(f"{a.topic}: six head cells from what has finished")
all_first = True
for c in ("T1", "FLAIR"):
    have = rows[c]
    missing = [k for k in ARMS if k not in have]
    print(f"\n{c}: arms present {len(have)} / {len(ARMS)}; missing {missing}")
    if "gs_joint_p3" not in have:
        print("  MC-GS not finished")
        all_first = False
        continue
    for m, best in HEAD:
        vals = {k: float(r[m]) for k, r in have.items() if k in ARMS and r.get(m) is not None}
        mc = vals.pop("gs_joint_p3")
        k2 = best(vals, key=vals.get)
        margin = (mc - vals[k2]) if best is max else (vals[k2] - mc)
        rank = 1 + sum(1 for v in vals.values() if (v > mc if best is max else v < mc))
        all_first &= rank == 1
        lam = f" (lam {have[k2].get('lam')}, {have[k2].get('n')} points)" if have[k2].get("lam") is not None else ""
        print(f"  {m:10s} MC-GS {mc:.4f}  strongest other {k2} {vals[k2]:.4f}{lam}  margin {margin:+.4f}  "
              f"rank {rank}/{len(vals) + 1}")
print(f"\nMC-GS first on every cell computed so far: {all_first}")
