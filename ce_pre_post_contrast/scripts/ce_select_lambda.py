#!/usr/bin/env python3
"""ce_select_lambda.py -- pick lambda for CS-SENSE, Joint SENSE and L1-ESPIRiT on the held-out residual.

Rule (fixed before any run, 2026-09-14): among the --mask fit runs of a family, take the lambda with the lowest
held-out kres (alpha refit on the 5% held-out lines, as ce_dip_inr.py defines it). CS-SENSE and L1-ESPIRiT are
selected per scan; Joint SENSE has one lambda for the pair and is selected on the mean of the two scans' held-out
kres. A minimum on either end of the tested range is not accepted: the family is extended by a factor of 2 past
that end (at most 3 extensions per side) and selected again.

usage: ce_select_lambda.py            -> prints the table, writes lambda_selection.json
Also imported by ce_orchestrate.py (select()).
"""
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ce_common as C  # noqa: E402

FAMILIES = {
    "cs_pre":   dict(dir="tv_sense",  pattern="cssense_o200_lam*_fit_pre_meta.json", scans=("pre",)),
    "cs_post":  dict(dir="tv_sense",  pattern="cssense_o200_lam*_fit_post_meta.json", scans=("post",)),
    "joint":    dict(dir="tv_sense",  pattern="jsense_o200_lam*_fit_pre_meta.json", scans=("pre", "post")),
    "l1_pre":   dict(dir="l1espirit", pattern="l1espirit_i100_lam*_fit_pre_meta.json", scans=("pre",)),
    "l1_post":  dict(dir="l1espirit", pattern="l1espirit_i100_lam*_fit_post_meta.json", scans=("post",)),
    # L1-SPIRiT (2026-09-15): one family per kernel size and scan; the kernel size is chosen afterwards on the same
    # held-out criterion (ce_orchestrate.py plan spirit). lambda 0 is the grid floor, so a minimum there is accepted.
    "spirit_ks3_pre":  dict(dir="spirit", pattern="spirit_ks3_i150_lam*_fit_pre_meta.json", scans=("pre",)),
    "spirit_ks5_pre":  dict(dir="spirit", pattern="spirit_ks5_i150_lam*_fit_pre_meta.json", scans=("pre",)),
    "spirit_ks3_post": dict(dir="spirit", pattern="spirit_ks3_i150_lam*_fit_post_meta.json", scans=("post",)),
    "spirit_ks5_post": dict(dir="spirit", pattern="spirit_ks5_i150_lam*_fit_post_meta.json", scans=("post",)),
}
MAX_EXT = 3


def curve(fam):
    spec = FAMILIES[fam]
    rows = {}
    for p in glob.glob(str(C.TOPIC / spec["dir"] / spec["pattern"])):
        lam = float(re.search(r"_lam([0-9.eE+-]+)_fit_", p).group(1))
        vals = []
        for s in spec["scans"]:
            q = Path(p.replace("_fit_pre_", f"_fit_{s}_").replace("_fit_post_", f"_fit_{s}_"))
            if not q.exists():
                break
            vals.append(json.load(open(q))["kres_heldout"])
        if len(vals) == len(spec["scans"]):
            rows[lam] = dict(per_scan=dict(zip(spec["scans"], vals)), score=sum(vals) / len(vals))
    return dict(sorted(rows.items()))


def select(fam, grid):
    """grid: the family's initial lambdas. Returns a dict with best, at_edge and next_lam (None when settled)."""
    rows = curve(fam)
    lams = list(rows)
    res = dict(family=fam, rows={f"{k:g}": v for k, v in rows.items()}, best=None, at_edge=None, next_lam=None,
               complete=all(any(abs(l - g) <= 1e-12 * max(g, 1e-30) + 1e-15 for l in lams) for g in grid))
    if not res["complete"]:
        return res
    best = min(lams, key=lambda l: rows[l]["score"])
    res["best"] = best
    lo, hi = min(grid), max(grid)
    if best == lams[0] and best > lo / 2 ** MAX_EXT * 1.0001:
        res["at_edge"], res["next_lam"] = "low", float(f"{best / 2:.3g}")
    elif best == lams[-1] and best < hi * 2 ** MAX_EXT / 1.0001:
        res["at_edge"], res["next_lam"] = "high", float(f"{best * 2:.3g}")
    elif best in (lams[0], lams[-1]):
        res["at_edge"] = "limit"          # extension budget spent: accepted, flagged
    return res


if __name__ == "__main__":
    grids = json.load(open(C.TOPIC / "lambda_grids.json"))
    out = {}
    for fam in FAMILIES:
        r = select(fam, grids[fam])
        out[fam] = r
        print(f"== {fam}: best {r['best']}  edge {r['at_edge']}  next {r['next_lam']}  complete {r['complete']}")
        for lam, row in r["rows"].items():
            mark = "  <-" if r["best"] is not None and float(lam) == r["best"] else ""
            print(f"   lam {lam:>9s}  held-out " + "  ".join(f"{s} {v:.5f}" for s, v in row["per_scan"].items())
                  + (f"  mean {row['score']:.5f}" if len(row["per_scan"]) > 1 else "") + mark)
    C.dump_json(out, C.TOPIC / "lambda_selection.json")
