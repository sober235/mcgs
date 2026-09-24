"""Ledger of the widen-the-margin loop: head PSNR / SSIM of every wd_ run and its margin over GS-single at lambda_edge 0.

Margin = MC-GS (joint) - GS-single (lambda_edge 0, the user's 2026-09-24 baseline), per AF and contrast, head-only metrics
from the run's score json (the same scorer the AF topics register). Single arms read the plain channel (both channels
carry the same contrast). AF 16's single baselines are the coupling-attribution study's ca_s_*_edge0 runs.
"""
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CA = Path("/data0/congcong/code/GS/mcgs_hparam_wt/results_arc")


def arm_of(d):
    cfg = d["argv"][d["argv"].index("--base_cfg") + 1]
    af = int(re.search(r"_af(\d+)_", cfg).group(1))
    m = re.search(r"single(T1|FLAIR)", cfg)
    return af, (m.group(1) if m else "joint")


def knobs(d):
    """lambda_edge (last occurrence wins, as in cycle.py) plus every arg after --base_cfg."""
    a = d["argv"]
    le = [j for j, x in enumerate(a) if x == "--lambda_edge"]
    extra = a[a.index("--base_cfg") + 2:]
    return " ".join(["le=" + a[le[-1] + 1]]
                    + [x.replace("loss.", "") for x in extra if x not in ("--save_volumes", "--set")])


def load(p):
    d = json.load(open(p))
    af, arm = arm_of(d)
    rows = {}
    if arm == "joint":
        rows["T1"] = d["plain"]
        rows["FLAIR"] = d["contrast"]
    else:
        rows[arm] = d["plain"]
    return af, arm, rows, knobs(d)


runs = []
for p in sorted(glob.glob(str(ROOT / "results_arc/wd_*_score.json"))):
    if "smoke" in p:
        continue
    runs.append((Path(p).name[:-11],) + load(p))
for name in ("ca_s_t1_edge0", "ca_s_fl_edge0"):
    p = CA / f"{name}_score.json"
    if p.exists():
        runs.append((name,) + load(p))

base = {}
for name, af, arm, rows, kn in runs:
    if arm != "joint" and "le=0" == kn.split()[0] and len(kn.split()) == 1:
        base[(af, arm)] = rows[arm]["head_psnr"]

print("| exp | AF | arm | knobs | T1 head PSNR / SSIM | FLAIR head PSNR / SSIM | margin T1 | margin FLAIR |")
print("|---|---|---|---|---|---|---|---|")
for name, af, arm, rows, kn in sorted(runs, key=lambda r: (r[1], r[2] != "joint", r[0])):
    cell = {c: (f"{rows[c]['head_psnr']:.3f} / {rows[c]['head_ssim']:.4f}" if c in rows else "") for c in ("T1", "FLAIR")}
    mg = {}
    for c in ("T1", "FLAIR"):
        if arm == "joint" and (af, c) in base:
            mg[c] = f"{rows[c]['head_psnr'] - base[(af, c)]:+.3f}"
        else:
            mg[c] = ""
    print(f"| {name} | {af} | {arm} | {kn} | {cell['T1']} | {cell['FLAIR']} | {mg['T1']} | {mg['FLAIR']} |")
print()
print("GS-single baselines (lambda_edge 0):", {f"AF{k[0]} {k[1]}": round(v, 3) for k, v in sorted(base.items())})
