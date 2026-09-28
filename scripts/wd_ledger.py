"""Ledger of the widen-the-margin loop: head PSNR / SSIM of every wd_ run and its margin over GS-single at lambda_edge 0.

Margin = MC-GS (joint) - GS-single (best single run, lambda_edge 0; user choices A and (1), 2026-09-24), per AF and contrast, head-only metrics
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
    a = d["argv"]
    cfg = a[len(a) - 1 - a[::-1].index("--base_cfg") + 1]      # the LAST --base_cfg is the one cycle.py uses
    af = int(re.search(r"_af(\d+)_", cfg).group(1))
    m = re.search(r"single(T1|FLAIR)", cfg)
    return af, (m.group(1) if m else "joint")


def knobs(d):
    """lambda_edge (last occurrence wins, as in cycle.py) plus every arg after --base_cfg."""
    a = d["argv"]
    le = [j for j, x in enumerate(a) if x == "--lambda_edge"]
    extra = [x for i, x in enumerate(a[a.index("--base_cfg") + 2:]) if x != "--base_cfg"]
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


PROBE = ("wd_o_", "wd_a_", "wd_va_")       # oracle and asymmetric-mask probes: not part of the fair comparison
runs, probes = [], []
for p in sorted(glob.glob(str(ROOT / "results_arc/wd_*_score.json"))):
    name = Path(p).name[:-11]
    if "smoke" in name:
        continue
    (probes if name.startswith(PROBE) else runs).append((name,) + load(p))
for p in sorted(CA.glob("ca_s_*_edge0_score.json")) + sorted(CA.glob("ca_s4_*_score.json")):
    runs.append((p.name[:-11],) + load(p))

# choice A: GS-single = the best single run of the symmetric per-channel sweep (coupling knobs are never on in it)
base, base_exp = {}, {}
for name, af, arm, rows, kn in runs:
    if arm != "joint":
        assert kn.split()[0] == "le=0" and "dtv" not in kn and "guide" not in kn, name
        if rows[arm]["head_psnr"] > base.get((af, arm), -1):
            base[(af, arm)], base_exp[(af, arm)] = rows[arm]["head_psnr"], name

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
print("probe runs (oracle partners / guides, asymmetric probe masks): raw head metrics, no margin -- see the ceiling record")
print("| exp | arm | knobs | T1 head PSNR / SSIM | FLAIR head PSNR / SSIM |")
print("|---|---|---|---|---|")
for name, af, arm, rows, kn in probes:
    cell = {c: (f"{rows[c]['head_psnr']:.3f} / {rows[c]['head_ssim']:.4f}" if c in rows else "") for c in ("T1", "FLAIR")}
    print(f"| {name} | {arm} | {kn} | {cell['T1']} | {cell['FLAIR']} |")
print()
print("GS-single = best single run:", {f"AF{k[0]} {k[1]}": (round(v, 3), base_exp[k]) for k, v in sorted(base.items())})
