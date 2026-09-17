#!/usr/bin/env python3
"""bench_table.py -- reconstruction time per arm for one T1 + FLAIR pair at AF 12, one A800 80GB per job, solver only.

Sources (all this topic's own files):
  fast arms      logs/bench/fast_arms.json          bench_fast_arms.py on an exclusive card (ZF / CG-SENSE: median of 3)
  L1-SPIRiT      logs/spirit_<c>_bench.json         recon_spirit.py --suffix _bench, its own solver timer (calibration included)
  ZS-SSDU        logs/zsssdu_<c>.json               run of record (one card per job); its timer starts before loading, so the
                                                    measured Pair load is subtracted; spot check logs/zsssdu_T1_bench.json
  3D-DIP, 3D-INR logs/{dip,inr}_<c>.json            run of record, timer = the optimisation loop (5000 / 1500 iterations)
  MC-INR         recon/af12/meta.json (MC-INR topic) seconds_loop of the run of record, both contrasts in one run
  GS-single, MC-GS  results_arc/<exp>.log            "joint training done in" of the P14 runs of record (4000 iterations,
                                                    3.6M points); spot check = the run in logs/bench/run_bench_time.log
Per pair: joint arms reconstruct both contrasts in one run; single-contrast arms = T1 + FLAIR.
Writes logs/bench/recon_time_table.{md,json}.
"""
import json, re
from pathlib import Path
R = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_v3_jiangyizhou")
W = Path("/data0/congcong/code/GS/mcgs_af12_wt/results_arc")
MCINR = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_MCINR_Niessen2025_v3_jiangyizhou/recon/af12/meta.json")
P14 = {"MC-GS": "p3_e0001_ptv001_full4k", ("GS-single", "T1"): "p3_e0001_ptv001_singleT1_full4k",
       ("GS-single", "FLAIR"): "p3_e0001_ptv001_singleFLAIR_full4k"}


def jload(p):
    return json.load(open(p)) if Path(p).exists() else None


def train_seconds(logfile):
    t = Path(logfile).read_text() if Path(logfile).exists() else ""
    m = re.findall(r"joint training done in ([0-9.]+)s", t)
    return float(m[-1]) if m else None


fast = jload(R / "logs/bench/fast_arms.json")
F = {}
if fast:
    for r in fast["rows"]:
        F[(r["arm"], r["contrast"])] = r
rows = []                                                     # (arm, joint?, T1 s, FLAIR s, pair s, note)


def add(arm, joint, t1, fl, note="", pair=None):
    if pair is None:
        pair = None if (t1 is None or fl is None) else t1 + fl
    rows.append(dict(arm=arm, joint=joint, T1=t1, FLAIR=fl, pair=pair, note=note))


def f(key, c):
    r = F.get((key, c)); return None if r is None else r["seconds"]


shared = sorted({tuple(r["shared"]) for r in (fast or {"rows": []})["rows"] if r["shared"]})
add("ZF", False, f("zf", "T1"), f("zf", "FLAIR"), "adjoint SENSE of the zero-filled k-space; median of 3")
add("CG-SENSE", False, f("cgsense", "T1"), f("cgsense", "FLAIR"), "60 CG iterations, Tikhonov 0.01; median of 3")
add("CS-SENSE", False, f("cssense_o200", "T1"), f("cssense_o200", "FLAIR"), "ADMM isotropic 3-D TV, 200 outer x 8 CG, lambda 7e-4")
jt = f("jcssense_o200", "T1+FLAIR")
add("Joint SENSE", True, None, None, "vectorial TV over both contrasts, 200 outer x 8 CG, lambda 1e-3; one run for the pair", pair=jt)
sp = {c: jload(R / f"logs/spirit_{c}_bench.json") for c in ("T1", "FLAIR")}
add("L1-SPIRiT", False, *(None if sp[c] is None else sp[c]["rows"][0]["seconds"] for c in ("T1", "FLAIR")),
    "FISTA, 150 iterations, kernel 5, lambda 2e-3 / 1e-3, per-readout calibration included; re-timed on the exclusive card")
load = {c: f("load", c) for c in ("T1", "FLAIR")}
zs = {c: jload(R / f"logs/zsssdu_{c}.json") for c in ("T1", "FLAIR")}
zs_s = {c: (None if zs[c] is None else zs[c]["rows"][0]["seconds"] - (load[c] or 0.0)) for c in ("T1", "FLAIR")}
zsb = jload(R / "logs/zsssdu_T1_bench.json")
add("ZS-SSDU", False, zs_s["T1"], zs_s["FLAIR"],
    f"self-supervised training to early stop (epochs {zs['T1']['rows'][0].get('epochs_run') if zs['T1'] else '?'} / "
    f"{zs['FLAIR']['rows'][0].get('epochs_run') if zs['FLAIR'] else '?'}) + inference; run of record minus the measured load"
    + (f"; spot check T1 {zsb['rows'][0]['seconds'] - (load['T1'] or 0):.0f} s" if zsb else ""))
for key, name, note in (("dip", "3D-DIP", "5000 iterations of the convolutional decoder, held-out selection online"),
                        ("inr", "3D-INR", "1500 iterations of the hash-grid INR, held-out selection online")):
    d = {c: jload(R / f"logs/{key}_{c}.json") for c in ("T1", "FLAIR")}
    add(name, False, *(None if d[c] is None else d[c]["rows"][0]["seconds"] for c in ("T1", "FLAIR")), note + "; run of record")
mi = jload(MCINR)
add("MC-INR", True, None, None, "Niessen 2025 per-slice 2-D multi-contrast INR, 256 readout slices x 200 epochs; run of record (A800, torch 1.13)",
    pair=None if mi is None else mi["seconds_loop"])
gs_t1, gs_fl = train_seconds(W / f"{P14[('GS-single', 'T1')]}.log"), train_seconds(W / f"{P14[('GS-single', 'FLAIR')]}.log")
add("GS-single", False, gs_t1, gs_fl, "P14 recipe, 4000 iterations, 3.6M Gaussians; training loop of the run of record (FLAIR single ran on an A100)")
mc = train_seconds(W / f"{P14['MC-GS']}.log"); mcb = train_seconds(R / "logs/bench/run_bench_time.log")   # the spot check's stdout
add("MC-GS", True, None, None, "P14 recipe, 4000 iterations, 3.6M Gaussians, both contrasts in one run; training loop of the run of record"
    + (f"; spot check on the exclusive card {mcb:.0f} s" if mcb else ""), pair=mc)


def fmt(s):
    if s is None: return "--"
    return f"{s:.2f} s" if s < 10 else (f"{s:.0f} s" if s < 600 else f"{s/60:.1f} min")


lines = ["| method | contrast coupling | T1 | FLAIR | one T1 + FLAIR pair | notes |", "|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['arm']} | {'joint' if r['joint'] else 'single'} | {fmt(r['T1'])} | {fmt(r['FLAIR'])} | **{fmt(r['pair'])}** | {r['note']} |")
hdr = (f"Reconstruction time per arm, one T1 + FLAIR pair, jiangyizhou AF 12 (26 coils, 256 x 256 x 176), one NVIDIA A800 80GB per job, "
       f"solver only (data loading {fmt(load['T1'])} / {fmt(load['FLAIR'])} and scoring excluded). "
       + ("Fast arms re-timed on an exclusive card; " if fast else "") + "long arms from the runs of record (one card per job)."
       + (f" Foreign processes seen on the card during the fast-arm timing: {shared}." if shared else " No foreign process on the card during the fast-arm timing."))
md = hdr + "\n\n" + "\n".join(lines) + "\n"
(R / "logs/bench").mkdir(exist_ok=True)
(R / "logs/bench/recon_time_table.md").write_text(md)
json.dump(dict(header=hdr, rows=rows, load=load), open(R / "logs/bench/recon_time_table.json", "w"), indent=1)
print(md)
