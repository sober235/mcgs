#!/usr/bin/env python3
"""autoresearch_gauss_row.py -- append one ledger row of the Gaussian-mask loop to autoresearch_gauss.tsv.

Reads results_arc/<exp>_score.json and the comparison bar (gauss topic logs/bar_gauss.json, built by
scripts/bar_gauss.py: best comparison arm per cell, GS-single at the frozen recipe included, MC-GS excluded).
Objective of this line: maximise min(dT1_mag, dFLAIR_mag) (head PSNR over the bar, magnitude gauge); target >= 0.5.
six_first = MC-GS beats the bar on all six head cells (PSNR / SSIM / HFEN x T1 / FLAIR).
The ledger is appended under an flock, so concurrent runs cannot interleave a row.
"""
import argparse, fcntl, json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAR = Path("/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_gauss_v3_jiangyizhou/logs/bar_gauss.json")
LEDGER = ROOT / "autoresearch_gauss.tsv"
COLS = ["exp", "status", "iters", "npts", "vram_gb", "min",
        "T1_mag", "FLAIR_mag", "dT1_mag", "dFLAIR_mag", "min_margin_mag",
        "T1_cplx", "FLAIR_cplx", "dT1_cplx", "dFLAIR_cplx", "min_margin_cplx",
        "T1_head_ssim", "FLAIR_head_ssim", "dT1_ssim", "dFLAIR_ssim",
        "T1_head_hfen", "FLAIR_head_hfen", "dT1_hfen", "dFLAIR_hfen", "six_first",
        "T1_sharp", "FLAIR_sharp", "T1_cos_phase", "FLAIR_cos_phase", "T1_e_unobs", "FLAIR_e_unobs",
        "commit", "args", "note"]

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True)
ap.add_argument("--status", default="pending")
ap.add_argument("--note", default="")
a = ap.parse_args()

bar = json.load(open(BAR))["bar"]
sj = ROOT / f"results_arc/{a.exp}_score.json"
row = {c: "" for c in COLS}
row.update(exp=a.exp, status=a.status, note=a.note)
try:
    row["commit"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                   capture_output=True, text=True).stdout.strip()
except Exception:
    pass
if sj.exists():
    d = json.load(open(sj))
    ch = {d.get("plain_name", "T1"): d["plain"], d.get("contrast_name", "FLAIR"): d["contrast"]}
    row.update(iters=d.get("iters"), npts=d.get("npts"), vram_gb=round(d.get("vram_gb", 0), 1),
               min=round(d.get("elapsed_s", 0) / 60.0, 1), args=" ".join(d.get("argv", [])))
    firsts = []
    for c in ("T1", "FLAIR"):
        s = ch.get(c)
        if s is None:
            continue
        row[f"{c}_mag"] = round(s["head_psnr"], 3)
        row[f"{c}_cplx"] = round(s["head_psnr_cplx"], 3)
        row[f"d{c}_mag"] = round(s["head_psnr"] - bar["head_psnr"][c]["value"], 3)
        row[f"d{c}_cplx"] = round(s["head_psnr_cplx"] - bar["head_psnr_cplx"][c]["value"], 3)
        row[f"{c}_head_ssim"] = round(s["head_ssim"], 4)
        row[f"{c}_head_hfen"] = round(s["head_hfen"], 4)
        row[f"d{c}_ssim"] = round(s["head_ssim"] - bar["head_ssim"][c]["value"], 4)
        row[f"d{c}_hfen"] = round(bar["head_hfen"][c]["value"] - s["head_hfen"], 4)   # positive = better than the bar
        row[f"{c}_sharp"] = round(s["sharp"], 3)
        row[f"{c}_cos_phase"] = round(s["cos_phase"], 4)
        row[f"{c}_e_unobs"] = round(s["e_unobs"], 4)
        firsts += [row[f"d{c}_mag"] > 0, row[f"d{c}_ssim"] > 0, row[f"d{c}_hfen"] > 0]
    for g in ("mag", "cplx"):
        ds = [row[f"d{c}_{g}"] for c in ("T1", "FLAIR") if row[f"d{c}_{g}"] != ""]
        row[f"min_margin_{g}"] = round(min(ds), 3) if ds else ""
    row["six_first"] = int(all(firsts)) if len(firsts) == 6 else ""
with open(LEDGER, "a+") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.seek(0)
    if not f.read(1):
        f.write("\t".join(COLS) + "\n")
    f.seek(0, 2)
    f.write("\t".join(str(row[c]) for c in COLS) + "\n")
    fcntl.flock(f, fcntl.LOCK_UN)
print("ledger row:", {k: row[k] for k in ("exp", "status", "T1_mag", "FLAIR_mag", "dT1_mag", "dFLAIR_mag",
                                          "min_margin_mag", "six_first")})
