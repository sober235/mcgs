#!/usr/bin/env python3
"""gauss_judge.py -- read autoresearch_gauss.tsv and apply the pre-registered Round-0 reading (autoresearch_gauss.md, rule 2).

Floors come from |control - control_s43| per cell (PSNR per contrast; SSIM / HFEN / sharpness floors are fixed at
0.0005 / 0.0005 / 0.010 because the seed repeat moved them by <= 0.0002 / 0.0002 / 0.001).
Verdicts (1500-it screens against g_r0_control):
  keep          both contrasts' head PSNR up by more than their floor, no cell (PSNR / SSIM / HFEN, either contrast) down
                by more than its floor, neither sharpness down by more than 0.010
  keep-one      one contrast up by more than its floor, the other inside its floor, no slips (usable in a stack)
  reshape       one contrast up beyond its floor, the other DOWN beyond its floor (a trade, not a gain)
  keep-marginal PSNR up (both or one) but a structural cell or sharpness slips (promoted only if 4000 it repairs it)
  discard       everything else
Usage: gauss_judge.py [--ref g_r0_control] [--seed g_r0_control_s43] [--md] [exp ...]
"""
import argparse, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--ref", default="g_r0_control")
ap.add_argument("--seed", default="g_r0_control_s43")
ap.add_argument("--md", action="store_true", help="print round-log rows in the program file's markdown format")
ap.add_argument("exps", nargs="*")
a = ap.parse_args()
rows = {r["exp"]: r for r in csv.DictReader(open(ROOT / "autoresearch_gauss.tsv"), delimiter="\t") if r["T1_mag"]}
ref, seed = rows[a.ref], rows[a.seed]
f = lambda r, k: float(r[k])
floor = {"T1": max(0.005, abs(f(ref, "T1_mag") - f(seed, "T1_mag"))),
         "FLAIR": max(0.005, abs(f(ref, "FLAIR_mag") - f(seed, "FLAIR_mag")))}
F_SSIM, F_HFEN, F_SHARP = 0.0005, 0.0005, 0.010
print(f"reference {a.ref}: T1 {f(ref,'T1_mag'):.3f} FLAIR {f(ref,'FLAIR_mag'):.3f} | floors PSNR {floor['T1']:.3f} / {floor['FLAIR']:.3f}, "
      f"SSIM {F_SSIM}, HFEN {F_HFEN}, sharp {F_SHARP}")
print(f"{'exp':22s} {'dT1':>7s} {'dFL':>7s} {'dSSIM T1/FL':>16s} {'dHFEN T1/FL':>16s} {'dsharp T1/FL':>15s} {'margin':>7s} verdict")
sel = a.exps or [e for e in rows if e not in (a.ref, a.seed, "g_smoke")]
for e in sel:
    r = rows.get(e)
    if r is None:
        print(f"{e:22s} (no row yet)"); continue
    d = {c: f(r, f"{c}_mag") - f(ref, f"{c}_mag") for c in ("T1", "FLAIR")}
    ds = {c: f(r, f"{c}_head_ssim") - f(ref, f"{c}_head_ssim") for c in ("T1", "FLAIR")}
    dh = {c: f(ref, f"{c}_head_hfen") - f(r, f"{c}_head_hfen") for c in ("T1", "FLAIR")}   # positive = better
    dsh = {c: f(r, f"{c}_sharp") - f(ref, f"{c}_sharp") for c in ("T1", "FLAIR")}
    up = {c: d[c] > floor[c] for c in d}; down = {c: d[c] < -floor[c] for c in d}
    slip = any(ds[c] < -F_SSIM or dh[c] < -F_HFEN for c in d) or any(dsh[c] < -F_SHARP for c in d)
    if any(down.values()) and any(up.values()):
        v = "reshape"
    elif any(down.values()):
        v = "discard"
    elif all(up.values()):
        v = "keep" if not slip else "keep-marginal"
    elif any(up.values()):
        v = "keep-one" if not slip else "keep-marginal"
    else:
        v = "discard (null)"
    if r["status"] != "pending":
        v += f" [{r['status']}]"
    if a.md:
        print(f"| {r['exp']} | {f(r,'T1_mag'):.3f} ({d['T1']:+.3f}) | {f(r,'FLAIR_mag'):.3f} ({d['FLAIR']:+.3f}) | "
              f"{f(r,'T1_head_ssim'):.4f} / {f(r,'FLAIR_head_ssim'):.4f} | {f(r,'T1_head_hfen'):.4f} / {f(r,'FLAIR_head_hfen'):.4f} | "
              f"{f(r,'T1_sharp'):.3f} / {f(r,'FLAIR_sharp'):.3f} | margin {r['min_margin_mag']} → **{v}** |")
    else:
        print(f"{e:22s} {d['T1']:+7.3f} {d['FLAIR']:+7.3f} {ds['T1']:+8.4f}/{ds['FLAIR']:+7.4f} {dh['T1']:+8.4f}/{dh['FLAIR']:+7.4f} "
              f"{dsh['T1']:+7.3f}/{dsh['FLAIR']:+7.3f} {r['min_margin_mag']:>7s} {v}")
