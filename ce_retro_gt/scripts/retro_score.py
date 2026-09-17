#!/usr/bin/env python3
"""retro_score.py -- score the retrospective reconstructions against the ground truth (2026-09-16).

What the prospective pair cannot answer and this can:

  image        in-brain PSNR and NRMSE of the magnitude, after the best complex scale against the truth
  enhancement  the mean post - pre inside the enhancing label (4), as a fraction of the true one: does the arm keep
               the uptake, or wash it out?
  false        the mean post - pre inside the necrotic core (1), which does not enhance, minus the true one: does
               the arm invent uptake where there is none? (the prospective lesion is exactly this situation)
  white matter the same in normal-appearing white matter, where the truth is near zero
  boundary     Dice of the enhancing region recovered by thresholding post - pre at half the true mean uptake

  python scripts/retro_score.py [--cases ...] [--arms ...]
"""
import argparse
import json
from pathlib import Path

import numpy as np

T = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--cases", default="")
ap.add_argument("--arms", default="zf,cgsense,cssense,jsense,spirit,zsssdu,dip,inr,mcinr,gs_single,gs_joint",
                help="the prospective topic's column order")
ap.add_argument("--lam", type=float, default=2e-3)
ap.add_argument("--out", default="retro_scores.json")
a = ap.parse_args()
CASES = a.cases.split(",") if a.cases else sorted(p.name for p in (T / "sim").iterdir() if p.is_dir())
LABEL = {"zf": "ZF", "cgsense": "CG-SENSE", "cssense": "CS-SENSE", "jsense": "Joint SENSE", "spirit": "L1-SPIRiT",
         "zsssdu": "ZS-SSDU", "dip": "3D-DIP", "inr": "3D-INR", "mcinr": "MC-INR", "gs_single": "GS-single",
         "gs_joint": "MC-GS"}


def more_sel(case):
    """the held-out choices of retro_more_arms.py (Joint SENSE lambda, L1-SPIRiT kernel and lambda)."""
    f = T / "recon" / case / "more_arms_selection.json"
    return json.load(open(f)) if f.exists() else {}


def cs_lambda(case, scan):
    """the held-out choice of retro_cs_lambda.py, or --lam if the selection has not run for this case."""
    f = T / "recon" / case / "cs_lambda.json"
    return json.load(open(f))[scan]["lam"] if f.exists() else a.lam


def paths(arm, case, scan):
    R = T / "recon" / case
    sel = more_sel(case)
    if arm == "jsense":
        l = sel.get("jsense", {}).get("lam")
        return (R / "tv" / f"jsense_o200_lam{l:g}_full_{scan}_complex.npy") if l else Path("/nonexistent"), False
    if arm == "spirit":
        q = sel.get(f"spirit_{scan}")
        return ((R / "spirit" / f"spirit_ks{q['ksize']}_i150_lam{q['lam']:g}_full_{scan}_complex.npy")
                if q else Path("/nonexistent")), False
    if arm == "zsssdu":
        return R / f"zsssdu_{scan}_complex.npy", False
    if arm == "mcinr":
        return R / "mcinr" / f"mcinr_full_{scan}_complex.npy", False
    if arm == "cssense":
        return R / "tv" / f"cssense_o200_lam{cs_lambda(case, scan):g}_full_{scan}_complex.npy", False
    if arm == "gs_single":
        return R / f"gs_single_{scan}" / f"gs_single_{scan}_raw_complex.npy", True
    if arm == "gs_joint":
        return R / "gs_joint" / f"gs_joint_{scan}_raw_complex.npy", True
    return R / f"{arm}_{scan}_complex.npy", False


def load(arm, case, scan, ref):
    p, gs = paths(arm, case, scan)
    if not p.exists():
        return None
    v = np.load(p)
    if gs:                                                  # the GS trainer renders (readout, PE2, PE1)
        v = np.transpose(v, (0, 2, 1))
    assert v.shape == ref.shape, f"{p}: {v.shape} vs {ref.shape}"
    al = np.vdot(v[BR], ref[BR]) / np.vdot(v[BR], v[BR])    # best complex scale against the truth
    return np.abs(al * v).astype(np.float32)


rows = []
for case in CASES:
    g = np.load(T / "sim" / case / "gt.npz")
    gt = {"pre": g["pre"], "post": g["post"]}
    BR, tum, wm = g["brain"], g["tumor"], g["wm"]
    enh, nec = tum == 4, tum == 1
    if not (enh.sum() and nec.sum()):
        print(f"{case}: no enhancing or necrotic label, skipped")
        continue
    peak = float(np.percentile(np.abs(gt["pre"][BR]), 99.9))
    d_true = np.abs(gt["post"]) - np.abs(gt["pre"])
    t_enh, t_nec, t_wm = float(d_true[enh].mean()), float(d_true[nec].mean()), float(d_true[wm].mean())
    print(f"\n{case}: enhancing {int(enh.sum())} mm3 (true uptake {t_enh:+.3f}), necrotic {int(nec.sum())} mm3 "
          f"(true {t_nec:+.3f}), white matter true {t_wm:+.3f}")
    print(f"{'arm':<11}{'PSNR pre':>9}{'PSNR post':>10}{'uptake':>9}{'kept':>7}{'false(nec)':>11}{'WM':>8}{'Dice':>7}")
    ref_mask = d_true > 0.5 * t_enh                          # where the truth says there is uptake
    for arm in a.arms.split(","):
        xs = {s: load(arm, case, s, gt[s]) for s in ("pre", "post")}
        if xs["pre"] is None or xs["post"] is None:
            print(f"{LABEL.get(arm, arm):<11}   missing")
            continue
        psnr = {s: 10 * np.log10(peak ** 2 / float(((xs[s] - np.abs(gt[s]))[BR] ** 2).mean())) for s in xs}
        d = xs["post"] - xs["pre"]
        r_enh, r_nec, r_wm = float(d[enh].mean()), float(d[nec].mean()), float(d[wm].mean())
        m = d > 0.5 * t_enh
        dice = 2 * float((m & ref_mask & BR).sum()) / max(float((m & BR).sum() + (ref_mask & BR).sum()), 1)
        rows.append(dict(case=case, arm=LABEL.get(arm, arm), psnr_pre=psnr["pre"], psnr_post=psnr["post"],
                         uptake=r_enh, uptake_true=t_enh, kept=r_enh / t_enh, uptake_err=r_enh - t_enh,
                         false_necrotic=r_nec - t_nec, wm=r_wm - t_wm, dice_enh=dice,
                         uptake_usable=abs(t_enh) >= 0.1))   # a case whose label 4 barely enhances cannot score "kept"
        print(f"{LABEL.get(arm, arm):<11}{psnr['pre']:>9.2f}{psnr['post']:>10.2f}{r_enh:>+9.3f}"
              f"{100 * r_enh / t_enh:>6.0f}%{r_nec - t_nec:>+11.3f}{r_wm - t_wm:>+8.3f}{dice:>7.3f}")
if rows:
    json.dump(dict(note="uptake/false/WM are post-pre in white-matter units; kept = uptake / true uptake; "
                        "false = necrotic-core uptake minus the true one; Dice of post-pre > half the true uptake",
                   rows=rows), open(T / a.out, "w"), indent=1)
    print(f"\nwrote {T / a.out}")
    arms = [LABEL.get(x, x) for x in a.arms.split(",")]
    nuse = len({r["case"] for r in rows if r["uptake_usable"]})
    print(f"\n{'arm':<11}{'PSNR':>8}{'kept':>8}{'|uptake err|':>14}{'false(nec)':>12}{'Dice':>7}   mean over "
          f"{len({r['case'] for r in rows})} case(s); kept over the {nuse} with real uptake")
    for arm in arms:
        rr = [r for r in rows if r["arm"] == arm]
        uu = [r for r in rr if r["uptake_usable"]]
        if rr:
            print(f"{arm:<11}{np.mean([(r['psnr_pre'] + r['psnr_post']) / 2 for r in rr]):>8.2f}"
                  f"{100 * np.mean([r['kept'] for r in uu]):>7.0f}%{np.mean([abs(r['uptake_err']) for r in rr]):>14.3f}"
                  f"{np.mean([r['false_necrotic'] for r in rr]):>+12.3f}{np.mean([r['dice_enh'] for r in rr]):>7.3f}")
