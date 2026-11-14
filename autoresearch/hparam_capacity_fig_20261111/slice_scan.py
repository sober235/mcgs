#!/usr/bin/env python
"""slice_scan.py -- per-slice head PSNR / SSIM / NMSE of the four capacity runs (2026-11-11).

User 2026-11-11: do not use the whole volume; support the capacity conclusion with one slice.
This scan computes the slice metrics for every slice on the three axes so that the slice can be chosen
with the whole picture in view, and so that the figure can state how typical the chosen slice is.

Slice metrics are those of the figure of record (fig_three_planes_compact.py): head mask
|GT| > 0.05 * P99.9 of the volume; PSNR at the volume's max|GT|; SSIM = mean of the 2-D SSIM map
(7 x 7 uniform window, data range max|GT|) over the head pixels; NMSE over the head pixels.

Pattern tested per slice (the sentence to be supported):
  A  under-capacity lowers both contrasts: 0.9M worse than 3.6M on PSNR and SSIM, T1 and FLAIR
  A+ ... and monotone 0.9M < 1.8M < 3.6M
  B  doubling to 7.2M is a trade-off: PSNR and SSIM move in opposite directions for T1 and FLAIR
     (B1 = T1 down / FLAIR up, the whole-head direction; B2 = T1 up / FLAIR down)
NMSE on one slice is a monotone transform of that slice's PSNR, so it never changes a verdict.
Writes slice_metrics.csv to MCGS_ARTIFACT_DIR. Reads saved volumes only. CPU only.
"""
import csv
import os
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity

ARC = Path("/data0/congcong/code/GS/mcgs_hparam_wt/results_arc")
OUT = Path(os.environ["MCGS_ARTIFACT_DIR"])
CAP = ["hp_n09", "hp_n18", "hp_n36", "hp_n72"]
KEYS = {"T1": ("recT1", "gtT1"), "FLAIR": ("recFL", "gtFL")}
ANAT = {0: "transverse", 1: "coronal", 2: "sagittal"}
HOUSE = {0: 180, 1: 124, 2: 76}
MIN_HEAD = 0.5            # a slice counts only if its head area is at least half of the largest on that axis

rec, gt = {c: {} for c in KEYS}, {}
for e in CAP:
    z = np.load(ARC / f"{e}_volumes.npz")
    for c, (rk, gk) in KEYS.items():
        gt.setdefault(c, z[gk].astype(np.float64))
        rec[c][e] = z[rk].astype(np.float64)
    z.close()
gmax = {c: float(gt[c].max()) for c in KEYS}
head = {c: gt[c] > 0.05 * float(np.quantile(gt[c], 0.999)) for c in KEYS}

rows = []
for axis in (0, 1, 2):
    area = {c: head[c].sum(axis=tuple(a for a in (0, 1, 2) if a != axis)) for c in KEYS}
    amax = max(area["T1"].max(), area["FLAIR"].max())
    for idx in range(gt["T1"].shape[axis]):
        if min(area["T1"][idx], area["FLAIR"][idx]) < MIN_HEAD * amax:
            continue
        sl = [slice(None)] * 3; sl[axis] = idx; sl = tuple(sl)
        row = {"axis": axis, "plane": ANAT[axis], "slice": idx, "head_px": int(min(area["T1"][idx], area["FLAIR"][idx]))}
        for c in KEYS:
            g, m = gt[c][sl], head[c][sl]
            for e in CAP:
                r = rec[c][e][sl]
                d = r[m] - g[m]
                mse = float((d * d).mean())
                row[f"{c}_{e}_psnr"] = 20 * np.log10(gmax[c]) - 10 * np.log10(mse)
                row[f"{c}_{e}_nmse"] = float((d * d).sum() / (g[m] ** 2).sum())
                _, smap = structural_similarity(g, r, data_range=gmax[c], full=True, win_size=7)
                row[f"{c}_{e}_ssim"] = float(smap[m].mean())
        v = lambda c, e, k: row[f"{c}_{e}_{k}"]
        A = all(v(c, "hp_n09", k) < v(c, "hp_n36", k) for c in KEYS for k in ("psnr", "ssim"))
        Ap = all(v(c, "hp_n09", k) < v(c, "hp_n18", k) < v(c, "hp_n36", k) for c in KEYS for k in ("psnr", "ssim"))
        dT = {k: v("T1", "hp_n72", k) - v("T1", "hp_n36", k) for k in ("psnr", "ssim")}
        dF = {k: v("FLAIR", "hp_n72", k) - v("FLAIR", "hp_n36", k) for k in ("psnr", "ssim")}
        B1 = all(dT[k] < 0 < dF[k] for k in dT)
        B2 = all(dF[k] < 0 < dT[k] for k in dT)
        both_up = all(dT[k] > 0 and dF[k] > 0 for k in dT)
        both_dn = all(dT[k] < 0 and dF[k] < 0 for k in dT)
        row.update(A=int(A), A_mono=int(Ap), B1=int(B1), B2=int(B2), both_up_72=int(both_up), both_down_72=int(both_dn),
                   dT_psnr_72=dT["psnr"], dF_psnr_72=dF["psnr"], dT_ssim_72=dT["ssim"], dF_ssim_72=dF["ssim"],
                   dT_psnr_09=v("T1", "hp_n09", "psnr") - v("T1", "hp_n36", "psnr"),
                   dF_psnr_09=v("FLAIR", "hp_n09", "psnr") - v("FLAIR", "hp_n36", "psnr"))
        rows.append(row)

with open(OUT / "slice_metrics.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

for axis in (0, 1, 2):
    R = [r for r in rows if r["axis"] == axis]
    n = len(R)
    cnt = lambda k: sum(r[k] for r in R)
    full = [r for r in R if r["A_mono"] and r["B1"]]
    print(f"== {ANAT[axis]}: {n} slices with head area >= {MIN_HEAD:.0%} of the largest "
          f"({R[0]['slice']}..{R[-1]['slice']})")
    print(f"   A (0.9M worse, both contrasts, PSNR+SSIM): {cnt('A')}/{n};  A monotone: {cnt('A_mono')}/{n}")
    print(f"   7.2M vs 3.6M (PSNR+SSIM agree): T1 down / FLAIR up {cnt('B1')}/{n};  T1 up / FLAIR down {cnt('B2')}/{n};  "
          f"both up {cnt('both_up_72')}/{n};  both down {cnt('both_down_72')}/{n};  "
          f"metrics disagree {n - cnt('B1') - cnt('B2') - cnt('both_up_72') - cnt('both_down_72')}/{n}")
    print(f"   full pattern (A monotone and T1 down / FLAIR up): {len(full)}/{n}")
    h = next((r for r in R if r["slice"] == HOUSE[axis]), None)
    if h:
        print(f"   house slice {HOUSE[axis]}: A={h['A']} A_mono={h['A_mono']} B1={h['B1']} | 0.9M dPSNR T1 {h['dT_psnr_09']:+.3f} "
              f"FLAIR {h['dF_psnr_09']:+.3f} | 7.2M dPSNR T1 {h['dT_psnr_72']:+.3f} FLAIR {h['dF_psnr_72']:+.3f} | "
              f"7.2M dSSIM T1 {h['dT_ssim_72']:+.5f} FLAIR {h['dF_ssim_72']:+.5f}")
    # clearest trade-off: the smaller of the two 7.2M PSNR moves is as large as possible
    full.sort(key=lambda r: -min(-r["dT_psnr_72"], r["dF_psnr_72"]))
    for r in full[:6]:
        print(f"   candidate {r['slice']:3d}: head_px {r['head_px']} | 0.9M dPSNR T1 {r['dT_psnr_09']:+.3f} FLAIR {r['dF_psnr_09']:+.3f} | "
              f"7.2M dPSNR T1 {r['dT_psnr_72']:+.3f} FLAIR {r['dF_psnr_72']:+.3f} | dSSIM T1 {r['dT_ssim_72']:+.5f} FLAIR {r['dF_ssim_72']:+.5f}")
print("done")
