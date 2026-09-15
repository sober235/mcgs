#!/usr/bin/env python3
"""slice_trend.py CONTRAST -- per-slice head metrics at AF 12 / 16 / 20 / 24 for every arm of record (read-only).

Extension of the AF 12 / 16 / 20 review's slice_trend.py (2026-09-14) to the AF 24 topic (2026-09-15); definitions unchanged
= the in-panel metrics of fig_three_planes_compact.py: head = |GT| > 0.05 P99.9, slice PSNR over head pixels against max|GT|
of the volume, SSIM map (7x7 uniform window, skimage) averaged over head pixels, NMSE over head pixels.
(a) the figure slices 180/124/76 and 174/110/47 exactly as the figure script computes them;
(b) every slice of every plane via an equivalent vectorised SSIM, checked against (a) on the figure slices at AF 20 and AF 24.
The arm -> volume mapping is read from logs/rescore_af<AF>_<C>.json (an arm without a volume at some AF is skipped there).
Writes logs/slice_trend_<C>.json.
"""
import json, sys, time
import numpy as np
from scipy.ndimage import uniform_filter
from skimage.metrics import structural_similarity

T = "/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_16_20_24_trend_v3_jiangyizhou"
ROOT = "/data0/congcong/code/GS/Results_20260908"
c = sys.argv[1]
AFS = ("12", "16", "20", "24")
# ORDER is what is SHOWN. L1-ESPIRiT stays in the rescore json (data) but is not shown: dropped from the comparison by
# the user on 2026-09-15; L1-SPIRiT, the k-space arm registered the same day, takes a row (as in the AF12_16_20 review).
ORDER = ["MC-GS", "GS-single", "CS-SENSE (converged TV)", "Joint SENSE", "CS-SENSE (35 it)",
         "ZS-SSDU", "L1-SPIRiT", "CG-SENSE", "MC-INR", "3D-DIP", "3D-INR", "ZF"]
SETS = {"180/124/76": (180, 124, 76), "174/110/47": (174, 110, 47)}
files = {af: {x["label"]: x["file"] for x in json.load(open(f"{T}/logs/rescore_af{af}_{c}.json")) if "file" in x} for af in AFS}

gt = np.abs(np.load(f"{ROOT}/T1FLAIR_AF24_v3_jiangyizhou/recon/gt_{c}_complex.npy")).astype(np.float32)
gmax = float(gt.max()); head = gt > 0.05 * float(np.quantile(gt, 0.999)); g64 = gt.astype(np.float64)
C1, C2, CN = (0.01 * gmax) ** 2, (0.03 * gmax) ** 2, 49.0 / 48.0
OTHER = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
cache = {}
for ax in (0, 1, 2):
    size = [7, 7, 7]; size[ax] = 1
    ux = uniform_filter(g64, size); uxx = uniform_filter(g64 * g64, size)
    n = head.sum(axis=OTHER[ax]); area = np.prod([gt.shape[a] for a in OTHER[ax]])
    cache[ax] = dict(size=size, ux=ux, vx=CN * (uxx - ux * ux), n=n, gg=((g64 ** 2) * head).sum(axis=OTHER[ax]),
                     keep=n >= 0.05 * area)

def figure_metrics(r, ax, idx):                         # verbatim logic of fig_three_planes_compact.py
    sl = [slice(None)] * 3; sl[ax] = idx; sl = tuple(sl)
    g2, m2 = g64[sl], head[sl]; rr = r[sl]; e = rr[m2] - g2[m2]
    p = 20 * np.log10(gmax) - 10 * np.log10(float((e * e).mean()))
    _, smap = structural_similarity(g2, rr, data_range=gmax, full=True, win_size=7)
    return p, float(smap[m2].mean()), float((e * e).sum() / (g2[m2] ** 2).sum())

def all_slices(r, ax):
    k = cache[ax]; size = k["size"]
    e2 = ((r - g64) ** 2) * head; sse = e2.sum(axis=OTHER[ax]); n = np.maximum(k["n"], 1)
    psnr = 20 * np.log10(gmax) - 10 * np.log10(np.maximum(sse / n, 1e-30))
    nmse = sse / np.maximum(k["gg"], 1e-30)
    uy = uniform_filter(r, size); uyy = uniform_filter(r * r, size); uxy = uniform_filter(g64 * r, size)
    vy = CN * (uyy - uy * uy); vxy = CN * (uxy - k["ux"] * uy)
    S = ((2 * k["ux"] * uy + C1) * (2 * vxy + C2)) / ((k["ux"] ** 2 + uy ** 2 + C1) * (k["vx"] + vy + C2))
    ssim = (S * head).sum(axis=OTHER[ax]) / n
    return psnr, ssim, nmse

out = dict(contrast=c, figure={}, allslices={}, keep={ax: int(cache[ax]["keep"].sum()) for ax in (0, 1, 2)}, validation=[])
t0 = time.time()
for lab in ORDER:
    for af in AFS:
        if lab not in files[af]:
            continue
        r = np.abs(np.load(f"{ROOT}/T1FLAIR_AF{af}_v3_jiangyizhou/recon/{files[af][lab]}")).astype(np.float64)
        for sname, idxs in SETS.items():
            for ax, idx in enumerate(idxs):
                out["figure"].setdefault(f"{lab}|{sname}|{ax}", {})[af] = figure_metrics(r, ax, idx)
        for ax in (0, 1, 2):
            p, s, nm = all_slices(r, ax)
            out["allslices"].setdefault(f"{lab}|{ax}", {})[af] = dict(psnr=p.tolist(), ssim=s.tolist(), nmse=nm.tolist())
            if af in ("20", "24") and lab in ("MC-GS", "L1-SPIRiT"):
                for sname, idxs in SETS.items():
                    fp, fs, fn = out["figure"][f"{lab}|{sname}|{ax}"][af]
                    out["validation"].append(dict(af=af, lab=lab, ax=ax, idx=idxs[ax], dpsnr=abs(fp - p[idxs[ax]]),
                                                  dssim=abs(fs - s[idxs[ax]]), dnmse=abs(fn - nm[idxs[ax]])))
    print(f"[{c}] {lab} done {time.time()-t0:.0f}s", flush=True)
json.dump(out, open(f"{T}/logs/slice_trend_{c}.json", "w"))
print(f"[{c}] DONE; worst figure-vs-vectorised: " + str({k: max(v[k] for v in out['validation']) for k in ('dpsnr', 'dssim', 'dnmse')}), flush=True)
