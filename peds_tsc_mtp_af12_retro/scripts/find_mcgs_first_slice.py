#!/usr/bin/env python3
"""find_mcgs_first_slice.py -- every slice on which one arm leads every other arm.

Same per-slice gauge the compact figure prints: head mask |GT| > 0.05 * P99.9(volume), PSNR at the
volume's max |GT|, SSIM as the 7x7 uniform-window map averaged over the head pixels, NMSE over the
head pixels. PSNR and NMSE are screened first over every slice of every plane, and SSIM is computed
only for the slices that survive, because the SSIM map is the expensive part.

  ./scripts/py scripts/find_mcgs_first_slice.py --contrast T2s --key gs_joint_p3f

This SELECTS a slice on the metric being reported. That is legitimate for illustration and not
evidence; the output therefore also prints how many slices the arm leads, out of how many.
"""
import argparse, json, re
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity

R = Path(__file__).resolve().parents[1]
PLANES = [(2, "transverse"), (0, "coronal"), (1, "sagittal")]

ap = argparse.ArgumentParser()
ap.add_argument("--contrast", required=True, choices=["T1w", "T2s"])
ap.add_argument("--key", default="gs_joint_p3f")
ap.add_argument("--min_head", type=float, default=0.15, help="skip slices whose head fraction is below this")
args = ap.parse_args()
c = args.contrast

rows = json.load(open(R / "logs/rescore.json"))["rows"]
def lam_of(method):
    for r in rows.values():
        if r["contrast"] == c and r["method"] == method:
            return r.get("lam")
    return None
def lam_file(method, prefix):
    lam = lam_of(method)
    if lam is None:
        return None
    for p in R.glob(f"recon/{prefix}_lam*_{c}_complex.npy"):
        m = re.fullmatch(re.escape(prefix) + r"_lam([0-9.]+)_" + c + r"_complex\.npy", p.name)
        if m and abs(float(m.group(1)) - float(lam)) < 1e-12:
            return p
    return None

cands = [("cgsense", R / f"recon/cgsense_{c}_complex.npy"),
         ("cssense_o200", lam_file("cssense_o200", "cssense_o200")),
         ("jcssense", lam_file("jcssense", "jcssense_o200")),
         ("l1espirit", lam_file("l1espirit", "l1espirit")),
         ("zsssdu", R / f"recon/zsssdu_{c}_complex.npy"),
         ("dip", R / f"recon/dip_{c}_complex.npy"),
         ("inr", R / f"recon/inr_{c}_complex.npy"),
         ("mcinr", R / f"recon/mcinr_{c}_complex.npy")]
for k in ("gs_single_ar", "gs_joint_ar", "gs_single_p3", "gs_joint_p3", "gs_single_p3f", "gs_joint_p3f"):
    p = R / f"recon/{k}_{c}_mag.npy"
    cands.append((k, p if p.exists() else R / f"recon/{k}_{c}_complex.npy"))
arms = {k: p for k, p in cands if p is not None and Path(p).exists()}
assert args.key in arms, f"{args.key} has no volume for {c}"

gt = np.abs(np.load(R / f"recon/gt_{c}_complex.npy")).astype(np.float64)
gmax = float(gt.max())
head = gt > 0.05 * float(np.quantile(gt, 0.999))
vols = {k: np.abs(np.load(p)).astype(np.float64) for k, p in arms.items()}
print(f"{c}: {len(vols)} arms, key = {args.key}")

survivors, considered = [], 0
for axis, pname in PLANES:
    n = gt.shape[axis]
    for idx in range(n):
        sl = [slice(None)] * 3; sl[axis] = idx; sl = tuple(sl)
        m2 = head[sl]
        if m2.mean() < args.min_head:
            continue
        considered += 1
        g = gt[sl][m2]
        den = float((g * g).sum())
        ps = {}
        for k, v in vols.items():
            e = v[sl][m2] - g
            ps[k] = float((e * e).mean())
        if min(ps, key=ps.get) != args.key:          # PSNR and NMSE share the ranking on one slice
            continue
        survivors.append((axis, pname, idx, ps))

print(f"{considered} slices with at least {args.min_head:.0%} head; "
      f"{len(survivors)} where {args.key} leads on PSNR/NMSE")

best = []
for axis, pname, idx, ps in survivors:
    sl = [slice(None)] * 3; sl[axis] = idx; sl = tuple(sl)
    m2 = head[sl]; g2 = gt[sl]
    ss = {}
    for k, v in vols.items():
        _, smap = structural_similarity(g2, v[sl], data_range=gmax, full=True, win_size=7)
        ss[k] = float(smap[m2].mean())
    if max(ss, key=ss.get) != args.key:
        continue
    others_ps = min(v for k, v in ps.items() if k != args.key)
    d_psnr = 10 * np.log10(others_ps / ps[args.key])
    d_ssim = ss[args.key] - max(v for k, v in ss.items() if k != args.key)
    best.append((d_psnr, d_ssim, pname, axis, idx,
                 20 * np.log10(gmax) - 10 * np.log10(ps[args.key]), ss[args.key]))

best.sort(reverse=True)
print(f"{len(best)} slices where it leads PSNR, SSIM and NMSE\n")
for d_psnr, d_ssim, pname, axis, idx, psnr, ssim in best[:10]:
    print(f"  {pname:11s} {idx:3d} (axis {axis})   PSNR {psnr:7.3f} dB  (+{d_psnr:.3f} over the best other arm)"
          f"   SSIM {ssim:.4f} (+{d_ssim:.4f})")
