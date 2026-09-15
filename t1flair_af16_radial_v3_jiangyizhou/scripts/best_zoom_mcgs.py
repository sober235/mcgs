#!/usr/bin/env python3
"""best_zoom_mcgs.py -- the (slice, square window) on which MC-GS leads the competing arms by the most.

Same gauge as best_slice_mcgs.py: head mask |GT| > 0.05 * P99.9 of the volume, MSE over the head voxels inside the
window. Because the PSNR peak is the same for every arm, the lead in dB is 10*log10(MSE_best_competitor / MSE_MC-GS)
and the head count cancels. "Competing arms" = the seven non-GS methods of the row figure; MC-GS is additionally
required to beat GS-single inside the window, so the figure's own single-contrast column cannot look better.
Score = min over the two contrasts of that lead. Windows must be at least --fill of head voxels, and their slice must
pass the same head-area gate as best_slice_mcgs.py.

  python scripts/best_zoom_mcgs.py --axis 0 --win 64 --mcgs_key gs_joint_p3 --single_key gs_single_p3
"""
import argparse, json, re
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--axis", type=int, default=0, choices=[0, 1, 2])
ap.add_argument("--win", type=int, nargs="+", default=[64], help="square window side(s) in voxels")
ap.add_argument("--stride", type=int, default=4)
ap.add_argument("--fill", type=float, default=0.85, help="minimum fraction of the window inside the head")
ap.add_argument("--min_area", type=float, default=0.5, help="slice head-area gate, as in best_slice_mcgs.py")
ap.add_argument("--mcgs_key", default="gs_joint_ar")
ap.add_argument("--single_key", default="gs_single_ar")
ap.add_argument("--top", type=int, default=6)
ap.add_argument("--gap", type=int, default=16, help="minimum distance between listed candidates (slice or in-plane)")
ap.add_argument("--tag", default="")
args = ap.parse_args()

R = Path(__file__).resolve().parents[1]
summary = json.load(open(R / "logs/summary.json"))
red = tuple(a for a in range(3) if a != args.axis)


def arms(c):
    rows = {r["method"]: r for r in summary if r["contrast"] == c}

    def lam_file(row_key, prefix):
        lam = rows.get(row_key, {}).get("lam")
        for p in R.glob(f"recon/{prefix}_lam*_{c}_complex.npy"):
            m = re.fullmatch(re.escape(prefix) + r"_lam([0-9.]+)_" + c + r"_complex\.npy", p.name)
            if lam is not None and m and abs(float(m.group(1)) - float(lam)) < 1e-12:
                return p
        raise FileNotFoundError(f"{prefix} {c} lam={lam}")

    comp = [("CG-SENSE", R / f"recon/cgsense_{c}_complex.npy"),
            ("CS-SENSE", lam_file("cssense_o200", "cssense_o200")),
            ("Joint SENSE", lam_file("jcssense", "jcssense_o200")),
            ("L1-ESPIRiT", lam_file("l1espirit", "l1espirit")),
            ("ZS-SSDU", R / f"recon/zsssdu_{c}_complex.npy"),
            ("3D-DIP", R / f"recon/dip_{c}_complex.npy"),
            ("3D-INR", R / f"recon/inr_{c}_complex.npy")]
    return comp, R / f"recon/{args.single_key}_{c}_mag.npy", R / f"recon/{args.mcgs_key}_{c}_mag.npy"


def integral(x):
    """summed-area table over the two in-plane axes; x is (n_slices, P, Q) after moveaxis."""
    a = np.cumsum(np.cumsum(x, axis=1), axis=2)
    return np.pad(a, ((0, 0), (1, 0), (1, 0)))


def winsum(A, w, st):
    """window sums for every top-left corner on the stride grid; A is the padded integral table."""
    n, P1, Q1 = A.shape
    ys = np.arange(0, P1 - 1 - w + 1, st)
    zs = np.arange(0, Q1 - 1 - w + 1, st)
    return (A[:, ys + w][:, :, zs + w] - A[:, ys][:, :, zs + w] - A[:, ys + w][:, :, zs] + A[:, ys][:, :, zs]), ys, zs


res = {}
for c in ("T1", "FLAIR"):
    gt = np.abs(np.load(R / f"recon/gt_{c}_complex.npy")).astype(np.float32)
    head = (gt > 0.05 * float(np.quantile(gt, 0.999)))
    hm = np.moveaxis(head, args.axis, 0).astype(np.float32)
    area = head.sum(axis=red).astype(np.float64)
    slice_ok = area >= args.min_area * area.max()
    Ih = integral(hm)
    comp, single_p, mcgs_p = arms(c)
    gtm = np.moveaxis(gt, args.axis, 0)
    hmb = hm > 0.5

    def se_table(path):
        v = np.abs(np.load(path)).astype(np.float32)
        s = (np.moveaxis(v, args.axis, 0) - gtm) ** 2
        return integral(s * hmb)

    Imc, Isg = se_table(mcgs_p), se_table(single_p)
    Icomp = [se_table(p) for _, p in comp]
    res[c] = dict(Ih=Ih, Imc=Imc, Isg=Isg, Icomp=Icomp, names=[n for n, _ in comp], slice_ok=slice_ok)

out = []
for w in args.win:
    per_c = {}
    for c in ("T1", "FLAIR"):
        d = res[c]
        N, ys, zs = winsum(d["Ih"], w, args.stride)
        Smc, _, _ = winsum(d["Imc"], w, args.stride)
        Ssg, _, _ = winsum(d["Isg"], w, args.stride)
        Sc = np.stack([winsum(I, w, args.stride)[0] for I in d["Icomp"]])
        best = Sc.min(axis=0)                                   # best competitor = smallest SE
        who = np.array(d["names"])[Sc.argmin(axis=0)]
        with np.errstate(divide="ignore", invalid="ignore"):
            lead = 10 * np.log10(best / Smc)
            over_single = 10 * np.log10(Ssg / Smc)
        ok = (N >= args.fill * w * w) & d["slice_ok"][:, None, None] & (over_single > 0) & np.isfinite(lead)
        per_c[c] = dict(lead=np.where(ok, lead, -np.inf), who=who, ok=ok)
    score = np.minimum(per_c["T1"]["lead"], per_c["FLAIR"]["lead"])
    score = np.where(per_c["T1"]["ok"] & per_c["FLAIR"]["ok"], score, -np.inf)
    flat = np.argsort(-score, axis=None)
    picked = []
    for f in flat:
        i, a, b = np.unravel_index(f, score.shape)
        if not np.isfinite(score[i, a, b]):
            break
        y, z = int(ys[a]), int(zs[b])
        if all(abs(int(i) - s0) >= args.gap or (abs(y - y0) >= args.gap or abs(z - z0) >= args.gap)
               for s0, y0, z0 in picked):
            picked.append((int(i), y, z))
            print(f"win {w:3d} slice {int(i):3d} y {y:3d}..{y+w:3d} z {z:3d}..{z+w:3d}  "
                  f"min-lead {score[i,a,b]:+.3f} dB   T1 {per_c['T1']['lead'][i,a,b]:+.3f} [{per_c['T1']['who'][i,a,b]}]  "
                  f"FLAIR {per_c['FLAIR']['lead'][i,a,b]:+.3f} [{per_c['FLAIR']['who'][i,a,b]}]")
            out.append(dict(win=w, axis=args.axis, slice=int(i), y=y, z=z,
                            min_lead=float(score[i, a, b]),
                            T1=float(per_c["T1"]["lead"][i, a, b]), FLAIR=float(per_c["FLAIR"]["lead"][i, a, b])))
        if len(picked) >= args.top:
            break
p = R / f"logs/best_zoom_mcgs_axis{args.axis}{args.tag}.json"
json.dump(out, open(p, "w"), indent=1)
print("wrote", p)
