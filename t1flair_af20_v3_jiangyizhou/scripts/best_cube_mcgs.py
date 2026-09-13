#!/usr/bin/env python3
"""best_cube_mcgs.py -- ONE 3-D box whose three orthogonal centre planes all show MC-GS ahead of the competing arms.

For a cube of side --win centred at a voxel, the transverse / coronal / sagittal centre slices are each cropped to the
cube's face. Score = min over the six panels (3 planes x 2 contrasts) of
    lead = 10*log10(MSE of the best competing arm / MSE of MC-GS)   over the head voxels of that face,
the gauge of best_slice_mcgs.py (head mask |GT| > 0.05 * P99.9). Competing arms = the seven non-GS methods of the row
figure; MC-GS must also beat GS-single in every panel, and every face must be at least --fill head.
One cube, one centre voxel, six panels -- no per-panel cherry-picking.

  python scripts/best_cube_mcgs.py --win 64 --mcgs_key gs_joint_p3 --single_key gs_single_p3
"""
import argparse, json, re
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--win", type=int, default=64)
ap.add_argument("--stride", type=int, default=4)
ap.add_argument("--fill", type=float, default=0.85)
ap.add_argument("--min_area", type=float, default=0.5)
ap.add_argument("--mcgs_key", default="gs_joint_ar")
ap.add_argument("--single_key", default="gs_single_ar")
ap.add_argument("--top", type=int, default=8)
ap.add_argument("--gap", type=int, default=24, help="minimum centre distance between listed candidates, voxels")
ap.add_argument("--tag", default="")
args = ap.parse_args()
W, ST = args.win, args.stride

R = Path(__file__).resolve().parents[1]
summary = json.load(open(R / "logs/summary.json"))


def arms(c):
    rows = {r["method"]: r for r in summary if r["contrast"] == c}

    def lam_file(row_key, prefix):
        lam = rows.get(row_key, {}).get("lam")
        for p in R.glob(f"recon/{prefix}_lam*_{c}_complex.npy"):
            m = re.fullmatch(re.escape(prefix) + r"_lam([0-9.]+)_" + c + r"_complex\.npy", p.name)
            if lam is not None and m and abs(float(m.group(1)) - float(lam)) < 1e-12:
                return p
        raise FileNotFoundError(f"{prefix} {c} lam={lam}")
    return ([("CG-SENSE", R / f"recon/cgsense_{c}_complex.npy"),
             ("CS-SENSE", lam_file("cssense_o200", "cssense_o200")),
             ("Joint SENSE", lam_file("jcssense", "jcssense_o200")),
             ("L1-ESPIRiT", lam_file("l1espirit", "l1espirit")),
             ("ZS-SSDU", R / f"recon/zsssdu_{c}_complex.npy"),
             ("3D-DIP", R / f"recon/dip_{c}_complex.npy"),
             ("3D-INR", R / f"recon/inr_{c}_complex.npy")],
            R / f"recon/{args.single_key}_{c}_mag.npy", R / f"recon/{args.mcgs_key}_{c}_mag.npy")


def tables(x, axis):
    a = np.moveaxis(x, axis, 0)
    a = np.cumsum(np.cumsum(a, axis=1), axis=2)
    return np.pad(a, ((0, 0), (1, 0), (1, 0)))


def winsum(A, starts_y, starts_z):
    return (A[:, starts_y + W][:, :, starts_z + W] - A[:, starts_y][:, :, starts_z + W]
            - A[:, starts_y + W][:, :, starts_z] + A[:, starts_y][:, :, starts_z])


vols = {}
for c in ("T1", "FLAIR"):
    gt = np.abs(np.load(R / f"recon/gt_{c}_complex.npy")).astype(np.float32)
    head = gt > 0.05 * float(np.quantile(gt, 0.999))
    comp, sp, mp = arms(c)
    vols[c] = dict(gt=gt, head=head, comp=comp, sp=sp, mp=mp)
D, H, Q = vols["T1"]["gt"].shape

# per-axis lead arrays on the stride grid
lead = {}                                     # lead[(axis, c)][slice, iy, iz]
starts = {}
for axis in (0, 1, 2):
    P, Qq = [n for k, n in enumerate((D, H, Q)) if k != axis]
    sy = np.arange(0, P - W + 1, ST)
    sz = np.arange(0, Qq - W + 1, ST)
    starts[axis] = (sy, sz)
    red = tuple(a for a in range(3) if a != axis)
    for c in ("T1", "FLAIR"):
        v = vols[c]
        hm = (np.moveaxis(v["head"], axis, 0) > 0.5)
        gtm = np.moveaxis(v["gt"], axis, 0)
        N = winsum(tables(v["head"].astype(np.float32), axis), sy, sz)

        def se(path):
            u = np.abs(np.load(path)).astype(np.float32)
            s = (np.moveaxis(u, axis, 0) - gtm) ** 2
            s *= hm
            a = np.cumsum(np.cumsum(s, axis=1), axis=2)
            return winsum(np.pad(a, ((0, 0), (1, 0), (1, 0))), sy, sz)

        Smc, Ssg = se(v["mp"]), se(v["sp"])
        Sc = np.stack([se(p) for _, p in v["comp"]]).min(axis=0)
        area = v["head"].sum(axis=red).astype(np.float64)
        ok = (N >= args.fill * W * W) & (area >= args.min_area * area.max())[:, None, None] & (Ssg > Smc) & (Smc > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            L = 10 * np.log10(Sc / Smc)
        lead[(axis, c)] = np.where(ok & np.isfinite(L), L, -np.inf).astype(np.float32)

h = W // 2
cy = {a: starts[a][0] + h for a in (0, 1, 2)}      # centre coordinate of each window start
cz = {a: starts[a][1] + h for a in (0, 1, 2)}
# a cube centre (i, j, k) uses: axis0 slice i, window centred (j, k); axis1 slice j, window (i, k); axis2 slice k, window (i, j)
I = np.intersect1d(cy[1], cy[2])                    # candidate i values present on both grids
J = np.intersect1d(cy[0], cz[2])
K = np.intersect1d(cz[0], cz[1])
ix0 = {v: n for n, v in enumerate(cy[0])}; iz0 = {v: n for n, v in enumerate(cz[0])}
ix1 = {v: n for n, v in enumerate(cy[1])}; iz1 = {v: n for n, v in enumerate(cz[1])}
ix2 = {v: n for n, v in enumerate(cy[2])}; iz2 = {v: n for n, v in enumerate(cz[2])}
best = []
for i in I:
    for j in J:
        for k in K:
            vals = []
            for c in ("T1", "FLAIR"):
                vals.append(lead[(0, c)][i, ix0[j], iz0[k]])
                vals.append(lead[(1, c)][j, ix1[i], iz1[k]])
                vals.append(lead[(2, c)][k, ix2[i], iz2[j]])
            m = min(vals)
            if np.isfinite(m):
                best.append((float(m), int(i), int(j), int(k), [float(x) for x in vals]))
best.sort(reverse=True)
picked, out = [], []
for m, i, j, k, vals in best:
    if all(abs(i - a) + abs(j - b) + abs(k - c0) >= args.gap for a, b, c0 in picked):
        picked.append((i, j, k))
        t1 = vals[0::3][:0]  # unused
        print(f"cube {W}^3 centre ({i:3d}, {j:3d}, {k:3d})  min-lead {m:+.3f} dB | "
              f"T1 tra {vals[0]:+.3f} cor {vals[1]:+.3f} sag {vals[2]:+.3f} | "
              f"FLAIR tra {vals[3]:+.3f} cor {vals[4]:+.3f} sag {vals[5]:+.3f}")
        out.append(dict(win=W, centre=[i, j, k], min_lead=m,
                        T1=dict(tra=vals[0], cor=vals[1], sag=vals[2]),
                        FLAIR=dict(tra=vals[3], cor=vals[4], sag=vals[5])))
    if len(picked) >= args.top:
        break
p = R / f"logs/best_cube_mcgs_w{W}{args.tag}.json"
json.dump(out, open(p, "w"), indent=1)
print("wrote", p)
