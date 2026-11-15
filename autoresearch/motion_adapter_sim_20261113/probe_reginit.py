#!/usr/bin/env python3
"""READ-ONLY probe, CPU: can a classical rigid registration of the UNDERSAMPLED inputs recover the inter-scan pose?

The adapter does not learn the pose from the data term (README, development record). This probe asks what a
registration done before the training would give: zero-filled T1 against zero-filled T2 of one condition (same mask
as the training), normalised gradient fields (NGF), multi-resolution Powell search over 6 parameters. The estimate
uses measured, undersampled data only; the brain mask comes from the zero-filled T1. The simulated truth is used
afterwards, for the error only. `--full` repeats it with the fully sampled images as a reference for the method.

    probe_reginit.py --af af12 --conds C0 D1 D2 [--full]

Writes probe_reginit_<af>[_full].json to ROOT/probe/ and MCGS_ARTIFACT_DIR. Prints no clock value.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import minimize

import evaluate
import geom
import lesions
import simulate
from common import CONDITIONS, DATA, HYW, MASKS, N, NATIVE_SHIFT_MM, ROOT

log = lambda *a: print(*a, flush=True)
CEN = (np.asarray(N, dtype=np.float64) - 1.0) / 2.0


def rigid(x):
    """x = (theta_deg[3], t_mm[3]) -> geom.Rigid mapping a canonical (T1) point to its T2 position."""
    return geom.Rigid(geom.rotmat(np.deg2rad(x[:3])), np.asarray(x[3:], dtype=np.float64))


def ngf_cost(ga, na, vol_b, r, x, mask, eps_b):
    """-mean NGF between the fixed gradients (ga, na = |ga|^2 + eps^2) and vol_b resampled at T(r)."""
    idx = np.moveaxis(rigid(x).apply(r) + CEN, -1, 0)
    b = ndi.map_coordinates(vol_b, idx, order=1, mode="nearest")
    gb = np.gradient(b)
    nb = sum(g ** 2 for g in gb) + eps_b ** 2
    dot = sum(u * w for u, w in zip(ga, gb))
    return -float(np.mean((dot ** 2 / (na * nb))[mask]))


def register(a, b, mask, factors=(4, 2, 1), x0=None):
    """a: fixed (T1), b: moving (T2), magnitudes on the full grid; mask: where the metric is read."""
    x = np.zeros(6) if x0 is None else np.asarray(x0, dtype=np.float64)
    r_full = geom.grid_mm(N)
    lo, hi = np.argwhere(mask).min(0), np.argwhere(mask).max(0) + 1
    for f in factors:
        sig = 0.5 * f + 0.5
        a_s, b_s = ndi.gaussian_filter(a, sig), ndi.gaussian_filter(b, sig)
        sl = tuple(slice(max(int(l) - 4, 0), int(h) + 4, f) for l, h in zip(lo, hi))
        a_l, m_l, r_l = a_s[sl], mask[sl], r_full[sl]
        ga = np.gradient(a_l)                            # per sample of this level, like the moving image's
        na = sum(g ** 2 for g in ga)
        na = na + (0.1 * float(np.median(np.sqrt(na[m_l])))) ** 2
        nb0 = sum(g ** 2 for g in np.gradient(b_s[sl]))
        eps_b = 0.1 * float(np.median(np.sqrt(nb0[m_l])))
        cost = lambda p: ngf_cost(ga, na, b_s, r_l, p, m_l, eps_b)
        res = minimize(cost, x, method="Powell", options=dict(xtol=0.01 / f * 2, ftol=1e-7, maxfev=1500))
        x = res.x
        log(f"  level {f}: cost {res.fun:.6f} evals {res.nfev} theta_deg {np.round(x[:3], 3).tolist()} "
            f"t_mm {np.round(x[3:], 3).tolist()}")
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--af", default="af12", choices=sorted(MASKS))
    ap.add_argument("--conds", nargs="+", default=["C0", "D1", "D2"])
    ap.add_argument("--full", action="store_true", help="fully sampled images (reference for the method)")
    args = ap.parse_args()
    import h5py
    with h5py.File(MASKS[args.af], "r") as h:
        mask3 = h["mask"][:].astype(np.float32)
    if args.full:
        mask3 = np.ones_like(mask3)
    w = simulate.FULL["fft_workers"]
    csm = {c: simulate._read(HYW / f"csm_espirit_{c}.h5", ("map", "csm")) for c in ("T1", "T2")}
    k1 = simulate._read(HYW / "ksp_T1.h5", ("kspace", "ksp"))
    t1 = np.abs(simulate.combine(k1 * mask3[None], csm["T1"], w))
    del k1
    brain_in = lesions.brain_mask(ndi.gaussian_filter(t1, 1.0), erode=simulate.FULL["brain_erode"],
                                  cut_below=simulate.FULL["brain_cut"])
    brain_true = np.load(DATA / "D1/brain.npy")
    log(f"mask from the input T1: {int(brain_in.sum())} voxels; overlap with the mask of the full T1 (Dice) "
        f"{2.0 * (brain_in & brain_true).sum() / (brain_in.sum() + brain_true.sum()):.3f}")
    out = dict(af=args.af, full=bool(args.full), brain_in_voxels=int(brain_in.sum()), conds={})
    est = {}
    for cond in args.conds:
        spec = CONDITIONS[cond]
        assert not spec["intra"], "inter-scan conditions only"
        path = HYW / "ksp_T2.h5" if cond == "C0" else DATA / cond / "ksp_T2.h5"
        k2 = simulate._read(path, ("kspace", "ksp"))
        t2 = np.abs(simulate.combine(k2 * mask3[None], csm["T2"], w))
        del k2
        log(f"[{cond}]")
        x = register(t1, t2, brain_in)
        est[cond] = rigid(x)
        rec = dict(theta_deg=x[:3].tolist(), t_mm=x[3:].tolist())
        if cond == "C0":
            rec["t_minus_native_mm"] = float(np.linalg.norm(x[3:] - np.asarray(NATIVE_SHIFT_MM)))
            rec["rot_deg"] = float(np.linalg.norm(x[:3]))
        else:
            truth = json.loads((DATA / cond / "motion_truth.json").read_text())
            g = geom.Rigid.from_json(truth["inter"])
            rec.update(evaluate.rigid_motion_errors(est[cond], g, est.get("C0"), brain_true))
            rec["truth_theta_deg"], rec["truth_t_mm"] = truth["inter"]["theta_deg"], truth["inter"]["t"]
            rec["note"] = "rigid part only; a non-rigid field is not estimated" if spec["nonrigid"] else ""
        out["conds"][cond] = rec
        log(json.dumps(rec, indent=1))
    name = f"probe_reginit_{args.af}{'_full' if args.full else ''}.json"
    (ROOT / "probe").mkdir(exist_ok=True)
    (ROOT / "probe" / name).write_text(json.dumps(out, indent=1))
    art = os.environ.get("MCGS_ARTIFACT_DIR")
    if art:
        (Path(art) / name).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
