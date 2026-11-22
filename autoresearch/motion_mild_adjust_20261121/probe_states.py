#!/usr/bin/env python3
"""Probe (amax, CPU; reads only): can a lasting shift inside the T2W scan be seen in the data-consistency residual
of a trained model, echo train by echo train?

Two statistics per echo train, from the model's T2W volume and the measured k-space of a condition:
  rho   the mean residual energy (what M2 uses);
  d     a linearised translation (3 numbers, voxels): the shift of the model volume that explains the train's residual
        best in the least-squares sense, y ~ yhat * (1 - i 2 pi k.d / N).
A lasting shift moves d by the same vector for every train after it, whatever share of the scan it covers; rho only
changes when the model fits one of the two positions better than the other.

    probe_states.py --model mm_af12_il_C0_B0 --order il --cond DTc

No clock value is read or printed.
"""
import argparse
import json

import numpy as np

import mcommon  # noqa: F401  (puts the first version's directory on the path)
import weights
from mcommon import DATA2, RES2, c1

log = lambda *a: print(*a, flush=True)


def line_sums(vol, ksp, csm, mask3, fft, workers=None):
    """Per (ky, kz) line, summed over coils and the readout (axis 0): A0 = sum |yhat|^2, A1 = sum kx |yhat|^2,
    A2 = sum kx^2 |yhat|^2, C0 = sum conj(yhat) y, C1 = sum kx conj(yhat) y, E = sum |y|^2; kx = k / N, centred."""
    nx = vol.shape[0]
    kx = ((np.arange(nx) - nx // 2) / nx)[:, None, None]
    out = {k: 0.0 for k in ("A0", "A1", "A2", "C0", "C1", "E")}
    for c in range(csm.shape[0]):
        yh = fft(csm[c] * vol, workers) * mask3
        y = ksp[c] * mask3
        p = (yh.real ** 2 + yh.imag ** 2).astype(np.float64)
        cy = (np.conj(yh) * y).astype(np.complex128)
        out["A0"] = out["A0"] + p.sum(0)
        out["A1"] = out["A1"] + (kx * p).sum(0)
        out["A2"] = out["A2"] + (kx * kx * p).sum(0)
        out["C0"] = out["C0"] + cy.sum(0)
        out["C1"] = out["C1"] + (kx * cy).sum(0)
        out["E"] = out["E"] + (y.real ** 2 + y.imag ** 2).astype(np.float64).sum(0)
    return out


def train_shift(sums, train_of, n_trains):
    """(rho[n_trains], d[n_trains, 3] in voxels, s) from the line sums. s = the complex scale between the model's
    k-space and the measured one (least squares over all sampled lines)."""
    ny, nz = train_of.shape
    ky = ((np.arange(ny) - ny // 2) / ny)[:, None] * np.ones((1, nz))
    kz = ((np.arange(nz) - nz // 2) / nz)[None, :] * np.ones((ny, 1))
    s = sums["C0"].sum() / sums["A0"].sum()
    q0, q1 = np.imag(np.conj(s) * sums["C0"]), np.imag(np.conj(s) * sums["C1"])
    e = sums["E"] - 2.0 * np.real(np.conj(s) * sums["C0"]) + abs(s) ** 2 * sums["A0"]
    rho, d = np.full(n_trains, np.nan), np.full((n_trains, 3), np.nan)
    for t in range(n_trains):
        m = train_of == t
        if not m.any() or sums["A0"][m].sum() == 0:
            continue
        a0, a1, a2, y, z = sums["A0"][m], sums["A1"][m], sums["A2"][m], ky[m], kz[m]
        G = np.array([[a2.sum(), (y * a1).sum(), (z * a1).sum()],
                      [(y * a1).sum(), (y * y * a0).sum(), (y * z * a0).sum()],
                      [(z * a1).sum(), (y * z * a0).sum(), (z * z * a0).sum()]]) * (2 * np.pi) ** 2 * abs(s) ** 2
        b = -2 * np.pi * np.array([q1[m].sum(), (y * q0[m]).sum(), (z * q0[m]).sum()])
        d[t] = np.linalg.solve(G, b)
        rho[t] = e[m].sum() / (abs(s) ** 2 * a0.sum())
    return rho, d, s


def split_stat(x, k):
    """The change-point statistic of weights.changepoints for one split of a series at train k."""
    a, b = x[:k], x[k:]
    mad = lambda v: 1.4826 * np.median(np.abs(v - np.median(v)))
    pooled = np.sqrt((a.size * mad(a) ** 2 + b.size * mad(b) ** 2) / x.size)
    return float(abs(np.median(a) - np.median(b)) / max(pooled * np.sqrt(1 / a.size + 1 / b.size), 1e-300))


def variants(sums, s, train_of, n, sampled, cont):
    """Candidate per-train energy statistics from the per-line residual energy: which one is level without motion
    (no train far above the rest) and still separates the contaminated trains?"""
    e = sums["E"] - 2.0 * np.real(np.conj(s) * sums["C0"]) + abs(s) ** 2 * sums["A0"]
    ok = sampled & (train_of >= 0)
    r = weights._radius(e.shape)

    def expected(n_shells):
        shells = weights.shell_index(e.shape, n_shells)
        cen, logs = [], []
        for k in np.unique(shells[ok]):
            sel = ok & (shells == k)
            cen.append(float(r[sel].mean()))
            logs.append(np.log(np.median(e[sel])))
        return np.exp(np.interp(r, cen, logs))

    def report(name, rho):
        z = weights.robust_z(np.where(np.isfinite(rho), rho, np.nanmedian(rho)))
        top = np.argsort(-z)[:6]
        a = weights.auc(z, cont) if cont.any() else float("nan")
        log(f"  {name:46s} clean max z {z[~cont].max():5.1f}, clean trains with z > 4: {int((z[~cont] > 4).sum()):2d}, > 6: "
            f"{int((z[~cont] > 6).sum()):2d}; contaminated min z {z[cont].min() if cont.any() else float('nan'):5.1f}, AUC {a:.3f}; "
            f"top {[(int(t), round(float(z[t]), 1)) for t in top]}")

    def per_train(fn, keep):
        out = np.full(n, np.nan)
        for t in range(n):
            m = keep & (train_of == t)
            if m.any():
                out[t] = fn(m)
        return out
    log("energy statistics per train:")
    ex24 = expected(24)
    report("sum(e) / sum(expected), 24 shells (in use)", per_train(lambda m: e[m].sum() / ex24[m].sum(), ok))
    for rmin in (0.05, 0.10):
        report(f"the same without the lines of radius < {rmin}", per_train(lambda m: e[m].sum() / ex24[m].sum(), ok & (r >= rmin)))
    report("median over the lines of e / expected", per_train(lambda m: np.median(e[m] / ex24[m]), ok))
    report("mean over the lines of log(e / expected)", per_train(lambda m: np.mean(np.log(e[m] / ex24[m])), ok))
    ex96 = expected(96)
    report("sum(e) / sum(expected), 96 shells", per_train(lambda m: e[m].sum() / ex96[m].sum(), ok))
    report("mean of log(e / expected), 96 shells", per_train(lambda m: np.mean(np.log(e[m] / ex96[m])), ok))


def reg_states(args):
    """Is a state's zero-filled image (the lines of that state only) enough to register it to the reference state?"""
    import h5py
    from scipy import ndimage as ndi

    import evaluate as ev1
    import geom
    import lesions
    import reg
    import simulate
    cdir = DATA2 / args.order / args.cond
    truth = json.loads((cdir / "motion_truth.json").read_text())
    sch = np.load(cdir / "schedule.npz")
    with h5py.File(c1.MASKS["af12"], "r") as h:
        mask3 = h["mask"][:].astype(np.float32)
    w = simulate.FULL["fft_workers"]
    csm = {c: simulate._read(c1.HYW / f"csm_espirit_{c}.h5", ("map", "csm")) for c in ("T1", "T2")}
    t1 = reg._zero_filled(reg.t1_path(args.order, args.cond), mask3, csm["T1"], w)
    brain = lesions.brain_mask(ndi.gaussian_filter(t1, 1.0), erode=simulate.FULL["brain_erode"],
                               cut_below=simulate.FULL["brain_cut"])
    raw = np.asarray(truth["state_of_train"])
    ref = int(truth["ref_state"])
    order = [ref] + [k for k in sorted(set(raw.tolist())) if k != ref]
    ksp = simulate._read(reg.t2_path(args.order, args.cond), ("kspace", "ksp"))
    s2d = np.where(sch["train_of"] >= 0, raw[np.maximum(sch["train_of"], 0)], -1)
    mags = []
    for k in order:
        sel = (s2d == k).astype(np.float32)
        mags.append(np.abs(simulate.combine(ksp * (mask3 * sel)[None], csm["T2"], w)))
        log(f"state {k}: {int((raw == k).sum())} trains, {int((mask3.max(0) * sel).sum())} sampled lines")
    poses = reg.register_states(mags, brain, say=log)
    est = [dict(theta_deg=np.rad2deg(geom.axis_angle(q.R)).tolist(), t_mm=np.asarray(q.t).tolist()) for q in poses]
    true = [geom.Rigid.from_json(truth["state_poses"][k]) for k in order]
    zero = [dict(theta_deg=[0.0] * 3, t_mm=[0.0] * 3)] * len(order)
    log("true poses:", [dict(theta_deg=np.round(truth["state_poses"][k]["theta_deg"], 3).tolist(),
                             t=np.round(truth["state_poses"][k]["t"], 3).tolist()) for k in order])
    for name, e in (("registration", est), ("no pose (identity)", zero)):
        r = ev1.segment_errors(e, true, 0, brain)
        log(f"{name}: largest rotation error {r['seg_rot_err_deg_max']:.3f} deg, translation {r['seg_t_err_mm_max']:.3f} mm, "
            f"mean brain TRE {r['seg_tre_mean_mm_max']:.3f} mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="experiment whose T2W volume (rec_raw) is the model")
    ap.add_argument("--order", default="il")
    ap.add_argument("--cond", required=True)
    ap.add_argument("--reg", action="store_true", help="only: register the zero-filled images of the TRUE states "
                                                       "(what M3's init \"reg\" does once the states are found)")
    args = ap.parse_args()
    if args.reg:
        return reg_states(args)
    import h5py

    import reg
    import simulate
    cdir = DATA2 / args.order / args.cond
    with np.load(RES2 / f"{args.model}_canon.npz") as z:
        vol = z["rec_raw"].astype(np.complex64)
    with h5py.File(c1.MASKS["af12"], "r") as h:
        mask3 = h["mask"][:].astype(np.float32)
    csm = simulate._read(c1.HYW / "csm_espirit_T2.h5", ("map", "csm"))
    ksp = simulate._read(reg.t2_path(args.order, args.cond), ("kspace", "ksp"))
    sch = np.load(cdir / "schedule.npz")
    truth = json.loads((cdir / "motion_truth.json").read_text())
    n = int(sch["n_trains"])
    sums = line_sums(vol, ksp, csm, mask3, simulate.fft3c, simulate.FULL["fft_workers"])
    rho, d, s = train_shift(sums, sch["train_of"], n)
    corr = abs(sums["C0"].sum()) / np.sqrt(sums["A0"].sum() * sums["E"].sum())
    log(f"model {args.model} on {args.order}/{args.cond}: scale |s| {abs(s):.4f}, correlation of model and data {corr:.4f}")
    state = np.asarray(truth.get("state_of_train", [0] * n))
    cont = np.asarray(truth.get("contaminated", [False] * n), dtype=bool)
    tp = np.array([p["t"] for p in truth.get("train_poses", [dict(t=[0.0, 0.0, 0.0])] * n)])
    for k in sorted(set(state.tolist())):
        m = state == k
        log(f"true state {k}: {int(m.sum())} trains; true t_mm median {np.round(np.median(tp[m], 0), 3).tolist()}; "
            f"d median {np.round(np.nanmedian(d[m], 0), 3).tolist()} voxels, scatter (1.48 MAD) "
            f"{np.round(1.4826 * np.nanmedian(np.abs(d[m] - np.nanmedian(d[m], 0)), 0), 3).tolist()}; "
            f"rho median {np.nanmedian(rho[m]):.5f}")
    b = [int(i) + 1 for i in np.flatnonzero(np.diff(state))]
    for k in b:
        log(f"true boundary at train {k}: split statistic of rho {split_stat(rho, k):.1f}; of d "
            f"{[round(split_stat(d[:, j], k), 1) for j in range(3)]}")
    log(f"change points found (min_frac 0.10, cp_t 8, max 3 states): rho {weights.changepoints(rho, 0.10, 8.0, 3)}; "
        f"d by axis {[weights.changepoints(d[:, j], 0.10, 8.0, 3) for j in range(3)]}")
    zr = weights.robust_z(rho) if hasattr(weights, "robust_z") else None
    if zr is not None:
        log(f"rho as a robust z: contaminated median {np.median(zr[cont]) if cont.any() else float('nan'):.2f}, "
            f"clean median {np.median(zr[~cont]):.2f}, AUC {weights.auc(zr, cont):.3f}")
    variants(sums, s, sch["train_of"], n, mask3.max(0) > 0, cont)
    med = np.nanmedian(d, 0)
    mad = 1.4826 * np.nanmedian(np.abs(d - med), 0)
    far = np.flatnonzero((np.abs(d - med) / mad > 6.0).any(1))
    log(f"d over all trains: median {np.round(med, 3).tolist()}, scatter {np.round(mad, 4).tolist()}, largest distance "
        f"from the median {np.round(np.nanmax(np.abs(d - med), 0), 3).tolist()}; trains more than 6 scatters away on "
        f"an axis: {far.tolist()} (truly contaminated: {np.flatnonzero(cont).tolist()})")
    log("d per train (voxels, axis 0/1/2): " + " ".join(f"{t}:{x[0]:+.2f}/{x[1]:+.2f}/{x[2]:+.2f}" for t, x in enumerate(d)))


if __name__ == "__main__":
    main()
