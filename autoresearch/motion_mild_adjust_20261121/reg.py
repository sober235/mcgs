#!/usr/bin/env python3
"""The inter-scan pose of M1 (runs on amax, CPU): a rigid registration of the zero-filled T1W and T2W inputs.

    reg.py --af af12 --order il --conds C0 D2 [--check-truth]

Inputs are what the training gets: the undersampled k-space of both contrasts under the training's mask and the
topic's coil maps. The brain mask comes from the zero-filled T1W image. The registration follows the first version's
probe (normalised gradient fields, three resolutions, Powell search over 6 parameters) with one change, the cost:
the probe's mean of the squared normalised dot product is not largest at the identity for two identical images
(it also grows when more strong edges of the moving image are pulled into the mask), which shows as a pose error
of a few tenths of a degree that cancels only in differences between two registrations. Here the sum is divided
by the two fields' own norms (register() below), so identical images give exactly no motion.
Writes REG2/<af>_<order>_<cond>.json = {theta_deg, t_mm, cost, brain_in_voxels}. --check-truth prints, afterwards,
the error against the simulated truth; nothing of the truth is written to that file. Prints no clock value.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from mcommon import CONDITIONS2, DATA2, ORDERS, REG2, c1  # first: puts the first version's folder on sys.path

import geom  # noqa: E402

log = lambda *a: print(*a, flush=True)


def t2_path(order, cond):
    spec = CONDITIONS2[cond]
    if spec["v1_data"]:
        return c1.DATA / spec["v1_data"] / "ksp_T2.h5"
    if spec["intra"]:
        return DATA2 / order / cond / "ksp_T2.h5"
    return c1.HYW / "ksp_T2.h5"


def t1_path(order, cond):
    return DATA2 / order / cond / "ksp_T1.h5" if CONDITIONS2[cond]["lesions"] else c1.HYW / "ksp_T1.h5"


def out_path(af, order, cond):
    return REG2 / f"{af}_{order}_{cond}.json"


def record(x, cost, brain_voxels):
    x = np.asarray(x, dtype=np.float64)
    return dict(theta_deg=x[:3].tolist(), t_mm=x[3:].tolist(), cost=float(cost), brain_in_voxels=int(brain_voxels))


def load_pose(path):
    """geom.Rigid of a registration file: canonical (T1) point -> its position in the T2W image."""
    d = json.loads(Path(path).read_text())
    return geom.Rigid(geom.rotmat(np.deg2rad(np.asarray(d["theta_deg"], dtype=np.float64))),
                      np.asarray(d["t_mm"], dtype=np.float64))


def ngfn_cost(ga, ea, vol_b, r, x, mask, eb, cen):
    """-sum (n_a . n_b)^2 / sqrt(sum |n_a|^4  sum |n_b|^4) over the mask, n = g / sqrt(|g|^2 + e^2), with the
    moving image resampled at T(r). By the Cauchy-Schwarz inequality the value is >= -1, reached only when the two
    normalised gradient fields are parallel with one common ratio of lengths: for identical images, at the identity."""
    from scipy import ndimage as ndi
    idx = np.moveaxis(rigid(x).apply(r) + cen, -1, 0)
    b = ndi.map_coordinates(vol_b, idx, order=1, mode="nearest")
    gb = np.gradient(b)
    na, nb = sum(g ** 2 for g in ga), sum(g ** 2 for g in gb)
    dot2 = sum(u * w for u, w in zip(ga, gb)) ** 2 / ((na + ea ** 2) * (nb + eb ** 2))
    pa, pb = (na / (na + ea ** 2))[mask], (nb / (nb + eb ** 2))[mask]
    return -float(dot2[mask].sum() / np.sqrt((pa ** 2).sum() * (pb ** 2).sum()))


def rigid(x):
    """x = (theta_deg[3], t_mm[3]) -> geom.Rigid mapping a point of the fixed image to its position in the moving one."""
    return geom.Rigid(geom.rotmat(np.deg2rad(np.asarray(x[:3], dtype=np.float64))), np.asarray(x[3:], dtype=np.float64))


def register(a, b, mask, factors=(4, 2, 1), x0=None, say=None):
    """a: fixed, b: moving, magnitudes on one grid; mask: where the cost is read. Resolution levels, smoothing, edge
    parameters and the Powell settings are the first version's probe's (probe_reginit.register); the cost is
    ngfn_cost. Returns (x = (theta_deg[3], t_mm[3]), cost of the finest level)."""
    from scipy import ndimage as ndi
    from scipy.optimize import minimize
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    x = np.zeros(6) if x0 is None else np.asarray(x0, dtype=np.float64)
    r_full = geom.grid_mm(a.shape)
    cen = (np.asarray(a.shape, dtype=np.float64) - 1.0) / 2.0
    lo, hi = np.argwhere(mask).min(0), np.argwhere(mask).max(0) + 1
    for f in factors:
        sig = 0.5 * f + 0.5
        a_s, b_s = ndi.gaussian_filter(a, sig), ndi.gaussian_filter(b, sig)
        sl = tuple(slice(max(int(l) - 4, 0), int(h) + 4, f) for l, h in zip(lo, hi))
        a_l, m_l, r_l = a_s[sl], mask[sl], r_full[sl]
        ga = np.gradient(a_l)
        ea = 0.1 * float(np.median(np.sqrt(sum(g ** 2 for g in ga)[m_l])))
        eb = 0.1 * float(np.median(np.sqrt(sum(g ** 2 for g in np.gradient(b_s[sl]))[m_l])))
        res = minimize(lambda p: ngfn_cost(ga, ea, b_s, r_l, p, m_l, eb, cen), x, method="Powell",
                       options=dict(xtol=0.01 / f * 2, ftol=1e-7, maxfev=1500))
        x = res.x
        if say:
            say(f"  level {f}: cost {res.fun:.6f} evals {res.nfev} theta_deg {np.round(x[:3], 3).tolist()} "
                f"t_mm {np.round(x[3:], 3).tolist()}")
    return x, float(res.fun)


def register_states(mags, brain, say=None):
    """Initial poses of the states of M3. mags[k]: magnitude of the zero-filled T2W image made from the lines of
    state k only (state 0 = the reference state); brain: where the metric is read. Each other state is registered
    to the reference state with the same method as the inter-scan pose. Returns [identity, pose_1, ...]:
    pose_k maps a point of the reference state to its position in state k."""
    poses = [geom.Rigid.identity()]
    for k in range(1, len(mags)):
        x, _ = register(mags[0], mags[k], brain)
        poses.append(rigid(x))
        if say:
            say(f"state {k}: theta_deg {np.round(x[:3], 3).tolist()} t_mm {np.round(x[3:], 3).tolist()}")
    return poses


def _zero_filled(path, mask3, csm, workers):
    import simulate
    k = simulate._read(path, ("kspace", "ksp"))
    return np.abs(simulate.combine(k * mask3[None], csm, workers))


def _true_inter(order, cond):
    """Simulated inter-scan transform of a condition (geom.Rigid), or None when it has none."""
    spec = CONDITIONS2[cond]
    p = (c1.DATA / spec["v1_data"] if spec["v1_data"] else DATA2 / order / cond) / "motion_truth.json"
    if not p.exists():
        return None
    inter = json.loads(p.read_text()).get("inter")
    return geom.Rigid.from_json(inter) if inter else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--af", default="af12", choices=sorted(c1.MASKS))
    ap.add_argument("--order", default="il", choices=ORDERS)
    ap.add_argument("--conds", nargs="+", required=True, choices=sorted(CONDITIONS2))
    ap.add_argument("--check-truth", action="store_true")
    args = ap.parse_args()
    import h5py
    from scipy import ndimage as ndi

    import evaluate
    import lesions
    import simulate
    with h5py.File(c1.MASKS[args.af], "r") as h:
        mask3 = h["mask"][:].astype(np.float32)
    w = simulate.FULL["fft_workers"]
    csm = {c: simulate._read(c1.HYW / f"csm_espirit_{c}.h5", ("map", "csm")) for c in ("T1", "T2")}
    REG2.mkdir(parents=True, exist_ok=True)
    t1_cache, est = {}, {}
    conds = sorted(args.conds, key=lambda c: c != "C0")                  # C0 first: the differential check needs it
    for cond in conds:
        p1 = t1_path(args.order, cond)
        if p1 not in t1_cache:
            t1 = _zero_filled(p1, mask3, csm["T1"], w)
            brain = lesions.brain_mask(ndi.gaussian_filter(t1, 1.0), erode=simulate.FULL["brain_erode"],
                                       cut_below=simulate.FULL["brain_cut"])
            t1_cache = {p1: (t1, brain)}
        t1, brain = t1_cache[p1]
        t2 = _zero_filled(t2_path(args.order, cond), mask3, csm["T2"], w)
        log(f"[{cond}] order {args.order}: brain mask from the zero-filled T1W image, {int(brain.sum())} voxels")
        x, cost = register(t1, t2, brain, say=log)
        rec = record(x, cost, brain.sum())
        out = out_path(args.af, args.order, cond)
        out.write_text(json.dumps(rec, indent=1))
        est[cond] = load_pose(out)
        log(f"[{cond}] theta_deg {np.round(x[:3], 3).tolist()} t_mm {np.round(x[3:], 3).tolist()} cost {cost:.6f} -> {out.name}")
        if not args.check_truth:
            continue
        if cond == "C0":
            log(f"[C0] check: |t - native shift| = {np.linalg.norm(x[3:] - np.asarray(c1.NATIVE_SHIFT_MM)):.3f} mm, "
                f"|rotation| = {np.linalg.norm(x[:3]):.3f} deg")
            continue
        g = _true_inter(args.order, cond)
        if g is None:
            log(f"[{cond}] check: no simulated inter-scan transform; difference to the C0 estimate: "
                + (f"t {np.round(est[cond].t - est['C0'].t, 3).tolist()} mm" if "C0" in est else "C0 not in this call"))
            continue
        vdir = c1.DATA / (CONDITIONS2[cond]["v1_data"] or "D1")
        brain_true = np.load(vdir / "brain.npy") if (vdir / "brain.npy").exists() else np.load(DATA2 / "il/C0/brain.npy")
        e = evaluate.rigid_motion_errors(est[cond], g, est.get("C0"), brain_true.astype(bool))
        log(f"[{cond}] check against the simulated rigid part: " + json.dumps({k: round(float(v), 4) for k, v in e.items()}))


if __name__ == "__main__":
    main()
