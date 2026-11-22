#!/usr/bin/env python3
"""The inter-scan pose of M1 (runs on amax, CPU): a rigid registration of the zero-filled T1W and T2W inputs.

    reg.py --af af12 --order il --conds C0 D2 [--check-truth]

Inputs are what the training gets: the undersampled k-space of both contrasts under the training's mask and the
topic's coil maps. The brain mask comes from the zero-filled T1W image. The registration is the first version's
probe (probe_reginit.register: normalised gradient fields, three resolutions, Powell search over 6 parameters).
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
    import probe_reginit
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
        x, cost = probe_reginit.register(t1, t2, brain, return_cost=True)
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
