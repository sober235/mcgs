#!/usr/bin/env python3
"""Step 0 of the plan (amax, CPU; reads only, no training): can a deformation inside the T2W scan be seen, echo
train by echo train, in the data-consistency residual of a model that was trained on data without motion?

    nprobe.py --cond DNa          (a) do the statistics of M2 separate the deformed trains?
    nprobe.py --cond DNc          (b) is the start of a lasting deformation found as a change point?
    both                          (c) can a coarse control grid represent the true field?

Statistics per echo train, from the model's T2W volume m and the measured k-space y:
  rho   M2's energy statistic (median over the train's lines of residual energy / normal energy, 24 shells);
  d     the second version's linearised translation (3 numbers, voxels);
  c     a linearised deformation on a G x G x G control grid with trilinear weights w_b (3 G^3 numbers, mm):
        y - s yhat ~ s F[S (-sum_b c_b . w_b grad m)], least squares per train with a small ridge.
The model is the second version's C0 / B0 run: a stand-in for a model that has learnt the reference state.
No clock value is read or printed.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from ncommon import DATA3, PROBE3, c1, c2

import probe_states as ps  # noqa: E402  (second version)
import shiftstat  # noqa: E402
import simulate  # noqa: E402
import warp  # noqa: E402
import weights  # noqa: E402

log = lambda *a: print(*a, flush=True)
CP = dict(min_frac=0.10, cp_t=8.0, max_states=3)             # the second version's frozen change-point settings


def hat_weights(shape, G):
    """Trilinear weights of a G x G x G control grid (align_corners, as adapter.MotionAdapter): list of 1-D factor
    arrays per axis, w_b = outer product of one factor per axis."""
    out = []
    for n in shape:
        x = (np.arange(n) + 0.5) / n * (G - 1)
        out.append(np.stack([np.clip(1.0 - np.abs(x - g), 0.0, None) for g in range(G)]).astype(np.float32))
    return out


def fit_grid(u, G, step=2):
    """Least-squares control values (G, G, G, 3) whose trilinear interpolant matches the field u (*shape, 3);
    returns (control values, interpolant on the full grid)."""
    shape = u.shape[:3]
    w = hat_weights(shape, G)
    ws = [a[:, ::step].astype(np.float64) for a in w]
    us = u[::step, ::step, ::step]
    # separable normal equations: (A0 x A1 x A2) c = u  ->  solve axis by axis with the pseudo-inverses
    c = us
    for ax, a in enumerate(ws):
        pinv = np.linalg.pinv(a.T)                           # (G, n_ax)
        c = np.moveaxis(np.tensordot(pinv, np.moveaxis(c, ax, 0), axes=(1, 0)), 0, ax)
    ctrl = full = c
    for ax, a in enumerate(w):
        full = np.moveaxis(np.tensordot(a.T.astype(np.float64), np.moveaxis(full, ax, 0), axes=(1, 0)), 0, ax)
    return ctrl, full


def deformation_stat(vol, ksp, csm, mask2, train_of, n_trains, s, G, workers, ridge=1e-3):
    """c[n_trains, 3 G^3] (mm = voxels of 1 mm): the linearised deformation of every train on the control grid."""
    shape = vol.shape
    w = hat_weights(shape, G)
    grads = [np.gradient(vol, axis=j).astype(np.complex64) for j in range(3)]
    sel = mask2 & (train_of >= 0)
    tr = train_of[sel]
    order = np.argsort(tr, kind="stable")
    starts = np.searchsorted(tr[order], np.arange(n_trains + 1))
    nb = 3 * G ** 3
    gram = np.zeros((n_trains, nb, nb))
    rhs = np.zeros((n_trains, nb))
    for c in range(csm.shape[0]):
        yh = simulate.fft3c(csm[c] * vol, workers)[:, sel]
        r = (ksp[c][:, sel] - s * yh)[:, order]
        basis = np.empty((nb, shape[0], int(sel.sum())), dtype=np.complex64)
        i = 0
        for a in range(G):
            for b in range(G):
                for d in range(G):
                    wb = w[0][a][:, None, None] * w[1][b][None, :, None] * w[2][d][None, None, :]
                    for j in range(3):
                        basis[i] = (-s) * simulate.fft3c(csm[c] * (wb * grads[j]), workers)[:, sel][:, order]
                        i += 1
        for t in range(n_trains):
            a0, a1 = starts[t], starts[t + 1]
            if a1 == a0:
                continue
            bt = basis[:, :, a0:a1].reshape(nb, -1)
            gram[t] += np.real(bt.conj() @ bt.T)
            rhs[t] += np.real(bt.conj() @ r[:, a0:a1].reshape(-1))
        log(f"  coil {c + 1}/{csm.shape[0]}")
    out = np.full((n_trains, nb), np.nan)
    for t in range(n_trains):
        if np.trace(gram[t]) > 0:
            lam = ridge * np.trace(gram[t]) / nb
            out[t] = np.linalg.solve(gram[t] + lam * np.eye(nb), rhs[t])
    return out


def best_component(x, state):
    """Largest split statistic at the true boundary over the components of x[n, k]."""
    b = [int(i) + 1 for i in np.flatnonzero(np.diff(state))]
    return {k: round(max(ps.split_stat(x[:, j][np.isfinite(x[:, j])], k) for j in range(x.shape[1])), 1) for k in b}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mm_af12_il_C0_B0", help="second-version run whose T2W volume is the model")
    ap.add_argument("--order", default="il")
    ap.add_argument("--cond", required=True)
    ap.add_argument("--grids", type=int, nargs="+", default=[2, 3], help="control grids of the deformation statistic")
    ap.add_argument("--fit-grids", type=int, nargs="+", default=[5, 7, 9])
    args = ap.parse_args()
    import h5py
    cdir = DATA3 / args.order / args.cond
    with np.load(c2.RES2 / f"{args.model}_canon.npz") as z:
        vol = z["rec_raw"].astype(np.complex64)
    with h5py.File(c1.MASKS["af12"], "r") as h:
        mask3 = h["mask"][:].astype(np.float32)
    csm = simulate._read(c1.HYW / "csm_espirit_T2.h5", ("map", "csm"))
    ksp = simulate._read(cdir / "ksp_T2.h5", ("kspace", "ksp"))
    sch = np.load(cdir / "schedule.npz")
    truth = json.loads((cdir / "motion_truth.json").read_text())
    brain = np.load(cdir / "brain.npy").astype(bool)
    n, train_of = int(sch["n_trains"]), sch["train_of"]
    w = simulate.FULL["fft_workers"]
    amp = np.asarray(truth["soft_amp"])
    state = np.asarray(truth["nr_state_of_train"])
    deformed = amp != 0.0
    res = dict(cond=args.cond, model=args.model, n_trains=n, ref_train=int(truth["ref_train"]),
               n_deformed=int(deformed.sum()), true_boundaries=[int(i) + 1 for i in np.flatnonzero(np.diff(state))])
    log(f"[nprobe {args.cond}] volume {vol.shape}, coils {csm.shape[0]}, trains {n}, deformed {res['n_deformed']}, "
        f"true boundaries {res['true_boundaries']}, reference train {res['ref_train']}")

    # ---- the true field and (c) its representation on a coarse grid
    f = truth["field"]
    u = warp.rbf_field(f["seed"], f["peak_mm"], brain, f["brain_max_mm"], shape=vol.shape,
                       sigma_mm=simulate.FULL["sigma_mm"]) * float(np.abs(amp).max())
    nu = np.linalg.norm(u, axis=-1)
    res["field"] = dict(peak_mm=float(nu.max()), brain_mean_mm=float(nu[brain].mean()), brain_max_mm=float(nu[brain].max()))
    log(f"true field at full amplitude: peak {nu.max():.2f} mm, in the brain mean {nu[brain].mean():.3f} / max "
        f"{nu[brain].max():.3f} mm; brain mean of the vector {np.round(u[brain].mean(0), 3).tolist()} mm")
    res["grid_fit"] = {}
    for G in args.fit_grids:
        e = np.linalg.norm(fit_grid(u, G)[1] - u, axis=-1)
        res["grid_fit"][G] = dict(brain_mean_mm=float(e[brain].mean()), brain_max_mm=float(e[brain].max()),
                                  all_max_mm=float(e.max()))
        log(f"(c) control grid {G}^3: residual in the brain mean {e[brain].mean():.3f} / max {e[brain].max():.3f} mm, "
            f"whole volume max {e.max():.3f} mm")

    # ---- the second version's statistics
    sums = ps.line_sums(vol, ksp, csm, mask3, simulate.fft3c, w)
    rho0, d, s = ps.train_shift(sums, train_of, n)
    e2d = sums["E"] - 2.0 * np.real(np.conj(s) * sums["C0"]) + abs(s) ** 2 * sums["A0"]
    mask2 = mask3.max(0) > 0 if mask3.ndim == 3 else mask3 > 0
    rho = weights.train_stat(e2d, mask2, train_of, n, weights.shell_index(e2d.shape, 24), kind="median")
    z = weights.robust_z(np.where(np.isfinite(rho), rho, np.nanmedian(rho)))
    res["m2"] = dict(auc=float(weights.auc(z, deformed)), deformed_min_z=float(z[deformed].min()),
                     deformed_median_z=float(np.median(z[deformed])), clean_max_z=float(z[~deformed].max()),
                     deformed_over_z0=int((z[deformed] > 6.0).sum()), clean_over_z0=int((z[~deformed] > 6.0).sum()))
    log(f"(a) M2 statistic (median, 24 shells): AUC {res['m2']['auc']:.3f}; deformed trains z min "
        f"{res['m2']['deformed_min_z']:.1f} / median {res['m2']['deformed_median_z']:.1f}, {res['m2']['deformed_over_z0']} "
        f"of {res['n_deformed']} above z0 = 6; clean trains z max {res['m2']['clean_max_z']:.1f}, "
        f"{res['m2']['clean_over_z0']} above 6")
    res["cp"] = dict(rho=weights.changepoints(rho, **CP), shift=shiftstat.changepoints_vec(d, **CP),
                     at_truth=dict(rho=best_component(rho[:, None], state), shift=best_component(d, state)))
    log(f"(b) change points: rho {res['cp']['rho']}, linearised shift {res['cp']['shift']}; statistic at the true "
        f"boundary: rho {res['cp']['at_truth']['rho']}, shift {res['cp']['at_truth']['shift']}")
    for k in sorted(set(state.tolist())):
        m = state == k
        log(f"  true state {k}: {int(m.sum())} trains, shift d median {np.round(np.nanmedian(d[m], 0), 3).tolist()} voxels")

    # ---- the linearised deformation on a control grid
    res["deform"] = {}
    for G in args.grids:
        log(f"linearised deformation, control grid {G}^3:")
        c = deformation_stat(vol, ksp, csm, mask2, train_of, n, s, G, w)
        nrm = np.linalg.norm(c - np.nanmedian(c[~deformed], 0), axis=1)
        zn = weights.robust_z(np.where(np.isfinite(nrm), nrm, np.nanmedian(nrm)))
        # the truth on the same grid, for the trains of the largest amplitude
        tru = fit_grid(u, G)[0].reshape(-1, 3)
        top = np.abs(amp) == np.abs(amp).max()
        est = (np.nanmedian(c[top], 0) - np.nanmedian(c[~deformed], 0)).reshape(-1, 3) * np.sign(amp[top][0])
        cos = float((est * tru).sum() / max(np.linalg.norm(est) * np.linalg.norm(tru), 1e-30))
        r = dict(norm_auc=float(weights.auc(zn, deformed)), cp=shiftstat.changepoints_vec(c, **CP),
                 at_truth=best_component(c, state), est_norm_mm=float(np.linalg.norm(est, axis=1).max()),
                 true_norm_mm=float(np.linalg.norm(tru, axis=1).max()), cosine_to_truth=cos)
        res["deform"][G] = r
        log(f"  |c - clean median| as a robust z: AUC {r['norm_auc']:.3f}, deformed min z {zn[deformed].min():.1f}, clean "
            f"max z {zn[~deformed].max():.1f}; change points {r['cp']}, statistic at the true boundary {r['at_truth']}")
        log(f"  estimate of the deformed trains against the truth's control values: largest node {r['est_norm_mm']:.3f} "
            f"vs {r['true_norm_mm']:.3f} mm, cosine {cos:.3f}")
    PROBE3.mkdir(parents=True, exist_ok=True)
    out = PROBE3 / f"nprobe_{args.order}_{args.cond}.json"
    out.write_text(json.dumps(res, indent=1))
    art = os.environ.get("MCGS_ARTIFACT_DIR")
    if art:
        (Path(art) / out.name).write_text(out.read_text())
    log(f"wrote {out.name}")


if __name__ == "__main__":
    main()
