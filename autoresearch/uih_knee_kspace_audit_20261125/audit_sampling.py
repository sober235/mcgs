"""Sampling audit of the UIH 3D knee k-space set (16 subjects x T1 / T2 / PD).

Read-only, CPU only. Data: /data2/congcong/data/3D/3D_data20261112/kspace_csm/knee/
(<date>/<subject>/UID_*_{t1,t2,pd}_mx3d_tra_iso_*.mat, MATLAB v7.3, keys kspace and s_maps,
h5py shape (n0, coils, n2, n3)). Subjects are numbered S01..S16 in sorted path order; no
names are printed or written.

For every file:
  1. grid, coil count, number of k-space points that are exactly zero;
  2. readout axis = the axis whose outer half, averaged per phase-encode (PE) position, leaves
     the largest share of PE positions below 1 % of the median ("empty" = not acquired);
  3. empty share of the PE plane, split into: outside / inside the inscribed ellipse, and
     partial-Fourier bands (runs of empty indices at either end of a PE axis, measured in a
     central strip of the other PE axis);
  4. scattered low positions inside the ellipse (below 10 % of the median of the same radius
     bin, PF bands excluded): the trace that a compressed-sensing reconstruction filled points;
  5. even / odd line prediction along each axis (predict a line from its two neighbours with a
     3 x 3 kernel over all coils, least squares, NRMSE on held-out points, measured points
     only): one parity far below the other is the trace of parallel-imaging synthesis;
  6. point-reflection test in the outer 15 % of each PE axis: correlation of |k| with the
     point-reflected |k|; near 1 means the band was filled by conjugate symmetry.

Writes audit.json and audit.tsv to $MCGS_ARTIFACT_DIR (or the working directory).
"""
import glob
import json
import os

import h5py
import numpy as np

ROOT = "/data2/congcong/data/3D/3D_data20261112/kspace_csm/knee"
OUT = os.environ.get("MCGS_ARTIFACT_DIR", ".")
EMPTY_REL = 0.01      # PE position counts as not acquired below 1 % of the median outer-readout energy
LOW_REL = 0.10        # scattered-low threshold, relative to the radius-bin median
N_PAR = 12000         # samples per parity in the line-prediction test
KAX = {0: 0, 1: 2, 2: 3}   # power-array axis -> h5 k-space axis (axis 1 of k is the coil axis)


def centres(P):
    return [int(np.argmax(P.sum(axis=tuple(j for j in range(3) if j != a)))) for a in range(3)]


def outer_mean(P, a, c):
    L = P.shape[a]
    idx = np.where(np.abs(np.arange(L) - c[a]) > 0.25 * L)[0]
    return np.take(P, idx, axis=a).mean(axis=a)


def runs_at_ends(flags):
    lo = 0
    while lo < len(flags) and flags[lo]:
        lo += 1
    hi = 0
    while hi < len(flags) and flags[len(flags) - 1 - hi]:
        hi += 1
    return lo, hi


def parity_test(k, measured, a, c, rng):
    arr = np.moveaxis(k, [KAX[a], 1], [0, 1])          # (L, coils, M, N)
    mm = np.moveaxis(measured, a, 0)                   # (L, M, N)
    L, C, M, N = arr.shape
    out = []
    for par in (0, 1):
        ts = np.array([t for t in range(1, L - 1) if abs(t - c[a]) > 0.15 * L and t % 2 == par])
        n_try = N_PAR * 8
        t = rng.choice(ts, n_try)
        m = rng.integers(1, M - 1, n_try)
        q = rng.integers(1, N - 1, n_try)
        ok = mm[t, m, q] & mm[t - 1, m, q] & mm[t + 1, m, q]
        t, m, q = t[ok][:N_PAR], m[ok][:N_PAR], q[ok][:N_PAR]
        if len(t) < 2000:
            out.append(None)
            continue
        X = np.concatenate([arr[t + dt, :, m + dm, q + dq]
                            for dt in (-1, 1) for dm in (-1, 0, 1) for dq in (-1, 0, 1)], axis=1)
        Y = arr[t, :, m, q]
        h = len(t) // 2
        W = np.linalg.lstsq(X[:h], Y[:h], rcond=None)[0]
        out.append(float(np.linalg.norm(Y[h:] - X[h:] @ W) / np.linalg.norm(Y[h:])))
    return out


def reflection_corr(A, measured, a, c):
    """Correlation of |k| with its point reflection in the outer 15 % at each end of axis a."""
    L = A.shape[a]
    res = []
    for side in ("low", "high"):
        if side == "low":
            band = np.arange(L)[(np.arange(L) - c[a]) < -0.85 * L / 2]
        else:
            band = np.arange(L)[(np.arange(L) - c[a]) > 0.85 * L / 2]
        vals, refl = [], []
        for i in band:
            j = 2 * c[a] - i
            if not 0 <= j < L:
                continue
            s1 = np.take(A, i, axis=a)
            s2 = np.take(A, j, axis=a)
            m1 = np.take(measured, i, axis=a)
            m2 = np.take(measured, j, axis=a)
            # reflect the two remaining axes about their centres
            others = [x for x in range(3) if x != a]
            r2 = s2
            mr2 = m2
            for d, ax in enumerate(others):
                n = A.shape[ax]
                idx = 2 * c[ax] - np.arange(n)
                valid = (idx >= 0) & (idx < n)
                idx = np.where(valid, idx, 0)
                r2 = np.take(r2, idx, axis=d)
                mr2 = np.take(mr2, idx, axis=d) & np.expand_dims(valid, axis=1 - d)
            sel = m1 & mr2
            if sel.sum() > 100:
                vals.append(s1[sel])
                refl.append(r2[sel])
        if vals:
            v = np.concatenate(vals)
            r = np.concatenate(refl)
            res.append(float(np.corrcoef(v, r)[0, 1]))
        else:
            res.append(None)
    return res


def audit(path, rng):
    with h5py.File(path, "r") as f:
        k = f["kspace"][()]
    shape = list(k.shape)
    A = np.sqrt((np.abs(k) ** 2).sum(axis=1))           # coil-combined |k|, axes (n0, n2, n3)
    P = A.astype(np.float64) ** 2
    exact_zero = int((P == 0).sum())
    c = centres(P)

    # readout axis and empty PE positions
    cand = []
    for a in range(3):
        Q = outer_mean(P, a, c)
        cand.append(((Q < EMPTY_REL * np.median(Q)).mean(), a, Q))
    frac_best, r, Q = max(cand, key=lambda z: z[0])
    pe = [x for x in range(3) if x != r]
    E = Q < EMPTY_REL * np.median(Q)                     # 2-D over (pe[0], pe[1])
    u = (np.arange(P.shape[pe[0]]) - c[pe[0]]) / (P.shape[pe[0]] / 2)
    v = (np.arange(P.shape[pe[1]]) - c[pe[1]]) / (P.shape[pe[1]] / 2)
    rho = np.sqrt(u[:, None] ** 2 + v[None, :] ** 2)
    inside = rho <= 1.0

    # partial-Fourier bands: empty runs at the ends of each PE axis, in a central strip of the other
    pf = {}
    strip1 = np.abs(v) <= 0.2
    strip0 = np.abs(u) <= 0.2
    flags0 = E[:, strip1].mean(axis=1) > 0.5
    flags1 = E[strip0, :].mean(axis=0) > 0.5
    pf[pe[0]] = runs_at_ends(flags0)
    pf[pe[1]] = runs_at_ends(flags1)
    band = np.zeros_like(E)
    lo0, hi0 = pf[pe[0]]
    lo1, hi1 = pf[pe[1]]
    if lo0:
        band[:lo0, :] = True
    if hi0:
        band[-hi0:, :] = True
    if lo1:
        band[:, :lo1] = True
    if hi1:
        band[:, -hi1:] = True

    # scattered low positions inside the ellipse, PF bands excluded
    sel = (rho <= 0.95) & ~band
    bins = np.minimum((rho / 0.95 * 5).astype(int), 4)
    low = np.zeros_like(E)
    for b in range(5):
        m = sel & (bins == b)
        if m.sum():
            low[m] = Q[m] < LOW_REL * np.median(Q[m])
    scattered = float(low[sel].mean())

    measured = np.broadcast_to(np.expand_dims(~E, axis=r), P.shape)
    parity = {a: parity_test(k, measured, a, c, rng) for a in range(3)}
    # the readout axis is fully acquired, so its value is the baseline from signal symmetry alone
    reflect = {a: reflection_corr(A, measured, a, c) for a in range(3)}

    return {
        "shape_h5": shape,
        "coils": shape[1],
        "grid": [shape[0], shape[2], shape[3]],
        "exact_zero_points": exact_zero,
        "kspace_centre": c,
        "readout_axis": r,
        "readout_empty_share_by_candidate": [float(z[0]) for z in cand],
        "pe_axes": pe,
        "measured_share_of_pe_plane": float(1 - E.mean()),
        "empty_share_outside_ellipse": float(E[~inside].mean()) if (~inside).any() else None,
        "empty_share_inside_ellipse": float(E[inside].mean()),
        "pf_empty_runs_lo_hi": {str(a): list(pf[a]) for a in pe},
        "scattered_low_share_inside_ellipse": scattered,
        "parity_nrmse_even_odd": {str(a): parity[a] for a in range(3)},
        "reflection_corr_low_high": {str(a): reflect[a] for a in pe},
    }


def main():
    files = sorted(glob.glob(ROOT + "/**/*.mat", recursive=True))
    subs = sorted({os.path.dirname(p) for p in files})
    rng = np.random.default_rng(0)
    rows = []
    for si, d in enumerate(subs, 1):
        for tag in ("t1", "t2", "pd"):
            ps = glob.glob(d + "/*_" + tag + "_mx3d*.mat")
            if len(ps) != 1:
                print("S%02d %s: %d files, skipped" % (si, tag, len(ps)), flush=True)
                continue
            res = audit(ps[0], rng)
            res.update({"subject": "S%02d" % si, "contrast": tag})
            rows.append(res)
            def r3(x):
                return None if x is None else round(x, 3)
            par = res["parity_nrmse_even_odd"]
            print("S%02d %-2s grid %s ro %d | measured %s | empty outside/inside ellipse %s/%s | "
                  "PF runs %s | scattered low %s | parity %s | reflect %s | zeros %d" % (
                      si, tag, res["grid"], res["readout_axis"], r3(res["measured_share_of_pe_plane"]),
                      r3(res["empty_share_outside_ellipse"]), r3(res["empty_share_inside_ellipse"]),
                      res["pf_empty_runs_lo_hi"], r3(res["scattered_low_share_inside_ellipse"]),
                      {a: [r3(x) for x in v] for a, v in par.items()},
                      {a: [r3(x) for x in v] for a, v in res["reflection_corr_low_high"].items()},
                      res["exact_zero_points"]), flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "audit.json"), "w") as f:
        json.dump(rows, f, indent=1)
    keys = ["subject", "contrast", "grid", "coils", "readout_axis", "measured_share_of_pe_plane",
            "empty_share_outside_ellipse", "empty_share_inside_ellipse", "pf_empty_runs_lo_hi",
            "scattered_low_share_inside_ellipse", "parity_nrmse_even_odd", "reflection_corr_low_high",
            "exact_zero_points"]
    with open(os.path.join(OUT, "audit.tsv"), "w") as f:
        f.write("\t".join(keys) + "\n")
        for row in rows:
            f.write("\t".join(json.dumps(row[k]) for k in keys) + "\n")
    print("files audited:", len(rows))


if __name__ == "__main__":
    main()
