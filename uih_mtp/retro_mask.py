"""Retrospective undersampling masks for the pediatric MTP line (stage-2 spec, section 4).

The house generator (MATLAB `vdPoisMex`) draws on a full rectangle, so a mask that lives inside the
acquired elliptical / partial-Fourier support is built by selection and trimming: take the draw with
the smallest number of points inside the support that still reaches the budget, then drop the points
farthest from k-space centre until exactly the budget remains. Calibration points are protected.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import h5py
import numpy as np
import scipy.io

from .bartio import write_cfl


def load_support(npz_path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Acquired support, known zeros and the ACS range from a stage-1 `masks_<c>.npz`."""
    with np.load(str(npz_path)) as d:
        support = np.asarray(d["envelope"], dtype=bool)
        zeros = np.asarray(d["m_zero"], dtype=bool)
        acs = tuple(int(v) for v in np.asarray(d["acs"]).reshape(-1)[:2])
    if (support & zeros).any() or not (support | zeros).all():
        raise ValueError("envelope and m_zero must partition the phase-encode plane")
    return support, zeros, acs


def calib_square(shape, centre, size: int) -> np.ndarray:
    """The `size` x `size` calibration block centred on `centre` (lower half included, as numpy slices)."""
    m = np.zeros(tuple(int(s) for s in shape), dtype=bool)
    h = int(size) // 2
    m[centre[0] - h:centre[0] - h + int(size), centre[1] - h:centre[1] - h + int(size)] = True
    return m


def load_draw_pool(directory) -> list[tuple[dict, np.ndarray]]:
    """Every `draw_*.mat` written by the MATLAB sweep, as (meta, boolean mask)."""
    pool = []
    for p in sorted(glob.glob(os.path.join(str(directory), "draw_*.mat"))):
        d = scipy.io.loadmat(p)
        m = np.asarray(d["mask"]).astype(bool)
        meta = dict(name=os.path.basename(p), nnz=int(m.sum()))
        for k in ("accelx", "accely", "calib", "ellipse", "pp"):
            if k in d:
                meta[k] = round(float(np.asarray(d[k]).squeeze()), 4)
        pool.append((meta, m))
    return pool


def select_draw(pool, support, target: int) -> tuple[dict, np.ndarray]:
    """The draw with the smallest in-support count that still reaches `target` (ties by name)."""
    cands = []
    for meta, m in pool:
        n = int((m & support).sum())
        meta = dict(meta, in_support=n)
        if n >= int(target):
            cands.append((meta, m))
    if not cands:
        best = max((int((m & support).sum()) for _, m in pool), default=0)
        raise ValueError(f"no draw reaches the target {target} inside the support (best {best}); "
                         "extend the accelx sweep downward")
    return min(cands, key=lambda mm: (mm[0]["in_support"], mm[0]["name"]))


def trim_to_budget(mask, support, target: int, centre, protect=None, seed: int = 0) -> np.ndarray:
    """Restrict `mask` to `support` and drop the outermost points until exactly `target` remain.

    Distance is Euclidean from `centre` in index units; ties are broken by a fixed RNG. Points in
    `protect` are never dropped.
    """
    m = np.asarray(mask, dtype=bool) & np.asarray(support, dtype=bool)
    keep = np.zeros_like(m) if protect is None else (np.asarray(protect, dtype=bool) & m)
    n_excess = int(m.sum()) - int(target)
    if n_excess < 0:
        raise ValueError(f"mask has {int(m.sum())} points inside the support, fewer than the target {target}")
    if n_excess == 0:
        return m
    droppable = m & ~keep
    if int(droppable.sum()) < n_excess:
        raise ValueError(f"cannot drop {n_excess} points without touching the protected {int(keep.sum())}")
    pe, spe = np.nonzero(droppable)
    r = np.hypot(pe - centre[0], spe - centre[1])
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(r.size), -r))       # farthest first, random among equal radii
    drop = order[:n_excess]
    out = m.copy()
    out[pe[drop], spe[drop]] = False
    return out


def broadcast_readout(mask2d, n_ro: int) -> np.ndarray:
    """(PE, SPE) -> (RO, PE, SPE) uint8, constant along the fully sampled readout."""
    m = np.asarray(mask2d, dtype=bool)
    return np.broadcast_to(m[None, :, :], (int(n_ro), *m.shape)).astype(np.uint8)


def write_mask_mat(path, mask2d) -> None:
    """MATLAB v5 file, key `mask`, float32 -- what the framework's external-mask loader reads."""
    scipy.io.savemat(str(path), {"mask": np.asarray(mask2d, dtype=bool).astype(np.float32)}, format="5")


def write_mask_h5(path, mask3d) -> None:
    with h5py.File(str(path), "w") as f:
        f.create_dataset("mask", data=np.asarray(mask3d, dtype=np.uint8), compression="gzip", compression_opts=1)


def write_support_h5(path, support2d) -> None:
    """Key `support`, uint8 -- what `paired_gs.data.known_zero.load_support_map` reads."""
    with h5py.File(str(path), "w") as f:
        f.create_dataset("support", data=np.asarray(support2d, dtype=bool).astype(np.uint8))


def write_pattern_cfl(base: str, mask2d, n_ro: int) -> None:
    """BART sampling pattern for `pics -p`, dims (RO, PE, SPE, 1) in the k-space's own axis order."""
    pat = broadcast_readout(mask2d, n_ro).astype(np.complex64)[..., None]
    Path(base).parent.mkdir(parents=True, exist_ok=True)
    write_cfl(base, pat)
