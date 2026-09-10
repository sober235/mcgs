"""Evaluation gauge for the free-running data (spec 5.7).

AF is relative to the 3D radial Nyquist count pi/2 * 112^2 = 19,704 spokes. Image metrics reuse the
brain-topic implementations unchanged: whole-volume PSNR/SSIM/NMSE from paired_gs.metrics, and the
masked PSNR/SSIM/HFEN from diag_artefact.headonly with its |ref| > 0.05 P99.9 mask (called "body"
here). The ranking metric is the held-out k-space NRMSE; the caller computes the prediction with BART.
"""
from __future__ import annotations

import numpy as np
import torch

from diag_artefact.headonly import head_hfen, head_mask, head_psnr, head_ssim
from paired_gs.metrics.metrics import evaluate_reconstruction

N_NYQ = int(round(np.pi / 2 * 112 ** 2))


def af(n_spokes: int) -> float:
    return N_NYQ / n_spokes


def heldout_split(ids, frac: float = 0.05, seed: int = 0):
    ids = np.sort(np.asarray(ids))
    ho = np.sort(np.random.default_rng(seed).choice(ids, size=int(round(frac * ids.size)), replace=False))
    return ho, np.setdiff1d(ids, ho)


def shot_subset(pool, target_af: float, n_seg: int = 22) -> np.ndarray:
    """Whole interleaves from `pool`: the j-th chosen interleave is shots[round(j * step)], with the
    step (grid 1e-3) whose spoke count is closest to N_NYQ / target_af; ties go to the smaller step."""
    pool = np.sort(np.asarray(pool))
    shots, counts = np.unique(pool // n_seg, return_counts=True)
    target = N_NYQ / target_af
    s0 = pool.size / target
    best = None
    for step in np.arange(max(1.0, 0.7 * s0), max(1.0, 1.3 * s0) + 1e-9, 1e-3):
        pick = np.unique(np.round(np.arange(0.0, shots.size / step) * step).astype(np.int64))
        pick = pick[pick < shots.size]
        err = abs(int(counts[pick].sum()) - target)
        if best is None or err < best[0]:
            best = (err, pick)
    return pool[np.isin(pool // n_seg, shots[best[1]])]


def norm_scale(ref_mag) -> float:
    return float(np.quantile(np.asarray(ref_mag), 0.999))


def box_mask(shape, box) -> np.ndarray:
    (x0, x1), (y0, y1), (z0, z1) = box
    m = np.zeros(shape, dtype=bool)
    m[x0:x1, y0:y1, z0:z1] = True
    return m


def image_metrics(rec_mag, ref_mag, heart_box=None) -> dict:
    """Both inputs already divided by the reference P99.9 (spec 5.7)."""
    rec = np.ascontiguousarray(rec_mag, dtype=np.float32)
    ref = np.ascontiguousarray(ref_mag, dtype=np.float32)
    met = evaluate_reconstruction(torch.from_numpy(rec), torch.from_numpy(ref), compute_extended=False)
    body, gmax = head_mask(ref), float(ref.max())
    out = {"psnr": float(met["psnr"]), "ssim": float(met["ssim"]), "nmse": float(met["nmse"]),
           "body_psnr": head_psnr(rec, ref, body, gmax), "body_ssim": head_ssim(rec, ref, body, gmax),
           "body_hfen": head_hfen(rec, ref, body, gmax), "body_frac": float(body.mean())}
    if heart_box is not None:
        hb = box_mask(ref.shape, heart_box)
        out["heart_psnr"] = head_psnr(rec, ref, hb, gmax)
        out["heart_ssim"] = head_ssim(rec, ref, hb, gmax)
    return out


def kspace_nrmse(pred, meas) -> float:
    meas = np.asarray(meas)
    return float(np.linalg.norm(np.asarray(pred) - meas) / np.linalg.norm(meas))


def error_map(rec_mag, ref_mag) -> np.ndarray:
    """|rec - ref| / P99.9(|ref|); display with jet on a fixed [0, 0.1] range."""
    return np.abs(np.asarray(rec_mag) - np.asarray(ref_mag)) / norm_scale(ref_mag)


def monotone_worse(values_by_af: dict, higher_is_better: bool) -> bool:
    v = [values_by_af[a] for a in sorted(values_by_af)]
    return all((b < a) if higher_is_better else (b > a) for a, b in zip(v, v[1:]))
