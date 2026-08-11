# diag_artefact/metrics.py
"""The five zero-cost metrics. Each is paired with a calibration stake in tests, not with a threshold.

§14's correction is the reason for that discipline: the S index ranked runs exactly as the eye did
while measuring global smoothing rather than staircasing (+14% shape excess, not +758%), and there
was no atom of exactly-zero gradients at all. Ranking correctly is not measuring the named thing.
"""
from __future__ import annotations

import numpy as np

from .volumes import air_mask


def psf_spread(mask2d: np.ndarray, mainlobe_radius: int = 2) -> dict:
    """M1 — is the mask's point-spread energy outside the mainlobe DISCRETE or DIFFUSE?

    Discrete replicas are a uniform-undersampling phenomenon; a blue-noise Poisson-disc pattern
    spreads its sidelobe energy, which is why its aliasing looks like snow rather than ghosts.
    """
    m = np.asarray(mask2d, dtype=np.float64)
    psf = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(m))))
    psf /= psf.max() + 1e-12
    c0, c1 = psf.shape[0] // 2, psf.shape[1] // 2
    gg0, gg1 = np.ogrid[: psf.shape[0], : psf.shape[1]]
    outside = ((gg0 - c0) ** 2 + (gg1 - c1) ** 2) > mainlobe_radius ** 2
    side = psf[outside]
    return {"peak_sidelobe": float(side.max()),
            "sidelobe_rms": float(np.sqrt((side ** 2).mean())),
            "mainlobe_radius": int(mainlobe_radius)}


def air_energy_curve(rec: np.ndarray, gt: np.ndarray, brain: np.ndarray,
                     thresholds=(0.02, 0.05, 0.10, 0.15), dilate: int = 8) -> list:
    """M3 — E_air(rec)/E_air(gt) reported as a CURVE over how air is thresholded, never a point.

    Air is the one region with no suppression on this pipeline: ecalib -c 0.0 leaves SoS==1 over
    the whole FOV, and the GT is built by the same conjugate-SENSE combine, so the GT's air already
    carries full-amplitude noise. A ratio near 1 therefore means "the grain is in the GT too".
    """
    base_air = air_mask(gt, brain, dilate=dilate)
    peak = float(np.percentile(gt, 99.9))
    rows = []
    for t in thresholds:
        a = base_air & (gt < t * peak)
        n = int(a.sum())
        if n == 0:
            rows.append({"threshold": float(t), "ratio_rec": float("nan"), "n_voxels": 0})
            continue
        e_rec = float((rec[a].astype(np.float64) ** 2).sum())
        e_gt = float((gt[a].astype(np.float64) ** 2).sum())
        rows.append({"threshold": float(t),
                     "ratio_rec": float(e_rec / (e_gt + 1e-30)),
                     "n_voxels": n})
    return rows


def ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Normalised cross-correlation of two fields inside `mask`."""
    x = a[mask].astype(np.float64)
    y = b[mask].astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30))


def autocorr_profile(err: np.ndarray, mask: np.ndarray, max_lag: int = 8) -> np.ndarray:
    """Autocorrelation vs lag along axis 0 of the PE plane, computed inside `mask` only.

    Index 0 is 1.0 by construction. A near-delta profile means the residual carries no long-range
    structure, i.e. it is incoherent rather than a coherent aliasing pattern.
    """
    e = np.where(mask, err.astype(np.float64), 0.0)
    e = e - (e[mask].mean() if mask.any() else 0.0) * mask
    denom = float((e[mask] ** 2).sum()) + 1e-30
    out = np.ones(max_lag + 1, dtype=np.float64)
    for lag in range(1, max_lag + 1):
        both = mask[lag:] & mask[:-lag]
        if not both.any():
            out[lag] = np.nan
            continue
        out[lag] = float((e[lag:][both] * e[:-lag][both]).sum() / denom)
    return out


def surrogate_band(err: np.ndarray, mask: np.ndarray, n: int = 64, seed: int = 0,
                   max_lag: int = 8, lo_pct: float = 2.5, hi_pct: float = 97.5):
    """The delta reference: permute the residual WITHIN `mask` n times and take the profile band.

    Permutation destroys every spatial relation while preserving the histogram exactly, so the
    resulting band is what "no structure" looks like for this specific residual — no threshold is
    invented. Default n=64 removes the estimation-noise component (band converges to true percentiles),
    but not the exchangeability component: the original profile's per-lag inside-probability still
    converges to 95%, not 100%, so single-realization tests must measure pooled rates, not all-lags-pass.
    """
    rng = np.random.default_rng(seed)
    vals = err[mask].astype(np.float64)
    profiles = np.empty((n, max_lag + 1), dtype=np.float64)
    scratch = np.zeros_like(err, dtype=np.float64)
    for i in range(n):
        scratch[mask] = rng.permutation(vals)
        profiles[i] = autocorr_profile(scratch, mask, max_lag=max_lag)
    return (np.nanpercentile(profiles, lo_pct, axis=0),
            np.nanpercentile(profiles, hi_pct, axis=0))
