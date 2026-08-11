# diag_artefact/metrics.py
"""The five zero-cost metrics. Each is paired with a calibration stake in tests, not with a threshold.

§14's correction is the reason for that discipline: the S index ranked runs exactly as the eye did
while measuring global smoothing rather than staircasing (+14% shape excess, not +758%), and there
was no atom of exactly-zero gradients at all. Ranking correctly is not measuring the named thing.
"""
from __future__ import annotations

import numpy as np

from .volumes import air_mask, brain_mask


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


def gradient_ratio_curve(rec: np.ndarray, gt: np.ndarray, wm: np.ndarray,
                         quantiles=(0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80,
                                    0.90, 0.95, 0.99, 0.999)) -> dict:
    """§14b's measurement: the q-quantile of in-WM per-axis gradient magnitude, over the GT's.

    The SHAPE of this curve names the mechanism with no further modelling — flat and below 1 is a
    low-pass (blur); rising and below 1 is selective suppression (flat interiors, edges kept);
    above 1 is under-regularised roughness.

    PORTED FROM THE PUBLISHED MEASUREMENT (`rc_figure.py`, the script that actually printed the
    numbers `.claude/memory.md` §14b quotes — recovered and verified 2026-08-11 after a first,
    prose-only port of this formula failed to reproduce them). Two things that reading missed, and
    which are restored here rather than re-derived:

    - amplitude projection: each candidate is least-squares rescaled onto GT inside the brain
      BEFORE differencing (`v *= (v[brain]*gt[brain]).sum() / (v[brain]**2).sum()`). An
      unregularised reconstruction (e.g. lambda=0, no TV at all) can drift in overall amplitude;
      without correcting for that, every gradient quantile is displaced by the scale factor
      instead of reflecting the SHAPE of the gradient distribution, which is the thing
      `selectivity` is actually meant to read. This is a no-op on the ground truth itself (its own
      least-squares scale onto itself is exactly 1).
    - both-endpoints-in-WM: a finite difference counts only if BOTH voxels forming it are inside
      `wm`, not just the lower-index one — a single-endpoint rule lets boundary-adjacent, high-
      gradient voxels leak into the high quantiles that `selectivity` reads.

    `brain` for the projection is `brain_mask(gt)` (the same published definition `wm`'s brain
    argument is normally built from) — recomputed here rather than threaded through as a parameter,
    so callers keep supplying only the one thing they already have (`wm`). Volumes are taken as
    `np.abs(...).astype(np.float64)`, matching the producer script (real MRI magnitude volumes are
    already non-negative, so the `abs` is a no-op in practice, but it is part of the faithful port).
    """
    rec = np.abs(rec).astype(np.float64)
    gt = np.abs(gt).astype(np.float64)
    brain = brain_mask(gt)

    def per_axis(v):
        denom = float((v[brain] ** 2).sum())
        scale = float((v[brain] * gt[brain]).sum() / denom) if denom > 0 else 1.0
        v = v * scale
        out = []
        for a in range(3):
            s0, s1 = [slice(None)] * 3, [slice(None)] * 3
            s0[a], s1[a] = slice(None, -1), slice(1, None)
            m = wm[tuple(s0)] & wm[tuple(s1)]
            out.append(np.abs(v[tuple(s1)] - v[tuple(s0)])[m])
        return np.concatenate(out)

    gr, gg = per_axis(rec), per_axis(gt)
    return {float(q): float(np.quantile(gr, q) / (np.quantile(gg, q) + 1e-30)) for q in quantiles}


def selectivity(rec: np.ndarray, gt: np.ndarray, wm: np.ndarray) -> float:
    """ratio@q0.999 / ratio@q0.10 — §14b's index, the one that survived its own calibration."""
    c = gradient_ratio_curve(rec, gt, wm, quantiles=(0.10, 0.999))
    return float(c[0.999] / (c[0.10] + 1e-30))


def half_voxel_shift(vol: np.ndarray, axis: int) -> np.ndarray:
    """Positive control for M4: an exact half-voxel Fourier shift along `axis`."""
    v = vol.astype(np.float64)
    n = v.shape[axis]
    k = np.fft.fftfreq(n)
    phase = np.exp(-2j * np.pi * k * 0.5)
    shape = [1, 1, 1]
    shape[axis] = n
    return np.real(np.fft.ifft(np.fft.fft(v, axis=axis) * phase.reshape(shape), axis=axis))


def gradient_correlation(err: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict:
    """M4 — correlation of the residual with the GT gradient, per axis.

    A sub-voxel displacement makes err ~= delta * d(gt)/dx, i.e. sign-paired across a structure;
    that is exactly a non-zero correlation here. Independent noise gives zero.
    """
    out = {}
    for a in range(3):
        g = np.gradient(gt.astype(np.float64), axis=a)
        out[f"axis{a}"] = ncc(err, g, mask)
    return out
