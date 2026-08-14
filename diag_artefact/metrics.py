# diag_artefact/metrics.py
"""The five zero-cost metrics. Each is paired with a calibration stake in tests, not with a threshold.

§14's correction is the reason for that discipline: the S index ranked runs exactly as the eye did
while measuring global smoothing rather than staircasing (+14% shape excess, not +758%), and there
was no atom of exactly-zero gradients at all. Ranking correctly is not measuring the named thing.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .volumes import air_mask, brain_mask, wm_mask


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
                     thresholds=(0.005, 0.01, 0.02, 0.04), dilate: int = 8) -> list:
    """M3 — E_air(rec)/E_air(gt) reported as a CURVE over how air is thresholded, never a point.

    Air is the one region with no suppression on this pipeline: ecalib -c 0.0 leaves SoS==1 over
    the whole FOV, and the GT is built by the same conjugate-SENSE combine, so the GT's air already
    carries full-amplitude noise. A ratio near 1 therefore means "the grain is in the GT too".

    TWO FIXES from the final whole-branch review (I3), both about making the sweep spec §7 asks for
    ("report the ratio as a curve over thresholds, never a single point") actually sweep something:

    - `peak` is the volume's own maximum, matching `volumes.brain_mask` (the published definition,
      `gt > 0.05 * gt.max()`), `provenance._render_error_panel` and `report.adjudication_figure`.
      It used to be `percentile(gt, 99.9)`, a second, quieter convention living in this one line.
    - the default thresholds sit BELOW the brain mask's own 0.05 cut. Above it the air mask is
      already fully included, so the old (0.02, 0.05, 0.10, 0.15) evaluated three of its four
      points on voxel sets within 0.25% of each other — a sweep in name only. On the shipped T1
      volume the defaults here span 2.6M to 9.2M voxels and a ratio range of 2.95 to 0.48.

    Note the ratio's own selection bias, which is exactly why the curve is reported rather than a
    point: air is selected by thresholding the GT, so a tighter threshold keeps only voxels where
    the GT happens to be small, shrinking the denominator without constraining the numerator. The
    ratio therefore RISES as the threshold tightens for any candidate, including the pure-aliasing
    zero-fill reference — which is why P4 is read against that reference rather than against 1.
    """
    base_air = air_mask(gt, brain, dilate=dilate)
    peak = float(gt.max())
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


def spatial_edge_error_autocorrelation(
    rec: np.ndarray,
    gt: np.ndarray,
    wm: np.ndarray | None = None,
    quantiles=(0.85, 0.90, 0.95),
    sigma: float = 0.5,
    max_lag: int = 4,
    flat_quantile: float = 0.50,
) -> dict:
    """Spatial coherence of reconstruction-only edge error on strong and flat GT regions.

    Patch/fragment artefacts are spatially coherent false edges.  This metric first removes a
    reconstruction-wide amplitude gauge, then measures the magnitude of the 3-D Sobel-gradient
    vector error after a small, fixed Gaussian prefilter.  Its autocorrelation is evaluated only
    on the upper tail of *GT* edge strength inside white matter, so a candidate cannot choose its
    own favourable support.  A complementary lower-tail reading catches false edges confined to
    flat GT regions; it also reports their RMS magnitude so incoherent errors remain visible.

    The amplitude projection and autocorrelation deliberately match the calibrated experiment
    convention:

    * ``rec`` is least-squares projected onto ``gt`` inside WM;
    * inputs and filtering use float32, Gaussian ``sigma=0.5`` and reflect boundaries;
    * every spatial-axis profile reuses :func:`autocorr_profile`, whose denominator is the total
      centred error energy over the whole threshold mask (it is not pairwise Pearson correlation);
    * ``score`` is the mean positive autocorrelation over axes and lags 1..``max_lag``.  Negative
      lobes are retained in ``rho_by_axis_lag`` but cannot cancel positive spatial coherence.

    The complete q={.85,.90,.95} strong-edge profile and the q=.50 flat-region reading are
    returned.  q=.90 is the primary strong-edge reading; lower is better, but it must be gated per
    contrast together with sharpness/grain/HFEN because blur and noise can game any edge-coherence
    scalar in isolation.
    """
    r = np.abs(np.asarray(rec)).astype(np.float32)
    g = np.abs(np.asarray(gt)).astype(np.float32)
    if r.shape != g.shape or r.ndim != 3:
        raise ValueError(f"rec and gt must be same-shape 3-D arrays, got {r.shape} and {g.shape}")
    if not np.isfinite(r).all() or not np.isfinite(g).all():
        raise ValueError("rec and gt must contain only finite values")
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    if isinstance(max_lag, bool) or not isinstance(max_lag, (int, np.integer)) or max_lag < 1:
        raise ValueError(f"max_lag must be a positive integer, got {max_lag!r}")
    if max_lag >= min(r.shape):
        raise ValueError(f"max_lag={max_lag} must be smaller than every volume dimension {r.shape}")

    qs = tuple(float(q) for q in quantiles)
    if 0.90 not in qs:
        raise ValueError("quantiles must include the primary q=0.90")
    if len(set(qs)) != len(qs) or any(not 0.0 < q < 1.0 for q in qs):
        raise ValueError(f"quantiles must be unique and strictly between 0 and 1, got {qs}")
    fq = float(flat_quantile)
    if not 0.0 < fq < 1.0:
        raise ValueError(f"flat_quantile must be strictly between 0 and 1, got {fq}")

    if wm is None:
        brain = brain_mask(g)
        w = wm_mask(g, brain)
    else:
        w = np.asarray(wm, dtype=bool)
        if w.shape != g.shape:
            raise ValueError(f"wm must match volume shape {g.shape}, got {w.shape}")
    if not w.any():
        raise ValueError("wm mask is empty")

    # Accumulate in float64 for a stable projection while preserving the calibrated float32
    # filter/Sobel path.  WM (not the whole brain) is the fixed projection region used by the
    # calibration anchors.
    rw = r[w].astype(np.float64)
    gw = g[w].astype(np.float64)
    projection_denom = float(rw @ rw)
    if projection_denom <= 0.0:
        raise ValueError("rec has zero energy inside wm; amplitude projection is undefined")
    scale = float((rw @ gw) / projection_denom)

    rs = ndimage.gaussian_filter(
        (r * scale).astype(np.float32), sigma=float(sigma), mode="reflect"
    )
    gs = ndimage.gaussian_filter(g, sigma=float(sigma), mode="reflect")
    dr = [ndimage.sobel(rs, axis=axis, mode="reflect") for axis in range(3)]
    dg = [ndimage.sobel(gs, axis=axis, mode="reflect") for axis in range(3)]
    gt_edge_magnitude = np.sqrt(sum(component * component for component in dg))
    edge_error_magnitude = np.sqrt(
        sum((rec_component - gt_component) ** 2 for rec_component, gt_component in zip(dr, dg))
    )

    rows = {}
    gt_edges_in_wm = gt_edge_magnitude[w]
    for q in qs:
        threshold = float(np.quantile(gt_edges_in_wm, q))
        support = w & (gt_edge_magnitude >= threshold)
        rho = np.empty((3, max_lag), dtype=np.float64)
        for axis in range(3):
            profile = autocorr_profile(
                np.moveaxis(edge_error_magnitude, axis, 0),
                np.moveaxis(support, axis, 0),
                max_lag=max_lag,
            )
            rho[axis] = profile[1:]
        if not np.isfinite(rho).all():
            raise ValueError(
                f"q={q} strong-edge support has insufficient adjacent voxels for all profiles"
            )
        rows[str(q)] = {
            "score": float(np.maximum(rho, 0.0).mean()),
            "rho_by_axis_lag": rho.tolist(),
            "n_voxels": int(support.sum()),
            "gt_edge_threshold": threshold,
        }

    flat_threshold = float(np.quantile(gt_edges_in_wm, fq))
    flat_support = w & (gt_edge_magnitude <= flat_threshold)
    flat_rho = np.empty((3, max_lag), dtype=np.float64)
    for axis in range(3):
        profile = autocorr_profile(
            np.moveaxis(edge_error_magnitude, axis, 0),
            np.moveaxis(flat_support, axis, 0),
            max_lag=max_lag,
        )
        flat_rho[axis] = profile[1:]
    if not np.isfinite(flat_rho).all():
        raise ValueError(
            f"q={fq} flat-region support has insufficient adjacent voxels for all profiles"
        )
    flat_error = edge_error_magnitude[flat_support].astype(np.float64)
    flat_region = {
        "quantile": fq,
        "gt_edge_threshold": flat_threshold,
        "n_voxels": int(flat_support.sum()),
        "rms": float(np.sqrt(np.mean(flat_error ** 2))),
        "score": float(np.maximum(flat_rho, 0.0).mean()),
        "rho_by_axis_lag": flat_rho.tolist(),
    }

    return {
        "quantiles": rows,
        "flat_region": flat_region,
        "primary_q": 0.90,
        "primary_score": rows[str(0.90)]["score"],
        "scale": scale,
        "sigma": float(sigma),
        "max_lag": int(max_lag),
    }


def half_voxel_shift(vol: np.ndarray, axis: int) -> np.ndarray:
    """Positive control for M4: an exact half-voxel Fourier shift along `axis`.

    Note: on even-length axes the Nyquist component is purely imaginary and discarded by np.real(),
    leaving a ~0.45% residual; this is standard FFT-based half-sample shift behaviour and negligible
    at the calibration margin (0.5 vs 0.02 correlation).
    """
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
