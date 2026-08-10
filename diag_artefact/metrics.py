# diag_artefact/metrics.py
"""The five zero-cost metrics. Each is paired with a calibration stake in tests, not with a threshold.

§14's correction is the reason for that discipline: the S index ranked runs exactly as the eye did
while measuring global smoothing rather than staircasing (+14% shape excess, not +758%), and there
was no atom of exactly-zero gradients at all. Ranking correctly is not measuring the named thing.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

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
