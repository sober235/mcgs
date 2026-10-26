"""Head-only image metrics for the AF12->AF10 gates (spec 2026-09-05 §2.1).

Definitions are frozen here because the reference values in gates.json and every candidate are
scored by this code; a change here silently moves the goalposts.

    head      = |GT| > 0.05 * P99.9(|GT|)            (same as the 2026-09-04 audit probe1)
    head_psnr = 20 log10(max|GT|) - 10 log10( mean over head of (rec - gt)^2 )   -- NO amplitude rescaling
    head_ssim = mean over head of the skimage SSIM map (7^3 uniform window, K1 .01, K2 .03, data_range max|GT|)
    head_hfen = || (LoG rec - LoG gt)[head] ||_2 / || LoG gt [head] ||_2, LoG sigma 1.5, 15 taps, DC-removed
    N         = rms(hp(rec)[wm]) / rms(hp(gt)[wm]),  hp(v) = v - gaussian_filter(v, 4),  wm = memory §14c mask
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.metrics import structural_similarity

from .volumes import brain_mask, wm_mask

HEAD_REL = 0.05
LOG_SIGMA = 1.5
LOG_SIZE = 15
HP_SIGMA = 4.0


def head_mask(gt: np.ndarray) -> np.ndarray:
    p999 = float(np.quantile(gt, 0.999))
    return gt > HEAD_REL * p999


def log_kernel(sigma: float = LOG_SIGMA, size: int = LOG_SIZE) -> np.ndarray:
    """Transcribed from paired_gs/metrics/metrics.py::_log3d_kernel (DC-removed 3-D LoG). Kept in
    sync by test_log_kernel_matches_paired_gs_definition."""
    if size % 2 == 0 or size < 3:
        raise ValueError(f"size must be odd and >= 3, got {size}")
    radius = size // 2
    ax = np.arange(-radius, radius + 1, dtype=np.float64)
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    r2 = x * x + y * y + z * z
    s2 = sigma * sigma
    gauss = np.exp(-r2 / (2.0 * s2))
    gauss /= gauss.sum()
    log = ((r2 / s2) - 3.0) / s2 * gauss
    log -= log.mean()
    return log


def head_psnr(rec: np.ndarray, gt: np.ndarray, head: np.ndarray, gmax: float) -> float:
    mse = float(np.mean((rec[head].astype(np.float64) - gt[head].astype(np.float64)) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(20.0 * np.log10(gmax) - 10.0 * np.log10(mse))


def head_ssim(rec: np.ndarray, gt: np.ndarray, head: np.ndarray, gmax: float) -> float:
    _, smap = structural_similarity(rec, gt, data_range=gmax, full=True)
    return float(smap[head].mean())


def head_hfen(rec: np.ndarray, gt: np.ndarray, head: np.ndarray, gmax: float) -> float:
    k = log_kernel()
    lr = ndimage.convolve(rec.astype(np.float64) / gmax, k, mode="reflect")
    lg = ndimage.convolve(gt.astype(np.float64) / gmax, k, mode="reflect")
    den = float(np.linalg.norm(lg[head]))
    if den == 0.0:
        return float("inf")
    return float(np.linalg.norm((lr - lg)[head]) / den)


def grain_ratio(rec: np.ndarray, gt: np.ndarray) -> float:
    brain = brain_mask(gt)
    wm = wm_mask(gt, brain)
    if not wm.any():
        return float("nan")

    def hp(v):
        v = v.astype(np.float32)
        return v - ndimage.gaussian_filter(v, HP_SIGMA)

    r = float(np.sqrt(np.mean(hp(rec)[wm] ** 2)))
    g = float(np.sqrt(np.mean(hp(gt)[wm] ** 2)))
    if g == 0.0:
        return float("inf")
    return r / g


def score_headonly(rec: np.ndarray, gt: np.ndarray) -> dict:
    rec = np.asarray(rec, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    head = head_mask(gt)
    gmax = float(gt.max())
    return {
        "head_psnr": head_psnr(rec, gt, head, gmax),
        "head_ssim": head_ssim(rec, gt, head, gmax),
        "head_hfen": head_hfen(rec, gt, head, gmax),
        "grain_ratio": grain_ratio(rec, gt),
        "head_frac": float(head.mean()),
        "gmax": gmax,
    }
