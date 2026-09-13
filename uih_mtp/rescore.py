"""Region-restricted rescoring on the frozen gauge (stage-2 spec, section 7).

The metric MAPS are built exactly as the frozen code builds them -- head mask, PSNR peak, SSIM data
range and the LoG kernel all come from the FULL-volume |GT| -- and only the averaging region changes.
With `region=None` every number equals the frozen gauge's, which the tests assert.

Frozen sources: diag_artefact/headonly.py (head metrics), paired_gs/metrics/metrics.py (whole-volume
PSNR / SSIM / NMSE), scripts/t1t2_common.py (sharpness). Do not "improve" any definition here.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.metrics import structural_similarity

from diag_artefact.headonly import HP_SIGMA, head_mask, log_kernel
from diag_artefact.volumes import brain_mask, wm_mask

SSIM_WIN = 7


def _mag(v) -> np.ndarray:
    a = np.asarray(v)
    return (np.abs(a) if np.iscomplexobj(a) else a).astype(np.float32)


def partition_region(shape, axis: int, lo: int, hi: int) -> np.ndarray:
    """Boolean volume selecting `lo <= index < hi` along `axis` (the vendor's central partitions)."""
    r = np.zeros(tuple(int(s) for s in shape), dtype=bool)
    sl = [slice(None)] * len(r.shape)
    sl[int(axis)] = slice(int(lo), int(hi))
    r[tuple(sl)] = True
    return r


def _gradient_magnitude(v: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(v); gy = np.zeros_like(v); gz = np.zeros_like(v)
    gx[:-1] = v[1:] - v[:-1]
    gy[:, :-1] = v[:, 1:] - v[:, :-1]
    gz[:, :, :-1] = v[:, :, 1:] - v[:, :, :-1]
    return np.sqrt(gx * gx + gy * gy + gz * gz + 1e-12)


def score_region(rec, gt, region=None) -> dict:
    """Every metric of the topic tables, averaged over `region` (None = the whole volume)."""
    rec = _mag(rec)
    gt = _mag(gt)
    if rec.shape != gt.shape:
        raise ValueError(f"shape mismatch {rec.shape} vs {gt.shape}")
    reg = np.ones_like(gt, dtype=bool) if region is None else np.asarray(region, dtype=bool)
    head = head_mask(gt) & reg
    gmax = float(gt.max())
    r64, g64 = rec.astype(np.float64), gt.astype(np.float64)

    mse = float(np.mean((r64[head] - g64[head]) ** 2))
    head_psnr = float("inf") if mse == 0 else float(20 * np.log10(gmax) - 10 * np.log10(mse))

    _, smap = structural_similarity(rec, gt, data_range=gmax, full=True)
    head_ssim = float(smap[head].mean())

    k = log_kernel()
    lr = ndimage.convolve(r64 / gmax, k, mode="reflect")
    lg = ndimage.convolve(g64 / gmax, k, mode="reflect")
    den = float(np.linalg.norm(lg[head]))
    head_hfen = float("inf") if den == 0 else float(np.linalg.norm((lr - lg)[head]) / den)

    brain = brain_mask(gt)
    wm = wm_mask(gt, brain) & reg
    if wm.any():
        hp = lambda v: v - ndimage.gaussian_filter(v.astype(np.float32), HP_SIGMA)
        gr = float(np.sqrt(np.mean(hp(rec)[wm] ** 2)))
        gg = float(np.sqrt(np.mean(hp(gt)[wm] ** 2)))
        grain = float("inf") if gg == 0 else gr / gg
    else:
        grain = float("nan")

    # whole-volume gauge (paired_gs.metrics): both magnitudes scaled by max|GT| and clipped to [0, 1]
    scale = max(gmax, 1e-8)
    ru, gu = np.clip(r64 / scale, 0.0, 1.0), np.clip(g64 / scale, 0.0, 1.0)
    mse_w = float(np.mean((ru[reg] - gu[reg]) ** 2))
    psnr = float("inf") if mse_w == 0 else float(-10 * np.log10(mse_w))
    _, smap_w = structural_similarity(ru, gu, data_range=1.0, win_size=SSIM_WIN, channel_axis=None, full=True)
    pad = (SSIM_WIN - 1) // 2
    interior = np.zeros_like(reg)
    interior[pad:-pad, pad:-pad, pad:-pad] = True                  # skimage crops this border from its mean
    ssim = float(smap_w[reg & interior].mean())
    den_n = float(np.sum(g64[reg] ** 2))
    nmse = float("inf") if den_n == 0 else float(np.sum((r64[reg] - g64[reg]) ** 2) / den_n)

    sb = brain & reg
    sharp = float(_gradient_magnitude(rec)[sb].mean() / (_gradient_magnitude(gt)[sb].mean() + 1e-9)) if sb.any() else float("nan")

    return dict(head_psnr=head_psnr, head_ssim=head_ssim, head_hfen=head_hfen, grain_ratio=grain,
                head_frac=float(head.sum() / reg.sum()), gmax=gmax,
                psnr=psnr, ssim=ssim, nmse=nmse, sharp=sharp, n_voxels=int(reg.sum()))
