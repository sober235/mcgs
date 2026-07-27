"""图像域指标:脑内掩膜 + 脑内统一动态范围(spec §4.1)。"""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity as ssim_sk


def brain_mask(gt_mag: np.ndarray, frac: float = 0.05) -> np.ndarray:
    return gt_mag > frac * float(gt_mag.max())


def brain_gauge(rec_mag: np.ndarray, gt_mag: np.ndarray, frac: float = 0.05) -> dict:
    """峰值取脑内 P99.9(不用全局离群峰),SSIM 用 full map 后在脑内平均。"""
    brain = brain_mask(gt_mag, frac)
    peak = float(np.quantile(gt_mag[brain], 0.999)) if brain.any() else 1.0
    peak = max(peak, 1e-8)
    g = (gt_mag / peak).astype(np.float64)
    r = (rec_mag / peak).astype(np.float64)
    valid = np.zeros_like(brain)
    valid[3:-3, 3:-3, 3:-3] = True
    _, s_map = ssim_sk(g, r, data_range=float(max(g.max(), 1.0)), win_size=7,
                       channel_axis=None, full=True)
    e = r - g
    mse = float((e[brain] ** 2).mean())
    return {
        "psnr_brain": float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse)),
        "ssim_brain": float(s_map[brain & valid].mean()),
        "nmse_brain": float((e[brain] ** 2).sum() / (g[brain] ** 2).sum()),
        "peak_used": peak,
    }
