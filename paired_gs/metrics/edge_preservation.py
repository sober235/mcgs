"""EdgePreservation metric — Sobel-3D gradient magnitude Pearson correlation.

Phase 9 GNG-03 5th metric (ROADMAP locked).
The other 4 metrics (NMSE, PSNR, SSIM, HFEN) reuse v1 metrics/metrics.py.

EdgePreservation is the cross-correlation between |grad recon| and |grad target|
Sobel-3D magnitude maps. Higher = recon preserves edge structure of GT.
Range: [-1.0, 1.0]; in practice [0.0, 1.0] because gradient magnitudes are
non-negative.

References
----------
- scipy.ndimage.sobel for 3-axis filter
- Pearson correlation: np.corrcoef
"""
from __future__ import annotations
from typing import Union

import numpy as np
import torch
from scipy.ndimage import sobel


ArrayLike = Union[torch.Tensor, np.ndarray]


def _to_magnitude_numpy(data: ArrayLike) -> np.ndarray:
    """Mirror metrics/metrics.py::_to_magnitude_numpy contract.

    Accepts: torch.Tensor (complex / real / 2-channel real),
             np.ndarray (complex / real / 2-channel real).
    Returns: float64 magnitude np.ndarray.
    """
    if torch.is_tensor(data):
        data = data.detach().cpu()
        if data.is_complex():
            data = torch.abs(data)
        elif data.ndim >= 1 and data.shape[0] == 2:
            data = torch.sqrt(data[0] ** 2 + data[1] ** 2 + 1e-12)
        return data.numpy().astype(np.float64, copy=False)
    arr = np.asarray(data)
    if np.iscomplexobj(arr):
        return np.abs(arr).astype(np.float64, copy=False)
    if arr.ndim >= 1 and arr.shape[0] == 2:
        return np.sqrt(arr[0] ** 2 + arr[1] ** 2 + 1e-12).astype(np.float64, copy=False)
    return arr.astype(np.float64, copy=False)


def sobel_gradient_magnitude_3d(volume: np.ndarray) -> np.ndarray:
    """3-axis Sobel filter; returns sqrt(gx^2 + gy^2 + gz^2).

    Input: 3D real-valued np.ndarray, shape (D, H, W).
    Output: same shape, float64, non-negative.
    """
    if volume.ndim != 3:
        raise ValueError(
            f"sobel_gradient_magnitude_3d expects 3D volume, got shape {volume.shape}"
        )
    gx = sobel(volume, axis=0, mode="reflect")
    gy = sobel(volume, axis=1, mode="reflect")
    gz = sobel(volume, axis=2, mode="reflect")
    return np.sqrt(gx * gx + gy * gy + gz * gz + 1e-24).astype(np.float64, copy=False)


def compute_edge_preservation(pred: ArrayLike, target: ArrayLike) -> float:
    """EdgePreservation: Pearson correlation between Sobel-3D gradient magnitudes.

    Parameters
    ----------
    pred : torch.Tensor | np.ndarray
        Reconstruction volume (complex / real / 2-channel real); 3D after magnitude.
    target : torch.Tensor | np.ndarray
        Ground-truth volume; same shape as pred after magnitude.

    Returns
    -------
    float
        Pearson correlation of |grad pred| and |grad target| flattened. Range [-1, 1].
        Returns float('nan') when either gradient map has zero variance
        (e.g. zero volume; degenerate constant).
    """
    pred_np = _to_magnitude_numpy(pred)
    target_np = _to_magnitude_numpy(target)
    if pred_np.shape != target_np.shape:
        raise ValueError(
            f"shape mismatch: pred {pred_np.shape} vs target {target_np.shape}"
        )
    if pred_np.ndim != 3:
        raise ValueError(
            f"compute_edge_preservation expects 3D inputs, got shape {pred_np.shape}"
        )
    grad_pred = sobel_gradient_magnitude_3d(pred_np).ravel()
    grad_target = sobel_gradient_magnitude_3d(target_np).ravel()
    if grad_pred.std() < 1e-12 or grad_target.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(grad_pred, grad_target)[0, 1])
