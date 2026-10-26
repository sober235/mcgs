"""Evaluation metrics and plateau monitoring for 3DGSMR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_sklearn


def _to_magnitude_numpy(data: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(data):
        data = data.detach().cpu()
        if data.is_complex():
            data = torch.abs(data)
        elif data.ndim >= 1 and data.shape[0] == 2:
            data = torch.sqrt(data[0] ** 2 + data[1] ** 2 + 1e-12)
        return data.numpy()

    arr = np.asarray(data)
    if np.iscomplexobj(arr):
        return np.abs(arr)
    if arr.ndim >= 1 and arr.shape[0] == 2:
        return np.sqrt(arr[0] ** 2 + arr[1] ** 2 + 1e-12)
    return arr


def _normalize_pair_to_unit_range(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize MRI magnitude volumes to [0, 1] using the reference target scale.

    The dataset pipeline normalizes by a percentile magnitude (typically P99.9),
    which is stable but does not guarantee a strict unit interval. For standard
    PSNR/SSIM reporting we explicitly map both prediction and reference to
    [0, 1] and then use a fixed data_range=1.0.
    """
    pred_np = _to_magnitude_numpy(pred).astype(np.float64, copy=False)
    target_np = _to_magnitude_numpy(target).astype(np.float64, copy=False)
    reference_scale = max(float(target_np.max()), 1e-8)
    pred_unit = np.clip(pred_np / reference_scale, 0.0, 1.0)
    target_unit = np.clip(target_np / reference_scale, 0.0, 1.0)
    return pred_unit, target_unit


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: Optional[float] = None) -> float:
    pred_np, target_np = _normalize_pair_to_unit_range(pred, target)
    mse = np.mean((pred_np - target_np) ** 2)
    if mse == 0:
        return float("inf")
    data_range = 1.0 if data_range is None else float(data_range)
    return float(10 * np.log10((data_range ** 2) / mse))


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: Optional[float] = None,
    win_size: int = 7,
) -> float:
    pred_np, target_np = _normalize_pair_to_unit_range(pred, target)
    data_range = 1.0 if data_range is None else float(data_range)
    min_dim = min(pred_np.shape)
    if win_size > min_dim:
        win_size = min_dim if min_dim % 2 == 1 else min_dim - 1
        win_size = max(3, win_size)
    return float(
        ssim_sklearn(
            pred_np,
            target_np,
            data_range=data_range,
            win_size=win_size,
            channel_axis=None,
        )
    )


def compute_nmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_np = _to_magnitude_numpy(pred)
    target_np = _to_magnitude_numpy(target)
    numerator = np.sum((pred_np - target_np) ** 2)
    denominator = np.sum(target_np ** 2)
    if denominator == 0:
        return float("inf")
    return float(numerator / denominator)


def _auto_roi_masks(
    target: torch.Tensor | np.ndarray,
    sig_quantile: float = 0.85,
    bg_low_q: float = 0.0,
    bg_high_q: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive background / signal / mid-tissue ROIs from intensity quantiles.

    CSM-aware reconstructions zero exactly outside FOV, making corner-based
    noise ROIs degenerate. Instead we sample background from low-intensity
    voxels inside the object support (CSF, low-signal tissue): values in
    [Q(bg_low_q), Q(bg_high_q)] of non-zero magnitudes.
    """
    mag = _to_magnitude_numpy(target).astype(np.float64, copy=False)
    flat = mag.flatten()
    nonzero = flat[flat > 1e-12]
    if nonzero.size == 0:
        return np.zeros_like(mag, dtype=bool), np.zeros_like(mag, dtype=bool), np.zeros_like(mag, dtype=bool)

    bg_lo = float(np.quantile(nonzero, bg_low_q)) if bg_low_q > 0 else float(nonzero.min())
    bg_hi = float(np.quantile(nonzero, bg_high_q))
    sig_thr = float(np.quantile(nonzero, sig_quantile))
    bg_mask = (mag > bg_lo - 1e-12) & (mag <= bg_hi)
    sig_mask = mag >= sig_thr
    mid_mask = (mag > bg_hi) & (mag < sig_thr)
    return bg_mask, sig_mask, mid_mask


def _robust_noise_std(values: np.ndarray) -> float:
    """Estimate noise std robust to all-zero / near-zero corner samples.

    Tries std, then MAD/0.6745, then mean absolute value. Returns 0.0 if all fail.
    """
    if values.size == 0:
        return 0.0
    s = float(values.std())
    if s > 1e-12:
        return s
    mad = float(np.median(np.abs(values - np.median(values))))
    if mad > 1e-12:
        return mad / 0.6745
    return float(np.mean(np.abs(values)))


def compute_snr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Signal-to-Noise Ratio (dB).

    SNR = 20 * log10( mean(signal_ROI) / noise_std(background_ROI) )
    Background = volume corner cubes; signal = top 25% intensity voxels.
    """
    mag_pred = _to_magnitude_numpy(pred).astype(np.float64, copy=False)
    bg_mask, sig_mask, _ = _auto_roi_masks(target)
    if sig_mask.sum() == 0 or bg_mask.sum() == 0:
        return float("nan")
    sig = float(mag_pred[sig_mask].mean())
    noise = _robust_noise_std(mag_pred[bg_mask])
    if noise < 1e-12 or sig <= 0:
        return float("inf")
    return float(20.0 * np.log10(sig / noise))


def compute_cnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Contrast-to-Noise Ratio.

    CNR = |mean(signal_ROI) - mean(background_ROI)| / noise_std(background_ROI)
    """
    mag_pred = _to_magnitude_numpy(pred).astype(np.float64, copy=False)
    bg_mask, sig_mask, _ = _auto_roi_masks(target)
    if sig_mask.sum() == 0 or bg_mask.sum() == 0:
        return float("nan")
    sig = float(mag_pred[sig_mask].mean())
    bg = float(mag_pred[bg_mask].mean())
    noise = _robust_noise_std(mag_pred[bg_mask])
    if noise < 1e-12:
        return float("inf")
    return float(abs(sig - bg) / noise)


def compute_cnr_fidelity(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Ratio of reconstructed CNR to target CNR (1.0 = perfect contrast preservation)."""
    cnr_pred = compute_cnr(pred, target)
    cnr_tgt = compute_cnr(target, target)
    if cnr_tgt < 1e-12 or not np.isfinite(cnr_tgt):
        return float("nan")
    return float(cnr_pred / cnr_tgt)


def _log3d_kernel(sigma: float, kernel_size: int) -> np.ndarray:
    # MC-3DGS roadmap Q4 spec: explicit 3D LoG, DC-removed so the kernel
    # is true bandpass (sum = 0) and does not respond to constant offsets.
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError(f"kernel_size must be odd and >= 3, got {kernel_size}")
    radius = kernel_size // 2
    ax = np.arange(-radius, radius + 1, dtype=np.float64)
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    r2 = x * x + y * y + z * z
    s2 = sigma * sigma
    gauss = np.exp(-r2 / (2.0 * s2))
    gauss /= gauss.sum()
    log = ((r2 / s2) - 3.0) / s2 * gauss
    log -= log.mean()
    return log


_HFEN_LOG_CACHE: Dict[tuple, np.ndarray] = {}


def compute_hfen(
    pred: torch.Tensor,
    target: torch.Tensor,
    sigma: float = 1.5,
    kernel_size: int = 15,
) -> float:
    """High-Frequency Error Norm (lower = better).

    MC-3DGS unified spec (roadmap Q4):
      - Inputs normalized to the same [0, 1] reference scale used by PSNR/SSIM
        via `_normalize_pair_to_unit_range`, so HFEN is cross-method comparable.
      - 3D Laplacian-of-Gaussian kernel, sigma=1.5 vox, kernel_size=15
        (radius 7 → fully resolves the LoG tails for sigma=1.5).
      - Kernel is DC-removed (sum = 0) so the filter is true bandpass.
      - Output = ‖LoG(pred) − LoG(target)‖_2 / ‖LoG(target)‖_2.
    """
    from scipy.ndimage import convolve as _nd_convolve

    pred_np, target_np = _normalize_pair_to_unit_range(pred, target)
    pred_np = pred_np.astype(np.float64, copy=False)
    target_np = target_np.astype(np.float64, copy=False)

    cache_key = (float(sigma), int(kernel_size))
    kernel = _HFEN_LOG_CACHE.get(cache_key)
    if kernel is None:
        kernel = _log3d_kernel(sigma=float(sigma), kernel_size=int(kernel_size))
        _HFEN_LOG_CACHE[cache_key] = kernel

    log_pred = _nd_convolve(pred_np, kernel, mode="reflect")
    log_target = _nd_convolve(target_np, kernel, mode="reflect")
    num = float(np.linalg.norm(log_pred - log_target))
    den = float(np.linalg.norm(log_target))
    if den < 1e-12:
        return float("inf")
    return float(num / den)


def compute_edge_l1(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute error of image gradient magnitude — measures edge fidelity."""
    pred_np = _to_magnitude_numpy(pred).astype(np.float64, copy=False)
    target_np = _to_magnitude_numpy(target).astype(np.float64, copy=False)

    def _grad_mag(x: np.ndarray) -> np.ndarray:
        gs = np.gradient(x)
        sq = sum(g ** 2 for g in gs)
        return np.sqrt(sq + 1e-12)

    return float(np.mean(np.abs(_grad_mag(pred_np) - _grad_mag(target_np))))


def _gradient_magnitude(x: np.ndarray) -> np.ndarray:
    gs = np.gradient(x)
    return np.sqrt(sum(g ** 2 for g in gs) + 1e-12)


def compute_snr_absolute(image: torch.Tensor | np.ndarray) -> float:
    """Absolute SNR (dB) computed using ROIs derived from the image itself."""
    mag = _to_magnitude_numpy(image).astype(np.float64, copy=False)
    bg_mask, sig_mask, _ = _auto_roi_masks(mag)
    if sig_mask.sum() == 0 or bg_mask.sum() == 0:
        return float("nan")
    sig = float(mag[sig_mask].mean())
    noise = _robust_noise_std(mag[bg_mask])
    if noise < 1e-12 or sig <= 0:
        return float("inf")
    return float(20.0 * np.log10(sig / noise))


def compute_cnr_absolute(image: torch.Tensor | np.ndarray) -> float:
    """Absolute CNR computed using ROIs derived from the image itself."""
    mag = _to_magnitude_numpy(image).astype(np.float64, copy=False)
    bg_mask, sig_mask, _ = _auto_roi_masks(mag)
    if sig_mask.sum() == 0 or bg_mask.sum() == 0:
        return float("nan")
    sig = float(mag[sig_mask].mean())
    bg = float(mag[bg_mask].mean())
    noise = _robust_noise_std(mag[bg_mask])
    if noise < 1e-12:
        return float("inf")
    return float(abs(sig - bg) / noise)


def compute_sharpness(image: torch.Tensor | np.ndarray) -> float:
    """Mean gradient magnitude — higher = sharper image edges."""
    mag = _to_magnitude_numpy(image).astype(np.float64, copy=False)
    return float(_gradient_magnitude(mag).mean())


def compute_entropy(image: torch.Tensor | np.ndarray, bins: int = 64) -> float:
    """Shannon entropy of intensity histogram — measures information content."""
    mag = _to_magnitude_numpy(image).astype(np.float64, copy=False).flatten()
    hist, _ = np.histogram(mag, bins=bins, density=False)
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def evaluate_reconstruction_absolute(image: torch.Tensor | np.ndarray) -> Dict[str, float]:
    """Compute GT-free quality metrics for a reconstruction.

    Returns SNR, CNR, sharpness, entropy — useful for real clinical data without GT.

    NOTE on units: SNR_abs / CNR_abs are scale-invariant dB / ratio measures
    and remain comparable across methods at any intensity scale. Sharpness and
    entropy are NOT scale-invariant: gradient magnitude and histogram shape
    scale linearly with the volume's intensity range, so two methods producing
    reconstructions at different absolute scales (e.g. CS-Wavelet at p99 ~ 2.6e5
    vs ZS-SSDU at p99 ~ 1.3e5 on the Huangyuwei T1 benchmark) are not
    comparable in raw sharpness units. To make sharpness/entropy cross-method
    comparable we normalize the input by the 99th-percentile magnitude before
    computing those two metrics only. SNR_abs / CNR_abs continue to use the
    unnormalized image.
    """
    metrics: Dict[str, float] = {}
    metrics["snr_abs"] = compute_snr_absolute(image)
    metrics["cnr_abs"] = compute_cnr_absolute(image)

    # Normalize for sharpness/entropy so they are cross-method comparable.
    mag_for_norm = _to_magnitude_numpy(image).astype(np.float64, copy=False)
    nonzero = mag_for_norm[mag_for_norm > 1e-12]
    p99 = float(np.percentile(nonzero, 99.0)) if nonzero.size else 0.0
    image_unit = mag_for_norm / p99 if p99 > 1e-12 else mag_for_norm
    metrics["sharpness"] = compute_sharpness(image_unit)
    metrics["entropy"] = compute_entropy(image_unit)
    return metrics


def compute_ssim_3d_slicewise(
    pred: torch.Tensor,
    target: torch.Tensor,
    axis: int = 0,
    data_range: Optional[float] = None,
) -> tuple[float, float]:
    pred_np, target_np = _normalize_pair_to_unit_range(pred, target)
    data_range = 1.0 if data_range is None else float(data_range)

    values = []
    for idx in range(pred_np.shape[axis]):
        pred_slice = np.take(pred_np, idx, axis=axis)
        target_slice = np.take(target_np, idx, axis=axis)
        if target_slice.max() - target_slice.min() < 1e-8:
            continue
        win_size = 7
        min_dim = min(pred_slice.shape)
        if win_size > min_dim:
            win_size = min_dim if min_dim % 2 == 1 else min_dim - 1
            win_size = max(3, win_size)
        values.append(ssim_sklearn(pred_slice, target_slice, data_range=data_range, win_size=win_size))
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


_LPIPS_MODELS: Dict[str, torch.nn.Module] = {}
_LPIPS_WARNING_EMITTED = False


def _get_lpips_model(device: torch.device, net: str = "alex"):
    key = f"{net}:{device}"
    if key in _LPIPS_MODELS:
        return _LPIPS_MODELS[key]
    try:
        import lpips  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "LPIPS evaluation requires the `lpips` package. Install it in the active environment to run paper-faithful reproduction metrics."
        ) from exc
    model = lpips.LPIPS(net=net).to(device)
    model.eval()
    _LPIPS_MODELS[key] = model
    return model


def compute_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    device: Optional[torch.device] = None,
    axis: int = 0,
    require: bool = False,
) -> Optional[float]:
    try:
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _get_lpips_model(dev)
    except ModuleNotFoundError:
        global _LPIPS_WARNING_EMITTED
        if require:
            raise
        if not _LPIPS_WARNING_EMITTED:
            print('[Metrics] LPIPS package not available; skipping LPIPS computation in this environment.')
            _LPIPS_WARNING_EMITTED = True
        return None

    pred_np = _to_magnitude_numpy(pred)
    target_np = _to_magnitude_numpy(target)
    t_min = float(target_np.min())
    t_max = float(target_np.max())
    scale = max(t_max - t_min, 1e-8)
    pred_norm = ((pred_np - t_min) / scale) * 2.0 - 1.0
    target_norm = ((target_np - t_min) / scale) * 2.0 - 1.0

    values = []
    with torch.no_grad():
        for idx in range(pred_norm.shape[axis]):
            pred_slice = np.take(pred_norm, idx, axis=axis)
            target_slice = np.take(target_norm, idx, axis=axis)
            pred_tensor = torch.from_numpy(pred_slice).float().to(dev).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
            target_tensor = torch.from_numpy(target_slice).float().to(dev).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
            values.append(float(model(pred_tensor, target_tensor).item()))
    if not values:
        return None
    return float(np.mean(values))


@dataclass
class MetricPlateauStopper:
    patience_evals: int
    min_iterations_before_stop: int
    metric_min_delta: Dict[str, float]
    best_metric: str = "psnr"
    best_values: Dict[str, float] = field(default_factory=dict)
    bad_eval_count: int = 0

    def _improved(self, name: str, value: float) -> bool:
        delta = self.metric_min_delta.get(name, 0.0)
        if name not in self.best_values:
            return True
        best = self.best_values[name]
        if name == "lpips":
            return value < best - delta
        return value > best + delta

    def update(self, iteration: int, metrics: Dict[str, float]) -> bool:
        tracked_value = metrics.get(self.best_metric)
        if tracked_value is None:
            raise ValueError(
                f"MetricPlateauStopper requires tracked metric '{self.best_metric}', "
                f"but only received: {sorted(metrics.keys())}"
            )

        improved = self._improved(self.best_metric, tracked_value)
        if improved:
            self.best_values[self.best_metric] = tracked_value

        # Keep best snapshots for auxiliary metrics without letting them affect
        # the plateau decision, which must follow best_metric only.
        for name, value in metrics.items():
            if name == self.best_metric or value is None:
                continue
            if self._improved(name, value):
                self.best_values[name] = value

        if improved:
            self.bad_eval_count = 0
        else:
            self.bad_eval_count += 1
        return iteration >= self.min_iterations_before_stop and self.bad_eval_count >= self.patience_evals


def evaluate_reconstruction(
    pred: torch.Tensor,
    target: torch.Tensor,
    compute_3d_ssim: bool = True,
    compute_lpips_metric: bool = False,
    lpips_device: Optional[torch.device] = None,
    require_lpips: bool = False,
    compute_extended: bool = True,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["psnr"] = compute_psnr(pred, target)
    if compute_3d_ssim:
        try:
            metrics["ssim"] = compute_ssim(pred, target)
        except Exception:
            mean_ssim, std_ssim = compute_ssim_3d_slicewise(pred, target)
            metrics["ssim"] = mean_ssim
            metrics["ssim_std"] = std_ssim
    else:
        mean_ssim, std_ssim = compute_ssim_3d_slicewise(pred, target)
        metrics["ssim"] = mean_ssim
        metrics["ssim_std"] = std_ssim
    metrics["nmse"] = compute_nmse(pred, target)
    if compute_extended:
        try:
            metrics["snr"] = compute_snr(pred, target)
            metrics["cnr"] = compute_cnr(pred, target)
            metrics["cnr_fidelity"] = compute_cnr_fidelity(pred, target)
            metrics["hfen"] = compute_hfen(pred, target)
            metrics["edge_l1"] = compute_edge_l1(pred, target)
        except Exception as exc:
            print(f"[Metrics] Extended metric computation failed: {exc}")
    if compute_lpips_metric:
        lpips_value = compute_lpips(pred, target, device=lpips_device, require=require_lpips)
        if lpips_value is not None:
            metrics["lpips"] = lpips_value
    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    if "psnr" in metrics:
        print(f"{prefix}PSNR: {metrics['psnr']:.2f} dB")
    if "ssim" in metrics:
        print(f"{prefix}SSIM: {metrics['ssim']:.4f}")
    if "nmse" in metrics:
        print(f"{prefix}NMSE: {metrics['nmse']:.6f}")
    if "snr" in metrics:
        print(f"{prefix}SNR:  {metrics['snr']:.2f} dB")
    if "cnr" in metrics:
        print(f"{prefix}CNR:  {metrics['cnr']:.3f}")
    if "cnr_fidelity" in metrics:
        print(f"{prefix}CNR fidelity: {metrics['cnr_fidelity']:.4f}")
    if "hfen" in metrics:
        print(f"{prefix}HFEN: {metrics['hfen']:.4f}")
    if "edge_l1" in metrics:
        print(f"{prefix}Edge L1: {metrics['edge_l1']:.6f}")
    if "snr_abs" in metrics:
        print(f"{prefix}SNR (abs): {metrics['snr_abs']:.2f} dB")
    if "cnr_abs" in metrics:
        print(f"{prefix}CNR (abs): {metrics['cnr_abs']:.3f}")
    if "sharpness" in metrics:
        print(f"{prefix}Sharpness: {metrics['sharpness']:.6f}")
    if "entropy" in metrics:
        print(f"{prefix}Entropy:   {metrics['entropy']:.4f}")
    if "lpips" in metrics:
        print(f"{prefix}LPIPS: {metrics['lpips']:.4f}")
