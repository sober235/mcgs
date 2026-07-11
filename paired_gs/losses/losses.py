"""Loss definitions for 3DGSMR."""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .kspace_loss import KSpaceConsistencyLoss
from .multiscale_loss import MultiScaleLoss

try:
    import ptwt
    PTWT_AVAILABLE = True
except ImportError:
    PTWT_AVAILABLE = False


def _to_magnitude(image: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(image):
        return torch.abs(image)
    if image.ndim >= 1 and image.shape[0] == 2:
        return torch.sqrt(image[0] ** 2 + image[1] ** 2 + 1e-12)
    return image


class MagnitudeTVLoss(nn.Module):
    """Anisotropic TV(|X|): sum |dx| + |dy| + |dz|."""

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        mag = _to_magnitude(image)
        diff_d = torch.abs(mag[1:, :, :] - mag[:-1, :, :])
        diff_h = torch.abs(mag[:, 1:, :] - mag[:, :-1, :])
        diff_w = torch.abs(mag[:, :, 1:] - mag[:, :, :-1])
        return diff_d.sum() + diff_h.sum() + diff_w.sum()


class IsotropicMagnitudeTVLoss(nn.Module):
    """Isotropic TV(|X|): sum sqrt(dx^2 + dy^2 + dz^2)."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        mag = _to_magnitude(image)
        base = mag[:-1, :-1, :-1]
        diff_d = mag[1:, :-1, :-1] - base
        diff_h = mag[:-1, 1:, :-1] - base
        diff_w = mag[:-1, :-1, 1:] - base
        return torch.sqrt(diff_d.square() + diff_h.square() + diff_w.square() + self.eps).sum()


class WaveletL1Loss(nn.Module):
    """L1 penalty on db4 3D wavelet detail subbands (coeffs[1:]).

    Applies ptwt.wavedec3(magnitude, "db4", level=2, mode="zero").
    Excludes approximation coefficients (LLL, coeffs[0]).
    Normalized by total number of detail elements so wavelet_weight
    is scale-independent of volume size.

    Per D-16: called in forward_with_loss() when wavelet_weight > 0.
    Requires: ptwt==0.1.9 installed in vediffusion0328 (WAVE-01).
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if not PTWT_AVAILABLE:
            raise ImportError(
                "ptwt is required for WaveletL1Loss. "
                "Install with: pip install ptwt==0.1.9"
            )
        mag = _to_magnitude(image)  # real float32 required by ptwt
        coeffs = ptwt.wavedec3(mag, "db4", level=2, mode="zero")
        # coeffs[0] = LLL approx -- excluded (per D-16)
        # coeffs[1] = level-2 detail dict (7 subbands)
        # coeffs[2] = level-1 detail dict (7 subbands)
        loss = torch.tensor(0.0, device=mag.device, dtype=mag.dtype)
        total_numel = 0
        for detail_dict in coeffs[1:]:  # skip coeffs[0]
            for subband in detail_dict.values():
                loss = loss + torch.abs(subband).sum()
                total_numel += subband.numel()
        return loss / total_numel  # normalized: ~ mean |detail coeff|


class SmoothMagnitudeTVLoss(nn.Module):
    """Smooth TV(|X|) using sqrt(diff^2 + eps) instead of abs(diff).

    Avoids gradient discontinuity at diff=0 (Huber-like behavior).
    Activated via loss.smooth_tv: true in YAML config (per D-05).
    Default OFF -- MagnitudeTVLoss remains the default (per D-04).
    Same scale as MagnitudeTVLoss when |diff| >> eps.

    Per WAVE-06: satisfies the optional smooth TV variant requirement.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        mag = _to_magnitude(image)
        diff_d = mag[1:, :, :] - mag[:-1, :, :]
        diff_h = mag[:, 1:, :] - mag[:, :-1, :]
        diff_w = mag[:, :, 1:] - mag[:, :, :-1]
        return torch.sqrt(diff_d ** 2 + self.eps).sum() + torch.sqrt(diff_h ** 2 + self.eps).sum() + torch.sqrt(diff_w ** 2 + self.eps).sum()


def build_tv_loss(tv_type: str = "anisotropic", smooth: bool = False, eps: float = 1e-8) -> nn.Module:
    """Factory for TV(|X|) regularizers.

    `anisotropic` matches the historical repo behavior:
    sum |dx| + |dy| + |dz|

    `isotropic` uses the vector magnitude at each voxel:
    sum sqrt(dx^2 + dy^2 + dz^2)
    """
    if tv_type == "anisotropic":
        return SmoothMagnitudeTVLoss(eps=eps) if smooth else MagnitudeTVLoss()
    if tv_type == "isotropic":
        return IsotropicMagnitudeTVLoss(eps=eps)
    raise ValueError(f"Unsupported tv_type: {tv_type}")


class ImageLoss(nn.Module):
    """Legacy non-paper image-domain auxiliary loss."""

    def __init__(self, loss_type: str = "l2"):
        super().__init__()
        if loss_type not in {"l1", "l2"}:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        self.loss_type = loss_type

    def forward(
        self,
        image_pred: torch.Tensor,
        image_target: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        diff = _to_magnitude(image_pred) - _to_magnitude(image_target)
        if weight_map is not None:
            diff = diff * weight_map
        if self.loss_type == "l1":
            return torch.abs(diff).sum()
        return (diff ** 2).sum()


class BackgroundSuppressionLoss(nn.Module):
    """Penalize residual signal in background voxels.

    The background mask is derived from the zero-filled reconstruction, so this
    regularizer remains self-supervised and targets visually cleaner output.
    """

    def __init__(self, loss_type: str = "l2"):
        super().__init__()
        if loss_type not in {"l1", "l2"}:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        self.loss_type = loss_type

    def forward(self, image: torch.Tensor, background_mask: torch.Tensor) -> torch.Tensor:
        mag = _to_magnitude(image)
        mask = background_mask.to(device=mag.device, dtype=mag.dtype)
        denom = torch.clamp(mask.sum(), min=1.0)
        if self.loss_type == "l1":
            return (torch.abs(mag) * mask).sum() / denom
        return (mag.square() * mask).sum() / denom


class PaperLoss(nn.Module):
    """Paper objective: ||A(X)-b|| + lambda * TV(|X|) + optional multi-scale loss."""

    def __init__(
        self,
        kspace_weight: float = 1.0,
        tv_weight: float = 0.1,
        loss_type: str = "l2",
        image_weight: float = 0.0,
        tv_type: str = "anisotropic",
        smooth_tv: bool = False,
        # NEW: Frequency-adaptive k-space
        kspace_freq_adaptive: bool = False,
        kspace_center_weight: float = 0.5,
        kspace_mid_weight: float = 1.0,
        kspace_high_weight: float = 2.0,
        kspace_center_fraction: float = 0.08,
        # NEW: Multi-scale image loss
        multiscale_image_weight: float = 0.0,
        multiscale_levels: list = None,
        multiscale_loss_type: str = "l1",
    ):
        super().__init__()
        self.kspace_weight = kspace_weight
        self.tv_weight = tv_weight
        self.image_weight = image_weight
        self.multiscale_image_weight = multiscale_image_weight

        self.kspace_loss = KSpaceConsistencyLoss(
            loss_type=loss_type,
            freq_adaptive=kspace_freq_adaptive,
            center_weight=kspace_center_weight,
            mid_weight=kspace_mid_weight,
            high_weight=kspace_high_weight,
            center_fraction=kspace_center_fraction,
        )
        self.tv_loss = build_tv_loss(tv_type=tv_type, smooth=smooth_tv)
        self.image_loss = ImageLoss(loss_type)

        if multiscale_image_weight > 0:
            if multiscale_levels is None:
                multiscale_levels = [1, 2, 4]
            self.multiscale_loss = MultiScaleLoss(
                scales=multiscale_levels,
                loss_type=multiscale_loss_type,
            )
        else:
            self.multiscale_loss = None

    def forward(
        self,
        kspace_pred: torch.Tensor,
        kspace_target: torch.Tensor,
        mask: torch.Tensor,
        image_pred: Optional[torch.Tensor] = None,
        image_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        total_loss = torch.tensor(0.0, device=kspace_pred.device)
        losses: Dict[str, torch.Tensor] = {}

        kspace_l = self.kspace_loss(kspace_pred, kspace_target, mask)
        total_loss = total_loss + self.kspace_weight * kspace_l
        losses["kspace_loss"] = kspace_l

        if self.tv_weight > 0 and image_pred is not None:
            tv_l = self.tv_loss(image_pred)
            total_loss = total_loss + self.tv_weight * tv_l
            losses["tv_loss"] = tv_l

        if self.image_weight > 0 and image_pred is not None and image_target is not None:
            image_l = self.image_loss(image_pred, image_target)
            total_loss = total_loss + self.image_weight * image_l
            losses["image_loss"] = image_l

        if self.multiscale_image_weight > 0 and self.multiscale_loss is not None:
            if image_pred is not None and image_target is not None:
                # Convert to magnitude for multi-scale comparison
                pred_mag = _to_magnitude(image_pred)
                target_mag = _to_magnitude(image_target)
                multiscale_l = self.multiscale_loss(pred_mag, target_mag)
                total_loss = total_loss + self.multiscale_image_weight * multiscale_l
                losses["multiscale_loss"] = multiscale_l

        losses["total_loss"] = total_loss
        return losses


class DataConsistencyLayer(nn.Module):
    def forward(self, kspace_pred, kspace_measured, mask):
        return kspace_pred * (1 - mask) + kspace_measured * mask


KSpaceLoss = KSpaceConsistencyLoss
TVLoss = MagnitudeTVLoss
IsotropicTVLoss = IsotropicMagnitudeTVLoss
CombinedLoss = PaperLoss
