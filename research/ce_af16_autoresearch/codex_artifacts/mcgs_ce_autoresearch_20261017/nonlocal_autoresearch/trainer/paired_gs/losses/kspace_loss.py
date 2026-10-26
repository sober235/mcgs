import torch
import torch.nn as nn


class KSpaceConsistencyLoss(nn.Module):
    """Paper loss term: data consistency in measured k-space only.

    Supports frequency-adaptive weighting to emphasize high-frequency regions.
    """

    def __init__(
        self,
        loss_type: str = "l2",
        freq_adaptive: bool = False,
        center_weight: float = 0.5,
        mid_weight: float = 1.0,
        high_weight: float = 2.0,
        center_fraction: float = 0.08,
    ):
        """
        Args:
            loss_type: 'l1' or 'l2'
            freq_adaptive: Enable frequency-adaptive weighting
            center_weight: Weight for low-freq center region
            mid_weight: Weight for mid-freq region
            high_weight: Weight for high-freq outer region
            center_fraction: Fraction of k-space considered as center (default 0.08 = 8%)
        """
        super().__init__()
        if loss_type not in {"l1", "l2"}:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        self.loss_type = loss_type
        self.freq_adaptive = freq_adaptive
        self.center_weight = center_weight
        self.mid_weight = mid_weight
        self.high_weight = high_weight
        self.center_fraction = center_fraction
        self._freq_mask_cache = None

    def _create_freq_mask(self, spatial_shape: tuple, device: torch.device) -> torch.Tensor:
        """Create frequency-adaptive weights for 2D or 3D centered k-space."""
        if len(spatial_shape) not in {2, 3}:
            raise ValueError(f"Expected 2D or 3D spatial mask shape, got {spatial_shape}")

        coords = []
        for size in spatial_shape:
            center = size // 2
            denom = max(center, 1)
            coords.append((torch.arange(size, device=device, dtype=torch.float32) - center) / denom)

        grids = torch.meshgrid(*coords, indexing="ij")
        radius = torch.sqrt(sum(grid.square() for grid in grids))

        weights = torch.full(spatial_shape, self.high_weight, device=device)
        weights[radius <= 0.30] = self.mid_weight
        weights[radius <= self.center_fraction] = self.center_weight
        return weights

    def forward(
        self,
        kspace_pred: torch.Tensor,
        kspace_target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            kspace_pred: Predicted k-space (H, W, D, C) complex
            kspace_target: Target k-space (H, W, D, C) complex
            mask: Sampling mask (H, W)

        Returns:
            Weighted k-space consistency loss
        """
        diff = (kspace_pred - kspace_target) * mask

        if self.freq_adaptive:
            spatial_shape = tuple(mask.shape[-3:] if mask.ndim >= 3 else mask.shape[-2:])
            if self._freq_mask_cache is None or tuple(self._freq_mask_cache.shape) != spatial_shape:
                self._freq_mask_cache = self._create_freq_mask(
                    spatial_shape, mask.device
                )

            view_shape = (1,) * (diff.ndim - len(spatial_shape)) + spatial_shape
            freq_weights = self._freq_mask_cache.view(view_shape)
            diff = diff * freq_weights

        if self.loss_type == "l1":
            return torch.abs(diff).sum()
        return (torch.abs(diff) ** 2).sum()
