import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleLoss(nn.Module):
    """Multi-scale image-domain loss for capturing structure at different resolutions.

    Computes L1 loss between reconstructed and target volumes at multiple scales
    (original, 1/2, 1/4 resolution) to enforce consistency across frequency bands.
    """

    def __init__(
        self,
        scales: list[int] = [1, 2, 4],
        loss_type: str = "l1",
        weights: list[float] | None = None,
    ):
        """
        Args:
            scales: Downsampling factors (1 = original resolution)
            loss_type: 'l1' or 'l2'
            weights: Per-scale weights (default: equal weighting)
        """
        super().__init__()
        if loss_type not in {"l1", "l2"}:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        self.scales = scales
        self.loss_type = loss_type

        # Default: equal weighting across scales
        if weights is None:
            weights = [1.0 / len(scales)] * len(scales)
        if len(weights) != len(scales):
            raise ValueError(f"weights length {len(weights)} != scales length {len(scales)}")

        self.weights = weights

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted volume (B, D, H, W) or (D, H, W)
            target: Target volume (B, D, H, W) or (D, H, W)

        Returns:
            Weighted sum of losses across scales
        """
        # Ensure 4D: (B, D, H, W)
        if pred.ndim == 3:
            pred = pred.unsqueeze(0)
        if target.ndim == 3:
            target = target.unsqueeze(0)

        total_loss = 0.0

        for scale, weight in zip(self.scales, self.weights):
            if scale == 1:
                # Original resolution
                pred_scaled = pred
                target_scaled = target
            else:
                # Downsample using 3D average pooling
                pred_scaled = F.avg_pool3d(pred, kernel_size=scale, stride=scale)
                target_scaled = F.avg_pool3d(target, kernel_size=scale, stride=scale)

            # Compute loss at this scale
            if self.loss_type == "l1":
                loss = F.l1_loss(pred_scaled, target_scaled, reduction='sum')
            else:  # l2
                loss = F.mse_loss(pred_scaled, target_scaled, reduction='sum')

            total_loss += weight * loss

        return total_loss
