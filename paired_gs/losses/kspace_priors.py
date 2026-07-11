"""
MRI-specific k-space regularization losses.

These priors exploit the physics and structure of MRI k-space data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoilConsistencyLoss(nn.Module):
    """
    Enforce consistency across multiple coils.

    Different coils should measure the same underlying image,
    just with different sensitivity profiles.
    """

    def __init__(self, loss_type='l2'):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, kspace_multi_coil, coil_maps):
        """
        Args:
            kspace_multi_coil: (C, D, H, W, 2) k-space from C coils
            coil_maps: (C, D, H, W) coil sensitivity maps

        Returns:
            loss: scalar consistency loss
        """
        # Convert to complex
        kspace_complex = torch.complex(
            kspace_multi_coil[..., 0],
            kspace_multi_coil[..., 1]
        )  # (C, D, H, W)

        # IFFT to image space
        image_multi_coil = torch.fft.ifftn(kspace_complex, dim=(-3, -2, -1))  # (C, D, H, W)

        # Coil combination (sum of squares)
        image_combined = torch.sum(
            image_multi_coil * torch.conj(coil_maps), dim=0
        )  # (D, H, W)

        # Re-encode with each coil
        image_reencoded = image_combined.unsqueeze(0) * coil_maps  # (C, D, H, W)
        kspace_reencoded = torch.fft.fftn(image_reencoded, dim=(-3, -2, -1))  # (C, D, H, W)

        # Should match original k-space
        if self.loss_type == 'l2':
            loss = F.mse_loss(torch.abs(kspace_reencoded), torch.abs(kspace_complex))
        elif self.loss_type == 'l1':
            loss = F.l1_loss(torch.abs(kspace_reencoded), torch.abs(kspace_complex))
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss


class KSpaceSupportLoss(nn.Module):
    """
    Encourage most energy to be in low frequencies (k-space center).

    Natural images have most energy concentrated in low frequencies.
    """

    def __init__(self, center_fraction=0.3):
        super().__init__()
        self.center_fraction = center_fraction

    def forward(self, kspace):
        """
        Args:
            kspace: (D, H, W, 2) or (C, D, H, W, 2) k-space values

        Returns:
            loss: negative ratio of center energy (to maximize)
        """
        # Convert to complex
        if kspace.dim() == 4:
            kspace_complex = torch.complex(kspace[..., 0], kspace[..., 1])
        else:  # Multi-coil
            kspace_complex = torch.complex(kspace[..., 0], kspace[..., 1])

        # Create center mask
        shape = kspace_complex.shape[-3:]  # (D, H, W)
        center_mask = self._create_center_mask(shape, self.center_fraction, kspace.device)

        # Compute energies
        energy_center = (torch.abs(kspace_complex) * center_mask).sum()
        energy_total = torch.abs(kspace_complex).sum()

        # Return negative ratio (we want to maximize center energy)
        return -energy_center / (energy_total + 1e-8)

    def _create_center_mask(self, shape, fraction, device):
        """Create binary mask for k-space center."""
        D, H, W = shape
        center_d = int(D * fraction)
        center_h = int(H * fraction)
        center_w = int(W * fraction)

        mask = torch.zeros(D, H, W, device=device)
        d_start = (D - center_d) // 2
        h_start = (H - center_h) // 2
        w_start = (W - center_w) // 2

        mask[d_start:d_start+center_d,
             h_start:h_start+center_h,
             w_start:w_start+center_w] = 1.0

        return mask


class PhaseSmoothnessLoss(nn.Module):
    """
    Encourage smooth phase variation in k-space.

    Abrupt phase changes often indicate artifacts.
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, kspace):
        """
        Args:
            kspace: (D, H, W, 2) or (C, D, H, W, 2) k-space values

        Returns:
            loss: phase smoothness penalty
        """
        # Convert to complex
        if kspace.dim() == 4:
            kspace_complex = torch.complex(kspace[..., 0], kspace[..., 1])
        else:  # Multi-coil
            kspace_complex = torch.complex(kspace[..., 0], kspace[..., 1])

        # Extract phase
        phase = torch.angle(kspace_complex)  # (..., D, H, W)

        # Compute gradients
        grad_d = phase[..., 1:, :, :] - phase[..., :-1, :, :]
        grad_h = phase[..., :, 1:, :] - phase[..., :, :-1, :]
        grad_w = phase[..., :, :, 1:] - phase[..., :, :, :-1]

        # Wrap phase differences to [-π, π]
        grad_d = torch.atan2(torch.sin(grad_d), torch.cos(grad_d))
        grad_h = torch.atan2(torch.sin(grad_h), torch.cos(grad_h))
        grad_w = torch.atan2(torch.sin(grad_w), torch.cos(grad_w))

        # Total variation of phase
        loss = (torch.abs(grad_d).mean() +
                torch.abs(grad_h).mean() +
                torch.abs(grad_w).mean())

        return loss


class KSpaceSparsityLoss(nn.Module):
    """
    Encourage sparsity in k-space (L1 penalty).

    Sparse k-space often corresponds to structured images.
    """

    def __init__(self):
        super().__init__()

    def forward(self, kspace):
        """
        Args:
            kspace: (D, H, W, 2) or (C, D, H, W, 2) k-space values

        Returns:
            loss: L1 norm of k-space magnitude
        """
        # Convert to complex
        if kspace.dim() == 4:
            kspace_complex = torch.complex(kspace[..., 0], kspace[..., 1])
        else:  # Multi-coil
            kspace_complex = torch.complex(kspace[..., 0], kspace[..., 1])

        # L1 penalty on magnitude
        return torch.abs(kspace_complex).mean()


class KSpaceSymmetryLoss(nn.Module):
    """
    Enforce Hermitian symmetry for real-valued images.

    For real image f(x), F(-k) = F*(k) (conjugate symmetry).
    """

    def __init__(self):
        super().__init__()

    def forward(self, kspace):
        """
        Args:
            kspace: (D, H, W, 2) k-space values

        Returns:
            loss: symmetry violation penalty
        """
        # Convert to complex
        kspace_complex = torch.complex(kspace[..., 0], kspace[..., 1])

        # Flip k-space
        kspace_flipped = torch.flip(kspace_complex, dims=(-3, -2, -1))

        # Should equal conjugate
        kspace_conj = torch.conj(kspace_complex)

        # Measure violation
        loss = F.mse_loss(torch.abs(kspace_flipped), torch.abs(kspace_conj))

        return loss


class CombinedKSpacePriors(nn.Module):
    """
    Combine multiple k-space priors with weights.
    """

    def __init__(self, config):
        super().__init__()

        self.coil_consistency = CoilConsistencyLoss() if config.get('coil_consistency', 0) > 0 else None
        self.support = KSpaceSupportLoss() if config.get('support', 0) > 0 else None
        self.phase_smooth = PhaseSmoothnessLoss() if config.get('phase_smooth', 0) > 0 else None
        self.sparsity = KSpaceSparsityLoss() if config.get('sparsity', 0) > 0 else None
        self.symmetry = KSpaceSymmetryLoss() if config.get('symmetry', 0) > 0 else None

        self.weights = config

    def forward(self, kspace, coil_maps=None):
        """
        Args:
            kspace: k-space values
            coil_maps: coil sensitivity maps (for coil consistency)

        Returns:
            total_loss: weighted sum of priors
            loss_dict: individual loss values
        """
        total_loss = 0.0
        loss_dict = {}

        if self.coil_consistency is not None and coil_maps is not None:
            loss = self.coil_consistency(kspace, coil_maps)
            total_loss += self.weights['coil_consistency'] * loss
            loss_dict['coil_consistency'] = loss.item()

        if self.support is not None:
            loss = self.support(kspace)
            total_loss += self.weights['support'] * loss
            loss_dict['support'] = loss.item()

        if self.phase_smooth is not None:
            loss = self.phase_smooth(kspace)
            total_loss += self.weights['phase_smooth'] * loss
            loss_dict['phase_smooth'] = loss.item()

        if self.sparsity is not None:
            loss = self.sparsity(kspace)
            total_loss += self.weights['sparsity'] * loss
            loss_dict['sparsity'] = loss.item()

        if self.symmetry is not None:
            loss = self.symmetry(kspace)
            total_loss += self.weights['symmetry'] * loss
            loss_dict['symmetry'] = loss.item()

        return total_loss, loss_dict
