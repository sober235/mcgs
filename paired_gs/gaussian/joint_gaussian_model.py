"""JointGaussianModel3D: shared geometry, dual densities for plain+contrast joint reconstruction.

Architecture:
    Shared parameters (geometry):  positions, scales, rotations
    Dual parameters (intensity):   density_plain (real, imag), density_contrast (real, imag)

The same Gaussian set describes anatomy seen from both scans; only per-Gaussian amplitudes
differ between plain and contrast renderings. This lets joint k-space supervision (plain ksp +
contrast ksp) double-constrain the geometry, suppressing aliasing artifacts.
"""
from __future__ import annotations

from typing import Tuple, List, Dict, Any, Optional

import torch
import torch.nn as nn

from .gaussian_model import GaussianModel3D


class JointGaussianModel3D(GaussianModel3D):
    """Shared-geometry, dual-density Gaussian model.

    Inherits position/scale/rotation/density (as 'plain' channel) from GaussianModel3D,
    and adds a parallel ('contrast') density set.
    """

    def __init__(
        self,
        num_points: int,
        volume_shape: Tuple[int, int, int],
        device: torch.device,
        initial_positions: torch.Tensor | None = None,
        initial_densities_plain: torch.Tensor | None = None,
        initial_densities_contrast: torch.Tensor | None = None,
        initial_scales: torch.Tensor | None = None,
    ):
        super().__init__(
            num_points=num_points,
            volume_shape=volume_shape,
            device=device,
            initial_positions=initial_positions,
            initial_densities=initial_densities_plain,
            initial_scales=initial_scales,
        )
        # parent already created density_real / density_imag (used as 'plain' channel)
        # add contrast channel
        if initial_densities_contrast is None:
            init_c = (self.density_real.detach().clone(), self.density_imag.detach().clone())
        else:
            init_c = (initial_densities_contrast.real.contiguous(),
                      initial_densities_contrast.imag.contiguous())
        self.density_real_contrast = nn.Parameter(init_c[0])
        self.density_imag_contrast = nn.Parameter(init_c[1])

    @property
    def density_plain(self) -> torch.Tensor:
        return torch.complex(self.density_real, self.density_imag)

    @property
    def density_contrast(self) -> torch.Tensor:
        return torch.complex(self.density_real_contrast, self.density_imag_contrast)

    def get_density(self, channel: str) -> torch.Tensor:
        if channel == "plain":
            return self.density_plain
        if channel == "contrast":
            return self.density_contrast
        raise ValueError(f"unknown channel '{channel}', expected 'plain' or 'contrast'")

    # -- alignment-aware contrast-private residual (MC-GS) --------------
    @property
    def has_private(self) -> bool:
        """True once a contrast-private residual Gaussian set has been attached."""
        return getattr(self, "num_private", 0) > 0

    def add_private_residual(
        self,
        positions: torch.Tensor,
        scales_log: torch.Tensor,
        rotations: torch.Tensor,
        density_real: torch.Tensor,
        density_imag: torch.Tensor,
    ) -> None:
        """Attach a contrast-private residual: a SEPARATE Gaussian set that renders ONLY into
        the contrast channel, added on top of the shared-geometry contrast render.

        The shared geometry is double-constrained by both contrasts' k-space DC and is thus
        plain-dominated; contrast-specific signal (lesions, contrast differences, residual
        misalignment) that the shared cloud cannot place is absorbed here. These Gaussians
        own their geometry (positions/scales/rotations) and amplitude, and train on the contrast
        DC term alone — the alignment-aware contrast-private residual of the MC-GS spec.

        scales_log is stored in log-space to match the parent's `self.scales` convention.
        """
        self.private_positions = nn.Parameter(positions.contiguous())
        self.private_scales = nn.Parameter(scales_log.contiguous())
        self.private_rotations = nn.Parameter(rotations.contiguous())
        self.private_density_real = nn.Parameter(density_real.contiguous())
        self.private_density_imag = nn.Parameter(density_imag.contiguous())
        self.num_private = int(positions.shape[0])

    def get_private_scale_values(self) -> torch.Tensor:
        return torch.exp(self.private_scales)

    def get_private_density(self) -> torch.Tensor:
        return torch.complex(self.private_density_real, self.private_density_imag)

    def _update_params(
        self,
        mask: torch.Tensor,
        new_pos: Optional[torch.Tensor] = None,
        new_scale: Optional[torch.Tensor] = None,
        new_rot: Optional[torch.Tensor] = None,
        new_dr: Optional[torch.Tensor] = None,
        new_di: Optional[torch.Tensor] = None,
        is_prune: bool = False,
    ):
        """Sync contrast density rows when shared geometry is densified/pruned."""
        has_contrast_params = (
            "density_real_contrast" in self._parameters
            and "density_imag_contrast" in self._parameters
        )
        old_contrast_re = self.density_real_contrast.data.clone() if has_contrast_params else None
        old_contrast_im = self.density_imag_contrast.data.clone() if has_contrast_params else None
        old_scales = self.get_scale_values().detach().clone()

        super()._update_params(mask, new_pos, new_scale, new_rot, new_dr, new_di, is_prune=is_prune)

        if not has_contrast_params:
            return

        if is_prune:
            new_contrast_re = old_contrast_re[mask]
            new_contrast_im = old_contrast_im[mask]
        else:
            kept_re = old_contrast_re[mask]
            kept_im = old_contrast_im[mask]
            parent_re = old_contrast_re[~mask]
            parent_im = old_contrast_im[~mask]
            n_appended = int(new_pos.shape[0]) if new_pos is not None else 0
            n_parent = int(parent_re.shape[0])
            if n_parent > 0 and new_scale is not None and n_appended >= n_parent:
                parent_scale = old_scales[~mask]
                child_scale = new_scale[:n_parent]
                volume_factor = torch.prod(
                    child_scale / parent_scale.clamp_min(1e-12),
                    dim=-1,
                ).clamp_min(1e-12)
                density_correction = 0.5 / volume_factor
                children_re = torch.cat(
                    [parent_re * density_correction, parent_re * density_correction],
                    dim=0,
                )
                children_im = torch.cat(
                    [parent_im * density_correction, parent_im * density_correction],
                    dim=0,
                )
            else:
                children_re = old_contrast_re.new_empty(0)
                children_im = old_contrast_im.new_empty(0)
            children_re = _align_rows(children_re, n_appended)
            children_im = _align_rows(children_im, n_appended)
            new_contrast_re = torch.cat([kept_re, children_re], dim=0)
            new_contrast_im = torch.cat([kept_im, children_im], dim=0)

        self.density_real_contrast = nn.Parameter(new_contrast_re)
        self.density_imag_contrast = nn.Parameter(new_contrast_im)

    def densify_and_clone(self, grads: torch.Tensor, grad_threshold: float, scale_threshold: float) -> int:
        has_contrast_params = (
            "density_real_contrast" in self._parameters
            and "density_imag_contrast" in self._parameters
        )
        with torch.no_grad():
            scales = self.get_scale_values()
            max_scales = scales.max(dim=-1)[0]
            mask = (grads > grad_threshold) & (max_scales <= scale_threshold)
            if mask.sum() == 0:
                return 0

            old_re = self.density_real_contrast.data.clone() if has_contrast_params else None
            old_im = self.density_imag_contrast.data.clone() if has_contrast_params else None

        clone_count = super().densify_and_clone(grads, grad_threshold, scale_threshold)
        if clone_count == 0:
            return 0
        if not has_contrast_params:
            return clone_count

        with torch.no_grad():
            new_re = torch.cat([old_re, old_re[mask] * 0.5], dim=0)
            new_im = torch.cat([old_im, old_im[mask] * 0.5], dim=0)
        self.density_real_contrast = nn.Parameter(new_re)
        self.density_imag_contrast = nn.Parameter(new_im)
        return clone_count

    def get_optimizable_params(
        self,
        lr_position: float = 1e-4,
        lr_density: float = 1e-3,
        lr_scale: float = 5e-4,
        lr_rotation: float = 1e-4,
        lr_private_position: float | None = None,
        lr_private_density: float | None = None,
        lr_private_scale: float | None = None,
        lr_private_rotation: float | None = None,
    ) -> List[Dict[str, Any]]:
        groups = [
            {"params": [self.positions], "lr": lr_position},
            {"params": [self.scales], "lr": lr_scale},
            {"params": [self.rotations], "lr": lr_rotation},
            {"params": [self.density_real], "lr": lr_density},
            {"params": [self.density_imag], "lr": lr_density},
            {"params": [self.density_real_contrast], "lr": lr_density},
            {"params": [self.density_imag_contrast], "lr": lr_density},
        ]
        if self.has_private:
            groups += [
                {"params": [self.private_positions], "lr": lr_private_position or lr_position},
                {"params": [self.private_scales], "lr": lr_private_scale or lr_scale},
                {"params": [self.private_rotations], "lr": lr_private_rotation or lr_rotation},
                {"params": [self.private_density_real], "lr": lr_private_density or lr_density},
                {"params": [self.private_density_imag], "lr": lr_private_density or lr_density},
            ]
        return groups

    def amplitude_difference(self) -> torch.Tensor:
        """|alpha_contrast| - |alpha_plain|, real-valued per-Gaussian.

        Used by sparsity (L1) and non-negativity (ReLU(plain - contrast)) priors.
        """
        return self.density_contrast.abs() - self.density_plain.abs()

    def parameters_summary(self) -> Dict[str, int]:
        summary = {
            "positions": self.positions.numel(),
            "scales": self.scales.numel(),
            "rotations": self.rotations.numel(),
            "density_plain": self.density_real.numel() * 2,
            "density_contrast": self.density_real_contrast.numel() * 2,
            "num_points": self.num_points,
        }
        if self.has_private:
            summary["num_private"] = self.num_private
        return summary


def _align_rows(rows: torch.Tensor, n_rows: int) -> torch.Tensor:
    if rows.shape[0] == n_rows:
        return rows
    if rows.shape[0] > n_rows:
        return rows[:n_rows]
    pad_shape = list(rows.shape)
    pad_shape[0] = n_rows - rows.shape[0]
    return torch.cat(
        [rows, torch.zeros(pad_shape, device=rows.device, dtype=rows.dtype)],
        dim=0,
    )
