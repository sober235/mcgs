"""Positive shared amplitudes over frozen Gaussian positions and log scales.

Only unit source phases and two global amplitude scales survive coefficient
initialization. No original per-point modulus enters the prediction path.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


class FullAmplitudeField(nn.Module):
    """Predict two positive moduli with a Fourier MLP or trilinear grid MLP.

    ``plain`` and ``contrast`` select columns zero and one. ``phase_delta`` is
    separate and frozen by default; amplitude optimizers never receive it.
    Source statistics use the entire input, including for indexed queries.
    """

    def __init__(
        self,
        positions: Tensor,
        log_scales: Tensor,
        plain_complex: Tensor,
        contrast_complex: Tensor,
        *,
        encoder: str = "mlp",
        bands: int = 8,
        width: int = 64,
        chunk_size: int = 131072,
    ) -> None:
        super().__init__()
        if encoder not in {"mlp", "grid"}:
            raise ValueError("encoder must be 'mlp' or 'grid'")
        for name, value, minimum in (("bands", bands, 0), ("width", width, 1),
                                     ("chunk_size", chunk_size, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name, value in (("positions", positions), ("log_scales", log_scales)):
            if not isinstance(value, Tensor) or value.ndim != 2 or value.shape[1] != 3:
                raise ValueError(f"{name} must have shape [N, 3]")
            if value.dtype not in {torch.float32, torch.float64}:
                raise TypeError(f"{name} must use float32 or float64")
        if positions.shape[0] == 0:
            raise ValueError("source must contain at least one point")
        if log_scales.shape != positions.shape:
            raise ValueError("log_scales must match positions shape")
        if log_scales.dtype != positions.dtype or log_scales.device != positions.device:
            raise ValueError("positions and log_scales must share dtype and device")
        expected_complex = torch.complex64 if positions.dtype == torch.float32 else torch.complex128
        for name, value in (("plain_complex", plain_complex), ("contrast_complex", contrast_complex)):
            if not isinstance(value, Tensor) or value.shape != (positions.shape[0],):
                raise ValueError(f"{name} must have shape [N]")
            if value.dtype != expected_complex:
                raise TypeError(f"{name} must match the source real precision as complex64/complex128")
            if value.device != positions.device:
                raise ValueError("all source tensors must share a device")
        for name, value in (("positions", positions), ("log_scales", log_scales),
                            ("plain_complex", plain_complex), ("contrast_complex", contrast_complex)):
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values")

        source = torch.stack((plain_complex.detach(), contrast_complex.detach()), dim=1)
        modulus = source.abs()
        # Finite complex components can still have an unrepresentable modulus.
        # Validate every point before deriving either scales or unit phases.
        if not torch.isfinite(modulus).all():
            raise ValueError("source moduli must be finite in the source real dtype")

        self.encoder = encoder
        self.bands = bands
        self.width = width
        self.chunk_size = chunk_size
        self.num_points = positions.shape[0]
        self.register_buffer("positions", positions.detach().clone())
        self.register_buffer("log_scales", log_scales.detach().clone())
        self.register_buffer("log_scale_mean", self.log_scales.mean(dim=0))
        # Population std is defined for a one-point source and constant axes.
        self.register_buffer("log_scale_std", self.log_scales.std(dim=0, unbiased=False).clamp_min(1e-6))
        self.register_buffer("amplitude_scale", modulus.median(dim=0).values.clamp_min(1e-12))
        safe_modulus = torch.where(modulus > 0, modulus, torch.ones_like(modulus))
        unit_phase = torch.where(modulus > 0, source / safe_modulus, torch.ones_like(source))
        self.register_buffer("unit_phase", unit_phase)
        factory = {"device": positions.device, "dtype": positions.dtype}
        self.phase_delta = nn.Parameter(torch.zeros(self.num_points, 2, **factory), requires_grad=False)
        self.register_buffer("frequencies", math.pi * 2.0 ** torch.arange(bands, **factory))
        self.grids = nn.ParameterList()
        if encoder == "grid":
            for resolution in (8, 16, 32, 64):
                grid = nn.Parameter(torch.empty(1, 2, resolution, resolution, resolution, **factory))
                nn.init.normal_(grid, mean=0, std=1e-4)
                self.grids.append(grid)
            feature_count = 6 + 8
        else:
            feature_count = 6 + 6 * bands
        self.network = nn.Sequential(
            nn.Linear(feature_count, width, **factory), nn.SiLU(),
            nn.Linear(width, width, **factory), nn.SiLU(),
            nn.Linear(width, 2, **factory),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, math.log(math.expm1(1.0)))

    def _magnitude_chunk(self, xyz: Tensor, logs: Tensor) -> Tensor:
        standardized_logs = (logs - self.log_scale_mean) / self.log_scale_std
        if self.encoder == "mlp":
            angles = xyz.unsqueeze(-1) * self.frequencies
            features = torch.cat((xyz, angles.sin().flatten(1), angles.cos().flatten(1),
                                  standardized_logs), dim=1)
        else:
            query = (xyz / 1.1).reshape(1, 1, 1, -1, 3)
            learned = [F.grid_sample(grid, query, mode="bilinear", padding_mode="border",
                                     align_corners=True).reshape(2, -1).T for grid in self.grids]
            features = torch.cat((xyz, standardized_logs, *learned), dim=1)
        return self.amplitude_scale * F.softplus(self.network(features))

    def _validate_indices(self, indices: Tensor | None) -> None:
        if indices is None:
            return
        if not isinstance(indices, Tensor) or indices.ndim != 1:
            raise ValueError("indices must be a one-dimensional integer tensor")
        if indices.dtype not in {torch.int32, torch.int64}:
            raise TypeError("indices must use int32 or int64")
        if indices.device != self.positions.device:
            raise ValueError("indices and source must share a device")
        if torch.any(indices < 0) or torch.any(indices >= self.num_points):
            raise IndexError("point index outside source")

    def magnitudes(self, indices: Tensor | None = None) -> Tensor:
        """Return shared positive amplitudes ``[n, 2]`` in query order."""
        self._validate_indices(indices)
        count = self.num_points if indices is None else indices.numel()
        if count == 0:
            return self.amplitude_scale.new_empty((0, 2))
        chunks = []
        for start in range(0, count, self.chunk_size):
            selection = slice(start, start + self.chunk_size) if indices is None else indices[start:start + self.chunk_size]
            args = (self.positions[selection], self.log_scales[selection])
            if torch.is_grad_enabled():
                # Frozen inputs still yield parameter gradients with nonreentrant
                # checkpointing, including grid parameters captured by the method.
                value = checkpoint(self._magnitude_chunk, *args, use_reentrant=False)
            else:
                value = self._magnitude_chunk(*args)
            chunks.append(value)
        return torch.cat(chunks, dim=0)

    def get_density(self, channel: str, indices: Tensor | None = None) -> Tensor:
        """Return complex coefficients with frozen source unit phases, ``[n]``."""
        if channel not in {"plain", "contrast"}:
            raise ValueError("channel must be 'plain' or 'contrast'")
        column = 0 if channel == "plain" else 1
        amplitude = self.magnitudes(indices)[:, column]
        selection = slice(None) if indices is None else indices
        phase = self.phase_delta[selection, column]
        rotation = torch.complex(phase.cos(), phase.sin())
        return amplitude * self.unit_phase[selection, column] * rotation

    def amplitude_parameters(self) -> list[nn.Parameter]:
        """Return network and grid parameters, excluding the independent phase."""
        return list(self.network.parameters()) + list(self.grids.parameters())

    def metadata(self) -> dict:
        """Describe the field recipe using JSON-safe complete-source values."""
        return {
            "model": type(self).__name__,
            "schema_version": 1,
            "encoder": self.encoder,
            "num_points": self.num_points,
            "bands": self.bands,
            "width": self.width,
            "chunk_size": self.chunk_size,
            "dtype": str(self.positions.dtype),
            "amplitude_scale": self.amplitude_scale.detach().cpu().tolist(),
            "log_scale_mean": self.log_scale_mean.detach().cpu().tolist(),
            "log_scale_std": self.log_scale_std.detach().cpu().tolist(),
            "domain": [-1.1, 1.1],
            "domain_outside_fraction": (self.positions.abs() > 1.1).any(dim=1).double().mean().item(),
            "grid_resolutions": [8, 16, 32, 64] if self.encoder == "grid" else [],
            "grid_channels": 2 if self.encoder == "grid" else 0,
            "geometry_frozen": True,
            "source_phases_frozen": True,
            "phase_delta_requires_grad": self.phase_delta.requires_grad,
            "amplitude_transform": "amplitude_scale * softplus(raw)",
            "checkpoint": "non_reentrant_per_chunk",
        }
