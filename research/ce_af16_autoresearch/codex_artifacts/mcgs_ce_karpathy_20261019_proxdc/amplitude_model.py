"""Frozen-G2 complex coefficients with trainable residual amplitude and phase.

The shared field keeps activation memory proportional to ``chunk_size`` by
recomputing each Fourier/MLP chunk during backward. No coordinate or original
coefficient is part of the optimization graph.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


class ResidualAmplitudeField(nn.Module):
    """Apply ``base * exp(bound*tanh(raw_gain)) * exp(1j*phase_delta)``.

    ``plain`` and ``contrast`` select columns 0 and 1 respectively. Both modes
    use the same gain bound; ``explicit`` replaces the shared Fourier MLP with
    an independent ``[N, 2]`` raw log-gain parameter. Input tensors are copied
    into detached buffers so subsequent caller mutations cannot change G2.

    A zero final MLP layer (or zero explicit gains) and zero phase reproduce
    the original finite complex coefficients elementwise exactly.
    """

    def __init__(
        self,
        positions: Tensor,
        plain_complex: Tensor,
        contrast_complex: Tensor,
        *,
        mode: str = "shared",
        bands: int = 4,
        width: int = 64,
        gain_bound: float = 0.25,
        chunk_size: int = 131072,
    ) -> None:
        super().__init__()
        if mode not in {"shared", "explicit"}:
            raise ValueError("mode must be 'shared' or 'explicit'")
        for name, value, minimum in (("bands", bands, 0), ("width", width, 1),
                                     ("chunk_size", chunk_size, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not math.isfinite(gain_bound) or gain_bound < 0:
            raise ValueError("gain_bound must be finite and nonnegative")
        if positions.ndim != 2 or positions.shape[1] != 3 or not positions.is_floating_point():
            raise ValueError("positions must be a real floating-point [N, 3] tensor")
        for name, coefficient in (("plain_complex", plain_complex),
                                  ("contrast_complex", contrast_complex)):
            if coefficient.ndim != 1 or coefficient.shape[0] != positions.shape[0]:
                raise ValueError(f"{name} must have shape [N]")
            if coefficient.dtype not in {torch.complex64, torch.complex128}:
                raise TypeError(f"{name} must use complex64 or complex128")
            if coefficient.device != positions.device:
                raise ValueError("positions and both coefficients must share a device")
        if plain_complex.dtype != contrast_complex.dtype:
            raise ValueError("both coefficient tensors must share a complex dtype")

        self.mode = mode
        self.bands = bands
        self.width = width
        self.gain_bound = float(gain_bound)
        self.chunk_size = chunk_size
        self.num_points = positions.shape[0]
        self.register_buffer("positions", positions.detach().clone())
        self.register_buffer("plain_complex", plain_complex.detach().clone())
        self.register_buffer("contrast_complex", contrast_complex.detach().clone())
        factory = {"device": plain_complex.device, "dtype": plain_complex.real.dtype}
        frequencies = math.pi * 2.0 ** torch.arange(bands, **factory)
        self.register_buffer("frequencies", frequencies)
        self.phase_delta = nn.Parameter(torch.zeros(self.num_points, 2, **factory))
        if mode == "shared":
            self.network = nn.Sequential(
                nn.Linear(3 + 6 * bands, width, **factory),
                nn.SiLU(),
                nn.Linear(width, width, **factory),
                nn.SiLU(),
                nn.Linear(width, 2, **factory),
            )
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)
            self.register_parameter("log_gain", None)
        else:
            self.network = None
            self.log_gain = nn.Parameter(torch.zeros(self.num_points, 2, **factory))

    def _features(self, xyz: Tensor) -> Tensor:
        xyz = xyz.to(dtype=self.frequencies.dtype)
        angles = xyz.unsqueeze(-1) * self.frequencies
        return torch.cat((xyz, angles.sin().flatten(1), angles.cos().flatten(1)), dim=1)

    def _shared_density_chunk(
        self, xyz: Tensor, base: Tensor, phase: Tensor, channel_index: int,
    ) -> Tensor:
        raw_gain = self.network(self._features(xyz))[:, channel_index]
        return self._transform(base, raw_gain, phase)

    def _transform(self, base: Tensor, raw_gain: Tensor, phase: Tensor) -> Tensor:
        gain = self.gain_bound * raw_gain.tanh()
        rotation = torch.complex(phase.cos(), phase.sin())
        return base * gain.exp() * rotation

    def get_density(self, channel: str) -> Tensor:
        """Return a complex ``[N]`` coefficient tensor for the requested scan."""
        if channel not in {"plain", "contrast"}:
            raise ValueError("channel must be 'plain' or 'contrast'")
        index = 0 if channel == "plain" else 1
        base = self.plain_complex if index == 0 else self.contrast_complex
        phase = self.phase_delta[:, index]
        if self.mode == "explicit":
            return self._transform(base, self.log_gain[:, index], phase)
        if self.num_points == 0:
            return base.clone()
        chunks = []
        for start in range(0, self.num_points, self.chunk_size):
            stop = min(start + self.chunk_size, self.num_points)
            args = (self.positions[start:stop], base[start:stop], phase[start:stop], index)
            if torch.is_grad_enabled():
                # Non-reentrant checkpoint preserves gradients of captured MLP
                # parameters even though the coordinate/base inputs are frozen.
                chunk = checkpoint(self._shared_density_chunk, *args, use_reentrant=False)
            else:
                chunk = self._shared_density_chunk(*args)
            chunks.append(chunk)
        return torch.cat(chunks)

    def parameter_groups(self, lr: float = 0.001) -> list[dict]:
        """Return disjoint named Adam groups; neither includes frozen buffers."""
        amplitude = list(self.network.parameters()) if self.mode == "shared" else [self.log_gain]
        return [
            {"name": "residual_amplitude", "params": amplitude, "lr": lr},
            {"name": "phase_delta", "params": [self.phase_delta], "lr": lr},
        ]

    def metadata(self) -> dict:
        """Describe the recipe; the runner persists it alongside ``state_dict``."""
        return {
            "model": type(self).__name__,
            "schema_version": 1,
            "mode": self.mode,
            "num_points": self.num_points,
            "bands": self.bands,
            "width": self.width,
            "gain_bound": self.gain_bound,
            "chunk_size": self.chunk_size,
            "coefficient_dtype": str(self.plain_complex.dtype),
            "geometry_frozen": True,
            "base_coefficients_frozen": True,
            "gain_transform": "gain_bound * tanh(raw_log_gain)",
            "coordinate_features": "raw_xyz,sin(pi*2**k*xyz),cos(pi*2**k*xyz);k=0..bands-1",
            "checkpoint": "non_reentrant_per_chunk" if self.mode == "shared" else None,
        }
