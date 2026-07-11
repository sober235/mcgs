"""PairedARCModel — shared anatomy Gaussian model with response-decoded densities.

Spec §4-§6. The container holds ONE shared anatomy Gaussian set G_A:

    G_A = { μ_n, Σ_n(via scales,rotations), z_n^A, π_n^A }        (§5)

and produces the per-contrast density NOT from independent tables but from the response
decoder (§6):

    α_{m,n}^A = D_ψ(z_n^A, e_m)

The image for contrast m is rendered by the (reused) voxelizer with SHARED geometry and
decoder-produced density:

    x_m^A = V( μ, Σ, α_m^A )                                     (§4, base term)

Named parameter groups (for the stage scheduler's freeze/unfreeze, §13):
    "geometry"  : positions, scales(log), rotations
    "latent"    : z_A  (+ confidence logits π_A)
    "decoder"   : the ResponseDecoder MLP
    ("change","phase","deformation","gabor" contributed by their modules in later layers)

The optional `decoder_enable=False` path falls back to two independent density tables — the
original MC-GS baseline (§17 row 2 / §3.2), kept so the ablation can be run within ONE model.
"""
from __future__ import annotations

from typing import Tuple, List, Dict, Any, Optional
import torch
import torch.nn as nn

from ..modules.response_decoder import ResponseDecoder
# Geometry->covariance and rendering are handled by the reused paired_gs voxelizer
# (gaussian/voxelizer.py, gaussian/tile_voxelizer.py); this model only holds parameters.


class PairedARCModel(nn.Module):
    def __init__(self, num_points: int, volume_shape: Tuple[int, int, int], device,
                 latent_dim: int = 12, n_contrasts: int = 2, use_confidence: bool = True,
                 decoder_enable: bool = True, decoder_hidden: int = 24, decoder_layers: int = 2,
                 condition_dim: int = 4, shared_trunk: bool = True,
                 initial_positions: Optional[torch.Tensor] = None,
                 initial_scales: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_points = int(num_points)
        self.volume_shape = tuple(volume_shape)
        self.device = device
        self.n_contrasts = int(n_contrasts)
        self.latent_dim = int(latent_dim)
        self.use_confidence = bool(use_confidence)
        self.decoder_enable = bool(decoder_enable)

        n = self.num_points
        # ---- shared geometry (§5): positions in [-1,1], scales in log-space, unit-quat rotations
        pos = initial_positions if initial_positions is not None else (torch.rand(n, 3, device=device) * 2 - 1)
        self.positions = nn.Parameter(pos.contiguous())
        if initial_scales is None:
            base = 2.0 / max(self.volume_shape)
            initial_scales = torch.full((n, 3), base, device=device)
        self.scales = nn.Parameter(torch.log(torch.clamp(initial_scales, min=1e-8)).contiguous())
        rot = torch.zeros(n, 4, device=device); rot[:, 0] = 1.0
        self.rotations = nn.Parameter(rot)

        # ---- anatomy latent z_A (§5) + confidence π_A (stored as logit; π = sigmoid) ----
        self.z_A = nn.Parameter(torch.randn(n, self.latent_dim, device=device) * 0.1)
        if self.use_confidence:
            self.pi_logit = nn.Parameter(torch.zeros(n, device=device))  # π_n^A = σ(logit), init 0.5

        # ---- response decoder (§6) OR fallback independent density tables (§3.2 baseline) ----
        if self.decoder_enable:
            self.decoder = ResponseDecoder(latent_dim=self.latent_dim, hidden_dim=decoder_hidden,
                                           n_contrasts=self.n_contrasts, condition_dim=condition_dim,
                                           n_layers=decoder_layers, shared_trunk=shared_trunk)
            self.density_tables = None
        else:
            self.decoder = None
            # independent per-contrast complex density tables (the MC-GS design §3.2)
            self.density_tables = nn.ParameterList([
                nn.Parameter(torch.randn(n, 2, device=device) * 0.1) for _ in range(self.n_contrasts)
            ])

    # ---------------------------------------------------------------- geometry accessors
    def get_scale_values(self) -> torch.Tensor:
        return torch.exp(self.scales)

    @property
    def confidence(self) -> Optional[torch.Tensor]:
        return torch.sigmoid(self.pi_logit) if self.use_confidence else None

    # ---------------------------------------------------------------- density (§6)
    def get_density(self, contrast_index: int) -> torch.Tensor:
        """Per-Gaussian complex density for contrast m (§6): decoder(z_A, e_m), or the
        fallback independent table when the decoder is disabled."""
        if self.decoder_enable:
            return self.decoder(self.z_A, contrast_index)
        t = self.density_tables[contrast_index]
        return torch.complex(t[:, 0], t[:, 1])

    # ---------------------------------------------------------------- param groups (§13)
    def param_group_names(self) -> List[str]:
        names = ["geometry", "latent"]
        if self.decoder_enable:
            names.append("decoder")
        return names

    def param_groups(self, lr_cfg: Dict[str, float]) -> Dict[str, List[Dict[str, Any]]]:
        """Return {group_name: [optim param-group dict]} so the trainer can build one Adam
        with per-group LRs and the scheduler can freeze a group by zeroing its LR."""
        groups: Dict[str, List[Dict[str, Any]]] = {}
        groups["geometry"] = [{"params": [self.positions, self.scales, self.rotations],
                               "lr": lr_cfg.get("geometry", 3e-4)}]
        latent_params = [self.z_A] + ([self.pi_logit] if self.use_confidence else [])
        groups["latent"] = [{"params": latent_params, "lr": lr_cfg.get("latent", 5e-3)}]
        if self.decoder_enable:
            groups["decoder"] = [{"params": list(self.decoder.parameters()),
                                  "lr": lr_cfg.get("decoder", 5e-3)}]
        else:
            groups["latent"].append({"params": list(self.density_tables.parameters()),
                                     "lr": lr_cfg.get("latent", 5e-3)})
        return groups

    def parameters_summary(self) -> Dict[str, int]:
        s = {"num_points": self.num_points, "latent_dim": self.latent_dim,
             "decoder": bool(self.decoder_enable), "n_contrasts": self.n_contrasts}
        if self.decoder_enable:
            s["decoder_params"] = sum(p.numel() for p in self.decoder.parameters())
        return s
