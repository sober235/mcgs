"""§6 Conditioned contrast-response decoder D_ψ(z_A, e_m).

Spec §6.1-§6.2: the same shared Gaussian, described by an anatomy latent z_n^A, produces a
DIFFERENT complex density in each imaging condition via a lightweight decoder:

    α_{m,n}^A = D_ψ(z_n^A, e_m)                                             (§6.1)

    h_n = σ(W1 z_n^A + b1)
    [Re(α_{m,n}), Im(α_{m,n})] = W_{2,m} h_n + b_{2,m}                      (§6.2)

Design constraints from §6.2 (enforced here + in config validator):
  * latent dim 8-16, hidden dim 16-32, TWO layers — deliberately LOW capacity, so the
    decoder cannot re-memorize each image (which would destroy the shared-anatomy constraint).
  * a shared trunk (W1) + PER-CONTRAST output heads (W_{2,m}) — T1/FLAIR and pre/post each
    get their own head. The imaging-condition code e_m is concatenated to z so ONE trunk can
    serve multiple pairs (spec's unified D_ψ(z, e_m) form).

This module REPLACES the two independent density tables of the original MC-GS
(density_real/imag + density_real_contrast/imag) that §3.2 critiques as "shared spatial
dictionary + independent coefficients" rather than a true shared-anatomy → contrast model.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn

from .interfaces import ResponseDecoderBase


class ResponseDecoder(ResponseDecoderBase):
    def __init__(self, latent_dim: int = 12, hidden_dim: int = 24, n_contrasts: int = 2,
                 condition_dim: int = 4, n_layers: int = 2, shared_trunk: bool = True):
        super().__init__()
        self.enabled = True
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_contrasts = int(n_contrasts)
        self.condition_dim = int(condition_dim)

        # imaging-condition codes e_m, one row per contrast (§6.1). Learnable so the model
        # can place the two conditions in the trunk's input space; small and fixed-size.
        self.condition_codes = nn.Parameter(torch.zeros(n_contrasts, condition_dim))
        nn.init.normal_(self.condition_codes, std=0.1)

        in_dim = self.latent_dim + self.condition_dim
        # shared trunk W1 (σ = tanh, bounded — keeps capacity low per §6.2)
        trunk_layers: List[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(max(0, n_layers - 2)):
            trunk_layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        self.trunk = nn.Sequential(*trunk_layers) if shared_trunk else None

        # per-contrast output heads W_{2,m}: hidden -> 2 (Re, Im)
        head_in = hidden_dim if shared_trunk else in_dim
        self.heads = nn.ModuleList([nn.Linear(head_in, 2) for _ in range(n_contrasts)])
        # init heads small so α starts near zero (ACS init supplies the coarse signal)
        for h in self.heads:
            nn.init.normal_(h.weight, std=1e-2); nn.init.zeros_(h.bias)

    def forward(self, z_A: torch.Tensor, contrast_index: int) -> torch.Tensor:
        """z_A: (N, latent_dim) -> (N,) complex density for the given contrast."""
        n = z_A.shape[0]
        e = self.condition_codes[contrast_index].unsqueeze(0).expand(n, -1)
        h = torch.cat([z_A, e], dim=1)
        if self.trunk is not None:
            h = self.trunk(h)
        out = self.heads[contrast_index](h)          # (N, 2)
        return torch.complex(out[:, 0], out[:, 1])

    def param_groups(self, lr: float) -> List[Dict[str, Any]]:
        return [{"params": list(self.parameters()), "lr": lr}]
