"""§11 Gaussian–Gabor hybrid representation.

Spec §11: a plain Gaussian has Fourier spectrum ∝ exp(−2π² kᵀΣk)·e^{−j2πkᵀμ}, whose energy is
naturally concentrated at low frequency — so the original MC-GS needs MANY tiny Gaussians and
a large high-freq DC weight to render sharp edges. Instead, split the shared representation:

    x_m^A = V_G(G_A, α_m^G) + V_B(B_A, α_m^B),

  * Gaussian primitives  → smooth tissue, low/mid frequencies (the existing anatomy model);
  * Gabor primitives      → local edges and high-frequency detail.

A Gabor primitive is a Gaussian envelope modulated by a local carrier frequency ξ_l:

    B_l(r) = G_l(r) · exp[ j 2π ξ_lᵀ (r − μ_l) ].

Per §11 the Gabor branch is NOT opened from the start: it is born AFTER the shared Gaussians
stabilize, from held-out high-frequency k-space error, with a STRICTLY limited primitive count.

Implementation: because the count is small (tens–hundreds), each Gabor is rendered analytically
on the voxel grid — an anisotropic Gaussian envelope (from μ, Σ via a rotation+scale) times the
complex carrier. This is exact (no reliance on the real-only tile voxelizer) and differentiable.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.utils.checkpoint

from .interfaces import PairedModule


def _quat_to_R(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion (w,x,y,z) -> 3x3 rotation. q: (...,4)."""
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return R


class GaborBranch(PairedModule):
    """Shared Gabor primitives B_A with per-contrast complex amplitudes α_m^B. Positions,
    scales (log), rotations (quat) and carrier frequencies ξ are SHARED (they describe a
    shared edge structure); amplitude is per-contrast. Starts empty; born at stage s6."""

    def __init__(self, n_contrasts: int, volume_shape, device,
                 max_primitives: int = 2000, init_scale_vox: float = 1.0):
        super().__init__()
        self.enabled = True
        self.n_contrasts = int(n_contrasts)
        self.volume_shape = tuple(int(s) for s in volume_shape)
        self.device = device
        self.max_primitives = int(max_primitives)
        D, H, W = self.volume_shape
        self._base_logscale = torch.log(torch.tensor(
            [init_scale_vox * 2.0 / D, init_scale_vox * 2.0 / H, init_scale_vox * 2.0 / W],
            device=device))
        # precompute normalized voxel coordinate grid in [-1,1] (axis order z,y,x)
        zs = torch.linspace(-1, 1, D, device=device)
        ys = torch.linspace(-1, 1, H, device=device)
        xs = torch.linspace(-1, 1, W, device=device)
        gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")
        self.register_buffer("_coords", torch.stack([gz, gy, gx], dim=-1))  # (D,H,W,3)
        # shared geometry (empty) + carrier ξ (cycles per normalized unit) + per-contrast α
        self.positions = nn.Parameter(torch.zeros(0, 3, device=device))
        self.log_scales = nn.Parameter(torch.zeros(0, 3, device=device))
        self.rotations = nn.Parameter(torch.zeros(0, 4, device=device))
        self.xi = nn.Parameter(torch.zeros(0, 3, device=device))
        self.alpha = nn.ParameterList([nn.Parameter(torch.zeros(0, 2, device=device)) for _ in range(self.n_contrasts)])

    def count(self) -> int:
        return self.positions.shape[0]

    @torch.no_grad()
    def add_primitives(self, positions, xi=None, scales_norm=None):
        k = positions.shape[0]
        if k == 0:
            return
        cap = self.max_primitives - self.count()
        if cap <= 0:
            return
        if k > cap:
            positions = positions[:cap]; k = cap
        logs = (torch.log(scales_norm) if scales_norm is not None
                else self._base_logscale.unsqueeze(0).expand(k, 3)).to(self.device)
        rot = torch.zeros(k, 4, device=self.device); rot[:, 0] = 1.0
        xi_new = xi[:k].to(self.device) if xi is not None else torch.zeros(k, 3, device=self.device)
        def cat(p, new):
            return nn.Parameter(torch.cat([p.data, new], dim=0))
        self.positions = cat(self.positions, positions.detach().to(self.device))
        self.log_scales = cat(self.log_scales, logs)
        self.rotations = cat(self.rotations, rot)
        self.xi = cat(self.xi, xi_new.detach())
        for ci in range(self.n_contrasts):
            self.alpha[ci] = nn.Parameter(torch.cat(
                [self.alpha[ci].data, torch.randn(k, 2, device=self.device) * 1e-3], dim=0))

    def forward(self, contrast_index: int, voxelizer=None, volume_shape=None) -> Optional[torch.Tensor]:
        """Render V_B for contrast m: Σ_l α_{m,l} · G_l(r) · exp[j2π ξ_lᵀ(r−μ_l)]  (D,H,W complex).
        Analytic anisotropic-Gaussian envelope × complex carrier, summed over primitives."""
        if not self.enabled or self.count() == 0:
            return None
        ci = contrast_index
        coords = self._coords                                   # (D,H,W,3)
        R = _quat_to_R(self.rotations)                          # (L,3,3)
        inv_s2 = torch.exp(-2.0 * self.log_scales)              # 1/scale^2 (L,3)
        alpha = torch.complex(self.alpha[ci][:, 0], self.alpha[ci][:, 1])  # (L,)
        D, H, W, _ = coords.shape

        def _chunk_sum(lo: int, hi: int) -> torch.Tensor:
            """Sum the V_B contribution of primitives [lo, hi). Same math as the exact
            per-primitive loop; isolated so it can be gradient-checkpointed (§11 render is
            per-primitive full-volume, so retaining every primitive's graph is O(L·DHW) —
            checkpointing bounds it to O(chunk·DHW))."""
            acc = torch.zeros(D, H, W, dtype=torch.complex64, device=coords.device)
            for l in range(lo, hi):
                d = coords - self.positions[l]                  # (D,H,W,3) centered
                dl = torch.einsum("ij,dhwj->dhwi", R[l], d)     # rotate into primitive frame
                quad = (dl * dl * inv_s2[l]).sum(-1)            # Mahalanobis (D,H,W)
                env = torch.exp(-0.5 * quad)                    # Gaussian envelope
                phase = 2 * torch.pi * (d * self.xi[l]).sum(-1) # carrier ξᵀ(r-μ)
                carrier = torch.exp(1j * phase.to(torch.float32))
                acc = acc + alpha[l] * env.to(torch.complex64) * carrier
            return acc

        L = self.count()
        # loop over primitives in chunks (count is strictly limited per §11); checkpoint each
        # chunk during training so peak memory is O(chunk_size · DHW), not O(L · DHW)
        chunk = int(getattr(self, "render_chunk", 16))
        out = torch.zeros(D, H, W, dtype=torch.complex64, device=coords.device)
        use_ckpt = torch.is_grad_enabled() and any(
            p.requires_grad for p in ([self.positions, self.log_scales, self.rotations, self.xi]
                                      + list(self.alpha)) if p.numel() > 0)
        for lo in range(0, L, chunk):
            hi = min(lo + chunk, L)
            if use_ckpt:
                out = out + torch.utils.checkpoint.checkpoint(_chunk_sum, lo, hi, use_reentrant=False)
            else:
                out = out + _chunk_sum(lo, hi)
        return out

    def param_groups(self, lr: float) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        ps = [self.positions, self.log_scales, self.rotations, self.xi] + list(self.alpha)
        ps = [p for p in ps if p.numel() > 0]
        return [{"params": ps, "lr": lr}] if ps else []
