"""Edge-preserving (edge-weighted) anisotropic TV — a self-contained edge-directed
regularizer that smooths flat-region grain while sparing sharp anatomical edges.
Detached Perona-Malik spatial weight => stable (magnitude/bounded-gradient convention).
Spec: docs/superpowers/specs/2026-07-23-edge-preserving-tv-csf-design.md"""
import numpy as np
import torch


def _edge_weight(g: torch.Tensor, kappa: float, w_floor: float) -> torch.Tensor:
    """~1 in flat regions, ->w_floor at sharp edges. |g| detached => spatial weight only."""
    gd = g.detach().abs()
    return torch.clamp(1.0 / (1.0 + (gd / kappa) ** 2), min=w_floor, max=1.0)


def edge_preserving_tv(vol: torch.Tensor, kappa: float, w_floor: float) -> torch.Tensor:
    """Edge-weighted anisotropic TV of |vol|. Per-axis finite differences (matches
    JointTrainer._tv_loss), each down-weighted at its own sharp gradients -> smoothing
    stays ON along an edge but ~OFF across it."""
    m = vol.abs()
    gx = m[1:] - m[:-1]
    gy = m[:, 1:] - m[:, :-1]
    gz = m[:, :, 1:] - m[:, :, :-1]
    return ((_edge_weight(gx, kappa, w_floor) * gx.abs()).mean()
            + (_edge_weight(gy, kappa, w_floor) * gy.abs()).mean()
            + (_edge_weight(gz, kappa, w_floor) * gz.abs()).mean())


def resolve_edge_kappa(vol: torch.Tensor, q: float, brain_frac: float = 0.05) -> float:
    """Absolute edge threshold = q-quantile of the in-brain per-axis gradient magnitude
    of the (initial) render. Fixed once so the edge bands do not drift during training."""
    m = vol.abs()
    thr = brain_frac * float(m.max())
    parts = [
        (m[1:] - m[:-1]).abs()[m[1:] > thr].reshape(-1),
        (m[:, 1:] - m[:, :-1]).abs()[m[:, 1:] > thr].reshape(-1),
        (m[:, :, 1:] - m[:, :, :-1]).abs()[m[:, :, 1:] > thr].reshape(-1),
    ]
    g = torch.cat(parts).detach().cpu().numpy()
    if g.size == 0:
        return max(thr, 1e-8)
    return max(float(np.quantile(g, q)), 1e-8)
