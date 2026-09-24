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


def charbonnier(g: torch.Tensor, eps: float) -> torch.Tensor:
    """|g| for eps == 0, else the Charbonnier / pseudo-Huber sqrt(g^2 + eps^2) - eps.

    L1's sub-gradient has magnitude 1 however small g is, so flattening a region to exactly
    zero gradient always pays: that is the staircase mechanism. Charbonnier's slope near zero
    is g/eps, so sub-eps gradients stop being worth removing, while |g| >> eps still costs
    |g| and real edges are untouched. The -eps keeps the penalty zero at g=0 so lambda means
    the same thing as before.
    """
    if eps <= 0.0:
        return g.abs()
    return torch.sqrt(g * g + eps * eps) - eps


# sum_a E|g_a| = 3*sigma*sqrt(2/pi) against E||g||_2 = 2*sigma*sqrt(2/pi) for an isotropic
# gradient field: without this factor the two forms would be compared at different effective
# strengths, and any difference between them could be a strength difference in disguise.
_VECTOR_SCALE = 1.5


def _axis_grads(m: torch.Tensor):
    return m[1:] - m[:-1], m[:, 1:] - m[:, :-1], m[:, :, 1:] - m[:, :, :-1]


def _colocated_grads(m: torch.Tensor):
    """The three differences anchored on the SAME voxel, so they form one gradient vector."""
    b = m[:-1, :-1, :-1]
    return m[1:, :-1, :-1] - b, m[:-1, 1:, :-1] - b, m[:-1, :-1, 1:] - b


def tv_magnitude(vol: torch.Tensor, eps: float = 0.0, vector: bool = False) -> torch.Tensor:
    """TV of |vol|. Defaults reproduce JointTrainer._tv_loss exactly.

    vector=True penalises ||grad||_2 per voxel instead of |gx|+|gy|+|gz|. The per-axis form
    charges a diagonal gradient sqrt(3) times more than an axis-aligned one of equal
    magnitude, so it prefers to lay edges (and plateau boundaries) along the voxel axes; the
    vector norm is orientation-neutral.
    """
    m = vol.abs()
    if not vector:
        gx, gy, gz = _axis_grads(m)
        return (charbonnier(gx, eps).mean()
                + charbonnier(gy, eps).mean()
                + charbonnier(gz, eps).mean())
    gx, gy, gz = _colocated_grads(m)
    gn = torch.sqrt(gx * gx + gy * gy + gz * gz)
    return _VECTOR_SCALE * charbonnier(gn, eps).mean()


def _guided_edge_weight(g: torch.Tensor, kappa: float, gg: torch.Tensor, guide_kappa: float,
                        alpha: float, w_floor: float) -> torch.Tensor:
    """Perona-Malik weight on the summed squared edge strength (g/kappa)^2 + alpha(gg/guide_kappa)^2:
    the channel's own edges count as before, and an edge the guide sees also relaxes the smoothing here.
    Both inputs detached."""
    r2 = (g.detach().abs() / kappa) ** 2 + alpha * (gg.detach().abs() / guide_kappa) ** 2
    return torch.clamp(1.0 / (1.0 + r2), min=w_floor, max=1.0)


def edge_preserving_tv(vol: torch.Tensor, kappa: float, w_floor: float,
                       eps: float = 0.0, vector: bool = False,
                       guide: torch.Tensor | None = None, guide_kappa: float = 1.0,
                       guide_alpha: float = 0.0) -> torch.Tensor:
    """Edge-weighted TV of |vol|. Per-axis finite differences (matches
    JointTrainer._tv_loss), each down-weighted at its own sharp gradients -> smoothing
    stays ON along an edge but ~OFF across it. eps/vector as in tv_magnitude; the defaults
    reproduce the shipped penalty exactly.

    guide (the other contrast's volume, detached) with guide_alpha > 0 mixes the guide's edges into
    the weight (per-axis form only), so a boundary the other contrast sees is not smoothed away here."""
    m = vol.abs()
    if guide is not None and guide_alpha > 0.0:
        if vector:
            raise ValueError("guided edge weights are implemented for the per-axis form only")
        gm = guide.detach().abs()
        terms = []
        for g, gg in zip(_axis_grads(m), _axis_grads(gm)):
            terms.append((_guided_edge_weight(g, kappa, gg, guide_kappa, guide_alpha, w_floor)
                          * charbonnier(g, eps)).mean())
        return terms[0] + terms[1] + terms[2]
    if not vector:
        gx, gy, gz = _axis_grads(m)
        return ((_edge_weight(gx, kappa, w_floor) * charbonnier(gx, eps)).mean()
                + (_edge_weight(gy, kappa, w_floor) * charbonnier(gy, eps)).mean()
                + (_edge_weight(gz, kappa, w_floor) * charbonnier(gz, eps)).mean())
    gx, gy, gz = _colocated_grads(m)
    gn = torch.sqrt(gx * gx + gy * gy + gz * gz)
    return _VECTOR_SCALE * (_edge_weight(gn, kappa, w_floor) * charbonnier(gn, eps)).mean()


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
