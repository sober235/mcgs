"""Structure-guided directional TV (Ehrhardt & Betcke, SIAM J. Imaging Sci. 2016), on magnitudes:

    R(u; v) = 1.5 * mean over voxels of rho( | (I - gamma * xi xi^T) grad|u| | ),
    xi = grad|v| / sqrt(|grad|v||^2 + eta^2),   eta = eta_scale * mean |grad|v|| over the guide's object

The guide v (the high-SNR contrast) is detached. Where the guide is flat, xi ~ 0 and R is the vector TV of u; across a
guide edge, the part of u's gradient parallel to the guide's gradient is (1 - gamma) cheap, so u may keep an edge where
v has one, with the same orientation, and is smoothed everywhere else. eta scales with the guide's own gradients, so the
penalty does not depend on the guide's intensity scale. The 1.5 is the project's vector-to-per-axis TV scale
(edge_tv._VECTOR_SCALE), so a flat guide gives exactly tv_magnitude(u, vector=True). rho is Charbonnier with eps
(eps = 0: the plain norm, with a zero-safe square root).
"""
import torch

from losses.edge_tv import _VECTOR_SCALE, _colocated_grads


def structure_guided_tv(vol: torch.Tensor, guide: torch.Tensor, gamma: float = 0.995, eta_scale: float = 1.0,
                        eps: float = 0.0, object_frac: float = 0.05) -> torch.Tensor:
    m = vol.abs()
    g = guide.detach().abs()
    ux, uy, uz = _colocated_grads(m)
    with torch.no_grad():
        vx, vy, vz = _colocated_grads(g)
        vn = torch.sqrt(vx * vx + vy * vy + vz * vz)
        obj = g[:-1, :-1, :-1] > object_frac * g.max()
        eta = eta_scale * (vn[obj].mean() if bool(obj.any()) else vn.mean()) + 1e-12
        den = torch.sqrt(vn * vn + eta * eta)
        xx, xy, xz = vx / den, vy / den, vz / den
    dot = ux * xx + uy * xy + uz * xz
    px, py, pz = ux - gamma * dot * xx, uy - gamma * dot * xy, uz - gamma * dot * xz
    n2 = px * px + py * py + pz * pz
    if eps > 0.0:
        rho = torch.sqrt(n2 + eps * eps) - eps
    else:
        rho = torch.sqrt(n2 + 1e-24) - 1e-12
    return _VECTOR_SCALE * rho.mean()
