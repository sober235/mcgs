"""R_ptv: phase-only TV — the part of the complex finite difference that the magnitude gauge drops.

Why (2026-09-11, jyz AF16 phase-2 loop): with hard DC the sampled k-space matches the data exactly, so all
image error lives in the unobserved region, and the complex gauge |recon - GT| additionally counts the phase
part of it, where the GS arms trail L1-ESPIRiT (head PSNR 31.95 / 31.20 against 32.20 / 31.72). In this
trainer TV, directional TV, edge-align and vectorial TV all act on |x| (losses/edge_tv.py), so phase is
constrained only by the k-space fit and the complex wavelet term.

This term penalises rapid phase variation on its own and leaves the tuned magnitude TV untouched. Per axis
the complex difference splits into a magnitude part and an orthogonal part:

    |dx|^2 = (d|x|)^2 + perp^2,   perp ~ |x| * dphi for small phase steps

so charging charbonnier(perp) is a phase penalty that is exactly zero for a constant-phase volume, whatever
its magnitude structure. The edge weight comes from the detached magnitude gradient, the same geometry
edge_preserving_tv uses, so a genuine anatomical edge (where the phase may also jump) is spared.
lambda_ptv = 0 means the term is never built and the run is bit-identical to before.
"""
import torch

from .edge_tv import _axis_grads, _edge_weight, _guided_edge_weight, charbonnier

_FLOOR = 1e-12          # keeps d/dx sqrt finite at perp = 0


def phase_tv(vol: torch.Tensor, kappa: float, w_floor: float, eps: float = 0.0,
             guide: torch.Tensor | None = None, guide_kappa: float = 1.0, guide_alpha: float = 0.0) -> torch.Tensor:
    """Edge-weighted TV of the phase-orthogonal component of the complex gradient of `vol`.

    Real input has no phase to penalise and returns 0. kappa / w_floor / eps are the edge-preserving TV
    parameters (resolve_edge_kappa, loss.edge_tv_wfloor, loss.tv_eps). guide / guide_kappa / guide_alpha as in
    edge_preserving_tv: with guide_alpha > 0 the other contrast's magnitude edges also spare the phase penalty.
    """
    m = vol.abs()
    if not torch.is_complex(vol):
        return torch.zeros((), device=vol.device, dtype=m.dtype)
    total = torch.zeros((), device=vol.device, dtype=m.dtype)
    guided = guide is not None and guide_alpha > 0.0
    gds = _axis_grads(guide.detach().abs()) if guided else (None, None, None)
    for cd, md, gd in zip(_axis_grads(vol), _axis_grads(m), gds):
        perp = torch.sqrt(torch.clamp(cd.abs().square() - md.square(), min=0.0) + _FLOOR)
        w = (_guided_edge_weight(md, kappa, gd, guide_kappa, guide_alpha, w_floor) if guided
             else _edge_weight(md, kappa, w_floor))
        total = total + (w * charbonnier(perp, eps)).mean()
    return total
