"""Multi-contrast pair-prior loss family for MC-3DGS.

Locked formula per ADR-0002 (complete pairwise contrast graph, mean over
valid pairs). For each pair ``(i, j)`` with ``i < j``::

    w_ij(v) = stopgrad(a_i(v) * a_j(v))
    ell_ij(v) = 1 - (g_i(v) . g_j(v))^2 / (||g_i(v)||^2 * ||g_j(v)||^2 + eps)
    L_ij    = sum_v w_ij(v) * ell_ij(v) / (sum_v w_ij(v) + eps)
    L_edge  = mean over pairs with sum_v w_ij / V > VALID_PAIR_MIN_MASS

At ``M=2`` this reduces bit-identically to the locked T1+FLAIR formula in
``MC_3DGS_ROADMAP.md`` Q1.

Scope of this module (issue #0002): formula only. ``tau_m`` is computed as
a naive full-volume quantile with ``p`` from the constructor. Foreground-
restricted quantile, ``p``-backoff cascade, hard-fail, and observability
are intentionally deferred to issue #0003.
"""

from __future__ import annotations

from abc import abstractmethod
from itertools import combinations
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn

VALID_PAIR_MIN_MASS = 0.01
EPS = 1e-8


def _grad_3d(x: torch.Tensor) -> torch.Tensor:
    """Centered finite-difference 3D gradient.

    Input ``x`` shape ``(D, H, W)``. Returns ``(3, D, H, W)`` stacking
    the gradients along the (D, H, W) axes in that order. Boundary
    handled by circular roll, consistent with k-space-centered FFT
    image conventions used elsewhere in the project.
    """
    gd = (torch.roll(x, -1, dims=0) - torch.roll(x, 1, dims=0)) * 0.5
    gh = (torch.roll(x, -1, dims=1) - torch.roll(x, 1, dims=1)) * 0.5
    gw = (torch.roll(x, -1, dims=2) - torch.roll(x, 1, dims=2)) * 0.5
    return torch.stack([gd, gh, gw], dim=0)


class RPairBase(nn.Module):
    """Abstract base for pair-prior losses on a list of M canonical-frame volumes.

    Subclasses must implement ``forward(x_m_list)`` returning a scalar
    ``torch.Tensor``. Inputs are interpreted as magnitude images; complex
    tensors are mapped to magnitude via ``.abs()``.
    """

    @abstractmethod
    def forward(self, x_m_list: List[torch.Tensor]) -> torch.Tensor:
        ...


class RPairNone(RPairBase):
    """Ablation pair-prior: always returns zero.

    The locked roadmap requires a no-prior baseline as a mandatory
    ablation alongside the common-edge prior. This class makes that
    baseline a single-line config swap.
    """

    def forward(self, x_m_list: List[torch.Tensor]) -> torch.Tensor:
        if x_m_list:
            return torch.zeros((), device=x_m_list[0].device, dtype=torch.float32)
        return torch.zeros(())


class RPairCommonEdgeAlign(RPairBase):
    """Common-edge cross-gradient prior on complete pairwise contrast graph.

    Parameters
    ----------
    pair_graph : ``"complete"`` or list of ``(i, j)`` tuples
        ``"complete"`` (default) enumerates all ``C(M, 2)`` pairs. An
        explicit pair list restricts to those pairs (for ablation).
    p : float
        Quantile used for the per-contrast gating threshold ``tau_m``.
        Default ``0.75`` — see ADR-0004 for calibration rationale.
        Issue #0002 implements naive full-volume quantile only; #0003
        upgrades to foreground-restricted quantile with backoff.
    kappa : float
        Sigmoid sharpness for the soft edge gate
        ``a_m(v) = sigmoid(kappa * (||g_m(v)|| - tau_m))``.
    valid_pair_min_mass : float
        Minimum normalized common-edge mass ``sum_v w_ij(v) / V`` for a
        pair to enter the reduction. Pairs below this are silently
        dropped so they cannot dilute the mean.

    Notes
    -----
    Edge weights ``w`` are detached (stopgrad) per the locked formula:
    gating is treated as a hard mask for backprop purposes. The
    pairwise alignment term ``ell_ij`` itself remains differentiable
    through both contrasts' gradients.
    """

    def __init__(
        self,
        pair_graph: Union[str, Sequence[Tuple[int, int]]] = "complete",
        p: float = 0.75,
        kappa: float = 10.0,
        valid_pair_min_mass: float = VALID_PAIR_MIN_MASS,
    ):
        super().__init__()
        if isinstance(pair_graph, str):
            if pair_graph != "complete":
                raise ValueError(
                    f"pair_graph string must be 'complete', got {pair_graph!r}"
                )
        else:
            pair_graph = list(pair_graph)
            for ij in pair_graph:
                if not (isinstance(ij, tuple) and len(ij) == 2):
                    raise ValueError(f"pair_graph entries must be (i,j) tuples, got {ij!r}")
                i, j = ij
                if i >= j:
                    raise ValueError(f"pair_graph entries must satisfy i<j, got {ij}")
        self.pair_graph = pair_graph
        self.p = float(p)
        self.kappa = float(kappa)
        self.valid_pair_min_mass = float(valid_pair_min_mass)

    def _pairs(self, M: int) -> List[Tuple[int, int]]:
        if self.pair_graph == "complete":
            return list(combinations(range(M), 2))
        # Defensive: drop any pair index that's out of range for this M.
        return [(i, j) for (i, j) in self.pair_graph if i < M and j < M]

    @staticmethod
    def _to_magnitude(x: torch.Tensor) -> torch.Tensor:
        return x.abs() if torch.is_complex(x) else x

    def forward(self, x_m_list: List[torch.Tensor]) -> torch.Tensor:
        if not x_m_list:
            return torch.zeros(())
        device = x_m_list[0].device
        M = len(x_m_list)
        if M < 2:
            return torch.zeros((), device=device, dtype=torch.float32)

        pairs = self._pairs(M)
        if not pairs:
            return torch.zeros((), device=device, dtype=torch.float32)

        # Per-contrast gradient field, gradient magnitude, soft gate.
        grads: List[torch.Tensor] = []
        gates: List[torch.Tensor] = []
        for m in range(M):
            xm = self._to_magnitude(x_m_list[m])
            if xm.ndim != 3:
                raise ValueError(
                    f"x_m_list[{m}] must be 3D (D,H,W) volume, got shape {tuple(xm.shape)}"
                )
            g = _grad_3d(xm)                       # (3, D, H, W)
            gmag = g.norm(dim=0)                   # (D, H, W)
            tau = torch.quantile(gmag.flatten(), self.p)
            a = torch.sigmoid(self.kappa * (gmag - tau))
            grads.append(g)
            gates.append(a)

        V = float(x_m_list[0].numel())
        pair_losses: List[torch.Tensor] = []
        for (i, j) in pairs:
            g_i, g_j = grads[i], grads[j]
            a_i, a_j = gates[i], gates[j]

            # Stopgrad on common-edge weight per locked formula.
            w = (a_i * a_j).detach()

            dot = (g_i * g_j).sum(dim=0)           # <g_i(v), g_j(v)>
            n_i_sq = (g_i * g_i).sum(dim=0)        # ||g_i(v)||^2
            n_j_sq = (g_j * g_j).sum(dim=0)
            ell = 1.0 - dot.pow(2) / (n_i_sq * n_j_sq + EPS)

            w_sum = w.sum()
            L_ij = (w * ell).sum() / (w_sum + EPS)
            mass = (w_sum / V).item()
            if mass > self.valid_pair_min_mass:
                pair_losses.append(L_ij)

        if not pair_losses:
            # No valid pair (e.g., empty gates everywhere). Treat as zero
            # loss so the trainer's total loss is well-defined.
            return torch.zeros((), device=device, dtype=torch.float32)
        return torch.stack(pair_losses).mean()
