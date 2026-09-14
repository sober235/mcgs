#!/usr/bin/env python3
"""Edge-preserving TV whose edge weights also drop at a guide's edges (paired_gs/losses/edge_tv.py, guide=...).

Stage 4 found that any smoothing added to T2s loses (weighted dTV on top -0.15 dB, structure-guided TV instead of
edge-TV -1.0 dB). This variant only ever REMOVES smoothing: the weight per axis is min(w(vol's own gradient),
w(guide's gradient)), so T2s keeps its own edges and may also sharpen where T1w has an edge. Pins: no guide is the
old penalty exactly; a flat guide changes nothing; a guide edge where the volume has none lowers the penalty and
never raises it anywhere; no gradient reaches the guide; the vector form follows the same rule.

Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_edge_tv_union_guide.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paired_gs"))
import torch
from losses.edge_tv import edge_preserving_tv

N = 20


def vol(seed, edge_axis=None):
    g = torch.Generator().manual_seed(seed)
    x = 0.5 + 0.05 * torch.randn(N, N, N, dtype=torch.float64, generator=g)
    if edge_axis is not None:
        sl = [slice(None)] * 3; sl[edge_axis] = slice(N // 2, None)
        x[tuple(sl)] += 0.6
    return x


def test_no_guide_is_the_old_penalty():
    u = vol(0, 0)
    for vector in (False, True):
        a = edge_preserving_tv(u, 0.1, 0.1, 0.04, vector)
        b = edge_preserving_tv(u, 0.1, 0.1, 0.04, vector, guide=None, guide_kappa=0.1)
        assert torch.equal(a, b)


def test_flat_guide_changes_nothing():
    u = vol(0, 0)
    flat = torch.full((N, N, N), 0.3, dtype=torch.float64)
    for vector in (False, True):
        a = edge_preserving_tv(u, 0.1, 0.1, 0.04, vector)
        b = edge_preserving_tv(u, 0.1, 0.1, 0.04, vector, guide=flat, guide_kappa=0.1)
        assert torch.allclose(a, b, rtol=0, atol=1e-15)


def test_guide_edge_only_lowers_the_penalty():
    u = vol(1, 0) + 0.2 * (torch.arange(N, dtype=torch.float64) / N).view(1, N, 1)   # a ramp across axis 1
    v = vol(2, 1)                                                                   # the guide's edge is on axis 1
    for vector in (False, True):
        a = float(edge_preserving_tv(u, 0.1, 0.1, 0.04, vector))
        b = float(edge_preserving_tv(u, 0.1, 0.1, 0.04, vector, guide=v, guide_kappa=0.1))
        assert b < a


def test_no_gradient_reaches_the_guide():
    u = vol(0, 0).requires_grad_(True)
    v = vol(3, 1).requires_grad_(True)
    edge_preserving_tv(u, 0.1, 0.1, 0.04, False, guide=v, guide_kappa=0.1).backward()
    assert v.grad is None and torch.isfinite(u.grad).all()
