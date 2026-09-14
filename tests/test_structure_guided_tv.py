#!/usr/bin/env python3
"""Structure-guided directional TV (paired_gs/losses/sgtv.py) - Ehrhardt & Betcke 2016, on magnitudes:

    R(u; v) = mean over voxels of rho( | (I - gamma xi xi^T) grad|u| | ),   xi = grad|v| / sqrt(|grad|v||^2 + eta^2)

The guide v (T1w) is detached. Where the guide is flat xi ~ 0 and R is the vector TV of u; where the guide has an edge,
the component of u's gradient along the guide's gradient is (almost) free, so u keeps edges where v has them and is
smoothed elsewhere. Pins: a flat guide gives exactly 1.5 x vector TV (the project's per-axis scale); u equal to the
guide costs far less than its TV; an edge perpendicular to the guide's edge costs its full TV; no gradient reaches the
guide; u's gradient is finite; eta is scaled by the guide's own gradient magnitude, so a scaled guide gives the same R.

Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_structure_guided_tv.py -q
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paired_gs"))
import torch
from losses.edge_tv import tv_magnitude
from losses.sgtv import structure_guided_tv

N = 24


def step(axis, where=12, lo=0.2, hi=1.0):
    x = torch.full((N, N, N), lo, dtype=torch.float64)
    sl = [slice(None)] * 3; sl[axis] = slice(where, None)
    x[tuple(sl)] = hi
    return x


def test_flat_guide_is_the_vector_tv():
    u = step(0) + 0.05 * torch.randn(N, N, N, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    v = torch.full((N, N, N), 0.7, dtype=torch.float64)
    r = float(structure_guided_tv(u.to(torch.complex128), v.to(torch.complex128)))
    assert math.isclose(r, float(tv_magnitude(u, vector=True)), rel_tol=1e-6)


def test_shared_edge_is_nearly_free():
    u = step(0)
    r = float(structure_guided_tv(u, u.clone(), gamma=0.995, eta_scale=0.01))
    assert r < 0.05 * float(tv_magnitude(u, vector=True))


def test_edge_across_the_guide_costs_its_full_tv():
    u, v = step(0), step(1)                      # u's edge is perpendicular to v's
    r = float(structure_guided_tv(u, v, gamma=0.995, eta_scale=0.01))
    assert r > 0.95 * float(tv_magnitude(u, vector=True))


def test_no_gradient_reaches_the_guide_and_u_gradient_is_finite():
    u = (step(0) + 0.01).requires_grad_(True)
    v = step(0).requires_grad_(True)
    structure_guided_tv(u, v).backward()
    assert v.grad is None
    assert torch.isfinite(u.grad).all() and float(u.grad.abs().sum()) > 0


def test_invariant_to_guide_scale():
    u = step(0) + 0.05 * torch.randn(N, N, N, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    v = step(2) + step(0)
    a = float(structure_guided_tv(u, v))
    b = float(structure_guided_tv(u, 7.0 * v))
    assert math.isclose(a, b, rel_tol=1e-9)
