#!/usr/bin/env python3
"""Per-contrast edge-TV strength (paired_gs/losses/edge_tv.py::geo_term), the phase-2 lever of the
jiangyizhou AF12 loop.

Pins: with both per-contrast strengths equal to the shared one the term is bit-identical to
lam * (l_p + l_c) and the logged value is l_p + l_c (the shipped objective); otherwise the term is
lam_p * l_p + lam_c * l_c and lam * logged == term; a one-sided override takes the weighted path;
gradients flow through both paths.

Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_geo_term.py -q
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paired_gs"))
import torch
from losses.edge_tv import geo_term


def _pair():
    g = torch.Generator().manual_seed(3)
    return (torch.rand((), generator=g) * 0.37 + 0.11), (torch.rand((), generator=g) * 0.53 + 0.07)


def test_shared_strength_is_bit_identical_to_shipped_objective():
    l_p, l_c = _pair()
    term, logged = geo_term(l_p, l_c, 0.10, 0.10, 0.10)
    assert torch.equal(term, 0.10 * (l_p + l_c))
    assert torch.equal(logged, l_p + l_c)


def test_per_contrast_strengths_weight_each_volume():
    l_p, l_c = _pair()
    term, logged = geo_term(l_p, l_c, 0.10, 0.13, 0.07)
    assert math.isclose(float(term), 0.13 * float(l_p) + 0.07 * float(l_c), rel_tol=1e-6)
    assert math.isclose(0.10 * float(logged), float(term), rel_tol=1e-6)


def test_one_sided_override_takes_the_weighted_path():
    l_p, l_c = _pair()
    term, _ = geo_term(l_p, l_c, 0.10, 0.10, 0.05)
    assert math.isclose(float(term), 0.10 * float(l_p) + 0.05 * float(l_c), rel_tol=1e-6)
    assert not torch.equal(term, 0.10 * (l_p + l_c))


def test_gradients_flow_on_both_paths():
    for lam_p, lam_c in ((0.10, 0.10), (0.13, 0.07)):
        l_p = torch.tensor(0.3, requires_grad=True)
        l_c = torch.tensor(0.2, requires_grad=True)
        term, _ = geo_term(l_p, l_c, 0.10, lam_p, lam_c)
        term.backward()
        assert math.isclose(float(l_p.grad), lam_p, rel_tol=1e-6)
        assert math.isclose(float(l_c.grad), lam_c, rel_tol=1e-6)
