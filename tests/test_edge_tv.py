#!/usr/bin/env python3
"""Standalone unit test for edge-preserving TV (no GPU/data).
Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python tests/test_edge_tv.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paired_gs"))
import torch
from losses.edge_tv import edge_preserving_tv, resolve_edge_kappa, _edge_weight

def approx(a, b, tol=1e-5): return abs(float(a) - float(b)) < tol

def test_weight_flat_and_edge():
    w_flat = _edge_weight(torch.zeros(4), kappa=1.0, w_floor=0.1)
    w_edge = _edge_weight(torch.full((4,), 100.0), kappa=1.0, w_floor=0.1)
    assert torch.all(w_flat > 0.99), f"flat weight ~1 expected, got {w_flat}"
    assert torch.all(w_edge <= 0.1 + 1e-6), f"edge weight should hit floor, got {w_edge}"
    print("PASS test_weight_flat_and_edge")

def test_wfloor_respected():
    w = _edge_weight(torch.linspace(0, 1000, 50), kappa=1.0, w_floor=0.2)
    assert torch.all(w >= 0.2 - 1e-6) and torch.all(w <= 1.0 + 1e-6)
    print("PASS test_wfloor_respected")

def test_flat_region_matches_isotropic_tv():
    torch.manual_seed(0)
    vol = 0.01 * torch.randn(8, 8, 8)                       # grain only, no edge
    m = vol.abs()
    iso = ((m[1:]-m[:-1]).abs().mean() + (m[:,1:]-m[:,:-1]).abs().mean() + (m[:,:,1:]-m[:,:,:-1]).abs().mean())
    ew = edge_preserving_tv(vol, kappa=1.0, w_floor=0.1)    # kappa>>grain -> w~1
    assert approx(ew, iso, tol=1e-3), f"flat edge-TV {float(ew)} should ~ iso {float(iso)}"
    print("PASS test_flat_region_matches_isotropic_tv")

def test_edge_penalized_less_than_isotropic():
    vol = torch.zeros(8, 8, 8); vol[4:] = 1.0              # sharp step across axis0
    m = vol.abs()
    iso = ((m[1:]-m[:-1]).abs().mean() + (m[:,1:]-m[:,:-1]).abs().mean() + (m[:,:,1:]-m[:,:,:-1]).abs().mean())
    ew = edge_preserving_tv(vol, kappa=0.1, w_floor=0.1)   # kappa<<step -> step ~w_floor
    assert float(ew) < float(iso), f"edge-TV {float(ew)} should be < iso {float(iso)}"
    print("PASS test_edge_penalized_less_than_isotropic")

def test_differentiable_and_finite():
    vol = torch.randn(8, 8, 8, requires_grad=True)
    loss = edge_preserving_tv(vol, kappa=0.5, w_floor=0.1); loss.backward()
    assert torch.isfinite(loss) and vol.grad is not None and torch.all(torch.isfinite(vol.grad))
    print("PASS test_differentiable_and_finite")

def test_resolve_kappa_positive_finite():
    vol = torch.zeros(10, 10, 10); vol[5:] = 1.0; vol += 0.05 * torch.randn(10, 10, 10)
    k = resolve_edge_kappa(vol, q=0.75)
    assert k > 0 and torch.isfinite(torch.tensor(k)), f"kappa positive finite expected, got {k}"
    print("PASS test_resolve_kappa_positive_finite")

if __name__ == "__main__":
    test_weight_flat_and_edge()
    test_wfloor_respected()
    test_flat_region_matches_isotropic_tv()
    test_edge_penalized_less_than_isotropic()
    test_differentiable_and_finite()
    test_resolve_kappa_positive_finite()
    print("ALL EDGE-TV UNIT TESTS PASSED")
