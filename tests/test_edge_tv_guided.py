"""Cross-contrast guided edge weights for edge-preserving TV (widen-mcgs-af16-20-24)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paired_gs"))
import torch
from losses.edge_tv import edge_preserving_tv


def _step(n=12, axis=0, at=6, height=1.0):
    v = torch.zeros(n, n, n)
    idx = [slice(None)] * 3
    idx[axis] = slice(at, None)
    v[tuple(idx)] = height
    return v


def test_alpha_zero_is_bit_identical():
    torch.manual_seed(0)
    vol = torch.randn(10, 11, 12)
    guide = torch.randn(10, 11, 12)
    a = edge_preserving_tv(vol, 0.3, 0.1, 0.04)
    b = edge_preserving_tv(vol, 0.3, 0.1, 0.04, guide=guide, guide_kappa=0.5, guide_alpha=0.0)
    assert torch.equal(a, b)


def test_guide_edge_relaxes_penalty_on_own_weak_edge():
    # a weak step in vol (below its kappa, so its own weight stays ~1) at the place the guide has a strong edge
    vol = _step(height=0.05)
    guide = _step(height=1.0)
    own = edge_preserving_tv(vol, 1.0, 0.1)
    guided = edge_preserving_tv(vol, 1.0, 0.1, guide=guide, guide_kappa=0.1, guide_alpha=1.0)
    assert guided < 0.2 * own


def test_guide_without_edge_changes_nothing_much():
    vol = _step(height=0.05)
    guide = torch.zeros_like(vol)
    own = edge_preserving_tv(vol, 1.0, 0.1)
    guided = edge_preserving_tv(vol, 1.0, 0.1, guide=guide, guide_kappa=0.1, guide_alpha=1.0)
    assert torch.allclose(own, guided)


def test_guide_is_detached():
    vol = _step(height=0.05).requires_grad_(True)
    guide = _step(height=1.0).requires_grad_(True)
    edge_preserving_tv(vol, 1.0, 0.1, guide=guide, guide_kappa=0.1, guide_alpha=0.5).backward()
    assert guide.grad is None
    assert vol.grad is not None


def test_phase_tv_guide_alpha_zero_bit_identical_and_detached():
    from losses.phase_tv import phase_tv
    torch.manual_seed(1)
    vol = torch.randn(9, 10, 11, dtype=torch.complex64)
    guide = torch.randn(9, 10, 11, dtype=torch.complex64, requires_grad=True)
    assert torch.equal(phase_tv(vol, 0.3, 0.1, 0.04),
                       phase_tv(vol, 0.3, 0.1, 0.04, guide=guide, guide_kappa=0.5, guide_alpha=0.0))
    v = vol.clone().requires_grad_(True)
    phase_tv(v, 0.3, 0.1, 0.04, guide=guide, guide_kappa=0.5, guide_alpha=0.5).backward()
    assert guide.grad is None and v.grad is not None


def test_phase_tv_guide_edge_relaxes():
    from losses.phase_tv import phase_tv
    n = 12
    ph = torch.zeros(n, n, n)
    ph[6:] = 1.0                                              # phase step, flat magnitude
    vol = torch.polar(torch.ones(n, n, n), ph)
    guide = _step(height=1.0).to(torch.complex64)             # guide has a magnitude edge at the same place
    own = phase_tv(vol, 1.0, 0.1)
    guided = phase_tv(vol, 1.0, 0.1, guide=guide, guide_kappa=0.1, guide_alpha=1.0)
    assert guided < 0.2 * own


def test_load_guide_volume_normalises_and_thresholds(tmp_path):
    import numpy as np
    from losses.edge_tv import load_guide_volume, resolve_edge_kappa
    vol = (3.0 * _step(height=1.0) + 0.01 * torch.randn(12, 12, 12)).numpy().astype(np.complex64)
    p = tmp_path / "g.npy"
    np.save(p, vol)
    g, kappa = load_guide_volume(str(p), torch.device("cpu"), 0.85)
    assert g.dtype == torch.complex64 and g.shape == (12, 12, 12)
    assert abs(float(torch.quantile(g.abs().flatten(), 0.999)) - 1.0) < 1e-4
    assert abs(kappa - resolve_edge_kappa(g, 0.85)) < 1e-9


def test_directional_tv_flat_guide_is_vector_tv():
    from losses.edge_tv import directional_tv, tv_magnitude
    torch.manual_seed(2)
    vol = torch.randn(10, 11, 12)
    a = directional_tv(vol, torch.zeros_like(vol), eta=0.1, eps=0.04)
    b = tv_magnitude(vol, eps=0.04, vector=True)
    assert torch.allclose(a, b, atol=1e-5)


def test_directional_tv_colocated_edge_costs_nothing_and_guide_is_detached():
    from losses.edge_tv import directional_tv
    vol = _step(height=1.0)
    guide = _step(height=1.0).requires_grad_(True)
    own = directional_tv(vol, torch.zeros_like(vol), eta=1e-3)
    guided = directional_tv(vol.clone().requires_grad_(True), guide, eta=1e-3)
    assert guided < 1e-3 * own
    guided.backward()
    assert guide.grad is None
