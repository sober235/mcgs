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
