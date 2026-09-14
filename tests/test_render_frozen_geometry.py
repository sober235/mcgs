#!/usr/bin/env python3
"""A render whose shared geometry is detached (JointTrainer._render_geometry_frozen).

Stage 4 found the joint arm losing T1w at T1w-only edges: T2s's smoothing reaches the shared cloud.
Regularisers evaluated on this render move the channel's densities and leave positions, scales and
rotations to the data terms. Pins, on a toy linear splatter standing in for the voxelizer: the frozen
render has the same values as the normal one; a loss on it gives density gradients and no geometry
gradients; the normal render still gives geometry gradients; a contrast-private residual set, when
present, keeps its own gradients.

Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_render_frozen_geometry.py -q
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paired_gs"))
import torch
from trainers.joint_trainer import JointTrainer

GRID = torch.stack(torch.meshgrid(*[torch.linspace(-1, 1, 6)] * 3, indexing="ij"), -1)   # (6,6,6,3)


def splat(positions, scales, rotations, density):
    d2 = ((GRID[..., None, :] - positions) ** 2).sum(-1) / (scales.mean(-1) ** 2 + 1e-3)  # (6,6,6,N)
    w = torch.exp(-d2) * (1.0 + 0.0 * rotations.sum(-1))
    return (w * density).sum(-1)


def fake(private=False):
    g = torch.Generator().manual_seed(0)
    n = 5
    m = SimpleNamespace(
        positions=torch.randn(n, 3, generator=g).mul(0.5).requires_grad_(True),
        scales=torch.full((n, 3), -1.0, requires_grad=True),
        rotations=torch.randn(n, 4, generator=g).requires_grad_(True),
        dens={"plain": torch.randn(n, generator=g).to(torch.complex64).requires_grad_(True),
              "contrast": torch.randn(n, generator=g).to(torch.complex64).requires_grad_(True)},
        has_private=private)
    m.get_scale_values = lambda: torch.exp(m.scales)
    m.get_density = lambda ch: m.dens[ch]
    if private:
        m.private_positions = torch.randn(2, 3, generator=g).requires_grad_(True)
        m.private_scales = torch.full((2, 3), -1.0, requires_grad=True)
        m.private_rotations = torch.randn(2, 4, generator=g).requires_grad_(True)
        m.private_density = torch.randn(2, generator=g).to(torch.complex64).requires_grad_(True)
        m.get_private_scale_values = lambda: torch.exp(m.private_scales)
        m.get_private_density = lambda: m.private_density
    return SimpleNamespace(model=m, voxelizer=splat)


def test_same_values_as_the_normal_render():
    t = fake()
    a = JointTrainer._render(t, "contrast")
    b = JointTrainer._render_geometry_frozen(t, "contrast")
    assert torch.allclose(a, b)


def test_loss_on_frozen_render_reaches_densities_only():
    t = fake()
    JointTrainer._render_geometry_frozen(t, "contrast").abs().pow(2).sum().backward()
    m = t.model
    assert m.positions.grad is None and m.scales.grad is None and m.rotations.grad is None
    assert m.dens["contrast"].grad is not None and float(m.dens["contrast"].grad.abs().sum()) > 0
    assert m.dens["plain"].grad is None


def test_normal_render_still_moves_the_geometry():
    t = fake()
    JointTrainer._render(t, "contrast").abs().pow(2).sum().backward()
    assert t.model.positions.grad is not None and float(t.model.positions.grad.abs().sum()) > 0


def test_private_residual_keeps_its_own_gradients():
    t = fake(private=True)
    a = JointTrainer._render(t, "contrast")
    b = JointTrainer._render_geometry_frozen(t, "contrast")
    assert torch.allclose(a, b)
    b.abs().pow(2).sum().backward()
    assert t.model.positions.grad is None
    assert t.model.private_positions.grad is not None and t.model.private_density.grad is not None
