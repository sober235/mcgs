#!/usr/bin/env python3
"""A per-Gaussian log-scale offset for the contrast channel (gaussian.contrast_scale_offset).

Positions and rotations stay shared; the contrast channel renders each Gaussian with
exp(log_scale + offset), so T2s can be smoother where it wants without merging the small Gaussians
T1w needs at its own edges (stage 4: the joint arm loses T1w exactly there). Pins, on the toy
splatter of test_render_frozen_geometry: a zero offset renders exactly as before; a non-zero offset
changes the contrast render only; a contrast loss trains the offset and a plain loss does not; the
geometry-frozen render still trains the offset but not the shared scales; the optimizer receives the
offset; the switch refuses to run with densification on.

Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_contrast_scale_offset.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paired_gs"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pytest
import torch
from test_render_frozen_geometry import fake
from trainers.joint_trainer import JointTrainer, attach_contrast_scale_offset


def with_offset(value=0.0):
    t = fake()
    t.model.scale_offset_contrast = torch.nn.Parameter(torch.full_like(t.model.scales, value))
    return t


def test_zero_offset_renders_as_before():
    t0, t1 = fake(), with_offset(0.0)
    assert torch.allclose(JointTrainer._render(t0, "contrast"), JointTrainer._render(t1, "contrast"))


def test_offset_changes_the_contrast_render_only():
    t0, t1 = fake(), with_offset(0.4)
    assert not torch.allclose(JointTrainer._render(t0, "contrast"), JointTrainer._render(t1, "contrast"))
    assert torch.allclose(JointTrainer._render(t0, "plain"), JointTrainer._render(t1, "plain"))


def test_contrast_loss_trains_the_offset_and_plain_loss_does_not():
    t = with_offset(0.1)
    JointTrainer._render(t, "plain").abs().pow(2).sum().backward()
    assert t.model.scale_offset_contrast.grad is None
    JointTrainer._render(t, "contrast").abs().pow(2).sum().backward()
    assert t.model.scale_offset_contrast.grad is not None and float(t.model.scale_offset_contrast.grad.abs().sum()) > 0


def test_frozen_render_trains_offset_not_shared_scales():
    t = with_offset(0.1)
    a = JointTrainer._render(t, "contrast")
    b = JointTrainer._render_geometry_frozen(t, "contrast")
    assert torch.allclose(a, b)
    b.abs().pow(2).sum().backward()
    assert t.model.scales.grad is None and t.model.positions.grad is None
    assert t.model.scale_offset_contrast.grad is not None


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scales = torch.nn.Parameter(torch.zeros(4, 3))


def test_attach_registers_a_zero_parameter_and_refuses_densification():
    m = _Model()
    attach_contrast_scale_offset(m, {"gaussian": {"contrast_scale_offset": True}})
    assert isinstance(m.scale_offset_contrast, torch.nn.Parameter)
    assert m.scale_offset_contrast.shape == m.scales.shape and float(m.scale_offset_contrast.abs().sum()) == 0
    assert any(p is m.scale_offset_contrast for p in m.parameters())
    m2 = _Model()
    attach_contrast_scale_offset(m2, {"gaussian": {}})
    assert getattr(m2, "scale_offset_contrast", None) is None
    with pytest.raises(ValueError):
        attach_contrast_scale_offset(_Model(), {"gaussian": {"contrast_scale_offset": True}, "adaptive_control": {"enable": True}})
