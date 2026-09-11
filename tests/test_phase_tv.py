"""R_ptv (losses/phase_tv.py): the phase-only TV term of the jyz AF16 phase-2 loop."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paired_gs"))
from losses.phase_tv import phase_tv


def _mag_structure(shape=(6, 7, 8)):
    g = torch.zeros(shape)
    g[2:5, 2:5, 2:5] = 1.0          # a block: plenty of magnitude gradient
    return g


def test_constant_phase_costs_nothing():
    """Magnitude structure alone must not be charged: that is the magnitude TV's job."""
    m = _mag_structure()
    for phi in (0.0, 0.7, -2.3):
        v = m * torch.exp(torch.tensor(1j * phi))
        assert float(phase_tv(v, kappa=0.3, w_floor=0.1, eps=0.0)) == pytest.approx(0.0, abs=1e-5)


def test_real_input_is_zero():
    assert float(phase_tv(_mag_structure(), kappa=0.3, w_floor=0.1)) == pytest.approx(0.0, abs=1e-12)


def test_phase_ramp_costs_about_magnitude_times_step():
    """Constant magnitude with a linear phase ramp along one axis: perp ~ |x| * dphi per step."""
    n, dphi, amp = 8, 0.05, 2.0
    ramp = torch.arange(n, dtype=torch.float32) * dphi
    v = amp * torch.exp(1j * ramp)[:, None, None].repeat(1, 4, 4)
    got = float(phase_tv(v, kappa=1e6, w_floor=1.0, eps=0.0))       # weight == 1 everywhere
    expect = amp * dphi                                              # one axis carries it, mean over that axis
    assert got == pytest.approx(expect, rel=0.02)


def test_gradients_are_finite_including_at_zero_phase():
    v = (_mag_structure() * torch.exp(torch.tensor(0j))).clone().requires_grad_(True)
    phase_tv(v, kappa=0.3, w_floor=0.1, eps=0.04).backward()
    assert torch.isfinite(v.grad).all()
