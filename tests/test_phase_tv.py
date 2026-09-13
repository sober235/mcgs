#!/usr/bin/env python3
"""Phase-only TV (paired_gs/losses/phase_tv.py), ported 2026-09-13 from the AF16 phase-2/3 line.

Pins the property the term exists for: it charges phase variation and ignores magnitude structure.
Zero on a real volume and on any constant-phase volume however rough its magnitude; positive on a
volume whose phase varies; invariant to a global phase rotation; scales with |x| for a fixed phase
ramp; finite gradients.

Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_phase_tv.py -q
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paired_gs"))
import torch
from losses.phase_tv import phase_tv

KAPPA, WFLOOR = 1.0, 0.1


def _mag(seed=0, shape=(16, 16, 12)):
    g = torch.Generator().manual_seed(seed)
    return (0.5 + torch.rand(shape, generator=g)).to(torch.float32)


def test_real_input_is_zero():
    assert float(phase_tv(_mag(), KAPPA, WFLOOR)) == 0.0


def test_constant_phase_is_zero_whatever_the_magnitude():
    """Exactly zero in exact arithmetic. In float32 the term computes sqrt(|dx|^2 - (d|x|)^2), a cancellation
    of two equal numbers, so it returns a small positive residue instead (measured 2026-09-13: 1.1e-4 for a
    rotated real volume, against 2.3e-1 for a realistic phase ramp -- 2000x below the signal, and 3000x below
    it again after the lambda_ptv ~ 0.03 the loop uses). float64 gives 1e-11, which is the proof that the
    residue is precision and not the formula; both are asserted here. The float64 value is not 0 either:
    it is the module's own _FLOOR = 1e-12 under the sqrt, ~2.6e-6, identical for every constant phase."""
    m = _mag()
    f64 = []
    for phi in (0.0, 0.7, -2.1):
        v = (m * complex(math.cos(phi), math.sin(phi))).to(torch.complex64)
        assert float(phase_tv(v, KAPPA, WFLOOR)) < 1e-3, phi
        f64.append(float(phase_tv(m.double() * complex(math.cos(phi), math.sin(phi)), KAPPA, WFLOOR)))
    # In float64 every constant phase gives the SAME number, and that number is the module's own
    # _FLOOR = 1e-12 inside the sqrt (sqrt(1e-12) = 1e-6 per axis, ~2.6e-6 after the edge weight and the
    # three axes) -- i.e. the phase part is exactly zero and only the regulariser is left.
    assert max(f64) < 1e-5, f64
    # they agree to four significant digits (measured spread 8e-6 relative, from the sqrt inside .abs());
    # what matters is that the value does not grow with the phase, i.e. no phase is being charged.
    assert (max(f64) - min(f64)) / max(f64) < 1e-4, f64


def test_varying_phase_is_positive():
    m = _mag()
    zz, yy, xx = torch.meshgrid(*[torch.linspace(0, 3.0, n) for n in m.shape], indexing="ij")
    v = (m * torch.exp(1j * xx)).to(torch.complex64)
    assert float(phase_tv(v, KAPPA, WFLOOR)) > 1e-3


def test_global_phase_rotation_does_not_change_it():
    m = _mag()
    zz, yy, xx = torch.meshgrid(*[torch.linspace(0, 3.0, n) for n in m.shape], indexing="ij")
    v = (m * torch.exp(1j * xx)).to(torch.complex64)
    rot = complex(math.cos(1.3), math.sin(1.3))
    # float32: the same cancellation leaves a 1e-4 relative wobble; float64 pins the exact invariance.
    assert math.isclose(float(phase_tv(v, KAPPA, WFLOOR)), float(phase_tv(v * rot, KAPPA, WFLOOR)), rel_tol=1e-3)
    v64 = v.to(torch.complex128)
    assert math.isclose(float(phase_tv(v64, KAPPA, WFLOOR)), float(phase_tv(v64 * rot, KAPPA, WFLOOR)), rel_tol=1e-8)


def test_scales_with_magnitude_for_a_fixed_phase_ramp():
    m = torch.ones((16, 16, 12))
    zz, yy, xx = torch.meshgrid(*[torch.linspace(0, 1.0, n) for n in m.shape], indexing="ij")
    v = (m * torch.exp(1j * xx)).to(torch.complex64)
    a = float(phase_tv(v, 1e6, WFLOOR))          # kappa huge -> edge weight ~1 everywhere
    b = float(phase_tv(2 * v, 1e6, WFLOOR))
    assert math.isclose(b, 2 * a, rel_tol=1e-3)


def test_gradients_are_finite():
    m = _mag().requires_grad_(True)
    zz, yy, xx = torch.meshgrid(*[torch.linspace(0, 3.0, n) for n in m.shape], indexing="ij")
    v = m * torch.exp(1j * xx)
    phase_tv(v, KAPPA, WFLOOR).backward()
    assert torch.isfinite(m.grad).all() and float(m.grad.abs().sum()) > 0
