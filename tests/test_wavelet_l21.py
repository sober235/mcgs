#!/usr/bin/env python3
"""Joint (group) wavelet sparsity across contrasts (paired_gs/losses/wavelet_l1.py::wavelet_l21).

    R(x_1..x_V) = (1/N) * sum over detail coefficients of rho( sum_v w_v^2 |W x_v|^2 )

Pins: with one volume all zero it is the weighted L1 of the other; two identical volumes cost sqrt(2)
times one; it never exceeds the separate weighted L1 sum and is strictly below it when the supports
overlap; a common global phase changes nothing; gradients are finite through cycle spinning; a
weight list of the wrong length and an unknown mode are rejected.

Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_wavelet_l21.py -q
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paired_gs"))
import pytest
import torch
from losses.wavelet_l1 import wavelet_l1, wavelet_l21


def _vol(seed=0, shape=(32, 32, 24)):
    g = torch.Generator().manual_seed(seed)
    zz, yy, xx = torch.meshgrid(*[torch.linspace(-1, 1, n) for n in shape], indexing="ij")
    smooth = torch.exp(-(zz ** 2 + yy ** 2 + xx ** 2) * 3.0) * torch.exp(1j * 0.5 * xx)
    noise = 0.05 * (torch.randn(shape, generator=g) + 1j * torch.randn(shape, generator=g))
    return (smooth + noise).to(torch.complex64)


def test_one_zero_volume_is_the_weighted_l1_of_the_other():
    a = _vol()
    z = torch.zeros_like(a)
    for sh in (None, (1, 0, 1)):
        g = float(wavelet_l21([a, z], "db4", 1, 0.0, sh, "complex", weights=(0.06, 0.10)))
        ref = 0.06 * float(wavelet_l1([a], "db4", 1, 0.0, sh, "complex"))
        assert math.isclose(g, ref, rel_tol=1e-4), (g, ref)


def test_identical_volumes_cost_sqrt2_times_one():
    a = _vol()
    g = float(wavelet_l21([a, a], "db4", 1))
    assert math.isclose(g, math.sqrt(2) * float(wavelet_l1([a], "db4", 1)), rel_tol=1e-4)


def test_never_above_the_separate_sum_and_below_it_on_overlap():
    a, b = _vol(0), 0.7 * _vol(1)
    w = (0.06, 0.10)
    g = float(wavelet_l21([a, b], "db4", 1, weights=w))
    s = float(wavelet_l1([a, b], "db4", 1, weights=w))
    assert g <= s * (1 + 1e-6)
    assert g < 0.95 * s                      # same smooth object in both: the supports overlap


def test_common_global_phase_is_invisible():
    a, b = _vol(0), _vol(1)
    rot = complex(math.cos(0.9), math.sin(0.9))
    x = float(wavelet_l21([a, b], "db4", 1, weights=(1.0, 0.5)))
    y = float(wavelet_l21([a * rot, b * rot], "db4", 1, weights=(1.0, 0.5)))
    assert math.isclose(x, y, rel_tol=1e-5)


def test_finite_gradients_through_cycle_spinning():
    a = _vol(0).to(torch.complex128).requires_grad_(True)
    b = _vol(1).to(torch.complex128).requires_grad_(True)
    r = wavelet_l21([a, b], "db4", 1, 0.0, (1, 1, 0), "complex", weights=(0.06, 0.10))
    r.backward()
    assert torch.isfinite(torch.view_as_real(a.grad)).all() and torch.isfinite(torch.view_as_real(b.grad)).all()
    assert float(a.grad.abs().sum()) > 0 and float(b.grad.abs().sum()) > 0


def test_rejects_bad_weights_and_mode():
    a = _vol()
    with pytest.raises(ValueError):
        wavelet_l21([a, a], "db4", 1, weights=(1.0,))
    with pytest.raises(ValueError):
        wavelet_l21([a, a], "db4", 1, on="phase")
