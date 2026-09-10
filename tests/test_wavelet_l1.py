#!/usr/bin/env python3
"""Wavelet-sparsity penalty (paired_gs/losses/wavelet_l1.py): the L1-wavelet ESPIRiT prior as an
MC-GS regulariser.

Pins: ~zero on a constant volume, positive homogeneity, invariance to a global phase, complex mode
equal to magnitude mode on real positive input, additivity over volumes, noise costs more than a
smooth image, finite gradients through the cycle-spinning path, a zero shift is the identity, and
an unknown mode is rejected.

Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_wavelet_l1.py -q
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paired_gs"))
import pytest
import torch
from losses.wavelet_l1 import wavelet_l1


def _vol(seed=0, shape=(32, 32, 24)):
    g = torch.Generator().manual_seed(seed)
    zz, yy, xx = torch.meshgrid(*[torch.linspace(-1, 1, n) for n in shape], indexing="ij")
    smooth = torch.exp(-(zz ** 2 + yy ** 2 + xx ** 2) * 3.0) * torch.exp(1j * 0.5 * xx)
    noise = 0.05 * (torch.randn(shape, generator=g) + 1j * torch.randn(shape, generator=g))
    return smooth.to(torch.complex64), noise.to(torch.complex64)


def test_constant_is_zero():
    v = torch.full((32, 32, 24), 0.7, dtype=torch.complex64)
    assert float(wavelet_l1([v])) < 1e-5


def test_positive_homogeneity():
    s, n = _vol()
    a = float(wavelet_l1([s + n]))
    b = float(wavelet_l1([2 * (s + n)]))
    assert math.isclose(b, 2 * a, rel_tol=1e-4)


def test_global_phase_invariance():
    s, n = _vol()
    x = s + n
    rot = complex(math.cos(1.1), math.sin(1.1))
    assert math.isclose(float(wavelet_l1([x])), float(wavelet_l1([x * rot])), rel_tol=1e-5)


def test_complex_equals_magnitude_on_real_positive():
    s, _ = _vol()
    r = s.abs()
    assert math.isclose(float(wavelet_l1([r.to(torch.complex64)], on="complex")),
                        float(wavelet_l1([r], on="magnitude")), rel_tol=1e-5)


def test_additive_over_volumes():
    s, n = _vol()
    s2, n2 = _vol(1)
    a, b = s + n, 0.5 * s2 + n2
    assert math.isclose(float(wavelet_l1([a, b])), float(wavelet_l1([a])) + float(wavelet_l1([b])), rel_tol=1e-5)


def test_noise_costs_more_than_smooth():
    s, n = _vol()
    assert float(wavelet_l1([s + n])) > 1.5 * float(wavelet_l1([s]))


def test_gradient_finite_and_shift_path():
    s, n = _vol()
    x = (s + n).clone().requires_grad_(True)
    loss = wavelet_l1([x], shifts=(3, 5, 1), eps=0.01)
    loss.backward()
    assert bool(torch.isfinite(x.grad).all()) and float(x.grad.abs().sum()) > 0
    assert math.isclose(float(wavelet_l1([s + n], shifts=(0, 0, 0))), float(wavelet_l1([s + n])), rel_tol=1e-6)


def test_bad_mode_rejected():
    s, _ = _vol()
    with pytest.raises(ValueError):
        wavelet_l1([s], on="phase")
