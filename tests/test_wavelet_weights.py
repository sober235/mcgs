#!/usr/bin/env python3
"""Per-level and per-sub-band multipliers of the wavelet prior (paired_gs/losses/wavelet_l1.py), the
phase-2 shape levers of the jiangyizhou AF12 loop.

Pins: all-ones multipliers reproduce the unweighted value; the finest level of a 2-level transform with the
coarse level weighted 0 equals the 1-level value; a sub-band weighted 0 removes exactly that band's share;
weights compose linearly with the per-volume strengths; wrong lengths and unknown bands are rejected;
gradients flow.

Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_wavelet_weights.py -q
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paired_gs"))
import pytest
import torch
from losses.wavelet_l1 import BANDS, wavelet_l1


def _vols(seed=0, shape=(32, 32, 24)):
    g = torch.Generator().manual_seed(seed)
    zz, yy, xx = torch.meshgrid(*[torch.linspace(-1, 1, n) for n in shape], indexing="ij")
    smooth = torch.exp(-(zz ** 2 + yy ** 2 + xx ** 2) * 3.0) * torch.exp(1j * 0.5 * xx)
    a = (smooth + 0.05 * (torch.randn(shape, generator=g) + 1j * torch.randn(shape, generator=g))).to(torch.complex64)
    b = (0.6 * smooth + 0.03 * (torch.randn(shape, generator=g) + 1j * torch.randn(shape, generator=g))).to(torch.complex64)
    return a, b


def test_all_ones_reproduce_the_unweighted_value():
    a, b = _vols()
    ref = float(wavelet_l1([a, b], "db4", 2))
    assert math.isclose(float(wavelet_l1([a, b], "db4", 2, level_weights=[1.0, 1.0])), ref, rel_tol=1e-5)
    assert math.isclose(float(wavelet_l1([a, b], "db4", 2, band_weights={k: 1.0 for k in BANDS})), ref, rel_tol=1e-5)


def test_default_path_untouched():
    a, b = _vols()
    assert torch.equal(wavelet_l1([a, b], "db4", 1), wavelet_l1([a, b], "db4", 1, level_weights=None, band_weights=None))


def test_coarse_level_weight_zero_equals_one_level_transform():
    a, b = _vols()
    two = wavelet_l1([a, b], "db4", 2, level_weights=[1.0, 0.0])      # finest first: keep level 1, drop level 2
    one = wavelet_l1([a, b], "db4", 1)
    assert math.isclose(float(two), float(one), rel_tol=1e-5)


def test_band_weight_zero_removes_exactly_that_band():
    a, b = _vols()
    full = float(wavelet_l1([a, b], "db4", 1))
    no_ddd = float(wavelet_l1([a, b], "db4", 1, band_weights={"ddd": 0.0}))
    only_ddd = float(wavelet_l1([a, b], "db4", 1, band_weights={k: (1.0 if k == "ddd" else 0.0) for k in BANDS}))
    assert only_ddd > 0
    assert math.isclose(full, no_ddd + only_ddd, rel_tol=1e-5)


def test_composes_with_per_volume_strengths():
    a, b = _vols()
    w = wavelet_l1([a, b], "db4", 1, weights=(0.12, 0.10), band_weights={"ddd": 0.5})
    ref = 0.12 * float(wavelet_l1([a], "db4", 1, band_weights={"ddd": 0.5})) + \
          0.10 * float(wavelet_l1([b], "db4", 1, band_weights={"ddd": 0.5}))
    assert math.isclose(float(w), ref, rel_tol=1e-5)


def test_rejects_bad_inputs():
    a, b = _vols()
    with pytest.raises(ValueError):
        wavelet_l1([a, b], "db4", 2, level_weights=[1.0])
    with pytest.raises(ValueError):
        wavelet_l1([a, b], "db4", 1, band_weights={"xyz": 1.0})


def test_gradients_flow_through_the_weighted_path():
    a, b = _vols()
    a = a.clone().requires_grad_(True)
    v = wavelet_l1([a, b], "db4", 2, level_weights=[1.0, 0.5], band_weights={"ddd": 0.5})
    v.backward()
    assert torch.isfinite(a.grad).all() and float(a.grad.abs().sum()) > 0
