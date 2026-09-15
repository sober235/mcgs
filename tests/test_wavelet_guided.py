#!/usr/bin/env python3
"""One-way guided wavelet sparsity (paired_gs/losses/wavelet_l1.py::wavelet_l1_guided).

The contrast volume's wavelet L1 with a per-coefficient weight that only ever drops where the guide (T1w) has a strong
coefficient at the same position: w = clamp(1 / (1 + (|W guide| / kappa)^2), w_floor, 1), guide detached, kappa the
q-quantile of the guide's detail magnitudes. It is the wavelet counterpart of edge_tv_union_guide; unlike the group
norm it leaves the guide's own penalty alone. Pins: a flat guide gives exactly the plain wavelet L1; it never exceeds
the plain L1; a guide with detail where the volume has detail lowers it; no gradient reaches the guide; the scale of
the guide does not matter (kappa is a quantile of its own coefficients).

Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_wavelet_guided.py -q
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paired_gs"))
import torch
from losses.wavelet_l1 import wavelet_l1, wavelet_l1_guided


def _vol(seed=0, shape=(32, 32, 24)):
    g = torch.Generator().manual_seed(seed)
    zz, yy, xx = torch.meshgrid(*[torch.linspace(-1, 1, n) for n in shape], indexing="ij")
    smooth = (torch.exp(-(zz ** 2 + yy ** 2 + xx ** 2) * 3.0) > 0.5).double() * torch.exp(1j * 0.5 * xx)
    noise = 0.05 * (torch.randn(shape, generator=g) + 1j * torch.randn(shape, generator=g))
    return (smooth + noise).to(torch.complex128)


def test_flat_guide_is_the_plain_l1():
    a = _vol()
    flat = torch.full(a.shape, 0.4, dtype=torch.complex128)
    for sh in (None, (1, 0, 1)):
        g = float(wavelet_l1_guided(a, flat, "db4", 1, 0.0, sh, "complex", q=0.85, w_floor=0.1))
        ref = float(wavelet_l1([a], "db4", 1, 0.0, sh, "complex"))
        assert math.isclose(g, ref, rel_tol=1e-9), (g, ref)


def test_never_above_plain_and_lower_with_shared_detail():
    a = _vol(0)
    guide = _vol(1)                                  # same object boundary, different noise
    g = float(wavelet_l1_guided(a, guide, "db4", 1, q=0.85, w_floor=0.1))
    ref = float(wavelet_l1([a], "db4", 1))
    assert g <= ref * (1 + 1e-12)
    assert g < 0.98 * ref


def test_no_gradient_to_guide_and_finite_gradient():
    a = _vol(0).requires_grad_(True)
    guide = _vol(1).requires_grad_(True)
    wavelet_l1_guided(a, guide, "db4", 1, 0.0, (1, 1, 0), "complex", q=0.85, w_floor=0.1).backward()
    assert guide.grad is None
    assert torch.isfinite(torch.view_as_real(a.grad)).all() and float(a.grad.abs().sum()) > 0


def test_guide_scale_does_not_matter():
    a, guide = _vol(0), _vol(1)
    x = float(wavelet_l1_guided(a, guide, "db4", 1, q=0.85, w_floor=0.1))
    y = float(wavelet_l1_guided(a, 5.0 * guide, "db4", 1, q=0.85, w_floor=0.1))
    assert math.isclose(x, y, rel_tol=1e-9)
