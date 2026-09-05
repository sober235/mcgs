"""Known-zero region Z (spec 2026-09-05 §4.1): estimated from ACQUIRED samples only, by the cliff rule.

Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_known_zero.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paired_gs"))
from data.known_zero import RING_WIDTH, KnownZeroInfo, estimate_known_zero, pe_radius_map  # noqa: E402

C, P0, P1, R = 4, 48, 40, 24


def _synth(rho_c=None, shutter=None, noise=2e-3, seed=0, ro_axis=2):
    """Multicoil k-space of a textured object (broad spectrum), optionally low-pass filtered in the PE plane
    (vendor filter, keeps noise) or hard-shuttered (exact zeros), vd-random mask broadcast along readout.
    Returns (kspace (C,P0,P1,R) undersampled, mask (P0,P1,R)) with ro_axis last, then moved to `ro_axis`."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.ogrid[:P0, :P1, :R]
    blob = np.exp(-(((zz - P0 / 2) / (P0 / 4)) ** 2 + ((yy - P1 / 2) / (P1 / 4)) ** 2 + ((xx - R / 2) / (R / 4)) ** 2))
    texture = ndimage.gaussian_filter(rng.normal(size=(P0, P1, R)), 0.7)
    obj = blob * (1.0 + 0.8 * texture / texture.std())          # texture -> nearly flat high-frequency spectrum
    ks = []
    for _ in range(C):
        cm = ndimage.gaussian_filter(rng.normal(size=(P0, P1, R)), 8.0)
        cm = cm / np.abs(cm).max() + 0.5
        k = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(cm * obj), norm="ortho"))
        ks.append(k)
    k = np.stack(ks).astype(np.complex64)
    ii, jj = np.meshgrid(np.arange(P0), np.arange(P1), indexing="ij")
    rho = np.sqrt(((ii - P0 // 2) / (P0 / 2)) ** 2 + ((jj - P1 // 2) / (P1 / 2)) ** 2)
    if rho_c is not None:
        k = k * (rho < rho_c)[None, :, :, None]
    if noise:
        k = k + noise * (rng.normal(size=k.shape) + 1j * rng.normal(size=k.shape)).astype(np.complex64) / np.sqrt(2)
    if shutter is not None:
        k = k * (rho < shutter)[None, :, :, None]                # exact zeros AFTER noise
    m2 = rng.random((P0, P1)) < 0.35
    m2[P0 // 2 - 4:P0 // 2 + 4, P1 // 2 - 4:P1 // 2 + 4] = True
    mask = np.repeat(m2[:, :, None], R, axis=2).astype(np.float32)
    k = k * mask[None]
    kt, mt = torch.from_numpy(k), torch.from_numpy(mask)
    return torch.movedim(kt, 3, ro_axis + 1).contiguous(), torch.movedim(mt, 2, ro_axis).contiguous()


def test_vendor_lowpass_is_found_within_one_ring():
    k, m = _synth(rho_c=0.8)
    Z, info = estimate_known_zero(k, m, ro_axis=2)
    assert isinstance(info, KnownZeroInfo)
    assert info.rho_c is not None and abs(info.rho_c - 0.8) <= RING_WIDTH + 1e-9
    assert info.kro_c is None                                    # nothing along readout
    rho = pe_radius_map(tuple(m.shape), info.dc, 2, m.device)
    assert Z.dtype == torch.bool and Z.shape == m.shape
    assert bool(Z[rho >= 0.9].all()) and not bool(Z[rho <= 0.7].any())
    assert info.frac_z == pytest.approx(float((rho >= info.rho_c).float().mean()), abs=1e-6)
    assert 0.0 < info.frac_z < 1.0


def test_unfiltered_data_has_no_known_zero():
    k, m = _synth(rho_c=None)
    Z, info = estimate_known_zero(k, m, ro_axis=2)
    assert info.rho_c is None and info.kro_c is None
    assert not bool(Z.any()) and info.frac_z == 0.0


def test_exact_shutter_is_found():
    k, m = _synth(rho_c=None, shutter=1.0)
    Z, info = estimate_known_zero(k, m, ro_axis=2)
    assert info.rho_c is not None and abs(info.rho_c - 1.0) <= RING_WIDTH + 1e-9
    assert info.n0_pe == 0.0


def test_uses_only_sampled_points():
    """Filling the UNSAMPLED points with garbage must not change the estimate."""
    k, m = _synth(rho_c=0.8, seed=3)
    _, info_a = estimate_known_zero(k, m, ro_axis=2)
    junk = torch.randn_like(k) * (1 - m).unsqueeze(0) * 10.0
    _, info_b = estimate_known_zero(k + junk, m, ro_axis=2)
    assert info_a.rho_c == info_b.rho_c and info_a.pe_profile == info_b.pe_profile


def test_readout_axis_position_does_not_matter():
    k0, m0 = _synth(rho_c=0.8, seed=5, ro_axis=2)
    k1, m1 = _synth(rho_c=0.8, seed=5, ro_axis=0)
    _, i0 = estimate_known_zero(k0, m0, ro_axis=2)
    _, i1 = estimate_known_zero(k1, m1, ro_axis=0)
    assert i0.rho_c == i1.rho_c and i0.dc[2] == i1.dc[0]
    assert np.allclose([p[1] for p in i0.pe_profile], [p[1] for p in i1.pe_profile], rtol=1e-5)   # reduction order may differ


def test_info_is_json_serializable():
    import dataclasses, json
    k, m = _synth(rho_c=0.8)
    _, info = estimate_known_zero(k, m, ro_axis=2)
    json.dumps(dataclasses.asdict(info))
