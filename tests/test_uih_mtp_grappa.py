"""3D GRAPPA for PE-only acceleration 2 with an ACS block (uih_mtp.grappa)."""
import numpy as np
import pytest
import torch


def _phantom_kspace(X=48, Y=64, Z=32, C=8, seed=0):
    """Smooth 3D phantom x 8 smooth coil maps -> full multi-coil k-space (C, X, Y, Z)."""
    rng = np.random.default_rng(seed)
    x, y, z = np.meshgrid(np.linspace(-1, 1, X), np.linspace(-1, 1, Y), np.linspace(-1, 1, Z), indexing="ij")
    obj = ((x / 0.8) ** 2 + (y / 0.7) ** 2 + (z / 0.9) ** 2 < 1).astype(np.float32)
    obj += 0.6 * (((x - 0.2) / 0.3) ** 2 + ((y + 0.1) / 0.25) ** 2 + (z / 0.5) ** 2 < 1)
    obj -= 0.4 * (((x + 0.3) / 0.2) ** 2 + ((y - 0.3) / 0.2) ** 2 + (z / 0.4) ** 2 < 1)
    obj = obj * np.exp(1j * 0.5 * x)                                          # mild phase
    maps = []
    for c in range(C):
        cx, cy, cz = rng.uniform(-1.2, 1.2, 3)
        m = np.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / 1.5) * np.exp(1j * rng.uniform(0, np.pi))
        maps.append(m)
    maps = np.stack(maps).astype(np.complex64)
    img = maps * obj[None]
    k = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img, axes=(1, 2, 3)), axes=(1, 2, 3)), axes=(1, 2, 3))
    return k.astype(np.complex64), maps


def _acq_mask(Y=64, Z=32, acs=(20, 44), pf_keep=56):
    """Even PE lines + ACS block, elliptical corner cut, partial Fourier (lines >= pf_keep dropped)."""
    yy, zz = np.meshgrid(np.arange(Y), np.arange(Z), indexing="ij")
    ellipse = ((yy - Y / 2) / (Y / 2)) ** 2 + ((zz - Z / 2) / (Z / 2)) ** 2 <= 1.0
    lines = (yy % 2 == 0) | ((yy >= acs[0]) & (yy < acs[1]))
    return ellipse & lines & (yy < pf_keep)


def _rss(k):
    img = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(k, axes=(1, 2, 3)), axes=(1, 2, 3)), axes=(1, 2, 3))
    return np.sqrt((np.abs(img) ** 2).sum(0))


@pytest.fixture(scope="module")
def case():
    k_full, _ = _phantom_kspace()
    m_acq = _acq_mask()
    k_us = k_full * m_acq[None, None]
    return k_full, k_us, m_acq


def test_target_mask_is_the_missing_lines_inside_the_acquired_envelope(case):
    from uih_mtp.grappa import target_mask
    _, _, m_acq = case
    targets = target_mask(m_acq, R=2)
    yy, zz = np.meshgrid(np.arange(64), np.arange(32), indexing="ij")
    assert not (targets & m_acq).any()                                        # never an acquired point
    assert targets[yy % 2 == 0].sum() == 0                                    # even lines are acquired
    assert targets[yy >= 55].sum() == 0                                       # beyond the last acquired line (54): partial Fourier
    inside = ((yy - 32) / 32) ** 2 + ((zz - 16) / 16) ** 2 <= 0.8             # well inside the ellipse
    assert targets[inside & (yy % 2 == 1) & (yy < 55) & ~m_acq].all()
    assert targets[np.isclose(yy, 32) & (zz == 0)].sum() == 0                # corner outside the envelope


def test_grappa_recovers_the_phantom_and_keeps_acquired_lines(case):
    from uih_mtp.grappa import grappa3d
    k_full, k_us, m_acq = case
    out = grappa3d(torch.from_numpy(k_us), torch.from_numpy(m_acq), acs_lines=(20, 44), R=2,
                   device=torch.device("cpu"), lam_rel=1e-4)
    k_est = out.kspace.numpy()
    m_target = out.m_grappa
    # acquired lines are bit-identical
    assert np.array_equal(k_est[:, :, m_acq], k_us[:, :, m_acq])
    # nothing written outside acquired + targets
    assert np.abs(k_est[:, :, ~(m_acq | m_target)]).max() == 0
    # reference = the full k-space restricted to the same support (ellipse and partial Fourier untouched)
    ref = k_full * (m_acq | m_target)[None, None]
    nrmse = np.linalg.norm(_rss(k_est) - _rss(ref)) / np.linalg.norm(_rss(ref))
    assert nrmse < 0.05, nrmse
    # and the interpolated lines themselves are close in k-space
    err = np.linalg.norm(k_est[:, :, m_target] - k_full[:, :, m_target]) / np.linalg.norm(k_full[:, :, m_target])
    assert err < 0.15, err


def test_grappa_hold_out_check_reports_small_error_on_hidden_acs_lines(case):
    from uih_mtp.grappa import acs_holdout_check
    _, k_us, m_acq = case
    rel = acs_holdout_check(torch.from_numpy(k_us), torch.from_numpy(m_acq), acs_lines=(20, 44), R=2,
                            device=torch.device("cpu"), lam_rel=1e-4)
    assert 0 < rel < 0.15, rel


def test_leave_one_out_reports_one_error_per_calibration_line(case):
    from uih_mtp.grappa import acs_leave_one_out
    _, k_us, m_acq = case
    loo = acs_leave_one_out(torch.from_numpy(k_us), torch.from_numpy(m_acq), acs_lines=(20, 44), R=2,
                            device=torch.device("cpu"), lam_rel=1e-4)
    assert loo["lines"] == [23, 25, 27, 29, 31, 33, 35, 37, 39]           # odd ACS lines with all sources inside
    assert len(loo["per_line"]) == 9
    assert all(0 < e < 0.15 for e in loo["per_line"]), loo["per_line"]
    assert 0 < loo["mean"] < 0.15
