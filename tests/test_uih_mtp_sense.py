"""SENSE combination and CG-SENSE (uih_mtp.sense), BART cfl I/O (uih_mtp.bartio)."""
import numpy as np
import pytest
import torch

from tests.test_uih_mtp_grappa import _acq_mask, _phantom_kspace


def _ifft3c(k):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(k, axes=(-3, -2, -1)), axes=(-3, -2, -1), norm="ortho"), axes=(-3, -2, -1))


@pytest.fixture(scope="module")
def case():
    k_full, maps = _phantom_kspace()
    sos = np.sqrt((np.abs(maps) ** 2).sum(0))
    maps_n = (maps / (sos + 1e-12)).astype(np.complex64)             # unit sum of squares, as ecalib -c 0 gives
    obj = (_ifft3c(k_full) * np.conj(maps_n)).sum(0)                    # the object seen through unit-SoS maps
    m_acq = _acq_mask()
    return k_full, maps_n, obj, m_acq


def test_sense_combine_returns_the_object_for_unit_sos_maps(case):
    from uih_mtp.sense import sense_combine
    k_full, maps, obj, _ = case
    x = sense_combine(torch.from_numpy(k_full), torch.from_numpy(maps)).numpy()
    assert x.shape == obj.shape and x.dtype == np.complex64
    np.testing.assert_allclose(x, obj, rtol=1e-3, atol=1e-3 * np.abs(obj).max())


def test_cg_sense_recovers_the_object_from_the_undersampled_data(case):
    from uih_mtp.sense import cg_sense
    k_full, maps, obj, m_acq = case
    y = k_full * m_acq[None, None]
    x = cg_sense(torch.from_numpy(y), torch.from_numpy(maps), torch.from_numpy(m_acq), iters=40,
                 device=torch.device("cpu")).numpy()
    # compare inside the support of the acquired k-space (partial Fourier and the corners are not recovered)
    ref = (_ifft3c(k_full * (m_acq | np.roll(m_acq, 1, axis=0))[None, None]) * np.conj(maps)).sum(0)
    support = np.abs(obj) > 0.2 * np.abs(obj).max()
    nrmse = np.linalg.norm((x - ref)[support]) / np.linalg.norm(ref[support])
    assert nrmse < 0.08, nrmse


def test_cfl_roundtrip(tmp_path):
    from uih_mtp.bartio import read_cfl, write_cfl
    rng = np.random.default_rng(1)
    a = (rng.standard_normal((5, 6, 7, 3)) + 1j * rng.standard_normal((5, 6, 7, 3))).astype(np.complex64)
    write_cfl(str(tmp_path / "a"), a)
    b = read_cfl(str(tmp_path / "a"))
    assert b.shape == a.shape and b.dtype == np.complex64
    np.testing.assert_array_equal(a, b)
    hdr = (tmp_path / "a.hdr").read_text().splitlines()
    assert hdr[0].startswith("# Dimensions") and hdr[1].split()[:4] == ["5", "6", "7", "3"]
