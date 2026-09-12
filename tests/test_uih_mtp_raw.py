"""UIH .raw reader for the 3D MTP Cartesian scan (uih_mtp.uih_raw).

The real-file tests read the pediatric TSC exam and are skipped when the raw file is absent.
The numbers asserted come from the header-only probe of 2026-09-12 (see the design spec).
"""
import os

import numpy as np
import pytest

RAW = ("/data0/congcong/data/3D_data/Children_data/TSC_20220124_age16/rawdata/"
       "UID_126819142341469_mtp_cartesian_.7x.7x1.raw")
needs_raw = pytest.mark.skipif(not os.path.exists(RAW), reason="MTP raw file not available")


@pytest.fixture(scope="module")
def index():
    from uih_mtp.uih_raw import scan_index
    return scan_index(RAW)


@needs_raw
def test_scan_index_counts_and_centres(index):
    assert index.n_imaging == 135084
    assert index.n_noise == 32
    assert index.samples == 640 and index.channels == 32
    assert (index.ro_center, index.pe_center, index.spe_center) == (160, 152, 52)
    assert index.pe_len == 304 and index.spe_len == 104          # protocol FT lengths


@needs_raw
def test_scan_index_masks_identical_across_contrasts(index):
    ref = index.mask(set_=1, con=0, u1=1)
    assert ref.shape == (304, 104) and ref.sum() == 11257
    for con in range(5):
        for u1 in (0, 1):
            assert np.array_equal(index.mask(set_=1, con=con, u1=u1), ref)


@needs_raw
def test_read_lines_fills_exactly_the_acquired_positions_and_centres_the_echo(index):
    from uih_mtp.uih_raw import read_lines
    k = read_lines(RAW, index, set_=1, con=0, u1=1)
    assert k.shape == (32, 640, 304, 104) and k.dtype == np.complex64
    energy = np.abs(k).sum(axis=(0, 1))                          # (PE, SPE)
    assert np.array_equal(energy > 0, index.mask(set_=1, con=0, u1=1))
    assert np.unravel_index(np.argmax(energy), energy.shape) == (152, 52)


def test_remove_readout_oversampling_keeps_the_central_field_of_view():
    from uih_mtp.uih_raw import remove_readout_oversampling
    rng = np.random.default_rng(0)
    img640 = np.zeros((3, 640, 2, 2), np.complex64)             # object only in the central 320 samples
    img640[:, 160:480] = rng.standard_normal((3, 320, 2, 2)) + 1j * rng.standard_normal((3, 320, 2, 2))
    k640 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(img640, axes=1), axis=1), axes=1)
    k320 = remove_readout_oversampling(k640, 320)
    expect = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(img640[:, 160:480], axes=1), axis=1), axes=1)
    assert k320.shape == (3, 320, 2, 2)
    np.testing.assert_allclose(k320, expect, rtol=1e-4, atol=1e-4)
