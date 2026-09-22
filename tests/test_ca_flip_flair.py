import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ca_flip_flair import flip_pair, fft3c, ifft3c


def _sense(ksp, csm):
    return (np.conj(csm) * ifft3c(ksp)).sum(axis=-1)


def test_flip_pair_flips_the_sense_image_and_keeps_the_support():
    rng = np.random.default_rng(0)
    img = rng.normal(size=(8, 8, 6)) + 1j * rng.normal(size=(8, 8, 6))
    csm = rng.normal(size=(8, 8, 6, 3)) + 1j * rng.normal(size=(8, 8, 6, 3))
    ksp = fft3c(img[..., None] * csm)
    support = np.zeros((8, 8, 6), bool)
    # symmetric about the DC SAMPLE (index N//2 = 4), which is the symmetry an image-space flip
    # preserves.  The brief's 1:7 window is symmetric about 3.5 instead -- half a sample off -- and
    # no correct implementation can return it unchanged (see test_flip_pair_mirrors_... below).
    support[:, 2:7, :] = True
    ksp = ksp * support[..., None]

    ksp_f, csm_f = flip_pair(ksp, csm, axis=1)

    got = _sense(ksp_f, csm_f)
    want = np.flip(_sense(ksp, csm), axis=1)
    assert np.allclose(got, want, atol=1e-8)
    got_support = np.abs(ksp_f).sum(axis=-1) > 0
    assert np.array_equal(got_support, support)    # known_zero stays valid


def test_flip_pair_mirrors_a_shutter_that_is_not_symmetric_about_dc():
    """The real FLAIR shutter is a DC-centred ellipse with a ONE-SIDED partial-Fourier cut in ky,
    so its support is mirrored about DC by the flip rather than preserved. Pin that law."""
    rng = np.random.default_rng(1)
    csm = rng.normal(size=(8, 8, 6, 3)) + 1j * rng.normal(size=(8, 8, 6, 3))
    ksp = fft3c(csm)
    support = np.zeros((8, 8, 6), bool)
    support[:, 2:8, :] = True                      # a one-sided cut: 2..7, DC at 4
    ksp = ksp * support[..., None]

    ksp_f, _ = flip_pair(ksp, csm, axis=1)

    got_support = np.abs(ksp_f).sum(axis=-1) > 0
    assert not np.array_equal(got_support, support)
    assert np.array_equal(got_support, np.roll(np.flip(support, axis=1), 1, axis=1))
