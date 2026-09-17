import numpy as np
import torch

import sim_cssense_af16 as SIM
from conftest import fft3c


def _toy(seed=0, G=(8, 8, 8), NC=2):
    rng = np.random.default_rng(seed)
    S = (rng.random((NC,) + G) + 1j * rng.random((NC,) + G)).astype(np.complex64)
    x = (rng.random(G) + 1j * rng.random(G)).astype(np.complex64)
    return rng, S, x


def test_sigma_from_residual_recovers_known_noise():
    rng, S, x = _toy(G=(16, 16, 16), NC=4)
    m0 = rng.random((16, 16)) < 0.5
    sigma0 = 0.05
    y = fft3c(S * x[None]) + sigma0 * (rng.standard_normal(S.shape) + 1j * rng.standard_normal(S.shape))
    y = (y * m0[None, None]).astype(np.complex64)
    alpha, kres, sigma = SIM.sigma_from_residual(torch.from_numpy(S), torch.from_numpy(x), torch.from_numpy(y),
                                                 torch.from_numpy(m0), torch.device("cpu"), chunk=3)
    assert abs(alpha - 1) < 0.02
    assert abs(sigma / sigma0 - 1) < 0.05
    n = m0.sum() * S.shape[0] * S.shape[1]  # BUGFIX vs brief: [..., m0] keeps the readout axis S.shape[1]; must be in n.
    expect_kres = np.sqrt(2 * n) * sigma0 / np.linalg.norm(y[:, :, m0])
    assert abs(kres / expect_kres - 1) < 0.1


def test_synth_kspace_noise_level_and_seed():
    rng, S, x = _toy()
    k1 = SIM.synth_kspace(torch.from_numpy(S), torch.from_numpy(x), 0.1, np.random.default_rng(7), torch.device("cpu"), chunk=1)
    k2 = SIM.synth_kspace(torch.from_numpy(S), torch.from_numpy(x), 0.1, np.random.default_rng(7), torch.device("cpu"), chunk=1)
    clean = fft3c(S * x[None])
    assert k1.dtype == np.complex64 and np.array_equal(k1, k2)
    noise = k1 - clean
    assert abs(noise.real.std() / 0.1 - 1) < 0.1 and abs(noise.imag.std() / 0.1 - 1) < 0.1
