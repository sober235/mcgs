import numpy as np
import torch


def test_to_gauge_units_roundtrip(tiny_topic):
    import run_mcinr as RM
    x_raw = torch.from_numpy((np.random.default_rng(0).random((4, 16, 16)) + 0j).astype(np.complex64))
    M, scale = 3.0, 0.5
    x_norm = x_raw / M * 100                       # the paper's normalisation of the data the net fitted
    back = RM.to_gauge(x_norm, M, scale)
    assert torch.allclose(back, x_raw / scale, atol=1e-6)


def test_assemble_volume_from_slices():
    import run_mcinr as RM
    sl = np.zeros((3, 5, 4), np.complex64); sl[1] = 1
    v = RM.assemble(sl, [10, 11, 12], nx=16)
    assert v.shape == (16, 5, 4) and v[11].real.sum() == 20 and v[0].sum() == 0
