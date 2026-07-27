import torch
from crisp.metrics.coil_domain import fft3c, e_coil_full
from crisp.recon.cg_sense import cg_sense


def _toy(seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(8, 8, 8, generator=g, dtype=torch.complex64)
    csm = torch.randn(4, 8, 8, 8, generator=g, dtype=torch.complex64)
    csm = csm / (csm.abs().pow(2).sum(0, keepdim=True).sqrt() + 1e-8)
    y = fft3c(csm * x.unsqueeze(0))
    return x, csm, y


def test_recovers_exactly_when_fully_sampled():
    x, csm, y = _toy()
    mask = torch.ones(8, 8, 8)
    xh = cg_sense(y, csm, mask, lam=0.0, iters=60)
    assert e_coil_full(xh, csm, y) < 1e-4


def test_regularization_shrinks_solution():
    x, csm, y = _toy()
    mask = torch.ones(8, 8, 8)
    x_small = cg_sense(y, csm, mask, lam=10.0, iters=60)
    x_big = cg_sense(y, csm, mask, lam=0.0, iters=60)
    assert torch.linalg.vector_norm(x_small) < torch.linalg.vector_norm(x_big)


def test_undersampled_run_is_finite_and_shaped():
    x, csm, y = _toy()
    mask = torch.zeros(8, 8, 8)
    mask[::2] = 1.0
    xh = cg_sense(y * mask.unsqueeze(0), csm, mask, lam=1e-3, iters=20)
    assert xh.shape == x.shape
    assert torch.isfinite(xh.abs()).all()
