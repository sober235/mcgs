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


def test_sampled_data_is_fit_under_undersampling():
    """遮罩若从 _normal_op 中掉落,CG 解将退化为零填充伴随解,已采样区残差会很大。"""
    x, csm, y = _toy()
    mask = torch.zeros(8, 8, 8)
    mask[::2] = 1.0
    yu = y * mask.unsqueeze(0)
    xh = cg_sense(yu, csm, mask, lam=0.0, iters=200)
    pred = fft3c(csm * xh.unsqueeze(0))
    m = mask.unsqueeze(0)
    sampled_err = (torch.linalg.vector_norm(m * (pred - y))
                   / torch.linalg.vector_norm(m * y))
    assert sampled_err < 1e-3
