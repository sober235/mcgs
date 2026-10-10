"""The subtraction (enhancement-sparsity) prior of the AR trainer copy (CE autoresearch phase 6, 2026-10-10).

The module lives in the trainer snapshot on the server (trainer_patches/2026-10-10-subtraction-prior.diff records
it); the test loads it by path and is skipped where the snapshot is absent."""
import importlib.util
import os
from pathlib import Path

import pytest
import torch

AR = Path(os.environ.get("RETRO_TRAINER_AR", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois/"
                         "trainer/_trainer_e778bf2_ar")) / "paired_gs/losses/sub_prior.py"
pytestmark = pytest.mark.skipif(not AR.exists(), reason="AR trainer snapshot not on this machine")


def _mod():
    spec = importlib.util.spec_from_file_location("sub_prior", AR)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_penalty_zero_at_zero_and_symmetric_shapes():
    m = _mod()
    d = torch.tensor([-3.0, -0.1, 0.0, 0.1, 3.0])
    for mode in ("cauchy", "charb"):
        p = m.penalty(d, 0.5, mode)
        assert float(p[2]) == 0.0 and torch.allclose(p, p.flip(0))
    neg = m.penalty(d, 0.5, "neg")
    assert float(neg[3]) == 0.0 and float(neg[4]) == 0.0 and float(neg[0]) > 0.0   # only negative d is charged


def test_cauchy_barely_shrinks_large_enhancement():
    """d/dd of the Cauchy penalty is d / (1 + (d/delta)^2): for |d| >> delta it falls like delta^2 / d, so a strong
    enhancement feels a much smaller pull than under the L1-like Charbonnier (whose slope tends to 1)."""
    m = _mod()
    d = torch.tensor([10.0], requires_grad=True)
    (gc,) = torch.autograd.grad(m.penalty(d, 0.5, "cauchy").sum(), d)
    d2 = torch.tensor([10.0], requires_grad=True)
    (gl,) = torch.autograd.grad(m.penalty(d2, 0.5, "charb").sum(), d2)
    assert float(gc) < 0.1 * float(gl)


def test_robust_scale_recovers_the_global_ratio_despite_sparse_enhancement():
    m = _mod()
    g = torch.Generator().manual_seed(0)
    pre = 1.0 + 0.2 * torch.rand(32, 32, 32, generator=g)
    post = 1.37 * pre
    post[10:12, 10:12, :] += 3.0                       # sparse strong enhancement (~0.4 % of the voxels)
    support = torch.ones_like(pre, dtype=torch.bool)
    s, tissue = m.robust_scale(pre, post, support)
    assert abs(s - 1.37) < 1e-3
    assert 1.3 < tissue < 1.7


def test_sub_loss_is_finite_and_differentiable_through_both_channels():
    m = _mod()
    g = torch.Generator().manual_seed(1)
    vp = torch.complex(torch.randn(8, 8, 8, generator=g), torch.randn(8, 8, 8, generator=g)).requires_grad_(True)
    vc = torch.complex(torch.randn(8, 8, 8, generator=g), torch.randn(8, 8, 8, generator=g)).requires_grad_(True)
    vp_air = vp.detach().clone(); vp_air[0] = 0                         # exactly-zero voxels (air) must not give NaN
    vp_air.requires_grad_(True)
    support = torch.ones(8, 8, 8, dtype=torch.bool)
    for mode in ("cauchy", "charb", "neg"):
        loss = m.sub_loss(vp_air, vc, 1.2, 0.3, support, mode)
        gp, gc = torch.autograd.grad(loss, (vp_air, vc))
        assert torch.isfinite(loss) and torch.isfinite(torch.view_as_real(gp)).all() and torch.isfinite(torch.view_as_real(gc)).all()
    loss = m.sub_loss(vp, vc, 1.2, 0.3, support, "cauchy", detach_plain=True)
    gp, gc = torch.autograd.grad(loss, (vp, vc), allow_unused=True)
    assert gp is None and gc is not None
