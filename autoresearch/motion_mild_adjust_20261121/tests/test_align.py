import numpy as np
import torch
import torch.nn.functional as F

import align
import geom
import warp
from helpers import blob_at, centroid, phantom

SHAPE = (64, 64, 48)


def _c(a):
    return torch.from_numpy(np.asarray(a)).to(torch.complex64)


def test_canonical_zf_undoes_known_shift():
    c0 = np.array([30.0, 28.0, 22.0])
    zf = blob_at(c0 + np.array([3.0, 0.0, 0.0]), SHAPE)                 # where the T2W image shows the blob
    inter = geom.Rigid(np.eye(3), np.array([3.0, 0.0, 0.0]))            # canonical point r -> r + 3 mm in the T2 image
    out = align.canonical_zf(_c(zf), inter)
    assert out.shape == SHAPE and out.dtype == torch.complex64
    assert np.abs(centroid(out.abs().numpy()) - c0).max() < 0.05
    wrong = align.canonical_zf(_c(zf), inter.inverse())                 # the opposite direction doubles the offset
    assert abs(centroid(wrong.abs().numpy())[0] - (c0[0] + 6.0)) < 0.05


def test_canonical_zf_inverts_the_simulators_warp():
    m = np.abs(phantom(SHAPE, seed=3) * blob_at((32, 32, 24), SHAPE, sigma=14.0))
    g = geom.rigid_from_pivot((0.0, 1.0, 2.0), (1.5, -1.0, 0.5), pivot_offset_mm=(-10.0, 5.0, 0.0))
    moved = warp.rigid(m.astype(np.complex128), g)                      # image = m o T^-1 (simulate.py)
    back = align.canonical_zf(_c(moved), g).abs().numpy()
    core = (slice(12, 52), slice(12, 52), slice(10, 38))
    err = np.abs(back - m)[core].max() / m[core].max()
    assert err < 0.05, err
    assert np.abs(np.abs(moved) - m)[core].max() / m[core].max() > 3 * err      # the warp itself is far larger
    assert torch.equal(align.canonical_zf(_c(moved), geom.Rigid.identity()), _c(moved))


def _img(seed=0):
    a = np.abs(phantom(SHAPE, seed=seed) * blob_at((32, 32, 24), SHAPE, sigma=12.0))
    return torch.from_numpy(a / a.max()).double()


def _shift(a, d):
    """a displaced by d voxels along axis 0, differentiable in d (grid_sample)."""
    n0, n1, n2 = a.shape
    z, y, x = torch.meshgrid(torch.linspace(-1, 1, n0, dtype=a.dtype), torch.linspace(-1, 1, n1, dtype=a.dtype),
                             torch.linspace(-1, 1, n2, dtype=a.dtype), indexing="ij")
    grid = torch.stack([x, y, z - d * 2.0 / (n0 - 1)], dim=-1)[None]
    return F.grid_sample(a[None, None], grid, mode="bilinear", padding_mode="border", align_corners=True)[0, 0]


def _mask():
    return torch.from_numpy(np.abs(blob_at((32, 32, 24), SHAPE, sigma=12.0)) > 0.2)


def test_ngf_loss_minimum_at_alignment():
    a, mask = _img(), _mask()
    vals = [float(align.ngf_loss(_shift(a, torch.tensor(d, dtype=a.dtype)), a, mask)) for d in (0.0, 0.5, 1.0, 2.0)]
    assert vals[0] < vals[1] < vals[2] < vals[3]
    assert 0.0 <= vals[0] and vals[3] <= 1.0
    # not 0 at alignment: where the gradient is below the edge parameter the ratio stays under 1 (flat regions)
    assert abs(vals[0] - float(align.ngf_loss(a, a, mask))) < 1e-9


def test_ngf_gradient_points_to_alignment():
    a, mask = _img(1), _mask()
    b = _shift(a, torch.tensor(1.0, dtype=a.dtype)).detach()            # the fixed image, 1 voxel away
    d = torch.zeros((), dtype=a.dtype, requires_grad=True)
    opt = torch.optim.Adam([d], lr=0.02)
    for _ in range(200):
        opt.zero_grad()
        align.ngf_loss(_shift(a, d), b, mask).backward()
        opt.step()
    assert abs(float(d) - 1.0) < 0.1, float(d)


def test_ngf_ignores_contrast_inversion():
    a, mask = _img(2), _mask()
    b = _shift(a, torch.tensor(0.7, dtype=a.dtype))
    assert abs(float(align.ngf_loss(a, b, mask)) - float(align.ngf_loss(a, b.max() - b, mask))) < 1e-6
    assert abs(float(align.ngf_loss(a, b, mask)) - float(align.ngf_loss(a, 3.0 * b, mask))) < 1e-6


def test_ngf_only_reads_inside_the_mask():
    a, mask = _img(4), _mask()
    b = a.clone()
    b[~mask] = torch.rand(int((~mask).sum()), dtype=a.dtype, generator=torch.Generator().manual_seed(0))
    inner = torch.from_numpy(np.abs(blob_at((32, 32, 24), SHAPE, sigma=12.0)) > 0.5)  # away from the mask's edge
    assert abs(float(align.ngf_loss(a, b, inner)) - float(align.ngf_loss(a, a, inner))) < 1e-9
    assert float(align.ngf_loss(a, b, torch.ones_like(mask))) > float(align.ngf_loss(a, a, torch.ones_like(mask))) + 0.05


# ---------------------------------------------------------------------------------- the alignment step (Aligner)
from adapter import MotionAdapter  # noqa: E402

REG = dict(ctrl_l2=1e-3, ctrl_smooth=1e-2, seg_l2=0.0, seg_smooth=0.0)


def _sample(a, ad):
    """a o T: the image read at the adapter-transformed voxel positions (differentiable in the control grid)."""
    n0, n1, n2 = a.shape
    ax = [(torch.arange(n, dtype=torch.float32) + 0.5) / n * 2.0 - 1.0 for n in (n0, n1, n2)]
    p = torch.stack(torch.meshgrid(*ax, indexing="ij"), dim=-1).reshape(-1, 3)
    q, _ = ad.transform(p)
    out = F.grid_sample(a[None, None].float(), q.flip(-1).reshape(1, n0, n1, n2, 3), mode="bilinear",
                        padding_mode="border", align_corners=False)
    return out[0, 0]


def _frozen_adapter(ctrl=None):
    ad = MotionAdapter(SHAPE, True, True, 0, 0)
    with torch.no_grad():
        ad.t_mm.copy_(torch.tensor([0.5, 0.1, -0.14]))
        if ctrl is not None:
            ad.ctrl.copy_(ctrl)
    for p in ad.parameters():
        p.requires_grad_(False)
    return ad


def _true_ctrl():
    g = torch.Generator().manual_seed(5)
    return 0.4 * torch.randn(5, 5, 5, 3, generator=g)


def test_ctrl_gets_no_gradient_from_main_loss():
    a, mask = _img(5), _mask()
    ref = _sample(a, _frozen_adapter(_true_ctrl())).detach()
    ad = _frozen_adapter()
    al = align.Aligner(ad, start=2, end=6, lr=0.05, weight=1.0, reg=REG)
    x = torch.ones((), requires_grad=True)                           # stands for the Gaussian parameters
    assert al.step(1, lambda: _sample(a, ad), ref, mask) is None     # before the start: nothing happens
    assert not bool(ad.ctrl.any())
    main = (x * _sample(a, ad)).pow(2).sum()                         # the graph of the main loss ...
    loss = al.step(2, lambda: _sample(a, ad), ref, mask)             # ... survives an alignment step taken meanwhile
    main.backward()
    assert x.grad is not None and ad.ctrl.grad is None and not ad.ctrl.requires_grad
    assert isinstance(loss, float) and bool(ad.ctrl.any())           # the alignment step itself moved the grid
    assert not any(p.requires_grad for p in ad.parameters())


def test_alignment_steps_reduce_the_mismatch_and_leave_the_rigid_pose():
    a, mask = _img(6), _mask()
    ref = _sample(a, _frozen_adapter(_true_ctrl())).detach()
    ad = _frozen_adapter()
    al = align.Aligner(ad, start=0, end=400, lr=0.05, weight=1.0, reg=dict(REG, ctrl_l2=0.0, ctrl_smooth=0.0))
    first = float(align.ngf_loss(_sample(a, ad), ref, mask))
    for it in range(150):
        al.step(it, lambda: _sample(a, ad), ref, mask)
    last = float(align.ngf_loss(_sample(a, ad), ref, mask))
    assert last < first - 0.3 * (first - float(align.ngf_loss(ref, ref, mask))), (first, last)
    np.testing.assert_allclose(ad.t_mm.numpy(), [0.5, 0.1, -0.14], atol=0)
    assert not bool(ad.theta.any())


def test_adapter_frozen_after_end():
    a, mask = _img(7), _mask()
    ref = _sample(a, _frozen_adapter(_true_ctrl())).detach()
    ad = _frozen_adapter()
    al = align.Aligner(ad, start=0, end=3, lr=0.05, weight=1.0, reg=REG)
    for it in range(3):
        assert al.step(it, lambda: _sample(a, ad), ref, mask) is not None
    kept = ad.ctrl.clone()
    for it in (3, 4, 100):
        assert al.step(it, lambda: _sample(a, ad), ref, mask) is None
    assert torch.equal(ad.ctrl, kept) and al.n_steps == 3
    assert not any(p.requires_grad for p in ad.parameters())


def test_aligner_needs_the_nonrigid_part():
    ad = MotionAdapter(SHAPE, True, False, 0, 0)
    try:
        align.Aligner(ad, start=0, end=3, lr=0.05, weight=1.0, reg=REG)
    except AssertionError:
        return
    raise AssertionError("an adapter without the control grid was accepted")
