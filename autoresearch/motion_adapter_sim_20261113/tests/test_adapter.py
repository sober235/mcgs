import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import geom
import warp
from common import SRC

if not (SRC / "paired_gs/gaussian/tile_voxelizer.py").exists():
    pytest.skip("frozen trainer source not on this machine", allow_module_level=True)
sys.path.insert(0, str(SRC / "paired_gs"))
from gaussian.tile_voxelizer import TileVoxelizer          # noqa: E402

from adapter import MotionAdapter, dense_displacement, render_transformed    # noqa: E402

SHAPE = (32, 32, 24)


def as_complex(v):
    return torch.complex(v[0], v[1]) if isinstance(v, tuple) else v


def cloud(n=300, smooth=False, seed=0, device="cpu", spread=0.45):
    g = torch.Generator().manual_seed(seed)
    pos = (torch.rand(n, 3, generator=g) - 0.5) * 2 * spread
    lo, hi = (0.16, 0.26) if smooth else (0.07, 0.15)
    scales = lo + (hi - lo) * torch.rand(n, 3, generator=g)
    rot = torch.randn(n, 4, generator=g)
    den = torch.complex(torch.rand(n, generator=g) + 0.2, 0.3 * torch.randn(n, generator=g))
    return [t.to(device) for t in (pos, scales, rot, den)]


def small_voxelizer_and_cloud(smooth=False, cuda=False):
    dev = "cuda:0" if cuda else "cpu"
    vox = TileVoxelizer(SHAPE, tile_size=8, use_cuda=cuda, device=dev).to(dev)
    return vox, cloud(smooth=smooth, device=dev)


def frozen(ad):
    for p in ad.parameters():
        p.requires_grad_(False)
    return ad


def test_identity_bitexact():
    vox, g = small_voxelizer_and_cloud()
    ad = frozen(MotionAdapter(SHAPE, True, True, 4, 1))
    assert ad.is_identity()
    assert torch.equal(render_transformed(vox, *g, ad), as_complex(vox(*g)))
    assert torch.equal(render_transformed(vox, *g, ad, seg=3), as_complex(vox(*g)))


def test_zero_but_trainable_adapter_keeps_the_graph_and_matches():
    vox, g = small_voxelizer_and_cloud()
    ad = MotionAdapter(SHAPE, True, True, 4, 1)
    out = render_transformed(vox, *g, ad, seg=2)
    ref = as_complex(vox(*g))
    assert (out - ref).abs().max() < 1e-5 * ref.abs().max()
    (out - ref.roll(1, 0)).abs().pow(2).sum().backward()
    for name in ("theta", "t_mm", "ctrl", "seg_theta", "seg_t_mm"):
        assert getattr(ad, name).grad is not None and torch.isfinite(getattr(ad, name).grad).all(), name
    assert ad.t_mm.grad.abs().max() > 0 and ad.seg_t_mm.grad[2].abs().max() > 0
    assert ad.seg_t_mm.grad[1].abs().max() == 0                 # the reference segment never moves


def test_translation_matches_shifted_render():
    vox, g = small_voxelizer_and_cloud()
    ad = MotionAdapter(SHAPE, True, False, 1, 0)
    with torch.no_grad():
        ad.t_mm.copy_(torch.tensor([2.0, 0.0, -1.0]))
    moved = render_transformed(vox, *g, ad).detach()
    ref = as_complex(vox(*g))
    torch.testing.assert_close(moved[4:-4, 4:-4, 4:-4], torch.roll(ref, (2, 0, -1), (0, 1, 2))[4:-4, 4:-4, 4:-4],
                               atol=1e-4, rtol=0)


def test_rotation_matches_resampled_render():
    vox, g = small_voxelizer_and_cloud(smooth=True)
    ad = MotionAdapter(SHAPE, True, False, 1, 0)
    with torch.no_grad():
        ad.theta.copy_(torch.deg2rad(torch.tensor([0.0, 0.0, 3.0])))
        ad.t_mm.copy_(torch.tensor([0.5, -0.25, 0.0]))
    moved = render_transformed(vox, *g, ad).detach().numpy()
    gt = geom.Rigid(geom.rotmat(np.deg2rad([0.0, 0.0, 3.0])), np.array([0.5, -0.25, 0.0]))
    ref = warp.rigid(as_complex(vox(*g)).numpy(), gt)             # render, then resample with the same transform
    assert np.abs(moved - ref)[4:-4, 4:-4, 4:-4].max() < 0.02 * np.abs(ref).max()


def test_segment_pose_composes_after_inter_scan_pose():
    vox, g = small_voxelizer_and_cloud(smooth=True)
    ad = MotionAdapter(SHAPE, True, False, 3, 0)
    with torch.no_grad():
        ad.theta.copy_(torch.deg2rad(torch.tensor([0.0, 2.0, 0.0])))
        ad.seg_theta[2].copy_(torch.deg2rad(torch.tensor([0.0, 0.0, 3.0])))
        ad.seg_t_mm[2].copy_(torch.tensor([1.0, 0.0, 0.0]))
    inter = geom.Rigid(geom.rotmat(np.deg2rad([0.0, 2.0, 0.0])), np.zeros(3))
    delta = geom.Rigid(geom.rotmat(np.deg2rad([0.0, 0.0, 3.0])), np.array([1.0, 0.0, 0.0]))
    d = dense_displacement(ad, SHAPE, seg=2)
    r = geom.grid_mm(SHAPE)
    np.testing.assert_allclose(d, delta.compose(inter).apply(r) - r, atol=1e-4)
    np.testing.assert_allclose(dense_displacement(ad, SHAPE, seg=0), inter.apply(r) - r, atol=1e-4)
    np.testing.assert_allclose(dense_displacement(ad, SHAPE), inter.apply(r) - r, atol=1e-4)


def _fd(f, p, idx, eps):
    with torch.no_grad():
        old = p[idx].item()
        p[idx] = old + eps
        a = f().item()
        p[idx] = old - eps
        b = f().item()
        p[idx] = old
    return (a - b) / (2 * eps)


def _check_gradients(cuda):
    vox, g = small_voxelizer_and_cloud(smooth=True, cuda=cuda)
    dev = g[0].device
    ad = MotionAdapter(SHAPE, True, True, 3, 0).to(dev)
    with torch.no_grad():
        ad.theta.copy_(torch.tensor([0.01, -0.02, 0.015]))
        ad.t_mm.copy_(torch.tensor([0.4, -0.3, 0.2]))
        ad.ctrl.normal_(0, 0.1, generator=torch.Generator().manual_seed(1)) if not cuda else ad.ctrl.fill_(0.05)
        ad.seg_theta[2].copy_(torch.tensor([0.0, 0.01, -0.01]))
        ad.seg_t_mm[2].copy_(torch.tensor([0.2, 0.0, -0.2]))
    target = as_complex(vox(*g)).detach()
    f = lambda: (render_transformed(vox, *g, ad, seg=2) - target).abs().pow(2).sum()
    f().backward()
    for name, idx, eps in (("theta", (2,), 2e-3), ("theta", (0,), 2e-3), ("t_mm", (0,), 2e-2), ("t_mm", (2,), 2e-2),
                           ("seg_theta", (2, 2), 2e-3), ("seg_t_mm", (2, 0), 2e-2), ("ctrl", (2, 2, 2, 0), 2e-2)):
        p = getattr(ad, name)
        num = _fd(f, p, idx, eps)
        ana = p.grad[idx].item()
        assert ana == pytest.approx(num, rel=0.08, abs=1e-3 * abs(num) + 1e-6), (name, idx, ana, num)


def test_gradients_match_finite_differences():
    _check_gradients(cuda=False)


@pytest.mark.gpu
def test_gradients_match_finite_differences_cuda():
    _check_gradients(cuda=True)


@pytest.mark.gpu
def test_cuda_identity_and_rotation():
    vox, g = small_voxelizer_and_cloud(smooth=True, cuda=True)
    ad = frozen(MotionAdapter(SHAPE, True, True, 4, 1).to(g[0].device))
    assert torch.equal(render_transformed(vox, *g, ad), as_complex(vox(*g)))
    ad = MotionAdapter(SHAPE, True, False, 1, 0).to(g[0].device)
    with torch.no_grad():
        ad.theta.copy_(torch.deg2rad(torch.tensor([0.0, 0.0, 3.0])))
    moved = render_transformed(vox, *g, ad).detach().cpu().numpy()
    ref = warp.rigid(as_complex(vox(*g)).cpu().numpy(), geom.Rigid(geom.rotmat(np.deg2rad([0.0, 0.0, 3.0])), np.zeros(3)))
    assert np.abs(moved - ref)[4:-4, 4:-4, 4:-4].max() < 0.02 * np.abs(ref).max()


def test_recovers_simulated_shift_direction():
    vox, g = small_voxelizer_and_cloud(smooth=True)
    sim = geom.Rigid(np.eye(3), np.array([1.5, 0.0, 0.0]))
    target = torch.from_numpy(warp.rigid(as_complex(vox(*g)).numpy(), sim)).to(torch.complex64)
    ad = MotionAdapter(SHAPE, True, False, 1, 0)
    opt = torch.optim.Adam(ad.parameters(), lr=0.05)
    for _ in range(150):
        opt.zero_grad()
        (render_transformed(vox, *g, ad) - target).abs().pow(2).sum().backward()
        opt.step()
    assert ad.t_mm[0].item() == pytest.approx(1.5, abs=0.1) and abs(ad.t_mm[1].item()) < 0.1   # same sign as the simulator


def test_nonrigid_has_zero_mean_and_is_bounded():
    ad = MotionAdapter(SHAPE, False, True, 1, 0)
    with torch.no_grad():
        ad.ctrl.normal_(0, 3.0, generator=torch.Generator().manual_seed(0))
    assert np.abs(dense_displacement(ad, SHAPE)).max() <= 2 * 3.0 + 1e-5      # |u_max tanh| <= 3, minus a mean <= 3
    with torch.no_grad():
        ad.ctrl.zero_()
        ad.ctrl[2, 2, 2, 0] = 10.0                                # only the centre control point, along axis 0
    d = dense_displacement(ad, SHAPE)
    mean = 3.0 / 125
    np.testing.assert_allclose(d[0, 0, 0], [-mean, 0, 0], atol=1e-4)
    p = (np.array([16, 16, 12]) + 0.5) / np.array(SHAPE) * 2 - 1    # voxel next to the FOV centre
    w = np.prod(1 - np.abs(p) * 2)                                # trilinear weight of the centre control point
    np.testing.assert_allclose(d[16, 16, 12], [3.0 * w - mean, 0, 0], atol=1e-4)


def test_set_truth_reproduces_a_simulated_field():
    brain = np.zeros(SHAPE, bool)
    brain[-16:-6, 10:22, 8:16] = True                      # superior side; the large displacements sit inferior
    u = warp.rbf_field(7, 1.0, brain, 0.8, shape=SHAPE, sigma_mm=(14.0, 20.0))
    g = geom.rigid_from_pivot((0.5, 0.5, 1.5), (0.8, -0.5, 0.3), pivot_offset_mm=(8, 3, 0))
    ad = MotionAdapter(SHAPE, True, True, 2, 0)
    seg = [geom.Rigid.identity(), geom.Rigid(geom.rotmat(np.deg2rad([0, 0, 1.0])), np.array([0.3, 0, 0]))]
    ad.set_truth(inter=g, dense_u=u, segments=seg, native_shift_mm=(0.6, 0.1, -0.1), brain=brain)
    assert not any(p.requires_grad for p in ad.parameters())
    r = geom.grid_mm(SHAPE)
    s = np.array([0.6, 0.1, -0.1])
    want = g.apply(r + s + warp._interp_field(u, r + s)) - r      # additive motion o native shift
    got = dense_displacement(ad, SHAPE)
    assert np.abs(got - want)[brain].max() < 0.25 and ad.oracle_fit_mm < 0.25
    got1 = dense_displacement(ad, SHAPE, seg=1)
    np.testing.assert_allclose(got1, seg[1].apply(r + got) - r, atol=1e-3)


def test_param_groups_regulariser_and_state():
    ad = MotionAdapter(SHAPE, True, True, 4, 1)
    groups = ad.param_groups(dict(rigid_theta=1e-3, rigid_t=2e-2, nonrigid=5e-3, seg_theta=5e-4, seg_t=1e-2))
    assert [g["name"] for g in groups] == ["rigid_theta", "rigid_t", "nonrigid", "seg_theta", "seg_t"]
    assert [g["lr"] for g in groups] == [1e-3, 2e-2, 5e-3, 5e-4, 1e-2]
    w = dict(ctrl_l2=1.0, ctrl_smooth=1.0, seg_l2=1.0, seg_smooth=1.0)
    assert ad.regulariser(w).item() == 0.0
    with torch.no_grad():
        ad.ctrl[0, 0, 0, 0] = 0.5
        ad.seg_t_mm[3, 0] = 1.0
    assert ad.regulariser(w).item() > 0
    st = ad.state()
    assert st["t_mm"] == [0.0, 0.0, 0.0] and len(st["segments"]) == 4 and st["segments"][3]["t_mm"][0] == 1.0
    only_rigid = MotionAdapter(SHAPE, True, False, 0, 0)
    assert [g["name"] for g in only_rigid.param_groups(dict(rigid_theta=1, rigid_t=1))] == ["rigid_theta", "rigid_t"]
    assert only_rigid.regulariser(w).item() == 0.0


def test_adam_eps_of_the_adapter_groups_decides_whether_a_tiny_gradient_moves_the_pose():
    """The trainer's Adam uses eps 1e-8. The pose gradients of the r1 runs were far below that, so the pose moved by
    a small fraction of its learning rate. A group-level eps restores steps of about one learning rate."""
    lrs = dict(rigid_theta=1e-3, rigid_t=2e-2)
    moved = {}
    for eps in (None, 1e-15):
        ad = MotionAdapter(SHAPE, True, False, 0, 0)
        groups = ad.param_groups(lrs, eps=eps)
        assert all(("eps" in g) == (eps is not None) for g in groups)
        opt = torch.optim.Adam([dict(params=[torch.nn.Parameter(torch.zeros(1))], lr=1.0), *groups])
        ad.t_mm.grad = torch.full((3,), 1e-11)
        ad.theta.grad = torch.zeros(3)
        opt.step()
        moved[eps] = float(ad.t_mm.detach().abs().max())
    assert moved[None] < 2e-2 * 2e-3 and moved[1e-15] == pytest.approx(2e-2, rel=1e-3)


def test_set_inter_matches_set_truth():
    g = geom.rigid_from_pivot((0.5, -0.3, 1.2), (0.8, -0.5, 0.3), pivot_offset_mm=(8, 3, 0))
    a, b = MotionAdapter(SHAPE, True, False, 0, 0), MotionAdapter(SHAPE, True, False, 0, 0)
    a.set_truth(inter=g)
    b.set_inter(g)
    assert a.state()["theta_deg"] == b.state()["theta_deg"] and a.state()["t_mm"] == b.state()["t_mm"]
    assert all(p.requires_grad for p in b.parameters())            # the setter leaves requires_grad alone
    r = geom.grid_mm(SHAPE)
    np.testing.assert_allclose(dense_displacement(b, SHAPE), g.apply(r) - r, atol=1e-4)


def test_set_segment_pose_matches_set_truth():
    seg = [geom.Rigid.identity(), geom.Rigid(geom.rotmat(np.deg2rad([0, 0, 1.0])), np.array([0.3, 0, 0])),
           geom.Rigid(geom.rotmat(np.deg2rad([0.4, 0, 0])), np.array([0.0, -0.2, 0.1]))]
    a, b = MotionAdapter(SHAPE, True, False, 3, 0), MotionAdapter(SHAPE, True, False, 3, 0)
    a.set_truth(segments=seg)
    for k, p in enumerate(seg):
        b.set_segment_pose(k, p)
    assert a.state()["segments"] == b.state()["segments"]
    assert all(p.requires_grad for p in b.parameters())
    with pytest.raises(AssertionError):
        b.set_segment_pose(3, seg[1])
