#!/usr/bin/env python3
"""Cold-start cloud init from the ZF images (paired_gs/gaussian/init_cloud.py).

The uniform init leaves ~72 % of a fresh cloud in air on the (256,256,176) bundle. These tests
pin the replacement: every drawn point sits inside the ZF head support, `intensity` weights
bright voxels more than dark ones, the seed makes the draw reproducible, and the 3-NN scale
tracks the point spacing.

Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_init_cloud.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paired_gs"))
import torch
from gaussian.init_cloud import head_support, sample_positions, scales_3nn, union_magnitude


def _phantom(shape=(32, 40, 24), r=0.35):
    d, h, w = shape
    zz, yy, xx = torch.meshgrid(torch.linspace(-1, 1, d), torch.linspace(-1, 1, h),
                                torch.linspace(-1, 1, w), indexing="ij")
    ball = (zz ** 2 + yy ** 2 + xx ** 2) < r ** 2
    mag = torch.zeros(shape)
    mag[ball] = 1.0 + 0.8 * xx[ball] / r       # 0.2 on the -x side, 1.8 on the +x side
    mag += 0.001 * torch.rand(shape)           # air noise well below the 5 % rule
    return mag, ball


def test_union_magnitude_is_max_of_normalised_channels():
    a = torch.zeros(4, 4, 4); a[0, 0, 0] = 2.0
    b = torch.zeros(4, 4, 4); b[1, 1, 1] = 4.0
    u = union_magnitude(a.to(torch.complex64), b.to(torch.complex64))
    assert u[0, 0, 0] > 0.5 and u[1, 1, 1] > 0.5 and u[2, 2, 2] == 0.0


def test_head_support_matches_the_ball():
    mag, ball = _phantom()
    sup = head_support(mag, 0.05)
    assert (sup == ball).float().mean() > 0.995


def test_every_sample_inside_support_and_seeded():
    mag, ball = _phantom()
    g = torch.Generator().manual_seed(7)
    vox, pos = sample_positions(mag, 2000, "head", 0.05, generator=g)
    assert vox.shape == (2000, 3) and pos.shape == (2000, 3)
    assert bool(ball[vox[:, 0], vox[:, 1], vox[:, 2]].all())
    assert pos.min() >= -1.0 and pos.max() <= 1.0
    g2 = torch.Generator().manual_seed(7)
    vox2, pos2 = sample_positions(mag, 2000, "head", 0.05, generator=g2)
    assert torch.equal(vox, vox2) and torch.equal(pos, pos2)


def test_intensity_mode_prefers_bright_voxels():
    mag, ball = _phantom()
    g = torch.Generator().manual_seed(0)
    vox_h, _ = sample_positions(mag, 5000, "head", 0.05, generator=g)
    vox_i, _ = sample_positions(mag, 5000, "intensity", 0.05, generator=g)
    xh = vox_h[:, 2].float().mean()
    xi = vox_i[:, 2].float().mean()
    assert xi > xh + 0.5, (xh, xi)        # bright side is +x, so intensity sampling drifts right


def test_positions_jitter_stays_within_the_voxel():
    mag, _ = _phantom()
    g = torch.Generator().manual_seed(1)
    vox, pos = sample_positions(mag, 500, "head", 0.05, generator=g)
    d, h, w = mag.shape
    centre = torch.stack([((vox[:, 0].float() + 0.5) / d) * 2 - 1,
                          ((vox[:, 1].float() + 0.5) / h) * 2 - 1,
                          ((vox[:, 2].float() + 0.5) / w) * 2 - 1], dim=-1)
    half = torch.tensor([1.0 / d, 1.0 / h, 1.0 / w])
    assert bool(((pos - centre).abs() <= half + 1e-6).all())


def test_scales_3nn_track_spacing():
    g = torch.Generator().manual_seed(3)
    dense = torch.rand(4000, 3, generator=g) * 0.2 - 0.1
    sparse = torch.rand(4000, 3, generator=g) * 2 - 1
    s_dense = scales_3nn(dense, (64, 64, 64))
    s_sparse = scales_3nn(sparse, (64, 64, 64))
    assert s_dense.shape == (4000, 3) and bool((s_dense > 0).all())
    assert s_sparse.mean() > 2 * s_dense.mean()      # sparse cloud is coarser (the 0.03 cap binds it)
    assert torch.allclose(scales_3nn(dense, (64, 64, 64), scale_mult=2.0), (2 * s_dense).clamp(max=0.03))
    # an isolated point far from everything is clamped, not turned into a blob
    with_outlier = torch.cat([dense, torch.tensor([[0.9, 0.9, 0.9]])], dim=0)
    s_out = scales_3nn(with_outlier, (64, 64, 64))
    assert abs(float(s_out[-1, 0]) - 0.03) < 1e-6 and float(s_out[:-1, 0].max()) < 0.03


def test_bad_mode_rejected():
    mag, _ = _phantom()
    try:
        sample_positions(mag, 10, "uniform", 0.05)
    except ValueError:
        return
    raise AssertionError("init_mode 'uniform' must not reach the sampler")
