import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from common import SRC

if not (SRC / "paired_gs/trainers/joint_trainer.py").exists():
    pytest.skip("frozen trainer source not on this machine", allow_module_level=True)
sys.path.insert(0, str(SRC / "paired_gs"))

import patch                                              # noqa: E402
from adapter import MotionAdapter                         # noqa: E402
from helpers import fft3c_np, small_src                   # noqa: E402


def _masks(K=4, with_z=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    shape = (6, 16, 12)
    pe = (torch.rand(shape[1], shape[2], generator=g) < 0.4).float()
    mask = pe.unsqueeze(0).expand(shape).clone()
    z = torch.zeros(shape)
    if with_z:
        z[:, :2, :] = 1.0                                  # known-zero band, never sampled
        mask[:, :2, :] = 0.0
    mask_fit = (mask + z).clamp(max=1.0)
    seg2d = (torch.arange(shape[2]) * K // shape[2]).unsqueeze(0).expand(shape[1], shape[2]).clone()
    seg2d[:2, :] = -1                                      # outside the shutter: no segment
    return mask, mask_fit, seg2d


def test_segment_masks_partition_and_reference_holds_known_zero():
    mask, mask_fit, seg2d = _masks()
    masks = patch.segment_masks(mask_fit, seg2d, ref=1, K=4)
    assert len(masks) == 4 and torch.equal(sum(masks), mask_fit)
    z = (mask_fit - mask).clamp_min(0)
    assert bool((masks[1] >= z).all())                     # Z and unassigned points stay with the reference term
    for s in (0, 2, 3):
        assert torch.equal(masks[s], mask * (seg2d == s).unsqueeze(0))


def test_segment_loss_matches_full_when_static():
    mask, mask_fit, seg2d = _masks()
    masks = patch.segment_masks(mask_fit, seg2d, ref=1, K=4)
    g = torch.Generator().manual_seed(1)
    resid = torch.randn(mask_fit.shape, generator=g)
    loss_fn = lambda vol, m: (vol * resid * m).pow(2).sum() / 100.0
    vol = torch.ones(mask_fit.shape)
    full = loss_fn(vol, mask_fit)
    all_segments = patch.segmented_dc(loss_fn, lambda s: vol, vol, masks, ref=1, n_extra=3,
                                      gen=torch.Generator().manual_seed(0))
    assert all_segments.item() == pytest.approx(full.item(), rel=1e-5)
    est = torch.stack([patch.segmented_dc(loss_fn, lambda s: vol, vol, masks, ref=1, n_extra=1,
                                          gen=torch.Generator().manual_seed(i)) for i in range(900)]).mean()
    assert est.item() == pytest.approx(full.item(), rel=0.03)


def test_segmented_dc_renders_only_the_sampled_segments():
    mask, mask_fit, seg2d = _masks()
    masks = patch.segment_masks(mask_fit, seg2d, ref=1, K=4)
    seen = []
    vol = torch.ones(mask_fit.shape)
    patch.segmented_dc(lambda v, m: (v * m).sum(), lambda s: seen.append(s) or vol, vol, masks, ref=1, n_extra=2,
                       gen=torch.Generator().manual_seed(0))
    assert len(seen) == 2 and 1 not in seen and len(set(seen)) == 2


def test_param_groups_appended_last():
    ad = MotionAdapter((32, 32, 24), True, True, 4, 1)
    w = torch.nn.Parameter(torch.zeros(3))
    lrs = dict(rigid_theta=1e-3, rigid_t=2e-2, nonrigid=5e-3, seg_theta=5e-4, seg_t=1e-2)
    groups = patch.append_groups([{"params": [w], "lr": 1.0}], ad, lrs)
    assert [g.get("name") for g in groups] == [None, "rigid_theta", "rigid_t", "nonrigid", "seg_theta", "seg_t"]
    opt = torch.optim.Adam(groups)                          # extra key "name" is accepted
    assert len(opt.param_groups) == 6
    for p in ad.parameters():
        p.requires_grad_(False)
    assert len(patch.append_groups([{"params": [w], "lr": 1.0}], ad, lrs)) == 1      # oracle: nothing to optimise


def test_stage_gating():
    ad = MotionAdapter((32, 32, 24), True, True, 4, 1)
    patch.apply_stage(ad, it=0, stage_iters=(500, 1500))
    assert ad.theta.requires_grad and ad.t_mm.requires_grad and not ad.ctrl.requires_grad and not ad.seg_theta.requires_grad
    patch.apply_stage(ad, it=499, stage_iters=(500, 1500))
    assert not ad.ctrl.requires_grad
    patch.apply_stage(ad, it=500, stage_iters=(500, 1500))
    assert ad.ctrl.requires_grad and ad.seg_theta.requires_grad and ad.seg_t_mm.requires_grad
    seg_only = MotionAdapter((32, 32, 24), False, False, 4, 1)
    patch.apply_stage(seg_only, it=0, stage_iters=(500, 1500))
    assert seg_only.seg_theta.requires_grad                 # nothing rigid to wait for


def test_guide_choice():
    vol_p, vol_c = torch.zeros(2), torch.ones(2)
    last = {"plain": vol_p, "contrast": vol_c}
    assert patch.guide_frame(vol_c, vol_p, last) == "plain_in_contrast_frame"
    assert patch.guide_frame(vol_p, vol_c, last) == "contrast_in_canonical_frame"
    assert patch.guide_frame(vol_c, torch.zeros(2), last) is None          # a file guide: left alone
    assert patch.guide_frame(vol_c, None, last) is None


def _write(path, key, arr):
    import h5py
    with h5py.File(path, "w") as h:
        h.create_dataset(key, data=arr)


def test_intra_dataset_uses_reference_gt(tmp_path):
    from trainers.joint_trainer import _load_dataset
    src = small_src(shape=(32, 32, 24), coils=4)
    rng = np.random.default_rng(0)
    corrupt = (src["ksp_T2"] * np.exp(1j * rng.uniform(-0.3, 0.3, size=(1, 1, 32, 24)))).astype(np.complex64)
    pe = rng.random((32, 24)) < 0.3
    pe[12:20, 8:16] = True
    mask = np.broadcast_to(pe[None], (32, 32, 24)).astype(np.uint8)
    _write(tmp_path / "gt.h5", "kspace", src["ksp_T2"])
    _write(tmp_path / "corrupt.h5", "kspace", corrupt)
    _write(tmp_path / "csm.h5", "map", src["csm_T2"])
    _write(tmp_path / "mask.h5", "mask", mask)
    cfg = dict(data_path=str(tmp_path / "gt.h5"), csm_path=str(tmp_path / "csm.h5"), mask_path=str(tmp_path / "mask.h5"),
               mask_type="external", acceleration_factor=4, phase_axes=[1, 2], readout_axis=0, presampled=False, normalize_kspace=True,
               normalization_percentile=99.9, skip_readout_check=False)
    gt_only = _load_dataset(cfg)
    ds = patch.load_with_reference_gt(_load_dataset, dict(cfg, motion_ksp_path=str(tmp_path / "corrupt.h5")))
    assert torch.equal(ds.ground_truth_image, gt_only.ground_truth_image)
    assert torch.equal(ds.kspace_multicoil, gt_only.kspace_multicoil)
    assert ds.normalization_scale == gt_only.normalization_scale
    exp = torch.from_numpy(corrupt) / gt_only.normalization_scale * ds.mask.unsqueeze(0)
    torch.testing.assert_close(ds.kspace_undersampled_complex, exp.to(torch.complex64))
    assert not torch.equal(ds.kspace_undersampled_complex, gt_only.kspace_undersampled_complex)
    assert not torch.equal(ds.zero_filled_complex, gt_only.zero_filled_complex)
    plain = patch.load_with_reference_gt(_load_dataset, cfg)               # no marker: untouched
    assert torch.equal(plain.kspace_undersampled_complex, gt_only.kspace_undersampled_complex)


def test_motion_config_validation():
    ok = dict(arm="A2", channel="contrast", rigid=True, nonrigid=True, segments=0, n_extra=2, schedule_path=None,
              truth_path=None, disp_path=None, native_shift_mm=[0.59, 0.12, -0.13], oracle=False,
              lrs=dict(rigid_theta=1e-3, rigid_t=2e-2, nonrigid=5e-3, seg_theta=5e-4, seg_t=1e-2),
              reg=dict(ctrl_l2=1e-3, ctrl_smooth=1e-2, seg_l2=1e-3, seg_smooth=1e-2), stage_iters=[500, 1500],
              out_dir="/tmp/x", exp="e", seed=42)
    patch.validate(ok)
    with pytest.raises(AssertionError):
        patch.validate(dict(ok, segments=8))                # segments need a schedule
    with pytest.raises(AssertionError):
        patch.validate(dict(ok, oracle=True))               # the oracle needs the truth
    with pytest.raises(AssertionError):
        patch.validate(dict(ok, channel="flair"))


def test_append_groups_passes_the_adam_eps():
    ad = MotionAdapter((32, 32, 24), True, False, 0, 0)
    lrs = dict(rigid_theta=1e-3, rigid_t=2e-2)
    assert all(g["eps"] == 1e-12 for g in patch.append_groups([], ad, lrs, eps=1e-12))
    assert all("eps" not in g for g in patch.append_groups([], ad, lrs))


def test_grad_stats_report_size_and_sign_consistency_per_window():
    gs = patch.GradStats()
    for sign in (1.0, 1.0, -1.0, 1.0):
        gs.add("rigid_t", torch.tensor([2e-10 * sign, 0.0, 4e-10]))
    gs.add("nonrigid", torch.full((5, 5, 5, 3), -1e-12))
    out = gs.flush()
    t = out["rigid_t"]
    assert t["n"] == 4 and t["abs_mean"] == pytest.approx(2e-10, rel=1e-6)          # mean |g| over elements, steps
    assert t["mean"] == pytest.approx([1e-10, 0.0, 4e-10], rel=1e-6)                # mean over the window
    assert t["consistency"] == pytest.approx((1e-10 + 4e-10) / 6e-10, rel=1e-6)     # |mean| / mean |g|, 1 = one sign
    assert "mean" not in out["nonrigid"] and out["nonrigid"]["consistency"] == pytest.approx(1.0)
    assert gs.flush() == {}                                                          # the window starts over


def test_grad_stats_hooks_survive_the_stage_switch():
    """Stage 1 freezes the non-rigid grid and the segment poses before the hooks exist (r2: three runs died with
    'cannot register a hook on a tensor that doesn't require gradient')."""
    ad = MotionAdapter((32, 32, 24), True, True, 2, 0)
    patch.apply_stage(ad, 0, [5, 10])
    gs = patch.GradStats()
    patch.attach_grad_stats(ad, gs)
    assert not ad.ctrl.requires_grad and ad.t_mm.requires_grad            # the stage is left as it was
    loss = lambda: 2.0 * ad.t_mm.sum() + 3.0 * ad.ctrl.sum() + ad.seg_t_mm.sum()
    loss().backward()
    patch.apply_stage(ad, 5, [5, 10])
    loss().backward()
    out = gs.flush()
    assert out["rigid_t"]["n"] == 2 and out["rigid_t"]["abs_mean"] == pytest.approx(2.0)
    assert out["nonrigid"]["n"] == 1 and out["nonrigid"]["abs_mean"] == pytest.approx(3.0)
    assert out["seg_t"]["n"] == 1 and "rigid_theta" not in out
