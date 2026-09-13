"""Region-restricted rescoring (uih_mtp.rescore) against the frozen gauge."""
import numpy as np
import pytest
import torch


def _phantom(shape=(48, 40, 36), seed=0):
    """Two tissue levels inside a blob plus noise: brain and white-matter masks are both non-empty."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(np.float64)
    c = [(s - 1) / 2 for s in shape]
    r = np.sqrt(((zz - c[0]) / (shape[0] * 0.42)) ** 2 + ((yy - c[1]) / (shape[1] * 0.42)) ** 2
                + ((xx - c[2]) / (shape[2] * 0.42)) ** 2)
    gt = np.where(r < 0.55, 1.0, np.where(r < 1.0, 0.62, 0.0))
    gt = gt + 0.01 * rng.standard_normal(shape)
    gt = np.clip(gt, 0, None).astype(np.float32)
    rec = np.clip(gt + 0.03 * rng.standard_normal(shape), 0, None).astype(np.float32)
    return rec, gt


def test_full_region_reproduces_the_frozen_head_gauge():
    import sys
    sys.path.insert(0, "/data0/congcong/code/GS/mcgs_peds_wt")
    from diag_artefact.headonly import score_headonly
    from uih_mtp.rescore import score_region
    rec, gt = _phantom()
    got = score_region(rec, gt)
    ref = score_headonly(rec, gt)
    for k in ("head_psnr", "head_ssim", "head_hfen", "head_frac", "gmax"):
        assert got[k] == pytest.approx(ref[k], rel=1e-6, abs=1e-9), k
    if np.isnan(ref["grain_ratio"]):
        assert np.isnan(got["grain_ratio"])
    else:
        assert got["grain_ratio"] == pytest.approx(ref["grain_ratio"], rel=1e-6)


def test_full_region_reproduces_the_frozen_whole_volume_metrics():
    import sys
    sys.path.insert(0, "/data0/congcong/code/GS/mcgs_peds_wt")
    sys.path.insert(0, "/data0/congcong/code/GS/mcgs_peds_wt/paired_gs")
    from metrics import evaluate_reconstruction
    from uih_mtp.rescore import score_region
    rec, gt = _phantom()
    got = score_region(rec, gt)
    ref = evaluate_reconstruction(torch.from_numpy(rec), torch.from_numpy(gt), compute_extended=False)
    assert got["psnr"] == pytest.approx(float(ref["psnr"]), rel=1e-6)
    assert got["nmse"] == pytest.approx(float(ref["nmse"]), rel=1e-6)
    assert got["ssim"] == pytest.approx(float(ref["ssim"]), rel=1e-6)


def test_restricted_region_equals_a_direct_computation():
    from uih_mtp.rescore import partition_region, score_region
    from diag_artefact.headonly import head_mask
    rec, gt = _phantom()
    region = partition_region(gt.shape, 2, 6, 30)
    got = score_region(rec, gt, region)
    head = head_mask(gt) & region
    gmax = float(gt.max())
    mse = float(np.mean((rec[head].astype(np.float64) - gt[head].astype(np.float64)) ** 2))
    assert got["head_psnr"] == pytest.approx(20 * np.log10(gmax) - 10 * np.log10(mse), rel=1e-9)
    assert got["head_frac"] == pytest.approx(head.sum() / region.sum(), rel=1e-9)
    num = float(np.sum((rec[region].astype(np.float64) - gt[region].astype(np.float64)) ** 2))
    den = float(np.sum(gt[region].astype(np.float64) ** 2))
    assert got["nmse"] == pytest.approx(num / den, rel=1e-9)
    assert got["n_voxels"] == int(region.sum())


def test_partition_region_selects_the_half_open_band():
    from uih_mtp.rescore import partition_region
    r = partition_region((4, 5, 10), 2, 2, 5)
    assert r.shape == (4, 5, 10) and r.sum() == 4 * 5 * 3
    assert r[..., 2].all() and r[..., 4].all() and not r[..., 5].any() and not r[..., 1].any()


def test_score_region_accepts_complex_input():
    from uih_mtp.rescore import score_region
    rec, gt = _phantom()
    a = score_region(rec, gt)
    b = score_region((rec * np.exp(1j * 0.3)).astype(np.complex64), gt.astype(np.complex64))
    assert b["head_psnr"] == pytest.approx(a["head_psnr"], rel=1e-6)
