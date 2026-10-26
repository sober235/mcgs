"""CPU acceptance checks for the scan-internal patch low-rank kernel."""
from importlib import util
from pathlib import Path

import numpy as np
import pytest
import torch


SOURCE = Path(__file__).resolve().parents[1] / "trainer/paired_gs/losses/nonlocal_lowrank.py"


def kernel_module():
    if not SOURCE.exists():
        pytest.fail("PatchLowRankDenoiser has not been implemented")
    spec = util.spec_from_file_location("nonlocal_lowrank_under_test", SOURCE)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def denoiser(**kwargs):
    params = dict(patch_size=5, search_radius=2, search_stride=2,
                  anchor_stride=3, group_size=8, min_group_size=4,
                  device="cpu", batch_size=32)
    params.update(kwargs)
    return kernel_module().PatchLowRankDenoiser(**params)


def fixture(seed=12, shape=(16, 17, 18)):
    return np.random.default_rng(seed).uniform(0.5, 1.5, shape).astype(np.float32)


def test_zero_noise_and_constant_are_exact_identity_without_input_mutation():
    x = fixture()
    original = x.copy()
    d = denoiser()
    result = d(x, 0.0)
    np.testing.assert_array_equal(result, x)
    np.testing.assert_array_equal(x, original)
    assert result is not x
    assert d.last_stats["reason"] == "zero_sigma"
    constant = np.full((16, 16, 16), 1.25, np.float32)
    np.testing.assert_array_equal(d(constant, 0.1), constant)
    assert d.last_stats["reason"] == "constant_crop"


def test_finite_nonnegative_same_shape_and_real_target_change():
    x = fixture()
    original = x.copy()
    d = denoiser(distance_factor=1e5)
    result = d(x, 0.1)
    assert result.shape == x.shape and result.dtype == np.float32
    assert np.isfinite(result).all() and result.min() >= 0
    np.testing.assert_array_equal(x, original)
    assert d.last_stats["valid_group_count"] > 0
    assert d.last_stats["target_rms_difference"] > 0
    assert d.last_stats["target_relative_norm"] > 0
    assert d.last_stats["elapsed_seconds"] > 0
    assert d.last_stats["device"] == "cpu"
    assert "cuda_allocated_before" in d.last_stats["memory"]


def test_positive_scale_equivariance():
    x = fixture()
    d = denoiser(distance_factor=1e5)
    np.testing.assert_allclose(d(x * 3.0, 0.3), 3.0 * d(x, 0.1), rtol=2e-5, atol=3e-6)


def test_terminal_anchors_cover_boundaries_and_undersized_groups_stay_exact():
    x = fixture(shape=(16, 17, 18))
    d = denoiser(search_radius=0)
    result = d(x, 0.1)
    np.testing.assert_array_equal(result, x)
    assert d.last_stats["covered_voxels"] == x.size
    assert d.last_stats["uncovered_voxels"] == 0
    assert d.last_stats["valid_group_count"] == 0
    assert d.last_stats["fallback_group_count"] == d.last_stats["anchor_count"]
    assert d.last_stats["group_size_histogram"] == {1: d.last_stats["anchor_count"]}


def independent_matches(x, anchor, radius, stride, k, sigma, factor, p):
    shape = np.asarray(x.shape) - p + 1
    a = np.asarray(anchor)
    first = x[tuple(slice(v, v + p) for v in a)].astype(np.float64)
    first -= first.mean()
    candidates = [(0.0, int(np.ravel_multi_index(tuple(a), shape)))]
    for offset in np.ndindex(*([len(range(-radius, radius + 1, stride))] * 3)):
        delta = np.asarray([list(range(-radius, radius + 1, stride))[j] for j in offset])
        if not delta.any():
            continue
        q = a + delta
        if (q < 0).any() or (q >= shape).any():
            continue
        patch = x[tuple(slice(v, v + p) for v in q)].astype(np.float64)
        patch -= patch.mean()
        distance = np.mean((first - patch) ** 2)
        if distance <= factor * sigma ** 2:
            candidates.append((distance, int(np.ravel_multi_index(tuple(q), shape))))
    return sorted(candidates, key=lambda pair: pair[0])[:k]


def test_streamed_grouping_matches_independent_patch_distances_and_is_deterministic():
    x = fixture(shape=(16, 16, 16))
    d = denoiser(distance_factor=1e5)
    a, ids, distances = d._match_groups(torch.from_numpy(x), 0.1)
    again = d._match_groups(torch.from_numpy(x), 0.1)
    assert torch.equal(ids, again[1])
    assert torch.equal(distances, again[2])
    for row in (0, len(a) // 2, len(a) - 1):
        expected = independent_matches(x, a[row].tolist(), 2, 2, 8, 0.1, 1e5, 5)
        valid = torch.isfinite(distances[row])
        assert ids[row][valid].tolist() == [pair[1] for pair in expected]
        np.testing.assert_allclose(distances[row][valid].numpy(), [pair[0] for pair in expected], rtol=2e-5, atol=1e-7)
        assert len(set(ids[row][valid].tolist())) == int(valid.sum())
    # All equal matches retain the self match once, ahead of equal-distance neighbors.
    a, ids, distances = d._match_groups(torch.ones((16, 16, 16)), 0.1)
    assert ids[0, 0].item() == 0
    assert len(set(ids[0].tolist())) == 8
    assert torch.equal(distances, torch.zeros_like(distances))


def test_distance_gate_leaves_unique_texture_unchanged():
    x = fixture()
    d = denoiser(distance_factor=0.0)
    np.testing.assert_array_equal(d(x, 0.1), x)
    assert d.last_stats["fallback_group_count"] == d.last_stats["anchor_count"]


def test_batch_size_does_not_change_groups_or_result():
    x = fixture(shape=(16, 16, 16))
    small = denoiser(batch_size=3, distance_factor=1e5)
    full = denoiser(batch_size=256, distance_factor=1e5)
    np.testing.assert_allclose(small(x, 0.1), full(x, 0.1), rtol=3e-6, atol=3e-7)
    assert small.last_stats["group_size_histogram"] == full.last_stats["group_size_histogram"]
    assert small.last_stats["max_materialized_groups"] <= 3
    assert full.last_stats["max_materialized_groups"] <= 256


def test_gram_shrink_matches_independent_numpy_svd_and_preserves_means():
    d = denoiser()
    rng = np.random.default_rng(4)
    group = rng.normal(size=(3, 125, 7)).astype(np.float64) + np.arange(7)[None, None, :]
    means = group.mean(axis=1, keepdims=True)
    centered = group - means
    shared = centered.mean(axis=2, keepdims=True)
    residual = centered - shared
    expected = []
    threshold = 0.13 * (np.sqrt(125) + np.sqrt(7))
    for b in range(3):
        u, s, vh = np.linalg.svd(residual[b], full_matrices=False)
        expected.append((u * np.maximum(s - threshold, 0)) @ vh + shared[b] + means[b])
    actual = d._shrink_groups(torch.from_numpy(group), 0.13).numpy()
    np.testing.assert_allclose(actual, np.asarray(expected), rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(actual.mean(axis=1), means[:, 0], atol=1e-12)
    np.testing.assert_allclose((actual - means).mean(axis=2), shared[:, :, 0], atol=1e-12)


def test_repeat_pattern_noise_fixture_improves_mse_without_guidance():
    xyz = np.indices((16, 16, 16))
    clean = (1.5 + 0.2 * np.sin(np.pi * xyz[0] / 2)
             + 0.15 * np.cos(np.pi * xyz[1] / 2)
             + 0.1 * np.sin(np.pi * xyz[2] / 2)).astype(np.float32)
    sigma = 0.06
    noisy = clean + np.random.default_rng(21).normal(0, sigma, clean.shape).astype(np.float32)
    d = denoiser(search_radius=4, group_size=16)
    actual = d(noisy, sigma)
    assert np.mean((actual - clean) ** 2) < 0.8 * np.mean((noisy - clean) ** 2)
    assert d.last_stats["valid_group_count"] > 0


@pytest.mark.parametrize("x,sigma", [
    (np.zeros((16, 16), np.float32), 0.1),
    (np.zeros((0, 16, 16), np.float32), 0.1),
    (np.zeros((16, 16, 16), np.float64), 0.1),
    (np.full((16, 16, 16), np.nan, np.float32), 0.1),
    (np.full((16, 16, 16), -0.1, np.float32), 0.1),
    (np.zeros((16, 16, 16), np.float32), -0.1),
    (np.zeros((16, 16, 16), np.float32), np.inf),
])
def test_invalid_inputs_are_rejected(x, sigma):
    with pytest.raises((ValueError, TypeError)):
        denoiser()(x, sigma)


def test_small_crop_returns_owned_identity_and_reason():
    x = fixture(shape=(3, 16, 16))
    d = denoiser()
    result = d(x, 0.1)
    np.testing.assert_array_equal(result, x)
    assert result is not x
    assert d.last_stats["reason"] == "no_valid_patches"


def test_import_constructor_and_identity_do_not_initialize_torch():
    import subprocess
    import sys
    code = f'''import importlib.util, sys, numpy as np
spec = importlib.util.spec_from_file_location("isolated_lowrank", {str(SOURCE)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
d = module.PatchLowRankDenoiser(device="cuda:0")
d(np.ones((16, 16, 16), dtype=np.float32), 0.0)
assert "torch" not in sys.modules
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_numerical_failure_propagates_and_records_error(monkeypatch):
    d = denoiser(distance_factor=1e5)

    def fail_decomposition(*args, **kwargs):
        raise RuntimeError("decomposition failure injected for error-path check")

    monkeypatch.setattr(torch.linalg, "eigh", fail_decomposition)
    with pytest.raises(RuntimeError, match="decomposition failure"):
        d(fixture(), 0.1)
    assert d.last_stats["reason"] == "error"
    assert "decomposition failure" in d.last_stats["error"]
