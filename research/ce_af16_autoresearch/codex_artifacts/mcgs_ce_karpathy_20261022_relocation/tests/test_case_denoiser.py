"""CPU contracts for independently fitted current-case magnitude targets."""

import hashlib
import json
import random

import numpy as np
import pytest
import torch

import case_denoiser as cd


def raw_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def parameter_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().numpy()
        digest.update(name.encode() + b"\0")
        digest.update(array.dtype.str.encode() + b"\0")
        digest.update(json.dumps(list(array.shape)).encode() + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def signal(shape=(9, 10, 11)):
    x, y, z = np.indices(shape, dtype=np.float32)
    return (0.2 + x / 20 + y / 35 + z / 50).astype(np.float32)


def denoiser(tmp_path, **kwargs):
    options = dict(device="cpu", width=4, patch_size=5, batch_size=2,
                   updates=8, tile_size=4)
    options.update(kwargs)
    return cd.CaseMagnitudeDenoiser(tmp_path, **options)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def test_initial_network_is_exact_identity_with_residual_zero_initialization():
    # A nonzero final convolution or wrong residual sign breaks identity.
    model = cd.MagnitudeNoiseCNN(width=4)
    x = torch.linspace(-1, 2, 8 * 9 * 10).reshape(1, 1, 8, 9, 10)
    assert torch.equal(model(x), x)
    convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv3d)]
    assert len(convs) == 5
    assert [(m.in_channels, m.out_channels) for m in convs] == [
        (1, 4), (4, 4), (4, 4), (4, 4), (4, 1)]
    assert sum(isinstance(m, torch.nn.ReLU) for m in model.modules()) == 4
    assert all(m.kernel_size == (3, 3, 3) and m.padding == (1, 1, 1)
               for m in convs)
    assert all(p.dtype == torch.float32 for p in model.parameters())


def test_no_grad_caller_still_fits_and_checkpoint_reproduces_saved_target(tmp_path):
    # Missing enable_grad, a stale checkpoint, or a fabricated training record fails.
    z = signal()
    before = z.copy()
    fit = denoiser(tmp_path)
    with torch.no_grad():
        out = fit(z, 0.06)
    rec = fit.last_stats
    assert rec is fit.records[0]
    assert rec["status"] == "trained" and rec["updates"] == 8
    assert rec["gradients"]["all_finite"] is True
    assert rec["gradients"]["any_nonzero"] is True
    assert rec["gradients"]["steps_with_nonzero"] == 8
    assert rec["gradients"]["max_abs"] > 0
    assert rec["changed_parameters"]
    assert rec["initial_parameter_sha256"] != rec["final_parameter_sha256"]
    assert all(np.isfinite(rec["losses"][key]) for key in ["first", "last"])
    assert rec["elapsed_seconds"] > 0
    assert rec["normalization"] == pytest.approx(float(np.quantile(z, 0.99)))
    assert rec["normalized_sigma"] == pytest.approx(0.06 / rec["normalization"])
    np.testing.assert_array_equal(z, before)
    assert out.dtype == np.float32 and out.flags.owndata
    assert np.isfinite(out).all() and (out >= 0).all()
    assert not np.shares_memory(out, z)
    assert not np.array_equal(out, z)
    saved = np.load(rec["target"]["path"], allow_pickle=False)
    np.testing.assert_array_equal(saved, out)
    state = load_checkpoint(rec["checkpoint"]["path"])
    assert state["schema"] == "case_magnitude_denoiser_checkpoint_v1"
    assert state["call_index"] == rec["call_index"] == 0
    assert state["seed"] == rec["seed"] == 42
    assert state["input"] == rec["input"]
    assert state["output"] == rec["output"]
    assert state["updates"] == rec["updates"]
    assert state["normalization"] == rec["normalization"]
    assert state["sigma"] == rec["sigma"]
    assert state["normalized_sigma"] == rec["normalized_sigma"]
    assert parameter_hash(state["model_state_dict"]) == rec["final_parameter_sha256"]
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        model = cd.MagnitudeNoiseCNN(width=4)
    assert parameter_hash(model.state_dict()) == rec["initial_parameter_sha256"]
    model.load_state_dict(state["model_state_dict"])
    with torch.no_grad():
        raw = model(torch.from_numpy(z / rec["normalization"])[None, None])[0, 0]
    expected = np.maximum(raw.numpy() * rec["normalization"], 0)
    np.testing.assert_allclose(out, expected, rtol=2e-6, atol=2e-7)
    assert rec["output"]["clipped_count"] == int((raw.numpy() < 0).sum())
    assert rec["input"] == dict(shape=[9, 10, 11], dtype="float32", sha256=raw_hash(z))
    assert rec["output"]["sha256"] == raw_hash(out)
    manifest = json.loads(open(rec["manifest"]["path"]).read())
    assert manifest == {k: v for k, v in rec.items() if k != "manifest"}
    assert set(rec) == {
        "schema", "call_index", "seed", "status", "input", "sigma",
        "normalization", "normalized_sigma", "updates", "patch_size",
        "batch_size", "width", "lr", "tile_size", "losses", "gradients", "changed_parameters",
        "initial_parameter_sha256", "final_parameter_sha256", "elapsed_seconds",
        "output", "checkpoint", "target", "manifest"}
    assert rec["schema"] == "case_magnitude_denoiser_call_v1"
    for key in ("checkpoint", "target", "manifest"):
        descriptor = rec[key]
        assert set(descriptor) == {"path", "sha256"}
        assert hashlib.sha256(open(descriptor["path"], "rb").read()).hexdigest() == descriptor["sha256"]
    out[:] = 100
    np.testing.assert_array_equal(np.load(rec["target"]["path"]), saved)


def test_each_call_starts_fresh_seed_and_replays_independently(tmp_path):
    # Warm starts or shared per-scan targets break fresh initialization/replay.
    fit = denoiser(tmp_path / "a", updates=3)
    a = fit(signal(), 0.05)
    b = fit(signal(), 0.05)
    first, second = fit.records
    assert [r["seed"] for r in fit.records] == [42, 1051]
    assert [r["call_index"] for r in fit.records] == [0, 1]
    assert first["initial_parameter_sha256"] != second["initial_parameter_sha256"]
    assert first["final_parameter_sha256"] != second["initial_parameter_sha256"]
    replay = denoiser(tmp_path / "b", updates=3, seed=1051)
    np.testing.assert_array_equal(b, replay(signal(), 0.05))
    assert not np.shares_memory(a, b)
    assert "call_0000" in first["manifest"]["path"]
    assert "call_0001" in second["manifest"]["path"]


def test_training_preserves_all_caller_cpu_rng_and_backend_flags(tmp_path):
    # Accidental global crop/noise/model seeding changes the caller's stream.
    random.seed(804)
    np.random.seed(805)
    torch.manual_seed(806)
    py_state, np_state, th_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    flags = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
             torch.are_deterministic_algorithms_enabled())
    denoiser(tmp_path, updates=2)(signal(), 0.05)
    assert random.getstate() == py_state
    now = np.random.get_state()
    assert now[0] == np_state[0] and np.array_equal(now[1], np_state[1]) and now[2:] == np_state[2:]
    assert torch.equal(torch.get_rng_state(), th_state)
    assert flags == (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                     torch.are_deterministic_algorithms_enabled())


@pytest.mark.parametrize("shape,tile", [((13, 15, 17), 4), ((7, 9, 11), 32), ((12, 10, 9), 6)])
def test_tiling_matches_nonidentity_full_network_at_interiors_and_boundaries(shape, tile):
    # Missing halo or artificial padding at global boundaries changes these values.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(18)
        model = cd.MagnitudeNoiseCNN(width=3)
        with torch.no_grad():
            for p in model.parameters():
                p.uniform_(-0.12, 0.16)
    x = torch.from_numpy(signal(shape))[None, None]
    with torch.no_grad():
        full = model(x)
        tiled = cd.tiled_inference(model, x, tile_size=tile)
    assert not torch.equal(full, x)
    torch.testing.assert_close(tiled, full, rtol=2e-6, atol=2e-6)
    assert not tiled.requires_grad


@pytest.mark.parametrize("zero_input,sigma,status", [
    (False, 0.0, "identity_zero_sigma"), (True, 0.05, "identity_zero_input")])
def test_identity_is_independent_and_does_not_train_or_advance_rng(tmp_path, zero_input, sigma, status):
    # No-op paths must neither initialize a model nor invent a checkpoint.
    fit = denoiser(tmp_path)
    z = np.zeros((9, 10, 11), dtype=np.float32) if zero_input else signal()
    state = torch.get_rng_state().clone()
    out = fit(z, sigma)
    np.testing.assert_array_equal(out, z)
    assert not np.shares_memory(out, z)
    assert torch.equal(torch.get_rng_state(), state)
    r = fit.last_stats
    assert r["status"] == status and r["updates"] == 0
    assert r["checkpoint"] is None and r["losses"] is None and r["gradients"] is None
    assert r["initial_parameter_sha256"] is None and r["final_parameter_sha256"] is None
    assert r["changed_parameters"] == []
    assert list((tmp_path / "call_0000").glob("*.pt")) == []
    np.testing.assert_array_equal(np.load(r["target"]["path"]), z)


def test_percentile_floor_and_noncontiguous_input_are_safe(tmp_path):
    z = np.full((7, 8, 9), 1e-8, dtype=np.float64)[:, ::-1, :]
    saved = z.tobytes()
    fit = denoiser(tmp_path, updates=1)
    out = fit(z, 1e-9)
    assert fit.last_stats["normalization"] == 1e-6
    assert fit.last_stats["input"]["dtype"] == "float64"
    assert fit.last_stats["input"]["sha256"] == hashlib.sha256(saved).hexdigest()
    assert z.tobytes() == saved
    assert out.dtype == np.float32


@pytest.mark.parametrize("option", [
    {"updates": 0}, {"updates": -1}, {"updates": True}, {"updates": 1.2},
    {"patch_size": 0}, {"batch_size": 0}, {"width": 0}, {"tile_size": 0},
    {"lr": 0}, {"lr": float("nan")}, {"lr": float("inf")},
    {"seed": -1}, {"seed": True}, {"device": "meta"}, {"device": "nonsense"}])
def test_invalid_settings_rejected_before_files_are_created(tmp_path, option):
    destination = tmp_path / "never_created"
    with pytest.raises((ValueError, TypeError)):
        denoiser(destination, **option)
    assert not destination.exists()


@pytest.mark.parametrize("z,sigma", [
    (np.ones((5, 5)), 0.1), (np.ones((4, 7, 8)), 0.1),
    (np.full((6, 7, 8), np.nan), 0.1), (np.full((6, 7, 8), np.inf), 0.1),
    (-np.ones((6, 7, 8)), 0.1), (np.ones((6, 7, 8), dtype=complex), 0.1),
    (np.ones((6, 7, 8), dtype=bool), 0.1), (np.ones((6, 7, 8)), -0.1),
    (np.ones((6, 7, 8)), float("nan")), (np.ones((6, 7, 8)), float("inf")),
    (np.full((6, 7, 8), 1e100), 0.1), ([[[1]]], 0.1)])
def test_invalid_call_has_no_artifacts_records_or_rng_mutation(tmp_path, z, sigma):
    destination = tmp_path / "untouched"
    fit = denoiser(destination)
    state = torch.get_rng_state().clone()
    with pytest.raises((ValueError, TypeError)):
        fit(z, sigma)
    assert not destination.exists()
    assert fit.records == [] and fit.last_stats is None
    assert torch.equal(torch.get_rng_state(), state)


def test_existing_call_directory_is_refused_without_overwrite_or_state_change(tmp_path):
    call = tmp_path / "call_0000"
    call.mkdir()
    marker = call / "manifest.json"
    marker.write_text("preserved")
    fit = denoiser(tmp_path)
    state = torch.get_rng_state().clone()
    with pytest.raises(FileExistsError):
        fit(signal(), 0.05)
    assert marker.read_text() == "preserved"
    assert torch.equal(torch.get_rng_state(), state)
    assert fit.records == [] and fit.last_stats is None


def test_noop_float64_is_exact_independent_canonical_float32_copy(tmp_path):
    # Returning original float64 or its storage violates the canonical interface.
    z = np.full((6, 7, 8), 0.123456789, dtype=np.float64)
    original = z.copy()
    fit = denoiser(tmp_path)
    out = fit(z, 0)
    np.testing.assert_array_equal(out, z.astype(np.float32))
    np.testing.assert_array_equal(z, original)
    assert out.dtype == np.float32 and not np.shares_memory(out, z)
    assert fit.last_stats["input"]["dtype"] == "float64"
    assert fit.last_stats["input"]["sha256"] == raw_hash(z)


def test_outer_autocast_does_not_change_float32_fitting_or_inference(tmp_path):
    # An inherited AMP context must not alter weights or detached prior targets.
    fit = denoiser(tmp_path / "ordinary", updates=2)
    expected = fit(signal(), 0.05)
    other = denoiser(tmp_path / "autocast", updates=2)
    with torch.autocast("cpu", dtype=torch.bfloat16), torch.no_grad():
        actual = other(signal(), 0.05)
    np.testing.assert_array_equal(actual, expected)
    assert other.last_stats["final_parameter_sha256"] == fit.last_stats["final_parameter_sha256"]


def test_failed_fit_restores_rng_preserves_partial_directory_and_no_success_record(tmp_path):
    # A nonfinite loss must not become a successful checkpoint or consume caller RNG.
    fit = denoiser(tmp_path, updates=2)
    fit(signal(), 0)
    previous = fit.last_stats
    state = torch.get_rng_state().clone()
    with pytest.raises(FloatingPointError, match="loss"):
        fit(signal(), 1e30)
    assert torch.equal(torch.get_rng_state(), state)
    assert fit.records == [previous] and fit.last_stats is previous
    assert (tmp_path / "call_0001").is_dir()
    assert not (tmp_path / "call_0001" / "manifest.json").exists()
    with pytest.raises(FileExistsError):
        fit(signal(), 0.05)


def test_clipping_counts_actual_negative_inferences_and_target_stays_nonnegative(tmp_path):
    # Omitting the clamp or reporting zero unconditionally breaks this actual fit.
    z = signal()
    z[:3] = 0
    fit = denoiser(tmp_path, updates=8)
    out = fit(z, 0.2)
    assert fit.last_stats["output"]["clipped_count"] > 0
    assert int((out == 0).sum()) >= fit.last_stats["output"]["clipped_count"]
    assert (out >= 0).all()


def test_record_and_checkpoint_preserve_actual_tiling_setting(tmp_path):
    # Missing inference settings make exact checkpoint/target linkage ambiguous.
    fit = denoiser(tmp_path, tile_size=3, updates=1)
    fit(signal(), 0.05)
    assert fit.last_stats["tile_size"] == 3
    checkpoint = load_checkpoint(fit.last_stats["checkpoint"]["path"])
    assert checkpoint["tile_size"] == 3


def test_model_initialization_stays_on_cpu_under_ambient_meta_default():
    # Ambient device selection must not redirect the CPU-seeded initialization.
    original_device = torch.get_default_device()
    try:
        torch.set_default_device("meta")
        with torch.random.fork_rng(devices=[]):
            model = cd.MagnitudeNoiseCNN(width=1)
        assert all(p.device.type == "cpu" for p in model.parameters())
        assert all(p.dtype == torch.float32 for p in model.parameters())
        assert torch.get_default_device() == torch.device("meta")
    finally:
        torch.set_default_device(original_device)


def test_cpu_fit_ignores_ambient_meta_default_and_preserves_default_and_rng(tmp_path):
    # The reviewed 2-cubed CPU call must succeed without touching ambient state.
    z = np.ones((2, 2, 2), dtype=np.float32)
    options = dict(device="cpu", width=1, patch_size=2, batch_size=1,
                   updates=1, tile_size=2)
    expected_fit = cd.CaseMagnitudeDenoiser(tmp_path / "ordinary", **options)
    expected = expected_fit(z, 0.05)
    fit = cd.CaseMagnitudeDenoiser(tmp_path / "ambient_meta", **options)
    original_device = torch.get_default_device()
    state = torch.get_rng_state().clone()
    try:
        torch.set_default_device("meta")
        actual = fit(z, 0.05)
        assert torch.get_default_device() == torch.device("meta")
        assert torch.equal(torch.get_rng_state(), state)
    finally:
        torch.set_default_device(original_device)
    np.testing.assert_array_equal(actual, expected)
    assert fit.last_stats["status"] == "trained" and fit.last_stats["updates"] == 1
    assert fit.last_stats["initial_parameter_sha256"] == expected_fit.last_stats["initial_parameter_sha256"]
    assert fit.last_stats["final_parameter_sha256"] == expected_fit.last_stats["final_parameter_sha256"]


def test_failed_fit_preserves_ambient_meta_default_and_rng(tmp_path):
    # Exceptional exit must unwind both the CPU allocation and RNG scopes.
    fit = denoiser(tmp_path, width=1, patch_size=2, batch_size=1,
                   updates=1, tile_size=2)
    original_device = torch.get_default_device()
    state = torch.get_rng_state().clone()
    try:
        torch.set_default_device("meta")
        with pytest.raises(FloatingPointError, match="loss"):
            fit(np.ones((2, 2, 2), dtype=np.float32), 1e30)
        assert torch.get_default_device() == torch.device("meta")
        assert torch.equal(torch.get_rng_state(), state)
    finally:
        torch.set_default_device(original_device)
    assert fit.records == [] and fit.last_stats is None
    assert (tmp_path / "call_0000").is_dir()
    assert not (tmp_path / "call_0000" / "manifest.json").exists()
