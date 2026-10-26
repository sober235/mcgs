from pathlib import Path

import pytest
import torch

from runner import SolverConfig, optimize_density, reserve_run, validate_audit, write_solver_checkpoint


class ExplicitPhysics:
    def __init__(self):
        torch.manual_seed(9)
        self.initial_density = torch.randn(2, 27, dtype=torch.complex128)
        self.target = torch.randn_like(self.initial_density)
        self.weights = torch.linspace(.5, 2, 27)
        self.counters = {"render_pairs": 0, "renderer_calls": 0}

    def render(self, rho):
        self.counters["render_pairs"] += 1
        self.counters["renderer_calls"] += 2
        return rho[0].reshape(3, 3, 3), rho[1].reshape(3, 3, 3)

    def structural_support(self):
        p, c = self.render(torch.ones_like(self.initial_density))
        return p.real.bool(), c.real.bool()

    def data_loss(self, pre, post, zero_target=False):
        value = torch.stack((pre, post)).reshape(2, 27)
        target = 0 if zero_target else self.target
        residual = (value - target) * self.weights
        return residual.abs().square().sum()


PRIOR = dict(lambda_tv=0., lambda_phase=0., lambda_wavelet=0., kappa=1.)


def test_real_complex_spd_step_decreases_actual_merit_and_preserves_inputs():
    operator = ExplicitPhysics()
    original = operator.initial_density.clone()
    result = optimize_density(operator, prior_kwargs=PRIOR, edge_weight=0.,
                              config=SolverConfig(outer_steps=1, cg_steps=32, cg_rtol=1e-12, damping_rel=1e-8))
    assert result.status == "complete"
    assert result.accepted_steps == 1
    assert result.trace[0]["merit_after"] < result.trace[0]["merit_before"] * 1e-10
    torch.testing.assert_close(result.density, operator.target, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(operator.initial_density, original, rtol=0, atol=0)
    assert not result.density.requires_grad
    counts = result.work
    assert counts["render_pairs"] == operator.counters["render_pairs"]
    assert counts["vjp_pairs"] == counts["normal_evaluations"] + counts["gradient_evaluations"]
    assert counts["normal_evaluations"] == result.trace[0]["cg_matvecs"] + 1
    assert counts["merit_forward_evaluations"] == 1


def test_full_gradient_includes_crosscontrast_edge_term():
    a, b = ExplicitPhysics(), ExplicitPhysics()
    # Data gradients vanish, leaving the cross-contrast gradient alone.
    a.target = a.initial_density.clone()
    b.target = b.initial_density.clone()
    cfg = SolverConfig(outer_steps=1, cg_steps=8, damping_rel=.1)
    no_edge = optimize_density(a, prior_kwargs=PRIOR, edge_weight=0., config=cfg)
    edge = optimize_density(b, prior_kwargs=PRIOR, edge_weight=.5, config=cfg)
    assert no_edge.accepted_steps == 0
    assert edge.accepted_steps == 1
    assert edge.trace[0]["merit_after"] < edge.trace[0]["merit_before"]
    assert not torch.equal(edge.density, b.initial_density)


def test_budget_exhaustion_is_explicit_and_no_hidden_render_or_vjp():
    operator = ExplicitPhysics()
    result = optimize_density(operator, prior_kwargs=PRIOR, edge_weight=0.,
                              config=SolverConfig(outer_steps=10, max_vjp_pairs=2, max_render_pairs=4))
    assert result.status == "vjp_budget"
    assert result.work["vjp_pairs"] == 2
    assert result.work["render_pairs"] == operator.counters["render_pairs"] <= 4
    assert result.accepted_steps == 0


def test_zero_outer_is_a_true_noop():
    operator = ExplicitPhysics()
    result = optimize_density(operator, prior_kwargs=PRIOR, edge_weight=0., config=SolverConfig(outer_steps=0))
    assert result.work["render_pairs"] == result.work["vjp_pairs"] == 0
    torch.testing.assert_close(result.density, operator.initial_density, rtol=0, atol=0)


def test_failed_backtracking_retains_last_accepted_density_and_reports_status():
    operator = ExplicitPhysics()
    # A full step toward data optimum increases a strong nonlinear edge term;
    # forcing a nearly-full Armijo reduction rejects the damped direction.
    initial = operator.initial_density.clone()
    result = optimize_density(operator, prior_kwargs=PRIOR, edge_weight=0.,
                              config=SolverConfig(outer_steps=1, cg_steps=1, damping_rel=10,
                                                  armijo=.999, backtracks=1))
    assert result.status == "backtracking_failed" and result.accepted_steps == 0
    assert result.trace[0]["accepted"] is False
    assert len(result.trace[0]["backtracking"]) == 1
    torch.testing.assert_close(result.density, initial, rtol=0, atol=0)


def test_merit_forward_cap_is_measured_separately_from_vjp_budget():
    operator = ExplicitPhysics()
    result = optimize_density(operator, prior_kwargs=PRIOR, edge_weight=0.,
                              config=SolverConfig(outer_steps=2, cg_steps=1, max_merit_forwards=1))
    assert result.status == "merit_forward_budget"
    assert result.accepted_steps == 1 and result.work["merit_forward_evaluations"] == 1
    assert result.work["render_pairs"] == operator.counters["render_pairs"]


def test_nan_merit_failure_has_serializable_diagnostics():
    operator = ExplicitPhysics()
    original_loss = operator.data_loss
    calls = 0
    def nan_on_backtrack(pre, post, zero_target=False):
        nonlocal calls
        if not zero_target:
            calls += 1
        value = original_loss(pre, post, zero_target)
        return value * float("nan") if calls >= 2 else value
    operator.data_loss = nan_on_backtrack
    with pytest.raises(RuntimeError, match="nonfinite") as error:
        optimize_density(operator, prior_kwargs=PRIOR, edge_weight=0.)
    import json
    json.dumps(error.value.solver_trace, allow_nan=False)


def test_failure_carries_work_and_keeps_source_density():
    operator = ExplicitPhysics()
    initial = operator.initial_density.clone()
    def broken(*args, **kwargs):
        raise RuntimeError("physics failure")
    operator.data_loss = broken
    with pytest.raises(RuntimeError, match="physics failure") as error:
        optimize_density(operator, prior_kwargs=PRIOR, edge_weight=0.)
    assert error.value.solver_work["render_pairs"] == operator.counters["render_pairs"]
    torch.testing.assert_close(operator.initial_density, initial)


def test_fresh_outputs_audit_strictness_and_nonresumable_checkpoint(tmp_path):
    with pytest.raises(ValueError):
        reserve_run(tmp_path, "no")
    with pytest.raises(RuntimeError):
        validate_audit({"pre": {"passed": False}, "post": {"passed": True}})
    with pytest.raises(RuntimeError):
        validate_audit({"pre": {"passed": True}})
    operator = ExplicitPhysics()
    result = optimize_density(operator, prior_kwargs=PRIOR, edge_weight=0., config=SolverConfig(outer_steps=0))
    model = torch.nn.Linear(2, 2)
    artifact = tmp_path / "density_solver.pth"
    write_solver_checkpoint(artifact, model, result, source_checkpoint="frozen", source_checkpoint_sha256="abc")
    saved = torch.load(artifact, weights_only=True)
    assert saved["source_iteration"] == 4000 and saved["resumable_by_original_adam"] is False
    assert "iter" not in saved and "optim_state" not in saved
    assert saved["solver_counters"]["vjp_pairs"] == 0


@pytest.mark.parametrize("kwargs", [{"outer_steps": -1}, {"max_vjp_pairs": 0}, {"damping_rel": -1}, {"backtracks": 0}])
def test_invalid_work_or_damping_is_rejected(kwargs):
    with pytest.raises(ValueError):
        SolverConfig(**kwargs)
def test_active_subtraction_prior_is_rejected(monkeypatch):
    import sys
    import types
    import pytest
    from runner import source_prior_kwargs
    edge = types.ModuleType('losses.edge_tv')
    edge.resolve_edge_kappa = lambda *args: 1.
    monkeypatch.setitem(sys.modules, 'losses', types.ModuleType('losses'))
    monkeypatch.setitem(sys.modules, 'losses.edge_tv', edge)
    trainer = types.SimpleNamespace(config={'loss': {'lambda_sub': .1,
        'edge_tv': True, 'edge_tv_kappa_source': 'zf'}}, zf_p=None, zf_c=None)
    with pytest.raises(NotImplementedError, match='active source prior'):
        source_prior_kwargs(trainer, (1, 1, 1))


def _clinical_fixture(tmp_path, monkeypatch):
    import hashlib
    import types
    import gpu_preflight
    import runner
    trainer_root = tmp_path / "trainer"
    tile_path = trainer_root / "paired_gs/gaussian/tile_voxelizer.py"
    tile_path.parent.mkdir(parents=True)
    tile_path.write_text("# fixture with no compilation imports\n")
    binary = tile_path.parent / "cuda_ext/build/tile_voxelize_cuda_fixture.so"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"verified prebuilt binary fixture")
    extension = object()
    tile = types.SimpleNamespace(__file__=str(tile_path), _cuda_ext=None, _cuda_load_error="old error")
    events = []
    def load(name, path):
        assert name == binary.stem and path == binary
        events.append("load_binary")
        return extension
    def import_module(name):
        assert name == "gaussian.tile_voxelizer"
        events.append("import_normal_tile")
        return tile
    monkeypatch.setattr(gpu_preflight, "extension_path", lambda: binary)
    monkeypatch.setattr(gpu_preflight, "_load_module", load)
    monkeypatch.setattr(runner.importlib, "import_module", import_module)
    cfg = {"voxelizer": {"type": "tile_cuda"}, "mode": {"strict_cuda": False, "other": 7}}
    gate = {"invariants": {"prebuilt_extension": {"path": str(binary),
              "sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}}}
    base = types.SimpleNamespace(TRAINER=trainer_root)
    return cfg, gate, base, tile, extension, events


def test_verified_prebuilt_is_installed_before_normal_trainer_constructor(tmp_path, monkeypatch):
    import copy
    import types
    from runner import construct_strict_cuda_trainer, prepare_runtime_cuda
    cfg, gate, base, tile, extension, events = _clinical_fixture(tmp_path, monkeypatch)
    original = copy.deepcopy(cfg)
    override = prepare_runtime_cuda(cfg)
    assert override == {"mode.strict_cuda": {"source_present": True, "source": False, "effective": True}}
    original["mode"]["strict_cuda"] = True
    assert cfg == original  # no runtime config change beyond strict_cuda
    def trainer_factory(actual, *, device):
        assert tile._cuda_ext is extension and tile._cuda_load_error is None
        assert actual["mode"]["strict_cuda"] is True and device == "cpu-fixture"
        events.append("construct_trainer")
        return types.SimpleNamespace(voxelizer=types.SimpleNamespace(use_cuda=True,
            use_cupy=False, strict_cuda=True, backend="tile_cuda"), writer=None)
    trainer, proof = construct_strict_cuda_trainer(cfg, "cpu-fixture", base, gate, trainer_factory)
    assert events == ["import_normal_tile", "load_binary", "construct_trainer"]
    assert proof["prebuilt_extension"] == gate["invariants"]["prebuilt_extension"]
    assert proof["backend"] == "tile_cuda" and proof["strict_cuda"] is True


@pytest.mark.parametrize("mutation", ["missing", "stale_sha", "wrong_path", "wrong_source", "fallback"])
def test_missing_stale_or_fallback_backend_is_rejected_without_build(tmp_path, monkeypatch, mutation):
    import types
    from runner import construct_strict_cuda_trainer, prepare_runtime_cuda
    cfg, gate, base, tile, extension, events = _clinical_fixture(tmp_path, monkeypatch)
    prepare_runtime_cuda(cfg)
    if mutation == "missing":
        gate["invariants"].clear()
    elif mutation == "stale_sha":
        gate["invariants"]["prebuilt_extension"]["sha256"] = "stale"
    elif mutation == "wrong_path":
        gate["invariants"]["prebuilt_extension"]["path"] += ".other"
    elif mutation == "wrong_source":
        tile.__file__ += ".other"
    def trainer_factory(actual, *, device):
        events.append("construct_trainer")
        assert tile._cuda_ext is extension
        return types.SimpleNamespace(voxelizer=types.SimpleNamespace(use_cuda=False,
            use_cupy=False, strict_cuda=False, backend="tile"), writer=None)
    with pytest.raises((ValueError, RuntimeError), match="prebuilt|source|backend"):
        construct_strict_cuda_trainer(cfg, "cpu-fixture", base, gate, trainer_factory)
    if mutation != "fallback":
        assert "construct_trainer" not in events
    else:
        assert tile._cuda_ext is None and tile._cuda_load_error == "old error"


def test_reference_voxelizer_is_rejected_before_runtime_override():
    from runner import prepare_runtime_cuda
    cfg = {"voxelizer": {"type": "tile"}, "mode": {"strict_cuda": False}}
    with pytest.raises(ValueError, match="tile_cuda"):
        prepare_runtime_cuda(cfg)
    assert cfg["mode"]["strict_cuda"] is False


def test_uniform8_rejects_an_explicit_single_shift_and_zero_audit_builds_nothing(monkeypatch):
    import runner
    with pytest.raises(ValueError, match="shift"):
        runner.wavelet_protocol("uniform8", (1, 1, 1))
    assert runner.wavelet_protocol("single")["fixed_wavelet_shift"] == [1, 1, 1]
    assert runner.wavelet_protocol("uniform8")["fixed_wavelet_shift"] is None
    def forbidden(*args, **kwargs):
        raise AssertionError("zero audit built a prior")
    result = optimize_density(ExplicitPhysics(), prior_kwargs={}, wavelet_mode="uniform8",
                              prior_factory=forbidden, config=SolverConfig(outer_steps=0))
    assert result.work["wavelet_transform_batches"] == result.work["render_pairs"] == 0
    assert result.work["wavelet_mode"] == "uniform8"


def test_prior_factory_none_resolves_current_global_at_call_time(monkeypatch):
    import runner
    original = runner.build_frozen_prior
    calls = []
    def observed(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, "build_frozen_prior", observed)
    optimize_density(ExplicitPhysics(), prior_kwargs=PRIOR, edge_weight=0.)
    assert calls == [1]


def test_single_mode_arithmetic_matches_archived_v2_and_uniform8_counts_actual_transforms(monkeypatch):
    import importlib.util
    import sys
    import runner
    import surrogates
    archive = Path(__file__).resolve().parents[2] / "density_mm/source_versions/review_gate_v2"
    def load(name, file):
        spec = importlib.util.spec_from_file_location(name, file)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module
    old_prior = load("_fixture_v2_prior", archive / "surrogates.py")
    old_prior._SOURCE = surrogates._SOURCE
    old = load("_fixture_v2_runner", archive / "runner.py")
    old.build_frozen_prior = old_prior.build_frozen_prior
    class WaveletPhysics(ExplicitPhysics):
        def __init__(self):
            torch.manual_seed(99)
            self.initial_density = torch.randn(2, 216, dtype=torch.complex128) + 1
            self.target = torch.randn_like(self.initial_density)
            self.weights = torch.linspace(.5, 2, 216)
            self.counters = {"render_pairs": 0, "renderer_calls": 0}
        def render(self, rho):
            self.counters["render_pairs"] += 1
            self.counters["renderer_calls"] += 2
            return rho[0].reshape(6, 6, 6), rho[1].reshape(6, 6, 6)
        def data_loss(self, pre, post, zero_target=False):
            value = torch.stack((pre, post)).reshape(2, 216)
            return (((value - (0 if zero_target else self.target)) * self.weights).abs().square()).sum()
    options = dict(lambda_tv=.01, lambda_phase=.003, lambda_wavelet=(.12, .1), kappa=.5)
    cfg = dict(outer_steps=2, cg_steps=4)
    previous = old.optimize_density(WaveletPhysics(), prior_kwargs=options, edge_weight=.001, config=old.SolverConfig(**cfg))
    current = optimize_density(WaveletPhysics(), prior_kwargs=options, edge_weight=.001, config=SolverConfig(**cfg))
    torch.testing.assert_close(current.density, previous.density, rtol=0, atol=0)
    assert current.status == previous.status and current.accepted_steps == previous.accepted_steps
    for a, b in zip(current.trace, previous.trace):
        for key in ("merit_before", "merit_after", "gradient_norm", "damping", "cg_residual_history", "alpha"):
            assert a[key] == b[key]
        assert a["wavelet_mode"] == "single"
    assert {k:v for k,v in current.work.items() if k not in ("wavelet_mode", "wavelet_transform_batches")} == previous.work
    assert current.work["wavelet_transform_batches"] == 20
    actual_transform = surrogates.wavelet_details
    observed = []
    def counted(*args, **kwargs):
        observed.append(1)
        return actual_transform(*args, **kwargs)
    monkeypatch.setattr(surrogates, "wavelet_details", counted)
    uniform = optimize_density(WaveletPhysics(), prior_kwargs=options, edge_weight=.001,
                               wavelet_mode="uniform8", config=SolverConfig(**cfg))
    assert uniform.work["wavelet_mode"] == "uniform8"
    assert uniform.work["wavelet_transform_batches"] == len(observed) == 160
    assert [t["wavelet_transform_batches"] for t in uniform.trace] == [80, 80]


def _six_cube_counter_fixture():
    operator = ExplicitPhysics()
    torch.manual_seed(123)
    operator.initial_density = torch.randn(2, 216, dtype=torch.complex128)
    def render(rho):
        operator.counters["render_pairs"] += 1
        operator.counters["renderer_calls"] += 2
        return rho[0].reshape(6,6,6), rho[1].reshape(6,6,6)
    operator.render = render
    operator.data_loss = lambda p,c,zero_target=False: torch.stack((p,c)).abs().square().sum()
    return operator


def test_budget_before_next_outer_preserves_accepted_transform_count():
    operator = _six_cube_counter_fixture()
    result = optimize_density(operator, prior_kwargs=dict(PRIOR, lambda_wavelet=(.12,.1)), edge_weight=0.,
        wavelet_mode="uniform8", config=SolverConfig(outer_steps=2,cg_steps=1,max_vjp_pairs=3))
    assert result.status == "vjp_budget" and result.accepted_steps == 1
    assert result.work["wavelet_transform_batches"] == 56
    assert sum(t["wavelet_transform_batches"] for t in result.trace) == 56


def test_stationary_exit_keeps_construction_and_gradient_transform_work():
    operator = _six_cube_counter_fixture()
    operator.initial_density.zero_()
    result = optimize_density(operator, prior_kwargs=dict(PRIOR, lambda_wavelet=(.12,.1)), edge_weight=0.,
        wavelet_mode="uniform8", config=SolverConfig(outer_steps=2,cg_steps=1))
    assert result.status == "stationary" and result.accepted_steps == 0
    assert result.work["wavelet_transform_batches"] == result.trace[0]["wavelet_transform_batches"] == 32


def test_nonfinite_merit_exit_preserves_actual_transform_failure_diagnostics():
    operator = _six_cube_counter_fixture()
    original = operator.data_loss
    calls = 0
    def loss(p,c,zero_target=False):
        nonlocal calls
        if not zero_target:
            calls += 1
        value = original(p,c,zero_target)
        return value * float("nan") if calls >= 2 else value
    operator.data_loss = loss
    with pytest.raises(RuntimeError, match="nonfinite") as observed:
        optimize_density(operator, prior_kwargs=dict(PRIOR, lambda_wavelet=(.12,.1)), edge_weight=0.,
            wavelet_mode="uniform8", config=SolverConfig(outer_steps=2,cg_steps=1))
    assert observed.value.solver_work["wavelet_transform_batches"] == 56
    assert observed.value.solver_trace[0]["wavelet_transform_batches"] == 56


@pytest.mark.parametrize("mode,phase,phase_count", [("single", [0,0,0], 1), ("uniform8", None, 1)])
def test_direct_clinical_cli_rejects_gate_for_different_phase_protocol(tmp_path, monkeypatch, mode, phase, phase_count):
    import json
    import runner
    gate = tmp_path / "cuda_gate.json"
    gate.write_text(json.dumps(dict(passed=True,code_sha256={},wavelet_mode=mode,
                                    fixed_wavelet_shift=phase,wavelet_phase_count=phase_count)))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(runner, "_validate_review", lambda path: {"code_sha256": {}})
    def forbidden():
        raise AssertionError("invalid phase gate reached source/trainer loading")
    monkeypatch.setattr(runner, "_load_base", forbidden)
    with pytest.raises(SystemExit) as error:
        runner.main(["--mode","zero","--exp","fixture","--wavelet-mode",mode,
                     "--review-gate",str(tmp_path/"review.json"),"--cuda-operator-gate",str(gate)])
    assert error.value.code == 2

