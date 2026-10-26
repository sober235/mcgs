"""CPU contract tests for the isolated NLR continuation (no CUDA extension import)."""
import ast
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "trainer/paired_gs"))


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runner():
    return load_file("nlr_runner_test", ROOT / "runner.py")


@pytest.fixture
def red():
    return load_file("nlr_red_test", ROOT / "trainer/paired_gs/losses/nonlocal_red.py")


def test_local_factor_scales_only_actual_tv_and_both_wavelet_weights(runner, tmp_path):
    source = yaml.safe_load(runner.BASE_CONFIG.read_text())
    cfg = runner.build_config("all", 500, tmp_path, local_factor=.4, nlambda=.03, sigma_rel=.02)
    expected = copy.deepcopy(source)
    expected["loss"].update(tv_weight=.04, lambda_wav=.04, lambda_wav_plain=.048)
    for key in source["loss"]:
        assert cfg["loss"][key] == pytest.approx(expected["loss"][key]) if isinstance(expected["loss"][key], float) else cfg["loss"][key] == expected["loss"][key]
    assert cfg["loss"]["lambda_nl"] == .03
    assert cfg["loss"]["nl_start"] == 4001
    assert cfg["loss"]["nl_refresh"] == 250
    assert cfg["loss"]["nl_sigma_rel"] == .02
    assert cfg["loss"]["nl_denoiser"] == "patch_lowrank"
    assert cfg["training"]["resume_offset"] == 4000
    assert cfg["training"]["max_iterations"] == 4500
    assert cfg["campaign_overrides"]["local_regularization"]["contrast_wavelet"]["effective"] == pytest.approx(.04)
    assert cfg["campaign_overrides"]["local_regularization"]["tv"]["source_key"] == "loss.tv_weight"


@pytest.mark.parametrize("name,value", [("local_factor", -1.), ("nlambda", -1.), ("sigma_rel", -1.),
                                          ("local_factor", float("nan")), ("nlambda", float("inf")),
                                          ("sigma_rel", float("nan"))])
def test_config_rejects_invalid_nonlocal_controls(runner, tmp_path, name, value):
    with pytest.raises(ValueError):
        runner.build_config("all", 2, tmp_path, **{name: value})


def test_patch_factory_forwards_fixed_preset_without_importing_at_zero(runner, red, monkeypatch, tmp_path):
    class Denoiser:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def __call__(self, z, sigma):
            return z * self.kwargs["threshold_scale"]
    monkeypatch.setitem(sys.modules, "losses.nonlocal_lowrank", SimpleNamespace(PatchLowRankDenoiser=Denoiser))
    cfg = runner.build_config("all", 2, tmp_path, nlambda=.1)
    denoiser = red.make_denoiser(cfg["loss"]["nl_denoiser"], **cfg["loss"]["nl_denoiser_kwargs"])
    assert denoiser.kwargs == {"patch_size":5, "search_radius":8, "search_stride":2, "anchor_stride":3,
                               "group_size":16, "min_group_size":4, "distance_factor":3.,
                               "threshold_scale":1., "batch_size":256, "device":"cuda:0"}
    assert np.array_equal(denoiser(np.ones((2, 2, 2), np.float32), .01), np.ones((2, 2, 2)))


def test_refresh_history_records_current_detached_scan_sigma_and_change(red):
    class Shrink:
        last_stats = {"valid_group_count": 2, "reason": "denoised"}
        def __call__(self, z, sigma):
            return z * np.float32(.9)
    support = torch.ones((3, 3, 3), dtype=torch.bool)
    prior = red.NonLocalRED(support, Shrink(), sigma_rel=.015, start=4001, refresh=250)
    vol = torch.arange(1, 28, dtype=torch.float32).reshape(3, 3, 3).requires_grad_()
    assert [it for it in range(4001, 4501) if prior.due(it)] == [4001, 4251]
    first = prior.refresh("pre", vol, 4001)
    assert first["sigma"] == pytest.approx(.015 * float(torch.quantile(vol.detach().flatten(), .99)))
    assert first["target_rms_difference"] > 0
    assert first["kernel_stats"]["valid_group_count"] == 2
    assert not prior.targets["pre"].requires_grad
    prior.refresh("post", 2 * vol, 4001)
    prior.refresh("pre", vol, 4251)
    assert [row["iter"] for row in prior.history["pre"]] == [4001, 4251]
    assert prior.history["post"][0]["sigma"] == pytest.approx(first["sigma"] * 2)
    prior("pre", vol).backward()
    assert torch.isfinite(vol.grad).all() and vol.grad.abs().sum() > 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.])
def test_invalid_target_cannot_silently_enter_prior(red, value):
    prior = red.NonLocalRED(torch.ones((2, 2, 2), dtype=torch.bool),
                           lambda z, sigma: np.full_like(z, value), start=4001)
    with pytest.raises((RuntimeError, ValueError)):
        prior.refresh("pre", torch.ones((2, 2, 2)), 4001)
    assert "pre" not in prior.targets


def train_function(path):
    """Execute the actual trainer method, excluding CUDA renderer imports."""
    from trainers.resume_protocol import absolute_iterations, resumed_wavelet_generator
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "JointTrainer")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "train")
    namespace = dict(torch=torch, np=np, warnings=warnings, time=time,
                     absolute_iterations=absolute_iterations, resumed_wavelet_generator=resumed_wavelet_generator)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), namespace)
    return namespace["train"]


def tiny_trainer(path, loss_config, max_iterations=4002):
    model = torch.nn.Module()
    for name in ("positions", "scales", "rotations", "density_real", "density_imag", "density_real_contrast", "density_imag_contrast"):
        model.register_parameter(name, torch.nn.Parameter(torch.tensor([.4, .6])))
    def render(channel):
        value = sum(parameter.sum() * (i + 1) * .01 for i, parameter in enumerate(model.parameters()))
        return torch.ones((5, 5, 5), dtype=torch.complex64) * value * (1 if channel == "plain" else 1.2)
    trainer = SimpleNamespace(config={"training":{"resume_offset":4000,"max_iterations":max_iterations,"save_every":1000},
                                      "loss":{"lambda_geo":0., **loss_config}},
                              model=model, device=torch.device("cpu"), writer=None, out_dir=path,
                              name_plain="pre", name_contrast="post", zf_p=torch.ones((5, 5, 5)),
                              zf_c=torch.ones((5, 5, 5)), _dc_norm_p=1, _dc_norm_c=1,
                              csm_fit_p=None, csm_fit_c=None, kspace_fit_p=None, kspace_fit_c=None,
                              mask_fit_p=None, mask_fit_c=None, _render=render,
                              _build_freq_weight=lambda: None, _freq_weight_curriculum=lambda *args:None,
                              _kspace_loss=lambda vol,*args,**kwargs:(vol.abs() - .7).square().mean(),
                              _adaptive_density_control=lambda it: None)
    trainer.optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    trainer.scheduler = None
    path.mkdir()
    return trainer


def test_zero_coefficient_has_identical_cpu_steps_and_no_denoiser_construction(tmp_path, monkeypatch):
    from losses import nonlocal_red
    def forbidden(*args, **kwargs):
        raise AssertionError("zero coefficient constructed denoiser")
    monkeypatch.setattr(nonlocal_red, "make_denoiser", forbidden)
    old = tiny_trainer(tmp_path / "old", {})
    new = tiny_trainer(tmp_path / "new", {"lambda_nl":0., "nl_denoiser":"patch_lowrank"})
    train_function(ROOT / "verification/integration_joint_trainer_before.py")(old)
    train_function(ROOT / "trainer/paired_gs/trainers/joint_trainer.py")(new)
    for name, parameter in old.model.named_parameters():
        assert torch.equal(parameter, dict(new.model.named_parameters())[name])
    old_state, new_state = old.optimizer.state_dict(), new.optimizer.state_dict()
    assert old_state["param_groups"] == new_state["param_groups"]
    for parameter_id, state in old_state["state"].items():
        for key, value in state.items():
            assert torch.equal(value, new_state["state"][parameter_id][key])


def test_training_activates_even_one_step_resume_and_reports_finite_terms(tmp_path, monkeypatch):
    from losses import nonlocal_red
    monkeypatch.setattr(nonlocal_red, "make_denoiser", lambda *args, **kwargs: lambda z, sigma: z * .9)
    trainer = tiny_trainer(tmp_path / "active", {"lambda_nl":.03, "nl_denoiser":"patch_lowrank",
                                                "nl_start":4001,"nl_refresh":250}, max_iterations=4001)
    trace = []
    train_function(ROOT / "trainer/paired_gs/trainers/joint_trainer.py")(trainer, lambda phase,it,data: trace.append((phase,it,data)))
    row = next(data for phase,it,data in trace if phase == "pre_step")
    assert row["nonlocal"]["raw_by_channel"]["pre"] > 0
    assert row["nonlocal"]["weighted"] > 0
    assert [r["iter"] for r in trainer.nlr_summary["refresh_history"]["pre"]] == [4001]
    assert trainer.nlr_summary["validated_active"]
    assert row["gradients_finite"] is True


def test_resumed_500_step_training_refreshes_both_scans_at_4001_and_4251(tmp_path, monkeypatch):
    from losses import nonlocal_red
    monkeypatch.setattr(nonlocal_red, "make_denoiser", lambda *args, **kwargs: lambda z, sigma: z * .9)
    trainer = tiny_trainer(tmp_path / "schedule", {"lambda_nl":.03, "nl_start":4001,"nl_refresh":250},
                           max_iterations=4500)
    train_function(ROOT / "trainer/paired_gs/trainers/joint_trainer.py")(trainer)
    summary = trainer.nlr_summary
    assert summary["expected_refresh_iterations"] == [4001,4251]
    assert [row["iter"] for row in summary["refresh_history"]["pre"]] == [4001,4251]
    assert [row["iter"] for row in summary["refresh_history"]["post"]] == [4001,4251]
    assert summary["active_step_count"] == summary["nonzero_weighted_step_count"] == 500
    assert summary["weighted_R_nl_min"] > 0
    assert summary["validated_active"]


def test_positive_coefficient_with_identity_prior_is_not_a_valid_experiment(tmp_path, monkeypatch):
    from losses import nonlocal_red
    monkeypatch.setattr(nonlocal_red, "make_denoiser", lambda *args, **kwargs: lambda z, sigma: z.copy())
    trainer = tiny_trainer(tmp_path / "identity", {"lambda_nl":.03, "nl_start":4001}, max_iterations=4001)
    with pytest.raises(RuntimeError, match="no validated active prior"):
        train_function(ROOT / "trainer/paired_gs/trainers/joint_trainer.py")(trainer)


def test_trace_records_active_prior_values_and_runtime(runner):
    model = torch.nn.Module()
    for name in runner.NAMES:
        model.register_parameter(name, torch.nn.Parameter(torch.ones(2)))
    optimizer = torch.optim.Adam([{"params":[p]} for p in model.parameters()])
    stream = io.StringIO()
    trace = runner.StepTrace(SimpleNamespace(model=model,optimizer=optimizer,device=torch.device("cpu")),
                             "all",stream)
    trace("start",4001,{})
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    trace("pre_step",4001,{"nonlocal":{"enabled":True,"raw_by_channel":{"pre":.1,"post":.2},"weighted":.009}})
    optimizer.step()
    trace("end",4001,{})
    row = json.loads(stream.getvalue())
    assert row["nonlocal"]["weighted"] == .009
    assert row["gradients_finite"]
    assert row["wall_seconds"] >= 0
    assert row["absolute_iteration"] == 4001


def test_nan_gradient_is_rejected_before_adam(runner):
    model = torch.nn.Module()
    for name in runner.NAMES:
        model.register_parameter(name, torch.nn.Parameter(torch.ones(2)))
    optimizer = torch.optim.Adam([{"params":[p]} for p in model.parameters()])
    trainer = SimpleNamespace(model=model,optimizer=optimizer,device=torch.device("cpu"))
    trace = runner.StepTrace(trainer,"all",io.StringIO())
    trace("start",4001,{})
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    model.positions.grad[0] = float("nan")
    with pytest.raises(RuntimeError, match="finite"):
        trace("pre_step",4001,{})
