"""Behavioral contracts for a scalar geometry restart, with real CPU Adam."""
from __future__ import annotations

import copy
import importlib.util
import io
import math
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("positions", "scales", "rotations", "density_real", "density_imag",
         "density_real_contrast", "density_imag_contrast")
LRS = (.0003, .002, .0003, .015, .015, .015, .015)


def load_file(path, name):
    if not path.is_file():
        pytest.fail(f"Required implementation is absent: {path.name}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def helper():
    return load_file(ROOT / "overrides.py", "geometry_restart_test_overrides")


@pytest.fixture
def restored():
    # Real strict source restoration, with small tensors rather than clinical data.
    source = ROOT.parent / "nonlocal_autoresearch/trainer/paired_gs/trainers/resume_protocol.py"
    protocol = load_file(source, "geometry_restart_test_resume")
    model = torch.nn.Module()
    for index, name in enumerate(NAMES):
        model.register_parameter(name, torch.nn.Parameter(
            torch.tensor([index + .25, -.5], dtype=torch.float64)))
    optimizer = torch.optim.Adam([
        {"params": [param], "lr": lr}
        for param, lr in zip(model.parameters(), LRS)])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.9995)
    for group in optimizer.param_groups:
        param = group["params"][0]
        group["lr"] = group["initial_lr"] * .9995 ** 4000
        optimizer.state[param] = {
            "step": torch.tensor(4000.),
            "exp_avg": torch.tensor([.2, -.1], dtype=torch.float64),
            "exp_avg_sq": torch.tensor([.04, .09], dtype=torch.float64)}
    checkpoint = copy.deepcopy({"iter": 4000, "model_state": model.state_dict(),
                                "optim_state": optimizer.state_dict()})
    protocol.restore_checkpoint(model, optimizer, scheduler, checkpoint)
    return model, optimizer, scheduler


def serialized(model, optimizer, scheduler):
    stream = io.BytesIO()
    torch.save((model.state_dict(), optimizer.state_dict(), scheduler.state_dict()), stream)
    return stream.getvalue()


def raw_serialized(model, optimizer, scheduler):
    """Capture even malformed Adam mappings which its state_dict cannot pack."""
    stream = io.BytesIO()
    groups = [{key: ([id(p) for p in value] if key == "params" else value)
               for key, value in group.items()} for group in optimizer.param_groups]
    states = [(id(param), value) for param, value in optimizer.state.items()]
    torch.save((model.state_dict(), groups, states, scheduler.state_dict()), stream)
    return stream.getvalue()


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    else:
        assert left == right


def test_identity_is_byte_equal_read_only_and_can_be_repeated(restored):
    helper = load_file(ROOT / "overrides.py", "geometry_restart_test_overrides")
    model, optimizer, scheduler = restored
    before = serialized(*restored)
    opt_keys = set(optimizer.__dict__)
    sched_keys = set(scheduler.__dict__)
    for _ in range(2):
        actual = helper.apply_geometry_lr_factor(*restored, 1.)
        assert serialized(*restored) == before
        assert actual["is_noop"] is True
        assert actual["source"] == actual["after"]
        assert set(optimizer.__dict__) == opt_keys
        assert set(scheduler.__dict__) == sched_keys


@pytest.mark.parametrize("factor", [2., 4., 8., .5, .0001])
def test_exact_three_rates_change_while_density_and_tensor_states_are_retained(helper, restored, factor):
    model, optimizer, scheduler = restored
    old_model = copy.deepcopy(model.state_dict())
    old_opt = copy.deepcopy(optimizer.state_dict())
    old_sched = copy.deepcopy(scheduler.state_dict())
    params = list(model.parameters())
    tensors = {name: dict(optimizer.state[p]) for name, p in zip(NAMES, params)}
    actual = helper.apply_geometry_lr_factor(*restored, factor)
    assert actual["factor"] == factor
    assert actual["geometry_groups"] == list(NAMES[:3])
    assert_equal(model.state_dict(), old_model)
    assert_equal(optimizer.state_dict()["state"], old_opt["state"])
    assert list(model.parameters()) == params
    for index, (name, group) in enumerate(zip(NAMES, optimizer.param_groups)):
        prior = old_opt["param_groups"][index]
        if index < 3:
            assert group["lr"] == prior["lr"] * factor
            assert group["initial_lr"] == prior["initial_lr"] * factor
        else:
            assert_equal(optimizer.state_dict()["param_groups"][index], prior)
        for key in set(prior) - {"lr", "initial_lr", "params"}:
            assert_equal(group[key], prior[key])
        for key, value in tensors[name].items():
            assert optimizer.state[params[index]][key] is value
        observed = actual["after"]["groups"][index]
        assert observed["name"] == name
        assert observed["lr"] == group["lr"]
        assert observed["step"] == 4000
    for index, base in enumerate(old_sched["base_lrs"]):
        assert scheduler.base_lrs[index] == base * (factor if index < 3 else 1.)
    assert scheduler.get_last_lr() == [g["lr"] for g in optimizer.param_groups]
    for key in set(old_sched) - {"base_lrs", "_last_lr"}:
        assert_equal(scheduler.state_dict()[key], old_sched[key])


def test_next_adam_and_scheduler_steps_retain_source_clocks_and_moments(helper, restored):
    model, optimizer, scheduler = restored
    start = model.positions.detach().clone()
    helper.apply_geometry_lr_factor(*restored, 4.)
    for p in model.parameters():
        p.grad = torch.tensor([.3, -.4], dtype=torch.float64)
    optimizer.step()
    # Original Adam recurrence at step 4001; neither first nor second moment resets.
    m = torch.tensor([.21, -.13], dtype=torch.float64)
    v = torch.tensor([.04005, .09007], dtype=torch.float64)
    expected = start - (.0012 * .9995 ** 4000 / (1 - .9 ** 4001)) * m / (
        (v / (1 - .999 ** 4001)).sqrt() + 1e-8)
    torch.testing.assert_close(model.positions, expected, rtol=0., atol=1e-15)
    torch.testing.assert_close(optimizer.state[model.positions]["exp_avg"], m, rtol=0., atol=1e-16)
    torch.testing.assert_close(optimizer.state[model.positions]["exp_avg_sq"], v, rtol=0., atol=1e-16)
    scheduler.step()
    assert scheduler.last_epoch == 4001
    assert scheduler._step_count == 4002
    assert all(optimizer.state[p]["step"].item() == 4001 for p in model.parameters())
    for index, group in enumerate(optimizer.param_groups):
        expected_lr = LRS[index] * (4. if index < 3 else 1.) * .9995 ** 4001
        assert math.isclose(group["lr"], expected_lr, rel_tol=1e-14)
        assert scheduler.get_last_lr()[index] == group["lr"]


@pytest.mark.parametrize("factor", [0., -1., 8.0001, float("inf"), float("nan"),
                                    math.nextafter(0., 1.), True, "2", None])
def test_invalid_factor_is_rejected_without_mutation(helper, restored, factor):
    before = serialized(*restored)
    with pytest.raises(ValueError):
        helper.apply_geometry_lr_factor(*restored, factor)
    assert serialized(*restored) == before


@pytest.mark.parametrize("bad_iteration", [3999, 4001, True, 4000., float("nan")])
def test_other_checkpoint_clock_is_not_supported(helper, restored, bad_iteration):
    before = serialized(*restored)
    with pytest.raises(ValueError):
        helper.apply_geometry_lr_factor(*restored, 2., expected_iteration=bad_iteration)
    assert serialized(*restored) == before


@pytest.mark.parametrize("defect", [
    "missing_group", "duplicate_group", "wrong_order", "multiple_params", "foreign_parameter",
    "extra_parameter", "parameter_alias", "frozen_parameter", "extra_state", "missing_moment",
    "moment_shape", "moment_dtype", "nonfinite_moment", "negative_variance", "step", "step_nan",
    "beta", "beta_nan", "beta_length", "lr_nan", "initial_lr_nan", "wrong_nominal_lr",
    "wrong_current_lr", "eps_nan", "weight_decay_nan", "amsgrad", "clock", "step_count",
    "gamma", "last_lr", "base_lr", "short_last_lr", "short_base_lr", "scheduler_optimizer"])
def test_invalid_source_is_rejected_atomically(helper, restored, defect):
    model, optimizer, scheduler = restored
    group = optimizer.param_groups[-1]  # Late validation must precede any early-group mutation.
    state = optimizer.state[model.density_imag_contrast]
    if defect == "missing_group":
        optimizer.param_groups.pop()
    elif defect == "duplicate_group":
        group["params"] = [model.positions]
    elif defect == "wrong_order":
        optimizer.param_groups[0], optimizer.param_groups[1] = optimizer.param_groups[1], optimizer.param_groups[0]
    elif defect == "multiple_params":
        group["params"].append(model.positions)
    elif defect == "foreign_parameter":
        group["params"] = [torch.nn.Parameter(torch.ones(2))]
    elif defect == "extra_parameter":
        model.register_parameter("private_density", torch.nn.Parameter(torch.ones(2)))
    elif defect == "parameter_alias":
        model.register_parameter("alias", model.positions)
    elif defect == "frozen_parameter":
        model.positions.requires_grad_(False)
    elif defect == "extra_state":
        optimizer.state[torch.nn.Parameter(torch.ones(2))] = {}
    elif defect == "missing_moment":
        del state["exp_avg"]
    elif defect == "moment_shape":
        state["exp_avg"] = torch.zeros(1, dtype=torch.float64)
    elif defect == "moment_dtype":
        state["exp_avg_sq"] = state["exp_avg_sq"].float()
    elif defect == "nonfinite_moment":
        state["exp_avg"][0] = float("nan")
    elif defect == "negative_variance":
        state["exp_avg_sq"][0] = -.1
    elif defect == "step":
        state["step"].fill_(3999.)
    elif defect == "step_nan":
        state["step"].fill_(float("nan"))
    elif defect == "beta":
        group["betas"] = (.9, .99)
    elif defect == "beta_nan":
        group["betas"] = (.9, float("nan"))
    elif defect == "beta_length":
        group["betas"] = (.9,)
    elif defect == "lr_nan":
        group["lr"] = float("nan")
    elif defect == "initial_lr_nan":
        group["initial_lr"] = float("nan")
    elif defect == "wrong_nominal_lr":
        optimizer.param_groups[0]["initial_lr"] *= 2.
        optimizer.param_groups[0]["lr"] *= 2.
        scheduler.base_lrs[0] *= 2.
        scheduler._last_lr[0] *= 2.
    elif defect == "wrong_current_lr":
        group["lr"] *= 2.
        scheduler._last_lr[-1] *= 2.
    elif defect == "eps_nan":
        group["eps"] = float("nan")
    elif defect == "weight_decay_nan":
        group["weight_decay"] = float("nan")
    elif defect == "amsgrad":
        group["amsgrad"] = True
    elif defect == "clock":
        scheduler.last_epoch = 3999
    elif defect == "step_count":
        scheduler._step_count = 4000
    elif defect == "gamma":
        scheduler.gamma = .999
    elif defect == "last_lr":
        scheduler._last_lr[-1] *= 2.
    elif defect == "base_lr":
        scheduler.base_lrs[-1] *= 2.
    elif defect == "short_last_lr":
        scheduler._last_lr.pop()
    elif defect == "short_base_lr":
        scheduler.base_lrs.pop()
    elif defect == "scheduler_optimizer":
        scheduler.optimizer = torch.optim.Adam([torch.nn.Parameter(torch.ones(1))])
    before = raw_serialized(*restored)
    with pytest.raises(ValueError):
        helper.apply_geometry_lr_factor(*restored, 2.)
    assert raw_serialized(*restored) == before


def test_nonidentity_reapplication_is_rejected_after_serialization(helper, restored):
    model, optimizer, scheduler = restored
    helper.apply_geometry_lr_factor(*restored, 2.)
    checkpoint = copy.deepcopy((optimizer.state_dict(), scheduler.state_dict()))
    model2 = copy.deepcopy(model)
    optimizer2 = torch.optim.Adam([{"params": [p], "lr": lr} for p, lr in zip(model2.parameters(), LRS)])
    scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer2, gamma=.9995)
    optimizer2.load_state_dict(checkpoint[0])
    scheduler2.load_state_dict(checkpoint[1])
    before = serialized(model2, optimizer2, scheduler2)
    with pytest.raises(ValueError):
        helper.apply_geometry_lr_factor(model2, optimizer2, scheduler2, 2.)
    assert serialized(model2, optimizer2, scheduler2) == before


def test_smallest_representable_nonidentity_factor_cannot_be_applied_twice(helper, restored):
    # A tolerance around nominal initial_lr could accidentally admit repeats.
    factor = math.nextafter(1., 2.)
    helper.apply_geometry_lr_factor(*restored, factor)
    before = serialized(*restored)
    with pytest.raises(ValueError):
        helper.apply_geometry_lr_factor(*restored, factor)
    assert serialized(*restored) == before


def test_unexpected_assignment_failure_rolls_back_already_written_rates(helper, restored):
    class FailingOnceDict(dict):
        def __setitem__(self, key, value):
            if key == "initial_lr":
                raise RuntimeError("injected assignment failure")
            super().__setitem__(key, value)

    model, optimizer, scheduler = restored
    optimizer.param_groups[1] = FailingOnceDict(optimizer.param_groups[1])
    before = serialized(*restored)
    with pytest.raises(RuntimeError, match="injected"):
        helper.apply_geometry_lr_factor(*restored, 4.)
    assert serialized(*restored) == before


def test_scheduler_assignment_failure_restores_both_optimizer_and_scheduler(helper, restored):
    class RejectLastLR(torch.optim.lr_scheduler.ExponentialLR):
        reject_assignment = False

        def __setattr__(self, name, value):
            if name == "_last_lr" and self.reject_assignment:
                raise RuntimeError("injected scheduler assignment failure")
            super().__setattr__(name, value)

    model, optimizer, scheduler = restored
    scheduler.__class__ = RejectLastLR
    scheduler.reject_assignment = True
    before = serialized(*restored)
    with pytest.raises(RuntimeError, match="injected"):
        helper.apply_geometry_lr_factor(*restored, 4.)
    assert serialized(*restored) == before
