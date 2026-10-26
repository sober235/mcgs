"""Post-restore geometry learning-rate factor; no optimizer or model step.

The caller must first perform the original strict checkpoint restoration. This
utility recognizes that specific iteration-4000 source; it does not restore a
checkpoint or provide an execution route for a reconstruction experiment.
"""
from __future__ import annotations

import math
from numbers import Real

import torch

PARAMETER_NAMES = ("positions", "scales", "rotations", "density_real", "density_imag",
                   "density_real_contrast", "density_imag_contrast")
GEOMETRY_NAMES = PARAMETER_NAMES[:3]
SOURCE_INITIAL_LRS = (.0003, .002, .0003, .015, .015, .015, .015)
SOURCE_ITERATION = 4000
SOURCE_BETAS = (.9, .999)
GAMMA = .9995


def _finite_real(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite real scalar")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0.):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))
    return value


def _validate_source(model, optimizer, scheduler, expected_iteration):
    if type(expected_iteration) is not int or expected_iteration != SOURCE_ITERATION:
        raise ValueError("only the formal source iteration 4000 is supported")
    if not isinstance(model, torch.nn.Module) or not isinstance(optimizer, torch.optim.Adam):
        raise ValueError("require the original model and seven-group Adam optimizer")
    # Do not silently hide aliased Parameters under named_parameters' default
    # de-duplication: one source parameter must map to exactly one group.
    named_items = list(model.named_parameters(remove_duplicate=False))
    named = dict(named_items)
    if (set(named) != set(PARAMETER_NAMES) or len(named_items) != 7
            or len({id(p) for p in named.values()}) != 7
            or len(optimizer.param_groups) != 7):
        raise ValueError("require exactly the original seven distinct model parameters and Adam groups")
    for name, group in zip(PARAMETER_NAMES, optimizer.param_groups):
        if (not isinstance(group, dict) or len(group.get("params", ())) != 1
                or group["params"][0] is not named[name] or not named[name].requires_grad):
            raise ValueError(f"original Adam parameter identity/order differs at {name}")
    if {id(p) for p in optimizer.state} != {id(p) for p in named.values()}:
        raise ValueError("Adam state must map exactly once to each original parameter")
    if (not isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)
            or scheduler.optimizer is not optimizer
            or scheduler.gamma != GAMMA
            or type(scheduler.last_epoch) is not int or scheduler.last_epoch != expected_iteration
            or type(scheduler._step_count) is not int or scheduler._step_count != expected_iteration + 1
            or not isinstance(scheduler.base_lrs, list) or len(scheduler.base_lrs) != 7
            or not isinstance(scheduler._last_lr, list) or len(scheduler._last_lr) != 7):
        raise ValueError("require the original ExponentialLR at iteration 4000 with gamma .9995")

    for index, (name, nominal, group) in enumerate(zip(
            PARAMETER_NAMES, SOURCE_INITIAL_LRS, optimizer.param_groups)):
        param = named[name]
        state = optimizer.state[param]
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError(f"require the original three Adam state fields for {name}")
        step = state["step"]
        if (not isinstance(step, torch.Tensor) or step.numel() != 1
                or float(step) != expected_iteration):
            raise ValueError(f"Adam step must equal 4000 for {name}")
        for key in ("exp_avg", "exp_avg_sq"):
            value = state[key]
            if (not isinstance(value, torch.Tensor) or value.shape != param.shape
                    or value.dtype != param.dtype or value.device != param.device
                    or not bool(torch.isfinite(value).all())):
                raise ValueError(f"invalid original Adam {key} for {name}")
        if bool((state["exp_avg_sq"] < 0).any()):
            raise ValueError(f"negative original Adam second moment for {name}")
        betas = group.get("betas")
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise ValueError(f"invalid original Adam betas for {name}")
        actual_betas = tuple(_finite_real(beta, f"{name} beta") for beta in betas)
        if actual_betas != SOURCE_BETAS:
            raise ValueError(f"source Adam betas must be (.9, .999) for {name}")
        eps = _finite_real(group.get("eps"), f"{name} eps", positive=True)
        decay = _finite_real(group.get("weight_decay"), f"{name} weight_decay")
        if eps != 1e-8 or decay != 0. or group.get("amsgrad") is not False:
            raise ValueError(f"require original Adam eps, weight decay and non-AMSGrad mode for {name}")
        for key in ("maximize", "capturable", "differentiable", "decoupled_weight_decay"):
            if group.get(key, False) is not False:
                raise ValueError(f"unsupported source Adam {key} for {name}")
        initial = _finite_real(group.get("initial_lr"), f"{name} initial_lr", positive=True)
        current = _finite_real(group.get("lr"), f"{name} lr", positive=True)
        base = _finite_real(scheduler.base_lrs[index], f"{name} scheduler base_lr", positive=True)
        last = _finite_real(scheduler._last_lr[index], f"{name} scheduler last_lr", positive=True)
        # Exact nominal equality also rejects applying the intervention twice,
        # including after an optimizer/scheduler serialization round trip.
        if (initial != nominal or base != initial or last != current
                or not math.isclose(current, nominal * GAMMA ** expected_iteration,
                                    rel_tol=1e-10, abs_tol=1e-14)):
            raise ValueError(f"source nominal/current Adam and scheduler learning rates disagree for {name}")


def _snapshot(optimizer, scheduler):
    """Only actual scalar metadata; tensor state is never copied or replaced."""
    return {
        "groups": [
            {"name": name, "lr": float(group["lr"]),
             "initial_lr": float(group["initial_lr"]),
             "betas": [float(beta) for beta in group["betas"]],
             "eps": float(group["eps"]), "weight_decay": float(group["weight_decay"]),
             "step": float(optimizer.state[group["params"][0]]["step"])}
            for name, group in zip(PARAMETER_NAMES, optimizer.param_groups)],
        "scheduler": {"last_epoch": scheduler.last_epoch, "step_count": scheduler._step_count,
                      "gamma": float(scheduler.gamma),
                      "base_lrs": [float(lr) for lr in scheduler.base_lrs],
                      "last_lrs": [float(lr) for lr in scheduler._last_lr]},
    }


def apply_geometry_lr_factor(model, optimizer, scheduler, factor, expected_iteration=4000):
    """Scale exactly the three geometry rates after strict source restoration.

    All source validation precedes mutation. The four density groups, model and
    Adam tensor states, every non-LR hyperparameter and scheduler clocks remain
    unchanged. The factor must be finite and satisfy ``0 < factor <= 8``. A
    nonidentity application changes the nominal rates, which rejects a second
    application without an optimizer marker. Factor 1 is fully read-only.
    """
    factor = _finite_real(factor, "geometry LR factor", positive=True)
    if factor > 8.:
        raise ValueError("geometry LR factor must not exceed 8")
    _validate_source(model, optimizer, scheduler, expected_iteration)
    source = _snapshot(optimizer, scheduler)
    result = {"factor": factor, "geometry_groups": list(GEOMETRY_NAMES),
              "source_iteration": expected_iteration, "is_noop": factor == 1.,
              "source": source, "scheduler_clock_preserved": True,
              "moment_policy": "retain original Adam step, exp_avg and exp_avg_sq; no reset"}
    if factor == 1.:
        result["after"] = _snapshot(optimizer, scheduler)
        return result

    old_rates = [(group["lr"], group["initial_lr"]) for group in optimizer.param_groups[:3]]
    old_base, old_last = scheduler.base_lrs, scheduler._last_lr
    new_base, new_last = list(old_base), list(old_last)
    new_rates = []
    for index, (current, initial) in enumerate(old_rates):
        new_current = _finite_real(current * factor, "scaled geometry LR", positive=True)
        new_initial = _finite_real(initial * factor, "scaled initial geometry LR", positive=True)
        new_rates.append((new_current, new_initial))
        new_base[index], new_last[index] = new_initial, new_current
    try:
        for group, (current, initial) in zip(optimizer.param_groups[:3], new_rates):
            group["lr"] = current
            group["initial_lr"] = initial
        scheduler.base_lrs = new_base
        scheduler._last_lr = new_last
        result["after"] = _snapshot(optimizer, scheduler)
    except BaseException:
        # Bypass a failing dict-subclass setter or scheduler __setattr__ while
        # restoring the exact pre-existing scalar objects and list references.
        for group, (current, initial) in zip(optimizer.param_groups[:3], old_rates):
            dict.__setitem__(group, "lr", current)
            dict.__setitem__(group, "initial_lr", initial)
        scheduler.__dict__["base_lrs"] = old_base
        scheduler.__dict__["_last_lr"] = old_last
        raise
    return result
