"""Small, CPU-testable continuation protocol for the formal seven-group model."""
from __future__ import annotations

import math
import torch

PARAMETER_NAMES = ("positions", "scales", "rotations", "density_real", "density_imag",
                   "density_real_contrast", "density_imag_contrast")
GEOMETRY_NAMES = PARAMETER_NAMES[:3]
DENSITY_NAMES = PARAMETER_NAMES[3:]


def absolute_iterations(offset: int, end: int) -> range:
    """Return absolute continued iterations; end is never an extra-step count."""
    if not isinstance(offset, int) or not isinstance(end, int) or not 0 <= offset <= end:
        raise ValueError("require integer 0 <= resume_offset <= max_iterations")
    return range(offset + 1, end + 1)


def resumed_wavelet_generator(seed: int, levels: int, offset: int, cycle_spin: bool = True):
    """Replay exactly the original per-iteration three-axis CPU randint calls."""
    absolute_iterations(offset, offset)
    if levels < 1:
        raise ValueError("wavelet levels must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed + 7919)
    if cycle_spin:
        for _ in range(offset):
            torch.randint(0, 2 ** levels, (3,), generator=generator)
    return generator


def validate_parameter_groups(model, optimizer):
    """Resolve unnamed groups by tensor identity and require the frozen order."""
    named = dict(model.named_parameters())
    if set(named) != set(PARAMETER_NAMES):
        raise ValueError(f"expected exactly seven parameter names {PARAMETER_NAMES}; got {tuple(named)}")
    if not isinstance(optimizer, torch.optim.Adam) or len(optimizer.param_groups) != 7:
        raise ValueError("continuation requires the original seven-group Adam optimizer")
    identities = {id(p): name for name, p in named.items()}
    group_names = []
    for group in optimizer.param_groups:
        if len(group["params"]) != 1 or id(group["params"][0]) not in identities:
            raise ValueError("each Adam group must contain exactly one original model parameter")
        group_names.append(identities[id(group["params"][0])])
    if tuple(group_names) != PARAMETER_NAMES:
        raise ValueError(f"Adam group order differs from the frozen checkpoint: {group_names}")
    if not all(p.requires_grad for p in named.values()):
        raise ValueError("all seven parameters must participate in forward/backward and global clipping")
    return {name: named[name] for name in PARAMETER_NAMES}


def restore_checkpoint(model, optimizer, scheduler, checkpoint, expected_iteration=4000, gamma=.9995):
    """Validate before mutation, then restore tensors/moments without replacing Parameters.

    The source checkpoint has no scheduler state. ExponentialLR is reconstructed
    from the absolute iteration and saved group LRs, without taking a dummy step.
    """
    named = validate_parameter_groups(model, optimizer)
    iteration = checkpoint.get("iter")
    if iteration != expected_iteration:
        raise ValueError(f"checkpoint iteration {iteration} != required iteration {expected_iteration}")
    model_state = checkpoint["model_state"]
    if set(model_state) != set(PARAMETER_NAMES):
        raise ValueError("checkpoint model must contain exactly the original seven tensors")
    saved = checkpoint["optim_state"]
    if len(saved["param_groups"]) != 7:
        raise ValueError("checkpoint must contain seven Adam groups")
    ids = []
    base_lrs = []
    for name, group, current in zip(PARAMETER_NAMES, saved["param_groups"], optimizer.param_groups):
        param = named[name]
        tensor = model_state[name]
        if tensor.shape != param.shape or tensor.dtype != param.dtype:
            raise ValueError(f"model shape/dtype mismatch for {name}")
        if len(group["params"]) != 1:
            raise ValueError(f"checkpoint Adam group {name} must contain exactly one tensor")
        ident = group["params"][0]
        ids.append(ident)
        state = saved["state"][ident]
        for key in ("exp_avg", "exp_avg_sq"):
            if state[key].shape != param.shape or state[key].dtype != param.dtype:
                raise ValueError(f"Adam {key} shape/dtype mismatch for {name}")
        if float(state["step"]) != expected_iteration:
            raise ValueError(f"Adam step for {name} does not equal checkpoint iteration")
        base_lr = group.get("initial_lr")
        if base_lr is None or not math.isclose(base_lr, current.get("initial_lr", current["lr"]), rel_tol=1e-12):
            raise ValueError(f"initial Adam LR mismatch for {name}")
        if not math.isclose(group["lr"], base_lr * gamma ** expected_iteration, rel_tol=1e-10, abs_tol=1e-14):
            raise ValueError(f"saved Adam LR does not match absolute exponential schedule for {name}")
        base_lrs.append(base_lr)
    if len(set(ids)) != 7 or set(saved["state"]) != set(ids):
        raise ValueError("checkpoint Adam state must map exactly once to each of the seven groups")
    if not isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR) or scheduler.gamma != gamma:
        raise ValueError("continuation requires the original ExponentialLR gamma")
    model.load_state_dict(model_state, strict=True)  # assign=False preserves parameter identity.
    optimizer.load_state_dict(saved)
    scheduler.base_lrs = base_lrs
    scheduler.last_epoch = expected_iteration
    scheduler._step_count = expected_iteration + 1
    scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
    validate_parameter_groups(model, optimizer)
    return {"iteration": iteration, "scheduler_last_epoch": scheduler.last_epoch,
            "scheduler_step_count": scheduler._step_count, "gamma": gamma,
            "group_names": list(PARAMETER_NAMES), "restored_lrs": scheduler.get_last_lr()}


def install_update_mask(mode, model, optimizer):
    """Mask geometry only inside Adam.step, after the original full clipping.

    Adam skips parameters whose grad is None, including their existing moments
    and step counters. A zero LR or zero gradient would still change that state.
    """
    if mode not in ("all", "density"):
        raise ValueError("update mode must be all or density")
    named = validate_parameter_groups(model, optimizer)
    if mode == "all":
        return None

    def before_step(opt, args, kwargs):
        for name in GEOMETRY_NAMES:
            named[name].grad = None

    return optimizer.register_step_pre_hook(before_step)
