"""Strict full-G2 continuation and measured state evidence, for numerical children.

This module is deliberately separate from the stdlib-only supervisor. It imports
Torch, but loads the unchanged original restore source only when restoring.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path

import torch


PARAMETER_NAMES = ('positions', 'scales', 'rotations', 'density_real', 'density_imag',
                   'density_real_contrast', 'density_imag_contrast')
BASE_LRS = (.0006, .004, .0006, .015, .015, .015, .015)
GAMMA = .9995
SOURCE_ITERATION = 4500
_FROZEN_SOURCE = Path('/home/congcongliu/codex_artifacts/mcgs_ce_autoresearch_20261017/nonlocal_autoresearch/trainer/paired_gs/trainers/resume_protocol.py')
RESUME_SOURCE = _FROZEN_SOURCE
RESUME_SOURCE_SHA256 = 'c30f4711c5abd581dda4832ba9d467e91d2158396262827ac13ca9c0907fcf1a'
_G2_RECIPE = {
    'version': 1, 'frequency_t': 1., 'geometry_factor': 2., 'local_factor': 1.,
    'nlambda': 0., 'wavelet_protocol': 'original_single_phase',
    'frequency_manifest_sha256': '5d1809c5d49d97503ee3d82f91144574749fbb7a44302b9fa178587438f41ce2',
    'requires_same_recipe_on_resume': True,
}
_ADAM_OPTIONS = {
    'betas': (.9, .999), 'eps': 1e-8, 'weight_decay': 0,
    'amsgrad': False, 'maximize': False, 'foreach': None,
    'capturable': False, 'differentiable': False, 'fused': None,
    'decoupled_weight_decay': False,
}


def _json(value):
    """Copy ordinary configuration/state values, refusing lossy serialization."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _json(item) for key, item in value.items()}
    raise ValueError('state/config contains a nonfinite or non-JSON value')


def _tensor_record(tensor):
    if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
        raise ValueError('expected a dense state tensor')
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return {'tensor_id': id(tensor), 'data_ptr': tensor.data_ptr(),
            'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
            'device': str(tensor.device), 'sha256': hashlib.sha256(raw).hexdigest()}


def _named_model(model):
    named = dict(model.named_parameters())
    if set(named) != set(PARAMETER_NAMES) or set(model.state_dict()) != set(PARAMETER_NAMES):
        raise ValueError('expected exactly seven original tensors; extra networks/buffers are forbidden')
    if len({id(param) for param in named.values()}) != 7:
        raise ValueError('seven distinct original Parameters are required')
    return {name: named[name] for name in PARAMETER_NAMES}


def snapshot_training_state(trainer):
    """Capture actual tensor bytes/identities, group membership and full clocks.

    Empty pre-restore optimizer state is represented explicitly. No update, RNG
    draw, gradient clearing, or mutation occurs while taking this snapshot.
    """
    named = _named_model(trainer.model)
    parameters = {}
    for name, param in named.items():
        parameters[name] = {**_tensor_record(param), 'requires_grad': param.requires_grad,
                            'gradient': None if param.grad is None else _tensor_record(param.grad)}
    optimizer = getattr(trainer, 'optimizer', None)
    opt_record = None
    if optimizer is not None:
        identities = {id(param): name for name, param in named.items()}
        groups = []
        for group in optimizer.param_groups:
            members = [identities.get(id(param)) for param in group['params']]
            if len(members) != 1 or members[0] is None:
                raise ValueError('optimizer group does not contain one original Parameter')
            groups.append({'name': members[0], 'parameter_ids': [id(param) for param in group['params']],
                           'options': _json({key: val for key, val in group.items() if key != 'params'})})
        if [group['name'] for group in groups] != list(PARAMETER_NAMES):
            raise ValueError('optimizer groups differ from canonical original order')
        if any(id(param) not in identities for param in optimizer.state):
            raise ValueError('optimizer has state for a foreign Parameter')
        states = {}
        for name, param in named.items():
            state = optimizer.state.get(param, {})
            states[name] = {}
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    states[name][key] = _tensor_record(value)
                    if key == 'step':
                        if value.numel() != 1:
                            raise ValueError('Adam step must be scalar')
                        states[name][key]['value'] = float(value.detach().cpu())
                else:
                    states[name][key] = _json(value)
        opt_record = {'type': type(optimizer).__module__ + '.' + type(optimizer).__name__,
                      'id': id(optimizer), 'groups': groups, 'states': states,
                      'defaults': _json(optimizer.defaults)}
    scheduler = getattr(trainer, 'scheduler', None)
    scheduler_record = None if scheduler is None else {
        'type': type(scheduler).__module__ + '.' + type(scheduler).__name__,
        'id': id(scheduler), 'state': _json(scheduler.state_dict()),
        'optimizer_id': id(scheduler.optimizer)}
    return {'schema': 'full_g2_training_state_v1', 'parameter_names': list(PARAMETER_NAMES),
            'parameters': parameters, 'optimizer': opt_record, 'scheduler': scheduler_record,
            'config': _json(trainer.config)}


def _load_original_resume():
    if Path(RESUME_SOURCE).resolve() != _FROZEN_SOURCE:
        raise ValueError('resume source path differs from the frozen original')
    digest = hashlib.sha256(RESUME_SOURCE.read_bytes()).hexdigest()
    if digest != RESUME_SOURCE_SHA256:
        raise ValueError('original resume source hash changed')
    spec = importlib.util.spec_from_file_location('_relocation_original_resume', RESUME_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.PARAMETER_NAMES != PARAMETER_NAMES:
        raise ValueError('original resume parameter interface changed')
    return module, {'path': str(RESUME_SOURCE), 'sha256': digest,
                    'function': 'restore_checkpoint', 'expected_iteration': SOURCE_ITERATION,
                    'gamma': GAMMA}


def _validate_checkpoint(trainer, checkpoint, named):
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
            'iter', 'model_state', 'optim_state', 'frequency_adam_recipe'}:
        raise ValueError('expected the exact original G2 checkpoint schema')
    if type(checkpoint['iter']) is not int or checkpoint['iter'] != SOURCE_ITERATION:
        raise ValueError('G2 source iteration must be4500')
    recipe = checkpoint['frequency_adam_recipe']
    if not isinstance(recipe, dict) or recipe != _G2_RECIPE or any(
            type(recipe[key]) is not type(value) for key, value in _G2_RECIPE.items()):
        raise ValueError('checkpoint does not have the frozen full G2 frequency/geometry recipe')
    config = trainer.config
    if not isinstance(config, dict) or not isinstance(config.get('training'), dict):
        raise ValueError('training configuration must be a mapping')
    training = config.get('training', {})
    end = training.get('max_iterations')
    if (type(training.get('resume_offset')) is not int or training['resume_offset'] != SOURCE_ITERATION
            or type(end) is not int or not SOURCE_ITERATION <= end <= 5000
            or type(training.get('seed')) is not int or training['seed'] != 42
            or training.get('lr_scheduler') != {'type': 'exponential', 'gamma': GAMMA}):
        raise ValueError('training configuration must retain G2 seed/4500 offset/exponential schedule')
    model_state = checkpoint['model_state']
    if not isinstance(model_state, dict) or set(model_state) != set(PARAMETER_NAMES):
        raise ValueError('checkpoint model must contain exactly the original seven tensors')
    n = named['positions'].shape[0] if named['positions'].ndim else 0
    if not n:
        raise ValueError('original point cloud must be nonempty')
    device = named['positions'].device
    for index, (name, param) in enumerate(named.items()):
        shape = (n, 3) if index < 2 else (n, 4) if index == 2 else (n,)
        value = model_state[name]
        if (tuple(param.shape) != shape or param.dtype != torch.float32 or param.device != device
                or not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or value.dtype != param.dtype or value.layout != torch.strided
                or not bool(torch.isfinite(value).all())):
            raise ValueError(f'original model shape/dtype/device/finiteness differs for {name}')
    saved = checkpoint['optim_state']
    if not isinstance(saved, dict) or set(saved) != {'state', 'param_groups'}:
        raise ValueError('invalid original Adam schema')
    groups, states = saved['param_groups'], saved['state']
    if (not isinstance(groups, list) or len(groups) != 7 or not isinstance(states, dict)
            or set(states) != set(range(7)) or any(type(key) is not int for key in states)):
        raise ValueError('Adam must have exactly seven canonical groups and state entries')
    for index, (name, base_lr, group) in enumerate(zip(PARAMETER_NAMES, BASE_LRS, groups)):
        if not isinstance(group, dict) or set(group) != set(_ADAM_OPTIONS) | {'params', 'lr', 'initial_lr'}:
            raise ValueError(f'Adam group option schema differs for {name}')
        # Equality alone admits scalar Tensors and other numeric lookalikes.
        # Validate the complete saved group before original restore can copy
        # model bytes or install options that the final snapshot cannot encode.
        _json(group)
        if group['params'] != [index] or any(type(key) is not int for key in group['params']):
            raise ValueError(f'Adam group membership/order differs for {name}')
        for key, expected in _ADAM_OPTIONS.items():
            value = group[key]
            if key == 'betas':
                if (not isinstance(value, (tuple, list)) or len(value) != 2
                        or any(type(beta) not in (int, float) or not math.isfinite(beta) for beta in value)
                        or tuple(value) != expected):
                    raise ValueError(f'Adam betas differ for {name}')
            elif type(expected) in (int, float):
                if type(value) not in (int, float) or not math.isfinite(value) or value != expected:
                    raise ValueError(f'Adam primitive numeric {key} differs for {name}')
            elif value != expected or (expected is None and value is not None) or (
                    isinstance(expected, bool) and type(value) is not bool):
                raise ValueError(f'Adam {key} differs for {name}')
        for key, expected in (('initial_lr', base_lr), ('lr', base_lr * GAMMA ** SOURCE_ITERATION)):
            value = group[key]
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not math.isclose(value, expected, rel_tol=1e-12, abs_tol=0)):
                raise ValueError(f'Adam saved {key} differs for {name}; never reapply geometryfactor2')
        state = states[index]
        if not isinstance(state, dict) or set(state) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError(f'Adam moment schema differs for {name}')
        step = state['step']
        if (not isinstance(step, torch.Tensor) or step.shape != torch.Size([])
                or step.dtype != torch.float32 or float(step) != SOURCE_ITERATION):
            raise ValueError(f'Adam scalar step differs for {name}')
        for key in ('exp_avg', 'exp_avg_sq'):
            value = state[key]
            if (not isinstance(value, torch.Tensor) or value.shape != named[name].shape
                    or value.dtype != named[name].dtype or value.layout != torch.strided
                    or not bool(torch.isfinite(value).all())
                    or (key == 'exp_avg_sq' and bool((value < 0).any()))):
                raise ValueError(f'Adam {key} shape/dtype/value differs for {name}')


def restore_training_state(trainer, checkpoint):
    """Unfreeze seven original Parameters and restore exact G2 Adam at4500.

    All source/schema/config checks precede live mutation. Construct rates already
    include G2's historical geometryfactor2; there is no LR intervention, dummy
    optimizer/scheduler update, or network installation here.
    """
    original, source = _load_original_resume()
    named = _named_model(trainer.model)
    _validate_checkpoint(trainer, checkpoint, named)
    before = snapshot_training_state(trainer)
    optimizer = torch.optim.Adam(
        [{'params': [named[name]], 'lr': lr} for name, lr in zip(PARAMETER_NAMES, BASE_LRS)],
        betas=(.9, .999), eps=1e-8, weight_decay=0, amsgrad=False,
        foreach=None, maximize=False, capturable=False, differentiable=False, fused=None)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)
    for param in named.values():
        param.requires_grad_(True)
    record = original.restore_checkpoint(trainer.model, optimizer, scheduler, checkpoint,
                                         expected_iteration=SOURCE_ITERATION, gamma=GAMMA)
    trainer.optimizer, trainer.scheduler = optimizer, scheduler
    return {**record, 'source': source, 'recipe': dict(_G2_RECIPE),
            'base_lrs': list(BASE_LRS), 'before': before, 'after': snapshot_training_state(trainer)}
