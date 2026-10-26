"""CPU continuation evidence; no MRI arrays, renderer, or synthetic gates."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


NAMES = ('positions', 'scales', 'rotations', 'density_real', 'density_imag',
         'density_real_contrast', 'density_imag_contrast')
BASE_LRS = (.0006, .004, .0006, .015, .015, .015, .015)
SOURCE = Path('/home/congcongliu/codex_artifacts/mcgs_ce_autoresearch_20261017/nonlocal_autoresearch/trainer/paired_gs/trainers/resume_protocol.py')
MANIFEST_HASH = '5d1809c5d49d97503ee3d82f91144574749fbb7a44302b9fa178587438f41ce2'


def runtime():
    path = Path(__file__).parents[1] / 'relocation_runtime.py'
    if not path.is_file():
        pytest.fail('strict relocation continuation utility is not implemented')
    spec = importlib.util.spec_from_file_location('_relocation_runtime_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TinySeven(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for index, name in enumerate(NAMES):
            shape = (5, 3) if index < 2 else (5, 4) if index == 2 else (5,)
            self.register_parameter(name, torch.nn.Parameter(
                torch.arange(torch.tensor(shape).prod(), dtype=torch.float32).reshape(shape) / 50 + .2))


def update(trainer):
    trainer.optimizer.zero_grad(set_to_none=True)
    sum(p.square().sum() for p in trainer.model.parameters()).backward()
    trainer.optimizer.step()
    trainer.scheduler.step()


@pytest.fixture
def fixture():
    source = TinySeven()
    optimizer = torch.optim.Adam([{'params': [getattr(source, name)], 'lr': lr, 'name': name}
                                  for name, lr in zip(NAMES, BASE_LRS)],
                                 betas=(.9, .999), eps=1e-8, weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.9995)
    update(SimpleNamespace(model=source, optimizer=optimizer, scheduler=scheduler))
    checkpoint = {'iter': 4500, 'model_state': copy.deepcopy(source.state_dict()),
                  'optim_state': copy.deepcopy(optimizer.state_dict()),
                  'frequency_adam_recipe': {
                      'version': 1, 'frequency_t': 1., 'geometry_factor': 2.,
                      'local_factor': 1., 'nlambda': 0.,
                      'wavelet_protocol': 'original_single_phase',
                      'frequency_manifest_sha256': MANIFEST_HASH, 'requires_same_recipe_on_resume': True}}
    for group, lr in zip(checkpoint['optim_state']['param_groups'], BASE_LRS):
        group['lr'] = lr * .9995 ** 4500
        checkpoint['optim_state']['state'][group['params'][0]]['step'].fill_(4500)
        # Inherited groups are unnamed; snapshots record canonical identities.
        group.pop('name')
    model = TinySeven()
    for param in model.parameters():
        param.requires_grad_(False)
    trainer = SimpleNamespace(model=model, optimizer=None, scheduler=None,
                              config={'training': {'seed': 42, 'resume_offset': 4500,
                                                   'max_iterations': 4502,
                                                   'lr_scheduler': {'type': 'exponential', 'gamma': .9995}},
                                      'loss': {'wav_levels': 1}})
    return trainer, checkpoint


def test_restore_uses_original_source_and_real_saved_state_without_update(fixture):
    trainer, checkpoint = fixture
    module = runtime()
    identities = {name: id(getattr(trainer.model, name)) for name in NAMES}
    config_before = copy.deepcopy(trainer.config)
    rng_before = torch.get_rng_state().clone()
    record = module.restore_training_state(trainer, checkpoint)
    assert record['iteration'] == 4500
    assert record['group_names'] == list(NAMES)
    assert record['source']['path'] == str(SOURCE)
    assert record['source']['sha256'] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert record['source']['sha256'] == 'c30f4711c5abd581dda4832ba9d467e91d2158396262827ac13ca9c0907fcf1a'
    assert trainer.config == config_before
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert all('name' not in group for group in trainer.optimizer.param_groups)
    assert [group['name'] for group in record['after']['optimizer']['groups']] == list(NAMES)
    assert trainer.scheduler.last_epoch == 4500 and trainer.scheduler._step_count == 4501
    assert trainer.scheduler.base_lrs == list(BASE_LRS)
    for name, group, saved_group in zip(NAMES, trainer.optimizer.param_groups, checkpoint['optim_state']['param_groups']):
        param = getattr(trainer.model, name)
        assert id(param) == identities[name] and param.requires_grad
        assert group['params'] == [param] and group['lr'] == saved_group['lr']
        assert torch.equal(param, checkpoint['model_state'][name])
        state = trainer.optimizer.state[param]
        saved = checkpoint['optim_state']['state'][saved_group['params'][0]]
        assert float(state['step']) == 4500
        assert torch.equal(state['exp_avg'], saved['exp_avg'])
        assert torch.equal(state['exp_avg_sq'], saved['exp_avg_sq'])
    assert tuple(dict(trainer.model.named_parameters())) == NAMES
    before = module.snapshot_training_state(trainer)
    after = module.snapshot_training_state(trainer)
    assert before == after == record['after']
    assert record['before']['optimizer'] is None
    json.dumps(record, allow_nan=False)


def test_snapshots_measure_tensor_gradient_moment_ids_hashes_and_two_real_steps(fixture):
    trainer, checkpoint = fixture
    module = runtime()
    module.restore_training_state(trainer, checkpoint)
    before = module.snapshot_training_state(trainer)
    for _ in range(2):
        update(trainer)
    after = module.snapshot_training_state(trainer)
    assert after['schema'] == 'full_g2_training_state_v1'
    assert after['parameter_names'] == list(NAMES)
    assert after['scheduler']['state']['last_epoch'] == 4502
    assert after['scheduler']['state']['_step_count'] == 4503
    assert after['config'] == trainer.config
    for name, group in zip(NAMES, trainer.optimizer.param_groups):
        param = getattr(trainer.model, name)
        assert after['parameters'][name]['tensor_id'] == id(param) == before['parameters'][name]['tensor_id']
        assert after['parameters'][name]['sha256'] != before['parameters'][name]['sha256']
        assert after['parameters'][name]['gradient']['tensor_id'] == id(param.grad)
        assert after['parameters'][name]['gradient']['sha256'] == hashlib.sha256(param.grad.numpy().tobytes()).hexdigest()
        assert after['optimizer']['states'][name]['step']['value'] == 4502
        for key in ('exp_avg', 'exp_avg_sq'):
            state = trainer.optimizer.state[param][key]
            observed = after['optimizer']['states'][name][key]
            assert observed['tensor_id'] == id(state) == before['optimizer']['states'][name][key]['tensor_id']
            assert observed['sha256'] == hashlib.sha256(state.numpy().tobytes()).hexdigest()
        assert group['lr'] == pytest.approx(group['initial_lr'] * .9995 ** 4502, rel=1e-12)
    assert after['optimizer']['groups'][0]['parameter_ids'] == [id(trainer.model.positions)]
    assert after['optimizer']['groups'][0]['options']['betas'] == [.9, .999]
    assert before != after
    json.dumps(after, allow_nan=False)


@pytest.mark.parametrize('defect', [
    'iteration', 'recipe_missing', 'recipe_geometry', 'recipe_manifest', 'recipe_extra',
    'model_missing', 'model_extra', 'model_dtype', 'model_shape', 'model_nonfinite',
    'group_missing', 'group_order', 'group_duplicate', 'state_extra', 'moment_missing',
    'moment_shape', 'moment_dtype', 'moment_nonfinite', 'moment_negative',
    'state_extra_key', 'step_wrong', 'step_nonscalar', 'step_dtype',
    'lr', 'base_lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'maximize',
    'capturable', 'differentiable', 'fused', 'foreach', 'decoupled_weight_decay', 'group_extra',
])
def test_invalid_checkpoint_fails_before_live_mutation(fixture, defect):
    trainer, checkpoint = fixture
    module = runtime()
    group = checkpoint['optim_state']['param_groups'][0]
    state = checkpoint['optim_state']['state'][group['params'][0]]
    if defect == 'iteration': checkpoint['iter'] = 4000
    elif defect == 'recipe_missing': checkpoint.pop('frequency_adam_recipe')
    elif defect == 'recipe_geometry': checkpoint['frequency_adam_recipe']['geometry_factor'] = 1.
    elif defect == 'recipe_manifest': checkpoint['frequency_adam_recipe']['frequency_manifest_sha256'] = '0' * 64
    elif defect == 'recipe_extra': checkpoint['frequency_adam_recipe']['extra'] = 1
    elif defect == 'model_missing': checkpoint['model_state'].pop('scales')
    elif defect == 'model_extra': checkpoint['model_state']['network.weight'] = torch.ones(1)
    elif defect == 'model_dtype': checkpoint['model_state']['scales'] = checkpoint['model_state']['scales'].double()
    elif defect == 'model_shape': checkpoint['model_state']['scales'] = torch.ones(5, 2)
    elif defect == 'model_nonfinite': checkpoint['model_state']['scales'][0, 0] = float('nan')
    elif defect == 'group_missing': checkpoint['optim_state']['param_groups'].pop()
    elif defect == 'group_order': checkpoint['optim_state']['param_groups'].reverse()
    elif defect == 'group_duplicate': checkpoint['optim_state']['param_groups'][1]['params'] = group['params'][:]
    elif defect == 'state_extra': checkpoint['optim_state']['state'][777] = copy.deepcopy(state)
    elif defect == 'moment_missing': state.pop('exp_avg')
    elif defect == 'moment_shape': state['exp_avg'] = torch.ones(5, 2)
    elif defect == 'moment_dtype': state['exp_avg'] = state['exp_avg'].double()
    elif defect == 'moment_nonfinite': state['exp_avg'][0, 0] = float('inf')
    elif defect == 'moment_negative': state['exp_avg_sq'][0, 0] = -1
    elif defect == 'state_extra_key': state['max_exp_avg_sq'] = state['exp_avg_sq'].clone()
    elif defect == 'step_wrong': state['step'].fill_(4499)
    elif defect == 'step_nonscalar': state['step'] = torch.full((2,), 4500.)
    elif defect == 'step_dtype': state['step'] = torch.tensor(4500, dtype=torch.int64)
    elif defect == 'lr': group['lr'] *= 2
    elif defect == 'base_lr': group['initial_lr'] *= 2
    elif defect == 'betas': group['betas'] = (.8, .999)
    elif defect == 'eps': group['eps'] = 1e-6
    elif defect == 'weight_decay': group['weight_decay'] = .01
    elif defect == 'foreach': group[defect] = True
    elif defect == 'fused': group[defect] = True
    elif defect == 'group_extra': group['custom_optimizer_option'] = True
    else: group[defect] = True
    before = module.snapshot_training_state(trainer)
    with pytest.raises(ValueError):
        module.restore_training_state(trainer, checkpoint)
    assert module.snapshot_training_state(trainer) == before


def test_extra_model_network_is_refused_before_any_change(fixture):
    trainer, checkpoint = fixture
    trainer.model.add_module('residual_field', torch.nn.Linear(1, 1))
    with pytest.raises(ValueError):
        runtime().restore_training_state(trainer, checkpoint)
    assert not trainer.model.positions.requires_grad and trainer.optimizer is None


def test_wrong_source_is_refused_before_any_change(fixture, monkeypatch, tmp_path):
    trainer, checkpoint = fixture
    module = runtime()
    alternate = tmp_path / 'resume_protocol.py'
    alternate.write_text(SOURCE.read_text())
    monkeypatch.setattr(module, 'RESUME_SOURCE', alternate)
    with pytest.raises(ValueError):
        module.restore_training_state(trainer, checkpoint)
    assert trainer.optimizer is None and not trainer.model.positions.requires_grad


@pytest.mark.parametrize('key,value', [('resume_offset', 4000), ('seed', 43), ('max_iterations', 5001)])
def test_wrong_training_clock_config_is_refused(fixture, key, value):
    trainer, checkpoint = fixture
    trainer.config['training'][key] = value
    with pytest.raises(ValueError):
        runtime().restore_training_state(trainer, checkpoint)
    assert trainer.optimizer is None and not trainer.model.positions.requires_grad


def test_invalid_restore_preserves_existing_optimizer_gradients_and_clocks(fixture):
    trainer, checkpoint = fixture
    module = runtime()
    module.restore_training_state(trainer, checkpoint)
    update(trainer)
    before = module.snapshot_training_state(trainer)
    checkpoint['optim_state']['param_groups'][0]['eps'] = 1e-5
    with pytest.raises(ValueError):
        module.restore_training_state(trainer, checkpoint)
    assert before == module.snapshot_training_state(trainer)


def test_restore_records_existing_frozen_baseline_optimizer(fixture):
    trainer, checkpoint = fixture
    trainer.optimizer = torch.optim.Adam([
        {'params': [getattr(trainer.model, name)], 'lr': lr}
        for name, lr in zip(NAMES, (.0003, .002, .0003, .015, .015, .015, .015))])
    trainer.scheduler = torch.optim.lr_scheduler.ExponentialLR(trainer.optimizer, gamma=.9995)
    module = runtime()
    before = module.snapshot_training_state(trainer)
    restored = module.restore_training_state(trainer, checkpoint)
    assert restored['before'] == before
    assert restored['before']['optimizer']['states'] == {name: {} for name in NAMES}
    assert restored['after']['optimizer']['groups'][0]['options']['initial_lr'] == .0006


@pytest.mark.parametrize('defect', ['bool_state_key', 'nondict_training', 'nondict_config'])
def test_malformed_schema_is_rejected_before_writes(fixture, defect):
    trainer, checkpoint = fixture
    if defect == 'bool_state_key':
        checkpoint['optim_state']['state'][False] = checkpoint['optim_state']['state'].pop(0)
    elif defect == 'nondict_training':
        trainer.config['training'] = None
    else:
        trainer.config = None
    with pytest.raises(ValueError):
        runtime().restore_training_state(trainer, checkpoint)
    assert trainer.optimizer is None and not trainer.model.positions.requires_grad


@pytest.mark.parametrize('defect', [
    'tensor_weight_decay', 'tensor_eps', 'tensor_beta0', 'tensor_beta1',
    'complex_weight_decay', 'bool_weight_decay',
])
@pytest.mark.parametrize('active_optimizer', [False, True])
def test_equal_valued_nonprimitive_adam_options_refuse_without_live_mutation(
        fixture, defect, active_optimizer):
    trainer, checkpoint = fixture
    module = runtime()
    if active_optimizer:
        module.restore_training_state(trainer, copy.deepcopy(checkpoint))
        update(trainer)
    # Checkpoint differs visibly from the live model, so an early model copy
    # cannot hide behind an exception from the eventual snapshot serializer.
    checkpoint['model_state']['positions'].add_(3)
    group = checkpoint['optim_state']['param_groups'][0]
    if defect == 'tensor_weight_decay':
        group['weight_decay'] = torch.tensor(0.)
    elif defect == 'tensor_eps':
        group['eps'] = torch.tensor(1e-8, dtype=torch.float64)
    elif defect.startswith('tensor_beta'):
        index = int(defect[-1])
        betas = list(group['betas'])
        betas[index] = torch.tensor(betas[index], dtype=torch.float64)
        group['betas'] = tuple(betas)
    elif defect == 'complex_weight_decay':
        group['weight_decay'] = complex(0.)
    else:
        group['weight_decay'] = False
    before = module.snapshot_training_state(trainer)
    with pytest.raises(ValueError):
        module.restore_training_state(trainer, checkpoint)
    assert module.snapshot_training_state(trainer) == before
