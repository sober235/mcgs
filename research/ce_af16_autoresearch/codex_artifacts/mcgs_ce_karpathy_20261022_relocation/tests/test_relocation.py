"""CPU behavior tests for the fixed-count, seven-parameter relocation event."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NAMES = ('positions', 'scales', 'rotations', 'density_real', 'density_imag',
         'density_real_contrast', 'density_imag_contrast')
LEGACY = ROOT.parent / 'mcgs_ce_autoresearch_20261017/nonlocal_autoresearch/trainer/paired_gs/gaussian'


def original_model_types():
    # Load only the original pure model sources; never import the renderer package.
    package = types.ModuleType('_relocation_test_gaussian')
    package.__path__ = [str(LEGACY)]
    sys.modules[package.__name__] = package
    modules = []
    for basename in ('gaussian_model', 'joint_gaussian_model'):
        name = package.__name__ + '.' + basename
        spec = importlib.util.spec_from_file_location(name, LEGACY / (basename + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    return modules[1].JointGaussianModel3D, modules[0].quaternion_to_rotation_matrix


Model, rotate = original_model_types()


def fixture(dtype=torch.float64):
    n = 8
    model = Model(n, (8, 8, 8), torch.device('cpu'),
                  initial_positions=torch.arange(n * 3, dtype=torch.float64).reshape(n, 3) / 30,
                  initial_scales=torch.full((n, 3), .1, dtype=torch.float64),
                  initial_densities_plain=torch.complex(torch.arange(1, n + 1, dtype=torch.float64),
                                                       torch.arange(11, 11 + n, dtype=torch.float64)),
                  initial_densities_contrast=torch.complex(torch.arange(21, 21 + n, dtype=torch.float64),
                                                          -torch.arange(31, 31 + n, dtype=torch.float64)))
    # Original constructor creates float32 quaternions; use one consistent model dtype.
    model.to(dtype=dtype)
    optimizer = torch.optim.Adam([{'params': [getattr(model, name)], 'name': name,
                                   'lr': lr} for name, lr in zip(NAMES, (.0006, .004, .0006, .015, .015, .015, .015))],
                                 betas=(.9, .999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.9995)
    for index, name in enumerate(NAMES):
        parameter = getattr(model, name)
        parameter.grad = torch.full_like(parameter, .25 + index)
    optimizer.step()
    scheduler.step()
    with torch.no_grad():
        model.positions[0] = torch.tensor([.2, .3, .4])
        model.scales[0] = torch.tensor([.4, .1, .2]).log()
        model.rotations[0] = torch.tensor([2 ** -.5, 0., 0., 2 ** -.5]) * 3
        model.density_real[0], model.density_imag[0] = 3., 4.
        model.density_real_contrast[0], model.density_imag_contrast[0] = -5., 12.
        model.positions[7] = 1.4  # untouched rows must never receive a global clamp
    return model, optimizer, scheduler


def digest(value):
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def snapshot(model, optimizer, scheduler):
    return {
        'parameters': {name: getattr(model, name).detach().clone() for name in NAMES},
        'gradients': {name: None if getattr(model, name).grad is None else getattr(model, name).grad.clone() for name in NAMES},
        'optimizer': copy.deepcopy(optimizer.state_dict()),
        'scheduler': copy.deepcopy(scheduler.state_dict()),
        'ids': [(id(getattr(model, name)), id(getattr(model, name).grad),
                 id(optimizer.state[getattr(model, name)]['exp_avg']),
                 id(optimizer.state[getattr(model, name)]['exp_avg_sq'])) for name in NAMES],
    }


def assert_nested_equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def api():
    import relocation
    assert callable(getattr(relocation, 'select_relocations', None)), 'missing selection helper'
    assert callable(getattr(relocation, 'relocate_rows', None)), 'missing fixed-shape transaction'
    return relocation


def test_helpers_exist():
    assert importlib.util.find_spec('relocation') is not None, 'missing pure relocation module'
    api()


def test_selection_stable_disjoint_and_fewer_donors_keeps_requested_exclusions():
    model, _, _ = fixture()
    with torch.no_grad():
        model.scales.zero_()
        for name in NAMES[3:]:
            getattr(model, name).fill_(1.)
    scores = torch.tensor([100., 99., 0., 8., 8., 0., 0., 0.], dtype=torch.float64)
    donors, victims, evidence = api().select_relocations(model, scores, 3)
    assert donors.tolist() == [3, 4]
    assert victims.tolist() == [0, 1]
    assert evidence['requested_victims'] == [0, 1, 2]
    assert evidence['eligible_donor_count'] == 2
    assert evidence['k_eff'] == 2
    assert evidence['requested_k'] == 3
    assert evidence['point_count'] == 8
    assert evidence['donors_sha256'] == digest(donors)
    json.dumps(evidence, allow_nan=False)


def test_proxy_uses_max_complex_magnitude_and_float64_scale_product():
    model, _, _ = fixture(torch.float32)
    with torch.no_grad():
        model.scales.fill_(-40.)  # float32 exp-product would underflow
        for name in NAMES[3:]:
            getattr(model, name).zero_()
        model.density_real[:] = torch.tensor([8., 7., 6., 5., 4., 3., 2., 1.])
        model.density_real_contrast[7] = 20.
    donors, victims, evidence = api().select_relocations(model, torch.ones(8, dtype=torch.float64), 2)
    assert victims.tolist() == [6, 5]
    assert donors.tolist() == [0, 1]
    assert evidence['victim_proxy'][0] > 0


def test_proxy_ranks_complex_magnitudes_rather_than_real_parts():
    model, _, _ = fixture()
    with torch.no_grad():
        model.scales.zero_()
        model.density_real.fill_(7.)
        for name in ('density_imag', 'density_real_contrast', 'density_imag_contrast'):
            getattr(model, name).zero_()
        model.density_real[0], model.density_imag[0] = 3., 4.
        model.density_real[1] = 4.5
        model.density_real[2], model.density_imag_contrast[2] = 1., 6.
    _, victims, evidence = api().select_relocations(model, torch.ones(8), 2)
    assert victims.tolist() == [1, 0]
    assert evidence['victim_proxy'] == [4.5, 5.]


def test_donor_ranking_descends_with_index_ties():
    model, _, _ = fixture()
    with torch.no_grad():
        model.scales.zero_()
        for name in NAMES[3:]:
            getattr(model, name).fill_(1.)
    donors, victims, _ = api().select_relocations(model, torch.tensor([90., 80., 2., 5., 5., 4., 0., 1.]), 2)
    assert victims.tolist() == [0, 1]
    assert donors.tolist() == [3, 4]


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_multiple_pairs_evidence_hashes_are_actual_and_zero_resets_are_recomputable(dtype):
    model, optimizer, scheduler = fixture(dtype)
    before = snapshot(model, optimizer, scheduler)
    donors, victims = torch.tensor([2, 0]), torch.tensor([6, 5])
    event = api().relocate_rows(model, optimizer, scheduler, donors, victims, rotate)
    changed, untouched = [0, 2, 5, 6], [1, 3, 4, 7]
    assert event['donors'] == [2, 0]
    assert event['victims'] == [6, 5]
    assert event['changed_indices'] == changed
    for i, name in enumerate(NAMES):
        p = getattr(model, name)
        assert event['before']['parameters'][name]['id'] == id(p)
        assert event['after']['parameters'][name]['sha256'] == digest(p)
        assert event['after']['parameters'][name]['changed_sha256'] == digest(p[changed])
        assert event['after']['gradients'][name]['id'] == id(p.grad)
        assert event['before']['gradients'][name]['untouched_sha256'] == event['after']['gradients'][name]['untouched_sha256']
        assert event['after']['gradients'][name]['changed_sha256'] == digest(torch.zeros_like(p[changed]))
        for key in ('exp_avg', 'exp_avg_sq'):
            record = event['after']['moments'][name][key]
            assert record['id'] == id(optimizer.state[p][key])
            assert record['untouched_sha256'] == digest(before['optimizer']['state'][i][key][untouched])
            assert record['changed_sha256'] == digest(torch.zeros_like(p[changed]))
    for donor, victim in zip(donors, victims):
        for name in NAMES[1:]:
            assert torch.equal(getattr(model, name)[donor], getattr(model, name)[victim])


def test_exact_two_channel_split_and_untouched_rows_state_and_next_step():
    model, optimizer, scheduler = fixture()
    old = snapshot(model, optimizer, scheduler)
    donors, victims = torch.tensor([0]), torch.tensor([6])
    evidence = api().relocate_rows(model, optimizer, scheduler, donors, victims, rotate)
    changed, untouched = [0, 6], [1, 2, 3, 4, 5, 7]
    expected_scales = torch.tensor([.2, .085, .17], dtype=torch.float64)
    for row, expected in ((0, [.2, .1, .4]), (6, [.2, .5, .4])):
        assert torch.allclose(model.positions[row], torch.tensor(expected, dtype=torch.float64), atol=3e-8, rtol=0)
        assert torch.allclose(model.scales[row], expected_scales.log(), atol=1e-7, rtol=0)
        assert torch.equal(model.rotations[row], old['parameters']['rotations'][0])
        for real, imag in (('density_real', 'density_imag'), ('density_real_contrast', 'density_imag_contrast')):
            parent = torch.complex(old['parameters'][real][0], old['parameters'][imag][0])
            child = torch.complex(getattr(model, real)[row], getattr(model, imag)[row])
            assert torch.allclose(child, parent * (200 / 144.5))
            assert torch.allclose(child / child.abs(), parent / parent.abs())
            assert torch.allclose(2 * child * model.scales[row].exp().prod(), parent * old['parameters']['scales'][0].exp().prod())
    for index, name in enumerate(NAMES):
        p = getattr(model, name)
        assert p.shape[0] == model.num_points == 8
        assert torch.equal(p[untouched], old['parameters'][name][untouched])
        assert torch.equal(p.grad[untouched], old['gradients'][name][untouched])
        assert torch.count_nonzero(p.grad[changed]) == 0
        for key in ('exp_avg', 'exp_avg_sq'):
            moment = optimizer.state[p][key]
            assert torch.count_nonzero(moment[changed]) == 0
            assert torch.equal(moment[untouched], old['optimizer']['state'][index][key][untouched])
        row = evidence['before']['parameters'][name]
        assert row['untouched_sha256'] == digest(old['parameters'][name][untouched])
        assert evidence['after']['parameters'][name]['untouched_sha256'] == row['untouched_sha256']
    assert snapshot(model, optimizer, scheduler)['ids'] == old['ids']
    assert_nested_equal(scheduler.state_dict(), old['scheduler'])
    assert_nested_equal(optimizer.state_dict()['param_groups'], old['optimizer']['param_groups'])
    assert evidence['before']['steps'] == evidence['after']['steps'] == dict.fromkeys(NAMES, 1.)
    assert evidence['changed_indices'] == [0, 6]
    assert evidence['k_eff'] == 1
    assert evidence['before']['point_count'] == evidence['after']['point_count'] == 8
    assert all(all(values.values()) for values in evidence['changed_rows_zero'].values())
    json.dumps(evidence, allow_nan=False)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    optimizer.step()
    scheduler.step()
    assert all(optimizer.state[p]['step'].item() == 2 for p in model.parameters())
    assert scheduler.last_epoch == 2


def test_only_children_are_clamped_and_missing_gradients_remain_missing():
    model, optimizer, scheduler = fixture()
    with torch.no_grad():
        model.positions[0] = torch.tensor([1.2, .95, -1.3])
    model.scales.grad = None
    api().relocate_rows(model, optimizer, scheduler, torch.tensor([0]), torch.tensor([6]), rotate)
    assert model.positions[[0, 6]].abs().max() <= 1.
    assert torch.equal(model.positions[7], torch.full((3,), 1.4, dtype=torch.float64))
    assert model.scales.grad is None


def test_original_unnamed_optimizer_groups_are_preserved():
    model, optimizer, scheduler = fixture()
    for group in optimizer.param_groups:
        del group['name']
    options = copy.deepcopy(optimizer.state_dict()['param_groups'])
    evidence = api().relocate_rows(model, optimizer, scheduler, torch.tensor([0]), torch.tensor([6]), rotate)
    assert_nested_equal(optimizer.state_dict()['param_groups'], options)
    assert evidence['before']['resolved_group_names'] == list(NAMES)


@pytest.mark.parametrize('requested_k,scores', [(0, torch.ones(8, dtype=torch.float64)), (2, torch.zeros(8, dtype=torch.float64))])
def test_zero_event_is_exact_noop(requested_k, scores):
    model, optimizer, scheduler = fixture()
    before = snapshot(model, optimizer, scheduler)
    donors, victims, selection = api().select_relocations(model, scores, requested_k)
    assert donors.numel() == victims.numel() == selection['k_eff'] == 0
    evidence = api().relocate_rows(model, optimizer, scheduler, donors, victims, lambda q: pytest.fail('no-op invoked rotation'))
    assert_nested_equal(snapshot(model, optimizer, scheduler), before)
    assert evidence['before'] == evidence['after']
    assert evidence['changed_indices'] == []


@pytest.mark.parametrize('case', ['overlap', 'duplicate', 'out_of_range', 'negative', 'wrong_dtype', 'different_lengths',
                                 'nonfinite_donor', 'overflow_child', 'underflow_child', 'bad_moment', 'bad_options',
                                 'bad_group', 'bad_rotation', 'bad_gradient', 'bad_step', 'bad_scheduler',
                                 'aliased_moment_parameter', 'aliased_gradient_parameter', 'aliased_moments'])
def test_invalid_transaction_is_atomic(case):
    model, optimizer, scheduler = fixture()
    donors, victims, rotation = torch.tensor([0]), torch.tensor([6]), rotate
    if case == 'overlap': victims = donors.clone()
    if case == 'duplicate': donors, victims = torch.tensor([0, 0]), torch.tensor([5, 6])
    if case == 'out_of_range': victims = torch.tensor([8])
    if case == 'negative': donors = torch.tensor([-1])
    if case == 'wrong_dtype': donors = torch.tensor([0.])
    if case == 'different_lengths': victims = torch.tensor([5, 6])
    if case == 'nonfinite_donor':
        with torch.no_grad(): model.density_imag[0] = float('nan')
    if case == 'overflow_child':
        with torch.no_grad(): model.density_real_contrast[0] = torch.finfo(torch.float64).max
    if case == 'underflow_child':
        with torch.no_grad(): model.scales[0] = -1000.
    if case == 'bad_moment': optimizer.state[model.positions]['exp_avg'] = torch.ones(1)
    if case == 'bad_options': optimizer.param_groups[0]['amsgrad'] = True
    if case == 'bad_group': optimizer.param_groups[0]['params'] = [model.scales]
    if case == 'bad_rotation': rotation = lambda q: torch.full((1, 3, 3), float('inf'), dtype=q.dtype)
    if case == 'bad_gradient': model.positions.grad = torch.sparse_coo_tensor(torch.tensor([[0], [0]]), torch.tensor([1.], dtype=torch.float64), (8, 3))
    if case == 'bad_step': optimizer.state[model.positions]['step'] = torch.tensor(float('nan'))
    if case == 'bad_scheduler': scheduler.optimizer = torch.optim.Adam([model.positions])
    if case == 'aliased_moment_parameter': optimizer.state[model.positions]['exp_avg'] = model.positions.detach()
    if case == 'aliased_gradient_parameter': model.positions.grad = model.positions.detach()
    if case == 'aliased_moments': optimizer.state[model.positions]['exp_avg_sq'] = optimizer.state[model.positions]['exp_avg']
    # Snapshot tensors as bytes so NaNs and sparse malformed inputs compare honestly.
    def atomic_state():
        tensors = {name: digest(getattr(model, name)) for name in NAMES}
        for name in NAMES:
            p = getattr(model, name)
            if p.grad is not None:
                tensors[name + '.grad'] = digest(p.grad.to_dense() if p.grad.is_sparse else p.grad)
            for key, value in optimizer.state[p].items():
                if torch.is_tensor(value): tensors[name + '.' + key] = digest(value)
        return tensors, [id(getattr(model, name)) for name in NAMES], copy.deepcopy(scheduler.state_dict())
    before = atomic_state()
    with pytest.raises((ValueError, TypeError)):
        api().relocate_rows(model, optimizer, scheduler, donors, victims, rotation)
    assert_nested_equal(atomic_state(), before)


@pytest.mark.parametrize('case', ['negative_evidence', 'nan_evidence', 'inf_evidence', 'score_shape', 'score_integer',
                                 'negative_k', 'bool_k', 'large_k', 'nonfinite_proxy'])
def test_selection_rejects_invalid_inputs(case):
    model, _, _ = fixture()
    scores, k = torch.ones(8, dtype=torch.float64), 2
    if case == 'negative_evidence': scores[2] = -1
    if case == 'nan_evidence': scores[2] = float('nan')
    if case == 'inf_evidence': scores[2] = float('inf')
    if case == 'score_shape': scores = scores[:, None]
    if case == 'score_integer': scores = scores.long()
    if case == 'negative_k': k = -1
    if case == 'bool_k': k = True
    if case == 'large_k': k = 9
    if case == 'nonfinite_proxy':
        with torch.no_grad(): model.scales[0] = 1000.
    before = {name: digest(getattr(model, name)) for name in NAMES}
    with pytest.raises((ValueError, TypeError)):
        api().select_relocations(model, scores, k)
    assert {name: digest(getattr(model, name)) for name in NAMES} == before


def complete_storage_snapshot(model, optimizer, scheduler):
    """Include full backing bytes, tensor identities and scalar clocks on rejection."""
    def record(tensor):
        if tensor is None:
            return None
        storage = tensor.untyped_storage()
        raw = torch.empty(0, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),), (1,))
        return (id(tensor), storage.data_ptr(), tensor.storage_offset(), tensor.stride(),
                tuple(tensor.shape), str(tensor.dtype), digest(tensor), digest(raw))
    return {
        'parameters': {name: record(getattr(model, name)) for name in NAMES},
        'gradients': {name: record(getattr(model, name).grad) for name in NAMES},
        'states': {name: {key: record(value) for key, value in optimizer.state[getattr(model, name)].items()}
                   for name in NAMES},
        'clock_values': {name: optimizer.state[getattr(model, name)]['step'].item() for name in NAMES},
        'groups': [{**{key: copy.deepcopy(value) for key, value in group.items() if key != 'params'},
                    'parameter_ids': [id(p) for p in group['params']]} for group in optimizer.param_groups],
        'scheduler': copy.deepcopy(scheduler.state_dict()),
        'object_ids': (id(model), id(optimizer), id(scheduler)),
    }


@pytest.mark.parametrize('case', [
    'expanded_parameter', 'expanded_exp_avg', 'expanded_exp_avg_sq', 'expanded_gradient',
    'strided_overlap_parameter', 'strided_overlap_moment', 'strided_overlap_gradient',
    'step_alias_parameter', 'step_alias_exp_avg', 'step_alias_exp_avg_sq', 'step_alias_gradient',
    'overlapping_parameters', 'overlapping_gradient_views', 'overlapping_differently_strided_parameters',
    'overlapping_differently_strided_moment_gradient',
])
def test_overlapping_writable_storage_and_aliased_clocks_reject_atomically(case):
    model, optimizer, scheduler = fixture()
    p = model.positions
    if case == 'expanded_parameter':
        model.scales.data = model.scales.detach()[:1].expand_as(model.scales)
    if case in ('expanded_exp_avg', 'expanded_exp_avg_sq'):
        key = case.removeprefix('expanded_')
        optimizer.state[p][key] = torch.ones(1, 3, dtype=p.dtype).expand_as(p)
    if case == 'expanded_gradient':
        p.grad = torch.ones(1, 3, dtype=p.dtype).expand_as(p)
    if case.startswith('strided_overlap_'):
        overlap = torch.ones(10, dtype=p.dtype).as_strided((8, 3), (1, 1))
        if case == 'strided_overlap_parameter': model.scales.data = overlap
        if case == 'strided_overlap_moment': optimizer.state[p]['exp_avg'] = overlap
        if case == 'strided_overlap_gradient': p.grad = overlap
    if case.startswith('step_alias_'):
        key = case.removeprefix('step_alias_')
        source = p.detach() if key == 'parameter' else p.grad if key == 'gradient' else optimizer.state[p][key]
        with torch.no_grad(): source[0, 1] = 1.
        optimizer.state[p]['step'] = source[0, 1]
    if case == 'overlapping_parameters':
        model.scales.data = model.positions.detach()
    if case == 'overlapping_gradient_views':
        backing = torch.ones(9, 3, dtype=p.dtype)
        p.grad = backing[:8]
        model.scales.grad = backing[1:]
    if case in ('overlapping_differently_strided_parameters', 'overlapping_differently_strided_moment_gradient'):
        backing = torch.ones(24, dtype=p.dtype)
        left = backing.as_strided((8,), (2,), 0)
        right = backing.as_strided((8,), (3,), 1)
        if case == 'overlapping_differently_strided_parameters':
            model.density_real.data, model.density_imag.data = left, right
        else:
            optimizer.state[model.density_real]['exp_avg'] = left
            model.density_imag.grad = right
    before = complete_storage_snapshot(model, optimizer, scheduler)
    with pytest.raises(ValueError):
        api().relocate_rows(model, optimizer, scheduler, torch.tensor([0]), torch.tensor([6]), rotate)
    assert_nested_equal(complete_storage_snapshot(model, optimizer, scheduler), before)


@pytest.mark.parametrize('layout', ['original_complex_views', 'sliced', 'column_major', 'interleaved_columns',
                                  'disjoint_differently_strided_views'])
def test_nonoverlapping_strided_parameters_moments_and_gradients_remain_supported(layout):
    model, optimizer, scheduler = fixture()
    if layout == 'original_complex_views':
        assert model.density_real.untyped_storage().data_ptr() == model.density_imag.untyped_storage().data_ptr()
        assert model.density_real.stride() == model.density_imag.stride() == (2,)
    elif layout == 'disjoint_differently_strided_views':
        backing = torch.zeros(81, dtype=model.positions.dtype)
        # Congruences have a solution, but outside these finite row ranges.
        real = backing.as_strided((8,), (8,), 0)
        imag = backing.as_strided((8,), (11,), 3)
        real.copy_(model.density_real.detach())
        imag.copy_(model.density_imag.detach())
        model.density_real.data, model.density_imag.data = real, imag
    else:
        def strided_copy(value):
            if value.ndim == 1:
                view = torch.zeros(value.numel() * 2, dtype=value.dtype)[::2]
            elif layout == 'sliced':
                view = torch.zeros(value.shape[0], value.shape[1] * 2, dtype=value.dtype)[:, ::2]
            elif layout == 'column_major':
                view = torch.zeros(value.shape[1], value.shape[0], dtype=value.dtype).T
            else:
                # Nonoverlapping despite intersecting column address intervals.
                width = value.shape[1]
                view = torch.zeros(7 * 3 + (width - 1) * 10 + 1, dtype=value.dtype).as_strided(value.shape, (3, 10))
                if width == 4:
                    # Axis stride3 would collide at the fourth column; use a coprime stride5.
                    view = torch.zeros(7 * 5 + (width - 1) * 11 + 1, dtype=value.dtype).as_strided(value.shape, (5, 11))
            view.copy_(value)
            return view
        for name in NAMES:
            p = getattr(model, name)
            p.data = strided_copy(p.detach())
            p.grad = strided_copy(p.grad)
            for key in ('exp_avg', 'exp_avg_sq'):
                optimizer.state[p][key] = strided_copy(optimizer.state[p][key])
    before = complete_storage_snapshot(model, optimizer, scheduler)
    event = api().relocate_rows(model, optimizer, scheduler, torch.tensor([0]), torch.tensor([6]), rotate)
    after = complete_storage_snapshot(model, optimizer, scheduler)
    assert event['before']['steps'] == event['after']['steps']
    assert event['before']['scheduler'] == event['after']['scheduler']
    for name in NAMES:
        for category in ('parameters', 'gradients'):
            assert before[category][name][:6] == after[category][name][:6]
            assert event['before'][category][name]['untouched_sha256'] == event['after'][category][name]['untouched_sha256']
        for key in ('exp_avg', 'exp_avg_sq'):
            assert before['states'][name][key][:6] == after['states'][name][key][:6]
            assert event['before']['moments'][name][key]['untouched_sha256'] == event['after']['moments'][name][key]['untouched_sha256']
        assert before['states'][name]['step'] == after['states'][name]['step']
    for p in model.parameters():
        p.grad.fill_(1.)
    optimizer.step()
    scheduler.step()
    assert all(optimizer.state[p]['step'].item() == 2. for p in model.parameters())


@pytest.mark.parametrize('case', [
    'gradient_wrappers', 'moment_wrappers', 'parameter_wrappers', 'moment_gradient_wrappers',
    'step_parameter_wrapper', 'step_exp_avg_wrapper', 'step_exp_avg_sq_wrapper', 'step_gradient_wrapper',
    'partial_byte_parameters', 'partial_byte_moment_gradient', 'partial_byte_float32_step', 'misaligned_float64_step',
])
def test_overlapping_physical_bytes_across_distinct_storage_wrappers_reject_atomically(case):
    model, optimizer, scheduler = fixture()
    p = model.positions
    if case in ('gradient_wrappers', 'moment_wrappers', 'parameter_wrappers', 'moment_gradient_wrappers'):
        backing = np.ones((9, 3), dtype=np.float64)
        left, right = torch.from_numpy(backing[:8]), torch.from_numpy(backing[1:])
        assert left.untyped_storage().data_ptr() != right.untyped_storage().data_ptr()
        if case == 'gradient_wrappers': p.grad, model.scales.grad = left, right
        if case == 'moment_wrappers': optimizer.state[p]['exp_avg'], optimizer.state[model.scales]['exp_avg_sq'] = left, right
        if case == 'parameter_wrappers': p.data, model.scales.data = left, right
        if case == 'moment_gradient_wrappers': optimizer.state[p]['exp_avg'], model.scales.grad = left, right
    elif case.startswith('step_'):
        backing = np.ones((8, 3), dtype=np.float64)
        source = torch.from_numpy(backing)
        key = case.removeprefix('step_').removesuffix('_wrapper')
        if key == 'parameter': p.data = source
        elif key == 'gradient': p.grad = source
        else: optimizer.state[p][key] = source
        optimizer.state[p]['step'] = torch.from_numpy(backing[0, 1:2]).reshape(())
        assert source.untyped_storage().data_ptr() != optimizer.state[p]['step'].untyped_storage().data_ptr()
    elif case in ('partial_byte_parameters', 'partial_byte_moment_gradient'):
        backing = bytearray(8 * 3 * 8 + 4)
        shape = (8,) if case == 'partial_byte_parameters' else (8, 3)
        left = torch.from_numpy(np.ndarray(shape, dtype=np.float64, buffer=backing, offset=0))
        right = torch.from_numpy(np.ndarray(shape, dtype=np.float64, buffer=backing, offset=4))
        assert right.data_ptr() - left.data_ptr() == 4 < left.element_size()
        if case == 'partial_byte_parameters': model.density_real.data, model.density_imag.data = left, right
        else: optimizer.state[p]['exp_avg'], model.scales.grad = left, right
    else:
        backing = bytearray(8 * 3 * 8)
        source = torch.from_numpy(np.ndarray((8, 3), dtype=np.float64, buffer=backing))
        step_array = np.ndarray((), dtype=np.float32 if case == 'partial_byte_float32_step' else np.float64,
                                buffer=backing, offset=12 if case == 'partial_byte_float32_step' else 4)
        step_array.fill(1.)
        p.data = source
        optimizer.state[p]['step'] = torch.from_numpy(step_array)
        assert source.data_ptr() != optimizer.state[p]['step'].data_ptr()
    before = complete_storage_snapshot(model, optimizer, scheduler)
    with pytest.raises(ValueError, match='overlap|writable storage'):
        api().relocate_rows(model, optimizer, scheduler, torch.tensor([0]), torch.tensor([6]), rotate)
    assert_nested_equal(complete_storage_snapshot(model, optimizer, scheduler), before)


@pytest.mark.parametrize('layout', ['adjacent', 'interleaved', 'different_strides', 'scalar_in_row_padding'])
def test_disjoint_physical_bytes_across_distinct_storage_wrappers_remain_supported(layout):
    model, optimizer, scheduler = fixture()
    if layout in ('adjacent', 'interleaved'):
        backing = np.ones((16, 3), dtype=np.float64)
        left, right = (backing[:8], backing[8:]) if layout == 'adjacent' else (backing[::2], backing[1::2])
        model.positions.grad, model.scales.grad = torch.from_numpy(left), torch.from_numpy(right)
    elif layout == 'different_strides':
        backing = np.zeros(81, dtype=np.float64)
        left = torch.from_numpy(np.ndarray((8,), dtype=np.float64, buffer=backing, offset=0, strides=(64,)))
        right = torch.from_numpy(np.ndarray((8,), dtype=np.float64, buffer=backing, offset=24, strides=(88,)))
        left.copy_(model.density_real.detach())
        right.copy_(model.density_imag.detach())
        model.density_real.data, model.density_imag.data = left, right
    else:
        backing = bytearray(8 * 4 * 8)
        positions = torch.from_numpy(np.ndarray((8, 3), dtype=np.float64, buffer=backing, strides=(32, 8)))
        positions.copy_(model.positions.detach())
        step_array = np.ndarray((), dtype=np.float64, buffer=backing, offset=24)
        step_array.fill(1.)
        model.positions.data = positions
        optimizer.state[model.positions]['step'] = torch.from_numpy(step_array)
    before = complete_storage_snapshot(model, optimizer, scheduler)
    event = api().relocate_rows(model, optimizer, scheduler, torch.tensor([0]), torch.tensor([6]), rotate)
    after = complete_storage_snapshot(model, optimizer, scheduler)
    assert event['before']['steps'] == event['after']['steps']
    assert event['before']['scheduler'] == event['after']['scheduler']
    for name in NAMES:
        for category in ('parameters', 'gradients'):
            assert before[category][name][:6] == after[category][name][:6]
            assert event['before'][category][name]['untouched_sha256'] == event['after'][category][name]['untouched_sha256']
        for key in ('exp_avg', 'exp_avg_sq'):
            assert before['states'][name][key][:6] == after['states'][name][key][:6]
            assert event['before']['moments'][name][key]['untouched_sha256'] == event['after']['moments'][name][key]['untouched_sha256']
        assert before['states'][name]['step'] == after['states'][name]['step']
    for p in model.parameters():
        p.grad.fill_(1.)
    optimizer.step()
    scheduler.step()
    assert all(optimizer.state[p]['step'].item() == 2. for p in model.parameters())
