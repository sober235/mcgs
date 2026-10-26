"""Real CPU fit math; adapters replace only the original CUDA rendering boundary."""
import importlib
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def api():
    assert (ROOT / 'representation_runner.py').is_file(), 'bounded representation runner is missing'
    return importlib.import_module('representation_runner')


class TinySource(torch.nn.Module):
    def __init__(self, duplicate=False):
        super().__init__()
        self.positions = torch.nn.Parameter(torch.tensor([[.1, .2, .3], [.1 if duplicate else -.2, .2, .3]]), requires_grad=False)
        self.scales = torch.nn.Parameter(torch.tensor([[-3., -4., -5.], [-3., -4., -5.]]), requires_grad=False)
        self.rotations = torch.nn.Parameter(torch.tensor([[1., 0., 0., 0.]] * 2), requires_grad=False)
        for name, value in [('density_real', [1., 4.]), ('density_imag', [1., 2.]),
                            ('density_real_contrast', [2., 7.]), ('density_imag_contrast', [1., -3.])]:
            setattr(self, name, torch.nn.Parameter(torch.tensor(value), requires_grad=False))

    def get_density(self, channel):
        suffix = '' if channel == 'plain' else '_contrast'
        return torch.complex(getattr(self, 'density_real' + suffix), getattr(self, 'density_imag' + suffix))


def tiny_render(model, voxelizer, density):
    # Nontrivial complex linear map: gradients and per-contrast relative energies matter.
    return torch.stack((2 * density[0] + .3j * density[1], density[1] - .1j * density[0])).reshape(1, 1, 2)


def anchors(image, scan):
    return dict(scan=scan, factor={'pre': .9997583823636317, 'post': 1.0003600786949414}[scan],
                array_equal=True, relative_l2=0., tolerance=1e-5, passed=True,
                anchor=f'/tiny/{scan}.npy', anchor_sha256='0' * 64)


def run_tiny(directory, encoder='mlp', *, full=False, duplicate=False, **kwargs):
    module = api()
    render = kwargs.pop('render', tiny_render)
    config = dict(module.FIT_CONFIG)
    if not full:
        config.update(coefficient_updates=3, coefficient_batch=4, raw_updates=2)
    if full:
        # Test-only adapter for identical source features: exact shared network
        # output, expanded through autograd, with real replacement-batch losses.
        # The public producer has no CPU mode or alternative field architecture.
        from full_amplitude import FullAmplitudeField
        from unittest.mock import patch
        original = FullAmplitudeField._magnitude_chunk
        def repeated(self, xyz, logs):
            assert torch.equal(xyz, xyz[:1].expand_as(xyz))
            assert torch.equal(logs, logs[:1].expand_as(logs))
            return original(self, xyz[:1], logs[:1]).expand(xyz.shape[0], -1)
        with patch.object(FullAmplitudeField, '_magnitude_chunk', repeated):
            return module.run_diagnostic(TinySource(duplicate), None, directory, stage='fit', encoder=encoder,
                                         config=config, render=render, verify_anchor=anchors, **kwargs)
    return module.run_diagnostic(TinySource(duplicate), None, directory, stage='fit', encoder=encoder,
                                 config=config, render=render, verify_anchor=anchors, **kwargs)


@pytest.mark.parametrize('encoder', ['mlp', 'grid'])
def test_real_fit_counts_updates_resets_adam_and_preserves_source_and_phase(tmp_path, encoder):
    result = run_tiny(tmp_path, encoder)
    assert result['updates_observed'] == {'coefficient': 3, 'raw': 2}
    assert result['work'] == {'gaussian_forwards': 12, 'gaussian_vjps': 4}
    assert result['optimizer_phases']['coefficient']['state_entries_initial'] == 0
    assert result['optimizer_phases']['coefficient']['step_final'] == 3
    assert result['optimizer_phases']['raw']['state_entries_initial'] == 0
    assert result['optimizer_phases']['raw']['step_final'] == 2
    assert result['source_state_before'] == result['source_state_after']
    assert result['field_invariants_before'] == result['field_invariants_after']
    assert result['phase_delta_zero'] and result['gradients_finite']
    assert result['amplitude_parameters_changed']
    assert result['field_metadata']['width'] == 64 and result['field_metadata']['bands'] == 8
    assert result['field_metadata']['encoder'] == encoder
    checkpoint = torch.load(result['final_checkpoint']['path'], weights_only=True)
    assert checkpoint['checkpoint_kind'] == 'final_fixed_budget'
    assert checkpoint['updates_observed'] == {'coefficient': 3, 'raw': 2}
    assert not any('density' in k for k in checkpoint['field_state'])
    trace = [__import__('json').loads(v) for v in (tmp_path / 'step_trace.jsonl').read_text().splitlines()]
    assert [v['phase'] for v in trace] == ['coefficient'] * 3 + ['raw'] * 2
    assert all(v['gradients_finite'] and v['amplitude_gradients_observed'] > 0 for v in trace)
    assert all(v['clip_max_norm'] == 1. for v in trace if v['phase'] == 'raw')


def test_two_channel_raw_loss_accumulates_double_precision_and_uses_each_energy():
    module = api()
    prediction = torch.tensor([1 + 3j, 4 - 2j], dtype=torch.complex64, requires_grad=True)
    teacher = torch.tensor([1 + 1j, 2 - 2j], dtype=torch.complex64)
    value = module.relative_squared_error(prediction, teacher)
    assert value.dtype == torch.float64
    assert value.item() == pytest.approx(.8)  # (4+4)/(2+8)
    value.backward()
    assert torch.allclose(prediction.grad, torch.tensor([.0 + .4j, .4 + 0j]))
    with pytest.raises(ValueError): module.relative_squared_error(prediction, torch.zeros_like(teacher))


def test_finite_low_capacity_fit_is_completed_negative_not_numerical_failure(tmp_path):
    result = run_tiny(tmp_path, duplicate=True)
    assert result['software_complete'] is True
    assert result['representation_admitted'] is False
    assert max(result['raw_relative_l2']['final'].values()) > .01


def test_render_control_never_builds_field_or_optimizer_and_requires_anchor(tmp_path):
    module = api()
    result = module.run_diagnostic(TinySource(), None, tmp_path, stage='render_control', encoder='none',
                                   render=tiny_render, verify_anchor=anchors)
    assert result['work'] == {'gaussian_forwards': 2, 'gaussian_vjps': 0}
    assert result['updates_observed'] == {'coefficient': 0, 'raw': 0}
    assert result['final_checkpoint'] is None and result['representation_admitted'] is None
    def wrong(image, scan):
        return {**anchors(image, scan), 'relative_l2': .1, 'passed': True}
    with pytest.raises(ValueError, match='anchor'):
        module.run_diagnostic(TinySource(), None, tmp_path / 'bad', stage='render_control', encoder='none',
                              render=tiny_render, verify_anchor=wrong)


@pytest.mark.parametrize('failure', ['detached', 'nonfinite', 'source_mutation'])
def test_fit_rejects_missing_vjp_nonfinite_or_source_mutation(tmp_path, failure):
    def broken(model, voxelizer, density):
        image = tiny_render(model, voxelizer, density)
        if density.requires_grad:
            if failure == 'detached': return image.detach()
            if failure == 'nonfinite': return image * float('nan')
            if failure == 'source_mutation':
                with torch.no_grad(): model.positions[0, 0] += 1
        return image
    with pytest.raises((ValueError, RuntimeError)):
        run_tiny(tmp_path, render=broken)


def test_actual_raw_update_loss_is_half_sum_of_independent_channel_relative_energies(tmp_path):
    module = api(); source = TinySource()
    # LR zero isolates arithmetic at the known initial constant-modulus field.
    config = {**module.FIT_CONFIG, 'coefficient_updates': 1, 'coefficient_batch': 4, 'raw_updates': 1, 'lr': 0.}
    expected = {}
    for scan, channel in [('pre', 'plain'), ('post', 'contrast')]:
        density = source.get_density(channel)
        initial = density / density.abs() * density.abs().min()
        teacher = tiny_render(source, None, density)
        difference = tiny_render(source, None, initial) - teacher
        expected[scan] = ((difference.real.double().square() + difference.imag.double().square()).sum()
                          / (teacher.real.double().square() + teacher.imag.double().square()).sum()).item()
    result = module.run_diagnostic(source, None, tmp_path, stage='fit', encoder='mlp', config=config,
                                   render=tiny_render, verify_anchor=anchors)
    import json
    row = json.loads((tmp_path / 'step_trace.jsonl').read_text().splitlines()[-1])
    assert row['loss'] == pytest.approx(.5 * (expected['pre'] + expected['post']), rel=2e-6)
    assert result['raw_relative_l2']['initial'] == pytest.approx({scan: value ** .5 for scan, value in expected.items()}, rel=2e-6)


def test_exact_constant_amplitudes_can_be_admitted_without_changing_the_bound(tmp_path):
    module = api(); source = TinySource()
    with torch.no_grad():
        source.density_real.copy_(torch.tensor([1., -1.])); source.density_imag.copy_(torch.tensor([0., 0.]))
        source.density_real_contrast.copy_(torch.tensor([0., 0.])); source.density_imag_contrast.copy_(torch.tensor([2., -2.]))
    config = {**module.FIT_CONFIG, 'coefficient_updates': 1, 'coefficient_batch': 4, 'raw_updates': 1}
    result = module.run_diagnostic(source, None, tmp_path, stage='fit', encoder='mlp', config=config,
                                   render=tiny_render, verify_anchor=anchors)
    assert result['software_complete'] is True and result['representation_admitted'] is True
    assert all(value <= .01 for value in result['raw_relative_l2']['final'].values())
