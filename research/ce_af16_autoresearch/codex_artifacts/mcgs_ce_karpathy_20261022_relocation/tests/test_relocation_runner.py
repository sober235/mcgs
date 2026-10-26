"""Tiny original-loop integration; never MRI controls or public recipe gates."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('positions', 'scales', 'rotations', 'density_real', 'density_imag',
         'density_real_contrast', 'density_imag_contrast')


def child():
    if not (ROOT / 'relocation_runner.py').exists():
        pytest.fail('relocation child production behavior is missing')
    import relocation_runner
    return relocation_runner


class TinyVoxelizer(torch.nn.Module):
    def forward(self, positions, scales, rotations, density):
        # Nonzero derivatives reach all seven tensors through two complex scans.
        weights = 1 + positions.sum(1) / 10 + scales.sum(1) / 10 + rotations.sum(1) / 10
        return (density * weights).sum().expand(3, 3, 3).contiguous()


def tiny_trainer(directory, steps=2, zero_evidence=False, wavelet=False):
    import runner as g2
    import relocation_runtime as rr
    sys.path[:0] = [str(g2.TRAINER / 'paired_gs'), str(g2.TRAINER)]
    from trainers.joint_trainer import JointTrainer
    from gaussian.tile_voxelizer import _quaternion_to_rotation_matrix
    from data.transforms import fft3c

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_points = 5
            for i, name in enumerate(NAMES):
                shape = (5, 3) if i < 2 else (5, 4) if i == 2 else (5,)
                value = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape) / 100 + .05
                if name == 'positions': value = value - .1
                if name == 'scales': value = value - 3
                if name == 'rotations': value = torch.tensor([[1., 0, 0, 0]]).repeat(5, 1)
                self.register_parameter(name, torch.nn.Parameter(value))
        has_private = False
        def get_scale_values(self): return self.scales.exp()
        def get_density(self, channel):
            suffix = '' if channel == 'plain' else '_contrast'
            return torch.complex(getattr(self, 'density_real' + suffix), getattr(self, 'density_imag' + suffix))

    class TinyOriginal(JointTrainer):
        def __init__(self):
            self.model, self.voxelizer = Model(), TinyVoxelizer()
            self.device = torch.device('cpu')
            self.volume_shape = (3, 3, 3)
            self.config = {'training': {'seed': 42, 'resume_offset': 4500,
                'max_iterations': 4500 + steps, 'lr_scheduler': {'type': 'exponential', 'gamma': .9995},
                'log_every': 20, 'save_every': 1000, 'eval_every': 1000},
                'loss': {'lambda_geo': 0., 'lambda_wav': 0., 'edge_tv_kappa_source': 'zf',
                         'kspace_freq_adaptive': True, 'kspace_high_weight_plain': 1.},
                'adaptive_control': {'enable': False}}
            if wavelet:
                self.config['loss'].update(lambda_wav=.001,wav_levels=1,wav_name='haar',wav_cycle_spin=True)
            self.writer = None
            self.name_plain, self.name_contrast = 'plain', 'contrast'
            self.out_dir = directory / 'training' / 'tiny'
            self.out_dir.mkdir(parents=True)
            self.optimizer = self.scheduler = None
            for tag in ('p', 'c'):
                csm = torch.full((2, 3, 3, 3), 0 if zero_evidence else 1, dtype=torch.complex64)
                mask = torch.zeros(3, 3, 3); mask[1] = 1
                data = torch.ones_like(csm) * mask
                for key, value in dict(csm=csm, csm_fit=csm, mask=mask, mask_fit=mask,
                        kspace=data, kspace_fit=data, kspace_dc=data, zf=data.sum(0)).items():
                    setattr(self, key + '_' + tag, value)
                setattr(self, '_dc_norm_' + tag, 1.)
            self._static_freq_w_plain = None
        def _build_freq_weight(self, high_override=None): return torch.ones(3, 3, 3)

    trainer = TinyOriginal()
    source = Model()
    optimizer = torch.optim.Adam([{'params': [getattr(source, name)], 'lr': lr}
                                  for name, lr in zip(NAMES, rr.BASE_LRS)])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.9995)
    sum(p.square().sum() for p in source.parameters()).backward(); optimizer.step(); scheduler.step()
    checkpoint = {'iter': 4500, 'model_state': copy.deepcopy(source.state_dict()),
                  'optim_state': copy.deepcopy(optimizer.state_dict()), 'frequency_adam_recipe': dict(rr._G2_RECIPE)}
    for group, lr in zip(checkpoint['optim_state']['param_groups'], rr.BASE_LRS):
        group['lr'] = lr * .9995 ** 4500
        checkpoint['optim_state']['state'][group['params'][0]]['step'].fill_(4500)
    for p in trainer.model.parameters(): p.requires_grad_(False)
    restore = rr.restore_training_state(trainer, checkpoint)
    frequency = g2.load_module(g2.OLD/'frequency_adam/runner.py','_tiny_actual_frequency')
    frequency.install_frequency_weights(trainer,1.,directory)
    return trainer, restore, frequency, _quaternion_to_rotation_matrix


@pytest.mark.parametrize('mode,steps,renders,vjps,events', [
    ('control', 0, 2, 0, 0), ('control', 2, 6, 4, 0), ('relocate', 2, 8, 4, 1),
    ('control', 500, 1002, 1000, 0), ('relocate', 500, 1004, 1000, 1)])
def test_original_restore_train_event_export_has_actual_clocks_and_work(tmp_path, mode, steps, renders, vjps, events):
    module = child()
    trainer, restore, frequency, rotation = tiny_trainer(tmp_path, steps)
    before = copy.deepcopy(restore['after'])
    result = module.run_continuation(trainer, frequency, tmp_path, mode=mode, steps=steps,
                                     rotation_matrix_fn=rotation, requested_k=1)
    assert result['before'] == before
    assert result['after']['scheduler']['state']['last_epoch'] == 4500 + steps
    assert result['after']['scheduler']['state']['_step_count'] == 4501 + steps
    assert result['counters']['optimizer_steps'] == result['counters']['scheduler_steps'] == steps
    assert result['counters']['gaussian_forwards'] == renders
    assert result['counters']['gaussian_vjps'] == vjps
    assert result['counters']['event_calls'] == events
    assert len(result['events']) == events
    for name in NAMES:
        assert result['after']['optimizer']['states'][name]['step']['value'] == 4500 + steps
        assert result['after']['parameters'][name]['tensor_id'] == before['parameters'][name]['tensor_id']
    if steps == 0:
        assert result['before'] == result['after']
    else:
        assert [r['iteration'] for r in result['trace']] == list(range(4501,4501+steps))
        assert all(set(r['gradient_norms']) == set(NAMES) for r in result['trace'])
    if events:
        event = result['events'][0]
        assert event['iteration'] == 4501
        assert set(event['transaction']['before']['steps'].values()) == {4501.}
        assert event['selection']['k_eff'] == 1
        assert event['transaction']['changed_row_count'] == 2
    gauges = {s:dict(scale_trainer=1., scale_gauge=1., factor_scale=1.) for s in ('pre','post')}
    # Export uses original canonical_control, so no extra render may be hidden.
    assert result['exports']['pre']['raw'].shape == (3, 3, 3)
    saved = module.save_exports(result.pop('exports'), gauges, tmp_path)
    for row in saved.values():
        for record in row.values():
            arr = np.load(record['path'], allow_pickle=False)
            assert np.isfinite(arr).all() and arr.dtype == np.complex64
    assert result['counters']['gaussian_forwards'] == renders
    json.dumps(result, allow_nan=False)


def test_missing_density_vjp_is_detected(tmp_path):
    module = child()
    trainer, _, frequency, rotation = tiny_trainer(tmp_path)
    class BadVoxelizer(TinyVoxelizer):
        def forward(self, positions, scales, rotations, density):
            return super().forward(positions, scales, rotations, density.detach())
    trainer.voxelizer = BadVoxelizer()
    with pytest.raises(ValueError, match='gradient|VJP|work'):
        module.run_continuation(trainer, frequency, tmp_path, mode='control', steps=2,
                                rotation_matrix_fn=rotation, requested_k=1)


def test_zero_eligible_donors_is_one_real_noop_event(tmp_path):
    module=child()
    trainer,_,frequency,rotation=tiny_trainer(tmp_path,zero_evidence=True)
    result=module.run_continuation(trainer,frequency,tmp_path,mode='relocate',steps=2,
                                   rotation_matrix_fn=rotation,requested_k=1)
    assert result['counters']['event_calls']==1
    assert result['counters']['roles']['event']==dict(forwards=2,vjps=0)
    assert result['events'][0]['selection']['k_eff']==0
    transaction=result['events'][0]['transaction']
    assert transaction['before']==transaction['after']
    assert transaction['changed_indices']==[]
    assert result['after']['scheduler']['state']['last_epoch']==4502


def test_original_wavelet_shift_generator_replays_4500_updates(tmp_path):
    module=child()
    trainer,_,frequency,rotation=tiny_trainer(tmp_path,wavelet=True)
    result=module.run_continuation(trainer,frequency,tmp_path,mode='control',steps=2,
                                  rotation_matrix_fn=rotation,requested_k=1)
    # Independently replay the declared Torch draws, then inspect the actual
    # original-loop payload, rather than invoking its resume helper for expected.
    generator=torch.Generator(device='cpu').manual_seed(7961)
    for _ in range(4500): torch.randint(0,2,(3,),generator=generator)
    expected=[torch.randint(0,2,(3,),generator=generator).tolist() for _ in range(2)]
    assert [list(row['wavelet_shift']) for row in result['trace']]==expected
