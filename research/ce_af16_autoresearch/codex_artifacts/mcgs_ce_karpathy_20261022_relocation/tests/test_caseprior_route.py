"""Real original Torch loop and Task1 fitting, with private small CPU fixtures."""
import copy
import importlib
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

from test_relocation_runner import tiny_trainer, NAMES

SMALL = dict(updates=2, patch_size=2, batch_size=1, width=2, tile_size=2)


def child():
    if importlib.util.find_spec('caseprior_runner') is None:
        pytest.fail('case-prior original-trainer integration is missing')
    return importlib.import_module('caseprior_runner')


def run_tiny(directory, mode='learned', steps=2, scan_names=('plain','contrast')):
    module = child()
    import relocation_runtime as runtime
    actual_restore=runtime.restore_training_state
    def restore_configured(trainer,checkpoint):
        module.configure_prior(trainer.config,mode)
        trainer.name_plain,trainer.name_contrast=scan_names
        trainer.config['data']={channel:dict(name=name) for channel,name in zip(('plain','contrast'),scan_names)}
        return actual_restore(trainer,checkpoint)
    # Match public construction order: complete config precedes actual restore
    # and immutable frequency installation. Preserve the returned restore record.
    with patch.object(runtime,'restore_training_state',restore_configured):
        trainer, restore, frequency, rotation = tiny_trainer(directory, steps)
    result = module.run_case_continuation(trainer, frequency, directory, mode=mode, steps=steps,
        rotation_matrix_fn=rotation, _denoiser_settings=SMALL)
    return trainer, restore, result


@pytest.mark.parametrize('mode,steps,calls', [('control',0,0),('control',2,0),('learned',2,2),('learned',500,10)])
def test_original_loop_fits_real_prior_and_saves_actual_work(tmp_path,mode,steps,calls):
    trainer, _, result = run_tiny(tmp_path, mode, steps)
    assert len(result['denoiser_calls']) == calls
    assert len(result['denoiser_rng']) == calls
    assert len(result['denoiser_checkpoint_checks']) == calls
    for index,row in enumerate(result['denoiser_checkpoint_checks']):
        assert row['call_index']==index and row['metadata_equal'] is True
        assert row['finite_float32_cpu'] is True
        assert row['checkpoint']==result['denoiser_calls'][index]['checkpoint']
        assert row['final_parameter_sha256']==result['denoiser_calls'][index]['final_parameter_sha256']
    for index,row in enumerate(result['denoiser_rng']):
        assert row['call_index']==index and row['schema']=='case_denoiser_rng_v1'
        assert row['cpu']['before_sha256']==row['cpu']['after_sha256'] and row['cpu']['preserved'] is True
        assert row['cuda']==[]
    assert sum(r['updates'] for r in result['denoiser_calls']) == calls*2
    assert result['counters']['gaussian_forwards'] == 2*steps+2
    assert result['counters']['gaussian_vjps'] == 2*steps
    assert result['counters']['event_calls'] == 0 and result['events'] == []
    assert result['counters']['optimizer_steps'] == result['counters']['scheduler_steps'] == steps
    assert result['after']['scheduler']['state']['last_epoch'] == 4500+steps
    assert result['after']['scheduler']['state']['_step_count'] == 4501+steps
    saved = torch.load(result['final_checkpoint'], weights_only=True)
    assert saved['iter'] == 4500+steps
    for name in NAMES:
        assert torch.equal(saved['model_state'][name],getattr(trainer.model,name))
        assert result['after']['optimizer']['states'][name]['step']['value'] == 4500+steps
    if calls:
        summary = result['nonlocal_summary']
        assert summary['validated_active'] is True
        assert summary['active_step_count'] == steps
        assert summary['weighted_R_nl_max'] > 0
        assert all(summary['raw_R_nl_by_channel_max'][s]>0 for s in ('plain','contrast'))
        for i,record in enumerate(result['denoiser_calls']):
            assert record['status']=='trained' and record['seed']==42+1009*i
            assert record['gradients']['all_finite'] and record['gradients']['any_nonzero']
            assert record['initial_parameter_sha256'] != record['final_parameter_sha256']
            saved_target=np.load(record['target']['path'],allow_pickle=False)
            assert np.isfinite(saved_target).all() and (saved_target>=0).all() and saved_target.max()>0
            scan=('plain','contrast')[i%2]
            row=summary['refresh_history'][scan][i//2]
            assert row['iter']==4501+100*(i//2) and row['kernel_stats']==record
        for row in result['trace']:
            assert all(np.isfinite(row['gradient_norms'][n]) for n in NAMES)
            assert row['nonlocal']['weighted']>0
        assert all(result['trace'][0]['gradient_norms'][n]>0 for n in NAMES)


def test_fixture_preserves_actual_original_restore_record_without_replacement(tmp_path):
    import relocation_runtime as runtime
    actual=runtime.restore_training_state
    observed=[]
    def capture(trainer,checkpoint):
        record=actual(trainer,checkpoint)
        observed.append(copy.deepcopy(record['after']))
        return record
    with patch.object(runtime,'restore_training_state',capture):
        _,restore,result=run_tiny(tmp_path)
    assert observed==[result['before']]
    assert restore['after']==observed[0]


def test_factory_restores_on_success_and_failure_and_delegates(tmp_path):
    module=child()
    import runner
    sys.path[:0]=[str(runner.TRAINER/'paired_gs'),str(runner.TRAINER)]
    import losses.nonlocal_red as red
    original=red.make_denoiser
    for failure in (False,True):
        try:
            with module.case_denoiser_factory(tmp_path/'calls',device='cpu',settings=SMALL) as instances:
                assert np.array_equal(red.make_denoiser('fake_identity')(np.ones((2,2,2)),1),np.ones((2,2,2)))
                real=red.make_denoiser('case_internal_cnn_v1')
                assert instances == [real]
                if failure: raise RuntimeError('fixture failure')
        except RuntimeError:
            assert failure
        assert red.make_denoiser is original


def test_control_does_not_construct_network(tmp_path,monkeypatch):
    module=child()
    import case_denoiser
    def forbidden(*a,**k): raise AssertionError('control constructed a CNN')
    monkeypatch.setattr(case_denoiser,'CaseMagnitudeDenoiser',forbidden)
    _,_,result=run_tiny(tmp_path,'control')
    assert result['denoiser_calls']==[]


def test_original_refresh_has_detached_independent_scan_targets(tmp_path,monkeypatch):
    module=child()
    import losses.nonlocal_red as red
    original=red.NonLocalRED.refresh
    targets=[]
    def observe(self,name,vol,it):
        result=original(self,name,vol,it)
        t=self.targets[name]
        assert t.requires_grad is False and t.grad_fn is None
        assert t.data_ptr()!=vol.data_ptr() and bool(torch.isfinite(t).all())
        targets.append(t)
        return result
    monkeypatch.setattr(red.NonLocalRED,'refresh',observe)
    run_tiny(tmp_path)
    assert len(targets)==2 and targets[0].data_ptr()!=targets[1].data_ptr()
    assert not torch.equal(targets[0],targets[1])


def test_original_training_failure_restores_factory_and_observers(tmp_path):
    module=child()
    import losses.nonlocal_red as red
    previous=red.make_denoiser
    scheduler_step=torch.optim.lr_scheduler.ExponentialLR.step
    # Fail inside original no_grad refresh, after its factory constructed the
    # real Task1 callable, by asking for a patch larger than the actual crop.
    from unittest.mock import patch
    import case_denoiser
    real=case_denoiser.CaseMagnitudeDenoiser
    def too_large(directory,**kwargs):
        return real(directory,**{**kwargs,'patch_size':4})
    (tmp_path/'failure').mkdir()
    with patch.object(case_denoiser,'CaseMagnitudeDenoiser',too_large):
        with pytest.raises(ValueError,match='patch'):
            run_tiny(tmp_path/'failure')
    assert red.make_denoiser is previous
    assert torch.optim.lr_scheduler.ExponentialLR.step is scheduler_step


@pytest.mark.parametrize('defect',['metadata','schema','weight','dtype','shape','key'])
def test_child_checkpoint_readback_refuses_corrupt_trained_artifact(tmp_path,defect):
    module=child()
    from case_denoiser import CaseMagnitudeDenoiser
    producer=CaseMagnitudeDenoiser(tmp_path,device='cpu',**SMALL)
    producer(np.arange(27,dtype=np.float32).reshape(3,3,3)+1,.1)
    record=copy.deepcopy(producer.records[0])
    path=Path(record['checkpoint']['path']);saved=torch.load(path,weights_only=True)
    if defect=='metadata':saved['seed']+=1
    elif defect=='schema':saved['schema']='other'
    elif defect=='weight':saved['model_state_dict']['network.8.weight'].fill_(float('nan'))
    elif defect=='dtype':saved['model_state_dict']['network.8.weight']=saved['model_state_dict']['network.8.weight'].double()
    elif defect=='shape':saved['model_state_dict']['network.8.weight']=torch.zeros(1)
    elif defect=='key':saved['model_state_dict'].pop('network.8.bias')
    torch.save(saved,path)
    record['checkpoint']=module.route.base.descriptor(path)
    with pytest.raises(ValueError):module._readback_checkpoint(record)
