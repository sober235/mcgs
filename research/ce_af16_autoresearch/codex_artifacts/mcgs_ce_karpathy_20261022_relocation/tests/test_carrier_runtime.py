"""Real original restore/loop/physics/export with explicit private tiny CPU native oracle."""
import copy
import importlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from test_relocation_runner import tiny_trainer, NAMES

ROOT = Path(__file__).resolve().parents[1]


def module():
    assert (ROOT/'carrier_runtime.py').exists(), 'carrier original-loop runtime is missing'
    return importlib.import_module('carrier_runtime')


class NativeCPU:
    """Test native boundary computes accepted dense double geometry oracle."""
    def __init__(self, fail=False):
        self.forwards = self.backwards = 0
        self.fail = fail
        self.gradients = []
    def forward(self, centers, m, density, frequency, ranges, ids, shape):
        from carrier_reference import _correction_from_geometry
        self.forwards += 1
        return _correction_from_geometry(centers.double(), m.double(), density.to(torch.complex128),
                                        frequency.double(), shape).to(torch.complex64)
    def backward(self, grad, centers, m, density, frequency, ranges, ids, shape):
        if self.fail: raise RuntimeError('native backward failed')
        from carrier_reference import _correction_from_geometry
        with torch.enable_grad():
            values = [x.detach().to(torch.complex128 if x.is_complex() else torch.float64).requires_grad_()
                      for x in (centers,m,density,frequency)]
            result = _correction_from_geometry(*values,shape)
            gradients = torch.autograd.grad(result, values, grad.to(torch.complex128))
        self.backwards += 1
        self.gradients.append([x.clone() for x in gradients])
        return tuple(g.to(x.dtype) for g,x in zip(gradients,(centers,m,density,frequency)))


def run(directory, mode='carrier', steps=2, native=None):
    m = module()
    import relocation_runtime as rr
    restore = rr.restore_training_state
    captured = []
    def configured(trainer, checkpoint):
        # Configuration is completed before the actual original restoration.
        trainer.config['data'] = {'plain':{'name':'pre'}, 'contrast':{'name':'post'}}
        checkpoint['model_state']['density_real_contrast'].mul_(1.5)
        checkpoint['model_state']['density_imag_contrast'].mul_(.8)
        record = restore(trainer, checkpoint)
        captured.append(copy.deepcopy(record))
        return record
    with patch.object(rr, 'restore_training_state', configured):
        trainer, restored, frequency, rotation = tiny_trainer(directory,steps)
    original_rng = torch.random.get_rng_state().clone()
    original = (trainer._render, trainer.train)
    native = native or NativeCPU()
    result = m._run_carrier_continuation(trainer,frequency,directory,mode=mode,steps=steps,
        rotation_matrix_fn=rotation, _cpu_extension=native, _test_k=2)
    assert trainer._render == original[0] and trainer.train == original[1]
    assert restored == captured[0]
    assert torch.equal(original_rng,torch.random.get_rng_state())
    return trainer, restored, result, native


@pytest.mark.parametrize('mode,steps,forwards,backwards,updates',[
    ('control',0,2,0,0),('control',2,0,0,0),('carrier',2,6,4,2),('carrier',500,1002,1000,500)])
def test_original_loop_restoration_export_and_separate_work(tmp_path,mode,steps,forwards,backwards,updates):
    trainer, restored, result, native = run(tmp_path,mode,steps)
    c = result['carrier']
    assert result['before'] == restored['after']
    assert result['counters']['gaussian_forwards'] == 2*steps+2
    assert result['counters']['gaussian_vjps'] == 2*steps
    assert result['counters']['event_calls'] == 0
    assert c['work']['forwards'] == native.forwards == forwards
    assert c['work']['backwards'] == native.backwards == backwards
    assert c['work']['optimizer_steps'] == updates and c['work']['scheduler_steps'] == 0
    assert c['export_state']['unchanged'] and c['export_state']['calls'] == (2 if forwards else 0)
    assert c['initial_state']['eta']['pre']['zero'] and c['initial_state']['eta']['post']['zero']
    assert c['final_state']['adam_steps'] == {'pre':updates,'post':updates}
    assert c['selection']['k'] == 2 and c['selection']['n'] == 5
    ids = torch.load(c['selection']['file']['path'], weights_only=True)
    expected = torch.randperm(5,generator=torch.Generator().manual_seed(42023))[:2].sort().values
    assert torch.equal(ids,expected)
    assert c['readback']['live_state_equal']
    assert c['checkpoint']['path'] != result['final_checkpoint']
    manifest = json.loads(Path(c['paired_checkpoint_manifest']['path']).read_text())
    assert manifest['iteration'] == 4500+steps
    assert manifest['gaussian'] == result['final_checkpoint_evidence']['file']
    assert tuple(dict(trainer.model.named_parameters())) == NAMES
    for name in NAMES:
        assert result['after']['parameters'][name]['tensor_id'] == restored['after']['parameters'][name]['tensor_id']
        assert result['after']['optimizer']['states'][name]['step']['value'] == 4500+steps
    if updates:
        assert all(row['pre_clip_norm']>0 and 0<row['post_clip_norm']<=1.00001 for row in c['trace'])
        for name in ('pre','post'):
            assert c['final_state']['eta'][name]['sha256'] != c['initial_state']['eta'][name]['sha256']
        assert c['final_state']['eta']['pre']['sha256'] != c['final_state']['eta']['post']['sha256']
        assert all(result['trace'][1]['gradient_norms'][n]>0 for n in NAMES)
        assert all(float(g.abs().sum())>0 for g in native.gradients[-1])
        assert [r['iteration'] for r in c['trace']] == list(range(4501,4501+steps))
        assert c['work']['roles']['training'] == {'forwards':2*steps,'backwards':2*steps}
    else: assert c['trace'] == []
    if steps==0:
        assert set(c['export_state']['zero_raw_parity'])=={'pre','post'}
        assert all(r['exact_equal'] and r['relative_l2']==0. for r in c['export_state']['zero_raw_parity'].values())
    json.dumps({k:v for k,v in result.items() if k!='exports'}, allow_nan=False)


def test_canonical_export_contains_reconstructed_saved_carrier_and_independent_states(tmp_path):
    m=module(); trainer,_,result,_=run(tmp_path)
    saved=m._validate_carrier_checkpoint(result['carrier']['checkpoint']['path'],expected_iteration=4502,
        expected_policy=result['carrier']['policy'],expected_selection_sha256=result['carrier']['selection']['sha256'],
        expected_binary_sha256=result['carrier']['extension']['binary_sha256'],_test_geometry=(5,2))
    import carrier_operator as op
    from carrier_reference import _correction_from_geometry
    ids=saved['ids']; p,s,q=[getattr(trainer.model,n)[ids] for n in ('positions','scales','rotations')]
    centers,matrix=op._geometry(p,s,q,trainer.volume_shape)
    assert saved['eta']['pre'].data_ptr()!=saved['eta']['post'].data_ptr()
    assert saved['optimizer']['state'][0]['exp_avg'].data_ptr()!=saved['optimizer']['state'][1]['exp_avg'].data_ptr()
    with pytest.raises(ValueError,match='shape|count'):
        m.validate_carrier_checkpoint(result['carrier']['checkpoint']['path'],expected_iteration=4502,
            expected_policy=result['carrier']['policy'],expected_selection_sha256=result['carrier']['selection']['sha256'],
            expected_binary_sha256=result['carrier']['extension']['binary_sha256'])
    with torch.no_grad():
        for channel,scan in [('plain','pre'),('contrast','post')]:
            original=trainer._render(channel)
            correction=_correction_from_geometry(centers.double(),matrix.double(),
                trainer.model.get_density(channel)[ids].to(torch.complex128),
                (.25*saved['eta'][scan].tanh()).double(),trainer.volume_shape).to(torch.complex64)
            assert torch.equal(result['exports'][scan]['raw'],original+correction)
            assert not torch.equal(result['exports'][scan]['raw'],original)


def test_carrier_adam_runs_after_original_adam_scheduler_and_only_once(tmp_path,monkeypatch):
    m=module(); trainer,_,frequency,rotation=tiny_trainer(tmp_path)
    order=[]
    trainer.optimizer.register_step_post_hook(lambda *a:order.append(('original',trainer.scheduler.last_epoch)))
    actual=m._optimizer
    def observed(parameters,policy):
        optimizer=actual(parameters,policy)
        optimizer.register_step_post_hook(lambda *a:order.append(('carrier',trainer.scheduler.last_epoch)))
        return optimizer
    monkeypatch.setattr(m,'_optimizer',observed)
    m._run_carrier_continuation(trainer,frequency,tmp_path,mode='carrier',steps=2,rotation_matrix_fn=rotation,
                               _cpu_extension=NativeCPU(),_test_k=2)
    assert order==[('original',4500),('carrier',4501),('original',4501),('carrier',4502)]


def test_resource_refusal_occurs_before_correction_pair_allocation(tmp_path,monkeypatch):
    m=module(); import carrier_operator as op
    trainer,_,frequency,rotation=tiny_trainer(tmp_path)
    def refuse(*a,**k): raise ValueError('pair resource limit')
    monkeypatch.setattr(m,'_pair_upper_bound',refuse)
    monkeypatch.setattr(op,'_tile_pairs',lambda *a:pytest.fail('correction allocated pairs after refusal'))
    with pytest.raises(ValueError,match='pair resource limit'):
        m._run_carrier_continuation(trainer,frequency,tmp_path,mode='carrier',steps=2,rotation_matrix_fn=rotation,
                                   _cpu_extension=NativeCPU(),_test_k=2)
    assert not (tmp_path/'paired_carrier_checkpoint.json').exists()


def test_public_cpu_has_no_fallback_and_frozen_k(tmp_path):
    m=module(); trainer,_,frequency,rotation=tiny_trainer(tmp_path)
    with pytest.raises((ValueError,RuntimeError),match='CUDA|verified|active'):
        m.run_carrier_continuation(trainer,frequency,tmp_path,mode='carrier',steps=2,rotation_matrix_fn=rotation)
    with pytest.raises(ValueError,match='18000|fixed|subset'):
        m.run_carrier_continuation(trainer,frequency,tmp_path,mode='carrier',steps=2,rotation_matrix_fn=rotation,requested_k=1)


def test_bounds_refuse_before_pair_allocation_and_use_full_volume_fallback(monkeypatch):
    m=module(); import carrier_operator as op
    p=torch.zeros(3,3); s=torch.zeros(3,3); q=torch.tensor([[1.,0,0,0]]).repeat(3,1)
    row=m._pair_upper_bound(p,s,q,(16,16,16),limit=24)
    assert row['upper_pairs']==24 and row['reliable_atoms']==3
    with pytest.raises(ValueError,match='pair.*limit|resource'):
        m._pair_upper_bound(p,s,q,(16,16,16),limit=23)
    monkeypatch.setattr(op,'_geometry',lambda *a:(p,torch.zeros(3,3,3)))
    assert m._pair_upper_bound(p,s,q,(16,16,16),limit=24)['fallback_atoms']==3


@pytest.mark.parametrize('defect',['schema','iteration','policy','ids','binary','shape','clock','moment','missing_pair'])
def test_checkpoint_corruption_refused(tmp_path,defect):
    m=module(); _,_,result,_=run(tmp_path)
    c=result['carrier']; path=Path(c['checkpoint']['path'])
    value=torch.load(path,weights_only=True)
    if defect=='schema': value['schema']='wrong'
    elif defect=='iteration': value['iteration']+=1
    elif defect=='policy': value['policy']['lr']=9
    elif defect=='ids': value['ids'][0]=value['ids'][1]
    elif defect=='binary': value['binary_sha256']='0'*64
    elif defect=='shape': value['eta']['pre']=torch.zeros(1)
    elif defect=='clock': value['optimizer']['state'][0]['step'].fill_(0)
    elif defect=='moment': value['optimizer']['state'][0]['exp_avg'].fill_(float('nan'))
    else: Path(result['final_checkpoint']).unlink()
    torch.save(value,path)
    with pytest.raises((ValueError,FileNotFoundError)):
        m._validate_carrier_checkpoint(path,expected_iteration=4502,expected_policy=c['policy'],
            expected_selection_sha256=c['selection']['sha256'],expected_binary_sha256=c['extension']['binary_sha256'],
            _test_geometry=(5,2))


def test_failed_native_backward_unwinds_instance_and_extension(tmp_path):
    m=module(); import carrier_operator as op
    trainer,_,frequency,rotation=tiny_trainer(tmp_path)
    original=(trainer._render,trainer.train,op._extension)
    with pytest.raises(RuntimeError,match='native backward failed'):
        m._run_carrier_continuation(trainer,frequency,tmp_path,mode='carrier',steps=2,
            rotation_matrix_fn=rotation,_cpu_extension=NativeCPU(fail=True),_test_k=2)
    assert trainer._render == original[0] and trainer.train == original[1]
    assert op._extension is original[2]
    assert not (tmp_path/'paired_carrier_checkpoint.json').exists()
