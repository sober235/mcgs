"""Restricted source4500 restore and export-only guards; no clinical data/GPU."""
import copy
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
PGS=ROOT.parent/'nonlocal_autoresearch/trainer/paired_gs'


def implementation():
    path=ROOT/'runner.py';assert path.is_file(),'export-only runner is not implemented'
    spec=importlib.util.spec_from_file_location('_proximal_test_runner',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def source_trainer():
    package=types.ModuleType('_proximal_test_gaussian');package.__path__=[str(PGS/'gaussian')]
    sys.modules[package.__name__]=package
    joint=importlib.import_module(package.__name__+'.joint_gaussian_model')
    shape=(6,6,6);n=3
    model=joint.JointGaussianModel3D(n,shape,'cpu',
        initial_positions=torch.tensor([[-.3,0.,.1],[.2,.2,-.1],[0.,-.3,.2]]),
        initial_densities_plain=torch.tensor([.1+.2j,.3-.1j,.2+.1j]),
        initial_densities_contrast=torch.tensor([.2+.1j,.1-.2j,.3+.1j]),
        initial_scales=torch.full((n,3),.15))
    optimizer=torch.optim.Adam(model.get_optimizable_params())
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=.9995)
    return types.SimpleNamespace(model=model,optimizer=optimizer,scheduler=scheduler,
                                 config={'test':'synthetic-source-model'},train=lambda:None)


def checkpoint_fixture(trainer):
    saved=trainer.optimizer.state_dict()
    named=dict(trainer.model.named_parameters())
    for (name,param),group in zip(named.items(),saved['param_groups']):
        ident=group['params'][0]
        saved['state'][ident]={'step':torch.tensor(4500.),'exp_avg':torch.full_like(param,.01),
                               'exp_avg_sq':torch.full_like(param,.02)}
        group['lr']=group['initial_lr']*.9995**4500
    recipe={'version':1,'frequency_t':1.,'geometry_factor':1.,'local_factor':1.,'nlambda':0.,
            'wavelet_protocol':'original_single_phase','frequency_manifest_sha256':'a'*64,'requires_same_recipe_on_resume':True}
    return {'iter':4500,'model_state':{k:v.detach().clone()+.01 for k,v in named.items()},
            'optim_state':saved,'frequency_adam_recipe':recipe},recipe


def test_actual_public_restore4500_preserves_parameter_identity_and_real_clocks():
    module=implementation();trainer=source_trainer();checkpoint,recipe=checkpoint_fixture(trainer)
    identities={k:id(v) for k,v in trainer.model.named_parameters()}
    proof=module.restore_export_only(trainer,checkpoint,recipe,{'path':'synthetic-test-only','sha256':'b'*64})
    assert proof['iteration']==4500 and proof['parameter_identities_before']==proof['parameter_identities_after']==identities
    assert trainer.scheduler.last_epoch==4500 and trainer.scheduler._step_count==4501
    for name,param in trainer.model.named_parameters():
        torch.testing.assert_close(param,checkpoint['model_state'][name],rtol=0,atol=0)
        assert float(trainer.optimizer.state[param]['step'])==4500
    assert module.snapshot(trainer)['parameter_sha256']


@pytest.mark.parametrize('defect',['iteration','recipe'])
def test_wrong_source_refused_before_restore_mutation(defect):
    module=implementation();trainer=source_trainer();checkpoint,recipe=checkpoint_fixture(trainer)
    before=module.snapshot(trainer)
    if defect=='iteration':checkpoint['iter']=4000
    else:checkpoint['frequency_adam_recipe']={**recipe,'frequency_t':0.}
    with pytest.raises(ValueError):module.restore_export_only(trainer,checkpoint,recipe,{'path':'test','sha256':'b'*64})
    assert module.snapshot(trainer)==before


def test_empty_Adam_snapshot_is_read_only():
    module=implementation();trainer=source_trainer()
    assert not trainer.optimizer.state
    a=module.snapshot(trainer);b=module.snapshot(trainer)
    assert a==b and not trainer.optimizer.state


@pytest.mark.parametrize('operation',['trainer_train','optimizer_step','scheduler_step'])
def test_export_only_blocks_steps_and_restores_bindings_on_error(operation):
    module=implementation();trainer=source_trainer();checkpoint,recipe=checkpoint_fixture(trainer)
    module.restore_export_only(trainer,checkpoint,recipe,{'path':'test','sha256':'b'*64})
    before=module.snapshot(trainer)
    with pytest.raises(RuntimeError,match='export.only'):
        with module.export_only(trainer) as calls:
            if operation=='trainer_train':trainer.train()
            elif operation=='optimizer_step':trainer.optimizer.step()
            else:trainer.scheduler.step()
    assert calls[operation]==1
    assert module.snapshot(trainer)==before


def test_scoped_canonical_observation_counts_source_FFT_and_captures_same_raw():
    module=implementation();trainer=source_trainer()
    tile=importlib.import_module('_proximal_test_gaussian.tile_voxelizer')
    trainer.voxelizer=tile.TileVoxelizer((6,6,6),use_cuda=False,device='cpu')
    trainer.device=torch.device('cpu')
    # Bind the actual source method, not a rewritten Gaussian reference.
    import ast
    path=PGS/'trainers/joint_trainer.py'
    node=next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='_render')
    namespace={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    trainer._render=types.MethodType(namespace['_render'],trainer)
    for tag in ('p','c'):
        setattr(trainer,'csm_'+tag,torch.ones((2,6,6,6),dtype=torch.complex64)/2**.5)
        setattr(trainer,'kspace_'+tag,torch.zeros((2,6,6,6),dtype=torch.complex64))
        setattr(trainer,'kspace_dc_'+tag,torch.zeros((2,6,6,6),dtype=torch.complex64))
        mask=torch.zeros(6,6,6);mask[:,:,::2]=1
        setattr(trainer,'mask_'+tag,mask);setattr(trainer,'mask_fit_'+tag,mask.clone())
    original=trainer._render
    result,work=module.canonical_control(trainer,backend_kind='cpu_test')
    assert trainer._render is original
    assert work['gaussian_renders']==2 and work['canonical_export_calls']==2
    assert work['canonical_fft_batches']==2 and work['canonical_ifft_batches']==4
    assert work['canonical_coil_fft_transforms']==4 and work['canonical_coil_ifft_transforms']==8
    assert set(result)=={'pre','post'} and all(set(v)=={'raw','canonical','zf'} for v in result.values())


def test_actual_lazy_strict_factory_resolves_verified_preflight_in_fresh_process():
    import subprocess
    code="""import importlib.util,sys
from pathlib import Path
p=Path('runner.py').resolve();s=importlib.util.spec_from_file_location('test_child',p);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
base,driver,preflight=m.reviewed_dependencies()
assert sys.modules['gpu_preflight'] is preflight
assert Path(preflight.__file__).resolve()==p.parent.parent/'phase_firstorder/gpu_preflight.py'
try:driver.construct_strict_cuda_trainer({},None,base,{},None)
except ValueError as e:assert 'strict tile_cuda' in str(e)
else:raise AssertionError('missing required CUDA config accepted')
"""
    result=subprocess.run([sys.executable,'-X','utf8','-B','-c',code],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_source_fit_masks_must_equal_acquired_before_export():
    module=implementation();trainer=source_trainer()
    for tag in ('p','c'):
        setattr(trainer,'mask_'+tag,torch.ones(6,6,6));setattr(trainer,'mask_fit_'+tag,torch.ones(6,6,6))
        setattr(trainer,'csm_'+tag,torch.ones(2,6,6,6,dtype=torch.complex64));setattr(trainer,'csm_fit_'+tag,getattr(trainer,'csm_'+tag))
        setattr(trainer,'kspace_'+tag,torch.ones(2,6,6,6,dtype=torch.complex64));setattr(trainer,'kspace_dc_'+tag,getattr(trainer,'kspace_'+tag));setattr(trainer,'kspace_fit_'+tag,getattr(trainer,'kspace_'+tag))
    assert module.validate_acquisition(trainer)
    trainer.mask_fit_p[0,0,0]=0
    with pytest.raises(ValueError,match='acquired'):module.validate_acquisition(trainer)


def test_fixed_published_gauge_uses_original_float32_conversion(tmp_path):
    import numpy as np
    module=implementation();x=torch.tensor([[[1.2345678+.1234567j]]],dtype=torch.complex64)
    gauge={'scale_trainer':183.53231811523438,'scale_gauge':183.57667347717228};gauge['factor_scale']=gauge['scale_trainer']/gauge['scale_gauge']
    info=module.save_image(tmp_path/'raw_pre_complex.npy',x,gauge)
    expected=(x.numpy().astype(np.complex64)*np.float32(gauge['factor_scale'])).astype(np.complex64)
    assert np.array_equal(np.load(info['path']),expected) and info['gauge']=='source_topic'
    with pytest.raises(FileExistsError):module.save_image(tmp_path/'raw_pre_complex.npy',x,gauge)


def test_tiny_physics_identity_costs_are_separate_from_solver():
    module=implementation();ph=module.physics();g=torch.Generator().manual_seed(42)
    shape=(4,4,4);csm=torch.complex(torch.randn((2,*shape),generator=g),torch.randn((2,*shape),generator=g))
    measured=torch.zeros_like(csm);mask=torch.ones(shape);op=ph.SenseOperator(csm,measured,mask)
    row=module.tiny_physics(op)
    for k in ('forward_adjoint_relative','normal_hermitian_relative','normal_linearity_relative'):assert row[k]<=2e-5
    assert row['work']['normal_calls']==3 and row['work']['forward_calls']==row['work']['adjoint_calls']==1
    assert row['work']['fft_batches']==row['work']['ifft_batches']==4
    assert row['work']['rhs_adjoint_calls']==row['work']['terminal_normal_calls']==row['work']['objective_calls']==0


def cpu_export_fixture():
    trainer=source_trainer();tile=importlib.import_module('_proximal_test_gaussian.tile_voxelizer')
    trainer.voxelizer=tile.TileVoxelizer((6,6,6),use_cuda=False,device='cpu');trainer.device=torch.device('cpu')
    import ast
    path=PGS/'trainers/joint_trainer.py';node=next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='_render')
    ns={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns);trainer._render=types.MethodType(ns['_render'],trainer)
    gen=torch.Generator().manual_seed(921)
    for tag in ('p','c'):
        csm=torch.ones((2,6,6,6),dtype=torch.complex64)/2**.5
        y=torch.complex(torch.randn((2,6,6,6),generator=gen),torch.randn((2,6,6,6),generator=gen))
        mask=torch.zeros(6,6,6);mask[:,:,::2]=1
        for name,value in (('csm',csm),('csm_fit',csm),('kspace',y),('kspace_dc',y),('kspace_fit',y),('mask',mask),('mask_fit',mask.clone())):setattr(trainer,name+'_'+tag,value)
    return trainer


def test_cpu_actual_source_tiny_flow_matches_parent_state_and_work_wire(monkeypatch):
    module=implementation();trainer=cpu_export_fixture();reference=copy.copy(trainer)
    reference._render=types.MethodType(trainer._render.__func__,reference)
    metadata={'extension_path':'/synthetic-cpu-test-only.so','extension_sha256':'a'*64}
    preflight=types.SimpleNamespace(make_tiny_cuda_operator=lambda **kwargs:(types.SimpleNamespace(trainer=trainer),types.SimpleNamespace(trainer=reference),metadata))
    monkeypatch.setattr(module,'reviewed_dependencies',lambda:(module.base_runner(),None,preflight))
    result=module.tiny_stage();parent=module.supervisor()
    parent.validate_states(result,'tiny',{})
    for row in result['scans'].values():parent.validate_scan(row['record'],row['work'])
    parent.validate_work_total(result,'tiny')
    assert result['arrays']=={} and result['source_restore_performed'] is False
    assert result['work_total']['gaussian_renders']==4 and result['work_total']['gaussian_vjps']==0


def test_canonical_render_hook_restores_class_binding_on_failure(monkeypatch):
    module=implementation();trainer=cpu_export_fixture()
    class Container:type_marker='CPU synthetic class binding'
    wrapped=Container();wrapped.__dict__.update(vars(trainer));del wrapped._render
    Container._render=trainer._render.__func__
    assert '_render' not in vars(wrapped)
    base=module.base_runner()
    def fail(*args):raise RuntimeError('deliberate CPU source-export failure')
    monkeypatch.setattr(base,'export_trainer_volume',fail)
    transforms=module.source_module('data.transforms','data/transforms.py');fft,ifft=transforms.fft3c,transforms.ifft3c
    with pytest.raises(RuntimeError,match='deliberate'):module.canonical_control(wrapped,backend_kind='cpu_test')
    assert '_render' not in vars(wrapped) and transforms.fft3c is fft and transforms.ifft3c is ifft


def test_wrong_runtime_refused_before_importing_numerics():
    import subprocess
    code="""import importlib.util,sys
s=importlib.util.spec_from_file_location('bad_runtime','runner.py');m=importlib.util.module_from_spec(s)
try:s.loader.exec_module(m)
except ValueError as e:assert 'GSTMI' in str(e)
else:raise AssertionError('wrong interpreter accepted')
assert 'torch' not in sys.modules
"""
    result=subprocess.run(['/home/congcongliu/anaconda3/envs/nvgen/bin/python','-X','utf8','-B','-c',code],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_wrong_torch_refused_by_numerical_snapshot(monkeypatch):
    module=implementation();trainer=source_trainer()
    monkeypatch.setattr(torch,'__version__','2.5.1+cu121')
    with pytest.raises(ValueError,match='GSTMI'):module.snapshot(trainer)


@pytest.mark.parametrize('stage',['canonical','prox'])
def test_cpu_actual_source_clinical_flow_preserves4500_state_and_json_wire(tmp_path,monkeypatch,stage):
    import json
    module=implementation();trainer=cpu_export_fixture();checkpoint,recipe=checkpoint_fixture(trainer)
    checkpoint_path=tmp_path/'synthetic_source4500.pth';torch.save(checkpoint,checkpoint_path)
    descriptor={'path':str(checkpoint_path),'sha256':module.sha256(checkpoint_path)}
    module.restore_export_only(trainer,checkpoint,recipe,descriptor)
    state=module.snapshot(trainer)
    metadata=module.load(ROOT.parent/'frequency_adam/overrides.py','_proximal_H_optimizer_metadata')._snapshot(trainer.optimizer,trainer.scheduler)
    config_path=tmp_path/'synthetic_config.yaml';config_path.write_text('output: {output_dir: synthetic}\nmode: {}\n')
    definition={'checkpoint':descriptor,'frequency_adam_recipe':recipe,'expected_model_state':state,'expected_optimizer_actual':metadata,
                'actual_config':{'path':str(config_path)},'scans':{}}
    controls,_=module.canonical_control(trainer,backend_kind='cpu_test')
    for scan,tag in (('pre','p'),('post','c')):
        reference=module.save_image(tmp_path/f'reference_{scan}.npy',controls[scan]['canonical'],{'factor_scale':1.})
        definition['scans'][scan]={'scale_trainer':1.,'scale_gauge':1.,'factor_scale':1.,'canonical_array':{'path':reference['path'],'sha256':reference['sha256']}}
        setattr(trainer,'ds_'+tag,types.SimpleNamespace(normalization_scale=1.))
    backend={'strict_cuda':True,'use_cuda':True,'compiler_branch_bypassed':True,'backend':'tile_cuda'} # synthetic CPU gate metadata; never published.
    def construct(cfg,*args):trainer.config=cfg;return trainer,backend
    monkeypatch.setattr(module,'reviewed_dependencies',lambda:(module.base_runner(),types.SimpleNamespace(construct_strict_cuda_trainer=construct),None))
    original_source=module.source_module
    monkeypatch.setattr(module,'source_module',lambda name,path:types.SimpleNamespace(JointTrainer=None) if name=='trainers.joint_trainer' else original_source(name,path))
    original_load=module.load
    h=original_load(ROOT.parent/'frequency_adam/runner.py','_proximal_H_checkpoint_validator')
    small_h=types.SimpleNamespace(SOURCE_SHAPES={n:list(p.shape) for n,p in trainer.model.named_parameters()},validate_checkpoint=h.validate_checkpoint)
    monkeypatch.setattr(module,'load',lambda path,name:small_h if name=='_proximal_H_checkpoint_validator' else original_load(path,name))
    output=tmp_path/'reconstruction';output.mkdir()
    result=module.clinical_stage(stage,output,definition,{})
    # Includes the real scheduler state after temporary forbidden-step hooks have gone.
    json.dumps(result,allow_nan=False)
    parent=module.supervisor();parent.validate_states(result,stage,definition);parent.validate_work_total(result,stage)
    for row in result['scans'].values():
        if stage=='prox':parent.validate_scan(row['record'],row['work'])
    assert result['source_state_before']==result['source_state_after']
    assert result['restore_proof']['optimizer_actual']==metadata
    assert all(set(row)==({'raw','canonical','candidate'} if stage=='prox' else {'raw','canonical'}) for row in result['arrays'].values())
