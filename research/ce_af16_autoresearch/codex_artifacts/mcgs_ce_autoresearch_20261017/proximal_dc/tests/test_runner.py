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
    def construct(cfg,*args):
        # Exercise actual source preparation in this mocked CPU constructor.
        scratch=tmp_path/'constructor_input';scratch.mkdir()
        prepared=cpu_export_fixture();prepared.config=simulated_source_data_config(scratch)
        module.source_module('trainers.joint_trainer','trainers/joint_trainer.py').JointTrainer._setup_data(prepared)
        trainer.config=cfg;return trainer,backend
    monkeypatch.setattr(module,'reviewed_dependencies',lambda:(module.base_runner(),types.SimpleNamespace(construct_strict_cuda_trainer=construct),None))
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


def simulated_source_data_config(tmp_path):
    import numpy as np,h5py
    root=tmp_path/'source_input';root.mkdir();rng=np.random.default_rng(328)
    for name,key,array in [('data','kspace',(rng.normal(size=(2,6,6,6))+1j*rng.normal(size=(2,6,6,6))).astype(np.complex64)),('map','map',np.ones((2,6,6,6),dtype=np.complex64)),('mask','mask',np.ones((6,6,6),dtype=np.float32))]:
        with h5py.File(root/f'{name}.h5','w') as stream:stream[key]=array
    row=dict(data_path=str(root/'data.h5'),csm_path=str(root/'map.h5'),mask_path=str(root/'mask.h5'),presampled=False,normalize_kspace=True,normalization_percentile=99.9,mask_type='external',skip_readout_check=True,readout_axis=0,phase_axes=[1,2],use_acs=False)
    return {'data':{'plain':dict(row),'contrast':dict(row)}}


def test_full_source_simulated_constructor_counts_scalar_and_coil_transforms(tmp_path):
    module=implementation();trainer=cpu_export_fixture();source=module.source_module('trainers.joint_trainer','trainers/joint_trainer.py')
    trainer.config=simulated_source_data_config(tmp_path)
    dataset=sys.modules['data.dataset'];transforms=sys.modules['data.transforms']
    original=[dataset.fft3c,dataset.ifft3c,source.ifft3c,transforms.fft3c,transforms.ifft3c]
    with module.observe_source_transforms() as construction:source.JointTrainer._setup_data(trainer)
    assert [dataset.fft3c,dataset.ifft3c,source.ifft3c,transforms.fft3c,transforms.ifft3c]==original
    assert construction==dict(fft_batches=2,ifft_batches=6,scalar_fft_transforms=2,scalar_ifft_transforms=0,coil_fft_transforms=0,coil_ifft_transforms=12)
    baseline=cpu_export_fixture();baseline.config=copy.deepcopy(trainer.config);source.JointTrainer._setup_data(baseline)
    for tag in ('p','c'):
        assert getattr(trainer,'ds_'+tag).normalization_scale==getattr(baseline,'ds_'+tag).normalization_scale
        for name in ('csm','kspace','kspace_dc','mask','mask_fit'):
            assert torch.equal(getattr(trainer,name+'_'+tag),getattr(baseline,name+'_'+tag))
    controls,canonical=module.canonical_control(trainer,backend_kind='cpu_test');parent=module.supervisor()
    scans={scan:{'n_coils':2,'work':dict.fromkeys(parent.PHYSICS_COUNTERS,0)} for scan in ('pre','post')}
    total=module.total_work(canonical,scans,construction=construction)
    assert total['construction']==construction and total['total_scalar_fft_transforms']==2
    assert total['total_fft_batches']==4 and total['total_ifft_batches']==10
    assert total['total_coil_fft_transforms']==4 and total['total_coil_ifft_transforms']==20
    parent.validate_work_total({'scans':scans,'work_total':total},'canonical')


def test_constructor_observer_keeps_partial_work_and_restores_aliases_on_error():
    module=implementation();transforms=module.source_module('data.transforms','data/transforms.py')
    original_fft,original_ifft=transforms.fft3c,transforms.ifft3c
    construction=dict.fromkeys(module.CONSTRUCTION_COUNTERS,0)
    with pytest.raises(RuntimeError,match='controlled'):
        with module.observe_source_transforms(construction):
            transforms.fft3c(torch.ones((6,6,6),dtype=torch.complex64))
            transforms.ifft3c(torch.ones((2,6,6,6),dtype=torch.complex64))
            raise RuntimeError('controlled constructor interruption')
    assert transforms.fft3c is original_fft and transforms.ifft3c is original_ifft
    assert construction==dict(fft_batches=1,ifft_batches=1,scalar_fft_transforms=1,scalar_ifft_transforms=0,coil_fft_transforms=0,coil_ifft_transforms=2)


@pytest.mark.parametrize('failure_at',['construction','canonical','rhs','initial_objective','second_normal'])
def test_main_numeric_failure_preserves_prior_scan_control_and_construction(tmp_path,monkeypatch,failure_at):
    import json,os
    module=implementation();ph=module.physics();real_parent=module.supervisor();out=tmp_path/'outputs';out.mkdir();exp='controlled_partial_only'
    verify=tmp_path/'verification';verify.mkdir()
    review=tmp_path/'unused_review.json';model=tmp_path/'unused_model.json';launch_path=verify/f'{exp}_launch.json'
    tokens=['--stage','prox','--exp',exp,'--output',str(out),'--launch-manifest',str(launch_path),'--review-gate',str(review),'--model-definition',str(model)]
    I={'accepted_h1':{'definition_manifest':{},'definition':{}}};reserved=types.SimpleNamespace(stage='prox',exp=exp,review_gate=review,model_definition=model)
    launch={'command':[str(module.GSTMI),'-X','utf8','-B','-u',str(tmp_path/'runner.py'),*tokens],
            'parent_pid':os.getppid(),'initial_visibility':os.environ.get('CUDA_VISIBLE_DEVICES'),'control_bindings':{},
            'stage':'prox','request':{'exp':exp},'code_sha256':{},'invariants':I,'reservation_token':'controlled-unit-only'}
    fake=types.SimpleNamespace(source_hashes=lambda:{},snapshot_invariants=lambda *a:I,validate_launch=lambda *a:(launch,reserved),stage_profile=real_parent.stage_profile,sha256=module.sha256,control_bindings=lambda *a:{})
    launch_path.write_text(json.dumps(launch))
    expected={}
    def fail_stage(stage,directory,definition,invariants,progress):
        trainer=cpu_export_fixture()
        folder=tmp_path/'source_data';folder.mkdir();trainer.config=simulated_source_data_config(folder)
        source=module.source_module('trainers.joint_trainer','trainers/joint_trainer.py')
        progress['active_domain']='construction'
        if failure_at=='construction':
            dataset=sys.modules['data.dataset'];original_ifft=dataset.ifft3c
            def interrupt_constructor(*a,**kw):original_ifft(*a,**kw);raise RuntimeError('controlled constructor interruption')
            monkeypatch.setattr(dataset,'ifft3c',interrupt_constructor)
        with module.observe_source_transforms(progress['construction']):source.JointTrainer._setup_data(trainer)
        progress['active_domain']='canonical'
        if failure_at=='canonical':
            original_render=trainer._render;renders=0
            def interrupt_canonical(*a,**kw):
                nonlocal renders
                value=original_render(*a,**kw);renders+=1
                if renders==2:raise RuntimeError('controlled canonical interruption')
                return value
            trainer._render=interrupt_canonical
        controls,control=module.canonical_control(trainer,backend_kind='cpu_test',work=progress['canonical'])
        pre=ph.SenseOperator(trainer.csm_p,trainer.kspace_p,trainer.mask_p)
        module.proximal_scan(pre,controls['pre']['raw'],progress,'pre')
        post=ph.SenseOperator(trainer.csm_c,trainer.kspace_c,trainer.mask_c)
        name={'rhs':'adjoint','initial_objective':'objective','second_normal':'normal'}[failure_at]
        original=getattr(post,name);count=0
        def injected(*args,**kwargs):
            nonlocal count
            value=original(*args,**kwargs);count+=1
            if count==(2 if failure_at=='second_normal' else 1):raise RuntimeError('controlled interrupted scan')
            return value
        setattr(post,name,injected)
        try:module.proximal_scan(post,controls['post']['raw'],progress,'post')
        except RuntimeError as error:expected.update(error.proximal_work);raise
    monkeypatch.setattr(module,'physics',lambda:ph)
    monkeypatch.setattr(module,'ROOT',tmp_path);monkeypatch.setattr(module,'supervisor',lambda:fake)
    monkeypatch.setattr(module,'clinical_stage',fail_stage);monkeypatch.setattr(sys,'argv',[str(tmp_path/'runner.py'),*tokens])
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True);monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    with pytest.raises(RuntimeError,match='controlled') as caught:module.main(tokens)
    failure=json.loads((out/exp/'failure.json').read_text())
    assert failure['status']=='failed' and failure['observation_status']=='partial' and failure['elapsed_seconds']>=0
    partial=failure['partial_work']
    if failure_at=='construction':
        assert partial['active_domain']=='construction' and partial['construction']['ifft_batches']==1
        assert partial['construction']['coil_ifft_transforms']==2 and partial['construction']['fft_batches']==0
        assert not any(partial['canonical'].values()) and partial['scans']=={}
    elif failure_at=='canonical':
        assert partial['active_domain']=='canonical' and partial['construction']['ifft_batches']==6 and partial['scans']=={}
        assert partial['canonical']['gaussian_renders']==partial['canonical']['canonical_export_calls']==2
        assert partial['canonical']['canonical_fft_batches']==1 and partial['canonical']['canonical_ifft_batches']==2
    else:
        assert partial['construction']['ifft_batches']==6
        assert partial['canonical']['canonical_export_calls']==2 and partial['canonical']['canonical_ifft_batches']==4
        assert partial['scans']['pre']['status']=='completed' and partial['scans']['pre']['record']['status']=='converged'
        assert partial['scans']['post']['status']=='failed' and partial['scans']['post']['record'] is None
        assert partial['scans']['post']['work']==expected and expected['terminal_normal_calls']==0
        assert partial['active_domain']=='proximal' and partial['current_scan']=='post'
    assert partial['unassigned_proximal_work'] is None and not (out/exp/'provenance.json').exists()
    # Bind the actual caught numerical artifact through the production parent,
    # then through failure aggregation. These owned selection/launch fixtures
    # live only under /tmp and do not create any real review/control gate.
    launch['parent_pid']=os.getpid();launch_path.write_text(json.dumps(launch))
    failure['launch_sha256']=module.sha256(launch_path)
    (out/exp/'failure.json').write_text(json.dumps(failure))
    selection={k:launch[k] for k in ('stage','request','code_sha256','invariants','control_bindings','reservation_token','parent_pid')}
    selection['process_start']=real_parent.process_start()
    (verify/f'{exp}_selection.json').write_text(json.dumps(selection))
    monkeypatch.setattr(real_parent,'ROOT',tmp_path);monkeypatch.setattr(real_parent,'VERIFY',verify)
    real_parent.record_failure(types.SimpleNamespace(exp=exp,stage='prox',reservation_token=launch['reservation_token']),caught.value)
    parent_failure=json.loads((verify/f'{exp}_failure.json').read_text())
    observation=parent_failure['child_observation']
    assert observation['binding_status']=='verified' and observation['partial_work']==partial
    assert observation['elapsed_seconds']==failure['elapsed_seconds']
    scorer=module.load(ROOT/'scoring/evaluate.py','_px_failure_integration_scorer')
    monkeypatch.setattr(scorer,'ROOT',tmp_path)
    setup,formal,bound=scorer.observed_failures()
    assert not setup and len(formal)==1
    assert formal[0]['partial_work']==partial and formal[0]['measured_elapsed_seconds']==failure['elapsed_seconds']
    assert formal[0]['child_observation']==observation
    assert bound[str(out/exp/'failure.json')]==module.sha256(out/exp/'failure.json')
