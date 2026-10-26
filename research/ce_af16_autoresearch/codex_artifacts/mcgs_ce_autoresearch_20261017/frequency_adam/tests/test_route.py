"""CPU launch-boundary contracts; no test selects or initializes a GPU."""
from __future__ import annotations

import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest
import torch

from test_overrides import NAMES, LRS, assert_equal, load_file, restored

ROOT = Path(__file__).resolve().parents[1]
NVGEN = Path('/home/congcongliu/anaconda3/envs/nvgen/bin/python')
GSTMI = Path('/home/congcongliu/anaconda3/envs/GSTMI/bin/python')


def module(name):
    return load_file(ROOT / (name + '.py'), 'geometry_restart_test_' + name)


def test_new_runner_exists_before_integration():
    runner = module('runner')
    assert callable(runner.configure_runner)


def test_supervisor_fresh_process_is_stdlib_only_even_when_loading_C_selector():
    code = """import importlib.util,json,sys
spec=importlib.util.spec_from_file_location('geo_supervisor',sys.argv[1])
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
mod.load_campaign()
print(json.dumps({'torch': 'torch' in sys.modules,'numpy': 'numpy' in sys.modules,'yaml': 'yaml' in sys.modules,'python':sys.executable,'version':sys.version_info[:3]}))
"""
    result = subprocess.run([str(NVGEN), '-X', 'utf8', '-B', '-c', code, str(ROOT/'run_gpu.py')],
                            env={**os.environ, 'CUDA_VISIBLE_DEVICES':'', 'PYTHONDONTWRITEBYTECODE':'1'},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    assert actual['torch'] is actual['numpy'] is actual['yaml'] is False
    assert Path(actual['python']) == NVGEN
    assert actual['version'] == [3, 11, 15]


def test_model_child_describes_actual_GSTMI_torch_without_cuda_work():
    result = subprocess.run([str(GSTMI), '-X', 'utf8', '-B', str(ROOT/'runner.py'), '--describe-runtime'],
                            env={**os.environ, 'CUDA_VISIBLE_DEVICES':'', 'PYTHONDONTWRITEBYTECODE':'1'},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    assert Path(actual['python']) == GSTMI
    assert actual['torch_version'] == '2.7.1+cu118'
    assert actual['cuda_version'] == '11.8'


def test_nvgen_cannot_run_model_child_even_for_description():
    result = subprocess.run([str(NVGEN), '-X', 'utf8', '-B', str(ROOT/'runner.py'), '--describe-runtime'],
                            env={**os.environ, 'CUDA_VISIBLE_DEVICES':'', 'PYTHONDONTWRITEBYTECODE':'1'},
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert 'GSTMI' in result.stderr


@pytest.mark.parametrize('stage,factor,steps,mode', [('audit0',1.,0,'audit'), ('noop2',1.,2,'all'),
                                                  ('trial500',2.,500,'all'),('trial500',1.,500,'all')])
def test_stage_command_uses_fixed_child_interpreter_and_no_other_tuning_knobs(stage,factor,steps,mode):
    supervisor=module('run_gpu')
    args=supervisor.parser().parse_args(['--stage',stage,'--exp','fresh','--geometry-factor',str(factor),
                                       '--review-gate','review.json'])
    cmd=supervisor.child_command(args, Path('/tmp/launch.json'),7)
    assert cmd[:5] == [str(GSTMI),'-X','utf8','-B','-u']
    assert cmd[5] == str(ROOT/'runner.py')
    assert cmd[cmd.index('--stage')+1] == stage
    assert cmd[cmd.index('--gpu')+1] == '7'
    assert '--density-lr-factor' not in cmd and '--local-factor' not in cmd
    assert supervisor.stage_profile(stage,factor) == {'steps':steps,'mode':mode,'geometry_factor':factor,
                                                     'frequency_t':0. if stage=='noop2' else 1.}


@pytest.mark.parametrize('stage,factor', [('audit0',2),('noop2',4),('trial500',4),('trial500',3),('trial500',float('nan'))])
def test_stage_refuses_unreviewed_controls(stage,factor):
    with pytest.raises(ValueError):
        module('run_gpu').stage_profile(stage,factor)


def test_review_gate_binds_exact_three_files_and_immutable_invariants(tmp_path):
    s=module('run_gpu'); hashes={'a':'a'*64,'b':'b'*64,'c':'c'*64}; invariant={'source':'frozen'}
    gate=tmp_path/'review.json'
    gate.write_text(json.dumps({'status':'pass','round':1,'actionable_findings':0,'unresolved_findings':0,
                                'code_sha256':hashes,'invariants':invariant}))
    assert s.validate_review(gate,hashes,invariant)['status']=='pass'
    for change in ({'code_sha256':{'a':'bad'}},{'invariants':{}},{'actionable_findings':1},{'status':'pending'}):
        value=json.loads(gate.read_text());value.update(change);bad=tmp_path/'bad.json';bad.write_text(json.dumps(value))
        with pytest.raises(ValueError): s.validate_review(bad,hashes,invariant)


def test_claim_ownership_checks_token_pid_start_and_initial_visibility(monkeypatch,tmp_path):
    s=module('run_gpu'); monkeypatch.setenv('CUDA_VISIBLE_DEVICES','7')
    monkeypatch.setattr(s,'initial_visibility',lambda:'7')
    monkeypatch.setattr(s,'process_start',lambda:'123')
    monkeypatch.setattr(s,'CLAIM_LOCK',tmp_path/'claim.lock')
    monkeypatch.setattr(s,'control_bindings',lambda args:{'review-only-claim-fixture':'unchanged'})
    args=s.parser().parse_args(['--stage','audit0','--exp','owned','--review-gate','review.json',
                                '--reserved-gpu','7','--reservation-token','token'])
    hashes={'h':'x'};invariants={'i':'v'}
    with s.claim_lock(s.CLAIM_LOCK) as descriptor:
        args.claim_fd=descriptor
        selection={'parent_pid':os.getpid(),'process_start':'123','reservation_token':'token',
                   'request':s.request(args),'resource':{'gpu':7},'code_sha256':hashes,'invariants':invariants,
                   'control_bindings':s.control_bindings(args)}
        s.validate_reservation(args,selection,hashes,invariants)
        for key,value in [('parent_pid',os.getpid()+1),('process_start','old'),('reservation_token','old'),('request',{}),('control_bindings',{})]:
            wrong={**selection,key:value}
            with pytest.raises(ValueError): s.validate_reservation(args,wrong,hashes,invariants)
        monkeypatch.setattr(s,'initial_visibility',lambda:None)
        with pytest.raises(ValueError): s.validate_reservation(args,selection,hashes,invariants)


def test_wait_interruption_keeps_actual_child_exit_code():
    s=module('run_gpu')
    class Child:
        pid=123;returncode=None
        def poll(self):return self.returncode
        def wait(self):self.returncode=17;raise KeyboardInterrupt('observed interruption')
    with pytest.raises(KeyboardInterrupt) as raised:s.wait_owned_worker(Child())
    assert raised.value.worker_returncode==17 and raised.value.worker_pid==123


def test_wrapper_applies_only_after_original_restore_and_restores_process_hooks(restored,tmp_path,monkeypatch):
    r=module('runner'); base=r.load_base_runner(); protocol=r.load_resume_protocol()
    model,opt,sched=restored
    checkpoint=copy.deepcopy({'iter':4000,'model_state':model.state_dict(),'optim_state':opt.state_dict()})
    before_state=copy.deepcopy(opt.state_dict()['state'])
    old_restore=protocol.restore_checkpoint
    launch={'code_sha256':r.code_hashes(),'invariants':{'test':'only'}}
    # Reservation is tested separately; this fixture needs no file publication.
    with r.configure_runner(base,protocol,4.,launch) as record:
        result=protocol.restore_checkpoint(model,opt,sched,checkpoint)
        assert result['geometry_restart']['factor']==4.
        assert record['restore']['source']['groups'][0]['initial_lr']==.0003
        assert record['restore']['after']['groups'][0]['initial_lr']==.0012
        assert_equal(opt.state_dict()['state'],before_state)
        for i in range(3,7): assert opt.param_groups[i]['initial_lr']==.015
        for p in model.parameters():p.grad=torch.ones_like(p)
        opt.step();sched.step()
        current=copy.deepcopy((model.state_dict(),opt.state_dict(),sched.state_dict()))
        with pytest.raises(ValueError,match='second restore'):
            protocol.restore_checkpoint(model,opt,sched,checkpoint)
        assert_equal(current,(model.state_dict(),opt.state_dict(),sched.state_dict()))
    assert protocol.restore_checkpoint is old_restore
    assert base.ROOT == ROOT.parent/'nonlocal_autoresearch'


def test_wrapper_source_failure_precedes_override(restored):
    r=module('runner');base=r.load_base_runner();protocol=r.load_resume_protocol()
    model,opt,sched=restored
    bad={'iter':3999};before=copy.deepcopy((model.state_dict(),opt.state_dict(),sched.state_dict()))
    with r.configure_runner(base,protocol,2.,{'code_sha256':r.code_hashes(),'invariants':{}}) as record:
        with pytest.raises(ValueError,match='checkpoint iteration'):
            protocol.restore_checkpoint(model,opt,sched,bad)
        assert 'restore' not in record
    assert_equal(before,(model.state_dict(),opt.state_dict(),sched.state_dict()))


def test_config_changes_only_declared_resume_output_and_strict_backend():
    r=module('runner');base=r.load_base_runner();protocol=r.load_resume_protocol()
    original=base.build_config('all',500,ROOT/'outputs/future')
    with r.configure_runner(base,protocol,2.,{'code_sha256':r.code_hashes(),'invariants':{}}):
        actual=base.build_config('all',500,ROOT/'outputs/future')
    assert actual['gaussian']==original['gaussian']
    assert actual['loss']==original['loss']
    assert actual['training']==original['training']
    assert actual['mode']['strict_cuda'] is True
    assert actual['optimizer_overrides']['geometry_factor']==2.
    assert actual['campaign_overrides']['geometry_restart']==actual['optimizer_overrides']


def test_fresh_output_reservation_is_scoped_to_GEO(tmp_path):
    r=module('runner');base=r.load_base_runner();protocol=r.load_resume_protocol()
    with r.configure_runner(base,protocol,1.,{'code_sha256':r.code_hashes(),'invariants':{}}):
        with pytest.raises(ValueError,match='inside'):
            base.reserve_run(tmp_path,'escape')
        with pytest.raises(ValueError):base.reserve_run(ROOT/'outputs','../escape')


def test_wavelet_replay_matches_original_first_two_and_all_500():
    r=module('runner');protocol=r.load_resume_protocol()
    expected=r.expected_wavelet_shifts(500,protocol)
    assert expected[:4]==[[1,1,1],[1,1,1],[1,0,0],[1,1,1]]
    generator=torch.Generator(device='cpu').manual_seed(7961)
    draws=[torch.randint(0,2,(3,),generator=generator).tolist() for _ in range(4500)]
    assert expected==draws[4000:]


def test_partial_source_provenance_cannot_mask_original_wrapper_failure():
    r=module('runner');base=r.load_base_runner();protocol=r.load_resume_protocol()
    with tempfile.TemporaryDirectory(dir=ROOT/'evidence') as directory:
        with pytest.raises(RuntimeError,match='primary failure'):
            with r.configure_runner(base,protocol,1.,{'code_sha256':r.code_hashes(),'invariants':{}}) as record:
                run=base.reserve_run(directory,'fresh')
                (run/'provenance.json').write_text('{')
                raise RuntimeError('primary failure')
        actual=json.loads((run/'geometry_provenance.json').read_text())
        assert actual['status']=='failed'
        assert 'primary failure' in actual['error']
        assert 'JSONDecodeError' in actual['provenance_read_error']
        assert (run/'provenance.json').read_text()=='{'


def test_failure_marking_requires_this_child_owned_fresh_output(tmp_path):
    r=module('runner')
    old=tmp_path/'old.json';old.write_text('{"status":"complete"}')
    r.fail_owned_run(None,RuntimeError('not owned'))
    r.fail_owned_run({'run_dir':str(tmp_path),'child_pid':os.getpid()},RuntimeError('outside'))
    assert old.read_text()=='{"status":"complete"}'
    with tempfile.TemporaryDirectory(dir=ROOT/'evidence') as directory:
        path=Path(directory)/'provenance.json';path.write_text('{"status":"complete"}')
        r.fail_owned_run({'run_dir':directory,'child_pid':os.getpid()+1},RuntimeError('stale'))
        assert json.loads(path.read_text())['status']=='complete'
        r.fail_owned_run({'run_dir':directory,'child_pid':os.getpid()},RuntimeError('owned failure'))
        assert json.loads(path.read_text())['status']=='failed'


def test_failure_record_preserves_partial_json_and_actual_exit_status(monkeypatch,tmp_path):
    s=module('run_gpu');monkeypatch.setattr(s,'ROOT',tmp_path);monkeypatch.setattr(s,'VERIFY',tmp_path/'verification')
    s.VERIFY.mkdir();run=tmp_path/'outputs/owned';run.mkdir(parents=True);(run/'provenance.json').write_text('{')
    args=s.parser().parse_args(['--stage','audit0','--exp','owned','--review-gate','review.json','--reservation-token','token'])
    s.write_new(s.VERIFY/'owned_selection.json',{'parent_pid':os.getpid(),'process_start':s.process_start(),'reservation_token':'token'})
    failure=KeyboardInterrupt('actual interrupted wait');failure.worker_returncode=17;failure.worker_pid=123
    s.record_failure(args,failure)
    actual=s.read(s.VERIFY/'owned_failure.json')
    assert actual['returncode']==17 and actual['worker_pid']==123
    assert 'KeyboardInterrupt' in actual['error'] and 'JSONDecodeError' in actual['artifact_read_error']
    stale={**vars(args),'exp':'stale','reservation_token':'old'}
    s.write_new(s.VERIFY/'stale_selection.json',{'parent_pid':os.getpid()+1,'process_start':s.process_start(),'reservation_token':'old'})
    s.record_failure(SimpleNamespace(**stale),failure)
    assert not (s.VERIFY/'stale_failure.json').exists()


def test_observed_control_gate_cannot_be_a_boolean_without_real_artifacts(tmp_path):
    s=module('run_gpu');path=tmp_path/'fake.json';hashes={'h':'v'};invariants={'i':'v'}
    path.write_text(json.dumps({'passed':True,'status':'complete','stage':'audit0','code_sha256':hashes,'invariants':invariants}))
    with pytest.raises(ValueError,match='artifacts'):s.validate_observed_gate(path,'audit0',hashes,invariants)


@pytest.fixture
def toy_completed_run(tmp_path,monkeypatch):
    """An actual CPU Adam trajectory and checkpoint, small shapes but all 7 groups."""
    r=module('runner');s=r.supervisor();protocol=r.load_resume_protocol();base=r.load_base_runner()
    geometry=load_file(ROOT/'overrides.py','geometry_route_fixture_override')
    model=torch.nn.Module();shapes={}
    for i,name in enumerate(NAMES):
        shape=(2,3) if i<2 else (2,4) if i==2 else (2,)
        model.register_parameter(name,torch.nn.Parameter(torch.full(shape,i+.25,dtype=torch.float32)))
        shapes[name]=list(shape)
    opt=torch.optim.Adam([{'params':[p],'lr':lr} for p,lr in zip(model.parameters(),LRS)])
    sched=torch.optim.lr_scheduler.ExponentialLR(opt,gamma=.9995)
    for group in opt.param_groups:
        p=group['params'][0];group['lr']=group['initial_lr']*.9995**4000
        opt.state[p]={'step':torch.tensor(4000.),'exp_avg':torch.full_like(p,.2),'exp_avg_sq':torch.full_like(p,.04)}
    checkpoint={'iter':4000,'model_state':model.state_dict(),'optim_state':opt.state_dict()}
    protocol.restore_checkpoint(model,opt,sched,copy.deepcopy(checkpoint))
    trainer=SimpleNamespace(model=model,optimizer=opt,scheduler=sched)
    before=base.state_snapshot(trainer)
    applied=geometry.apply_geometry_lr_factor(model,opt,sched,2.)
    initial=geometry._snapshot(opt,sched)
    from test_frequency_adam import source_trainer,actual_loss_pair
    frequency_trainer=source_trainer()
    run=tmp_path/'outputs/toy';run.mkdir(parents=True)
    r.install_frequency_weights(frequency_trainer,1.,run)
    traces=[];shifts=r.expected_wavelet_shifts(500,protocol)
    for index in range(500):
        _,_,frequency_use=actual_loss_pair(frequency_trainer,r,4001+index)
        lrs={n:g['lr'] for n,g in zip(NAMES,opt.param_groups)}
        for p in model.parameters():p.grad=torch.full_like(p,.01)
        opt.step();sched.step()
        traces.append({'mode':'all','absolute_iteration':4001+index,'wavelet_shift':shifts[index],
                       'loss':1.,'unclipped_grad_norm':.1,'postclip_gradient_norm':.1,
                       'gradients_finite':True,'full_gradient_group_count':7,'updated_parameter_groups':list(NAMES),
                       'learning_rates_used':lrs,'learning_rates_next':{n:g['lr'] for n,g in zip(NAMES,opt.param_groups)},
                       'nonlocal':{'enabled':False},'frequency_use':frequency_use})
    after=base.state_snapshot(trainer);final=geometry._snapshot(opt,sched)
    frequency=r.frequency_record(frequency_trainer,2.)
    (run/'frequency_provenance.json').write_text(json.dumps(frequency))
    (tmp_path/'frequency_weights.py').write_bytes((ROOT/'frequency_weights.py').read_bytes())
    import yaml
    (run/'actual_config.yaml').write_text(yaml.safe_dump({'frequency_overrides':{
        'frequency_t':1.,'mask_source':'original acquired mask_p/mask_c; never fit masks',
        'utility_sha256':s.sha256(ROOT/'frequency_weights.py')}}))
    ckpt=run/'latest.pth'
    torch.save({'iter':4500,'model_state':model.state_dict(),'optim_state':opt.state_dict()},ckpt)
    r.attach_checkpoint_recipe(ckpt,frequency)
    invariant={'source_checkpoint_sha256':'f'*64,'source_config_sha256':'e'*64,
               'input_manifest':{'source_checkpoint':'/original/latest.pth','source_config':'/original/config.yaml'},
               'prebuilt_extension':{'path':'/original/source.so','sha256':'d'*64}}
    provenance={'status':'complete','mode':'all','source_iteration':4000,'end_iteration':4500,
                'source_model_shapes_before_restore':shapes,'forward_backward_evaluations':500,
                'source_checkpoint_sha256':'f'*64,'source_config_sha256':'e'*64,
                'source_checkpoint':'/original/latest.pth','source_config':'/original/config.yaml',
                'runner_sha256':s.sha256(r.BASE_RUNNER),'torch_version':'2.7.1+cu118','cuda_version':'11.8'}
    metrics={'mode':'all','source_iteration':4000,'end_iteration':4500,'forward_backward_evaluations':500,
             'source_checkpoint_sha256':'f'*64,'before':before,'after':after,'final_checkpoint':str(ckpt)}
    observed={'status':'complete','restore':applied,'after_restore':initial,'final':final,
              'backend':{'status':'constructed','compiler_branch_bypassed':True,'prebuilt_extension':invariant['prebuilt_extension']},
              'code_sha256':r.code_hashes(),'runtime':{'torch_version':'2.7.1+cu118'}}
    for name,value in [('provenance.json',provenance),('metrics.json',metrics),('geometry_provenance.json',observed)]:
        value.update(method_variant='frequency_adam',frequency_t=1.,geometry_factor=2.,frequency_adam=frequency)
        (run/name).write_text(json.dumps(value))
    (run/'step_trace.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in traces))
    reference=tmp_path/'reference';reference.mkdir();(reference/'metrics.json').write_text(json.dumps({'before':before}))
    monkeypatch.setattr(s,'NOOP_REFERENCES',(reference,));monkeypatch.setattr(r,'ROOT',tmp_path)
    monkeypatch.setattr(r,'supervisor',lambda:s);monkeypatch.setattr(r,'SOURCE_SHAPES',shapes)
    args=SimpleNamespace(stage='trial500',exp='toy',geometry_factor=2.)
    launch={'code_sha256':observed['code_sha256'],'invariants':invariant}
    return r,s,run,args,launch,protocol


def test_completed_real_CPU_adam_checkpoint_and_all_500_traces_are_verified(toy_completed_run):
    r,s,run,args,launch,protocol=toy_completed_run
    result=r.verify_run(args,launch,protocol)
    assert result['passed'] is True
    assert result['checkpoint']['iter']==4500
    assert result['checkpoint']['all_seven_model_and_moment_hashes_match'] is True
    assert result['trace']['count']==500 and result['trace']['first_shifts'][:2]==[[1,1,1],[1,1,1]]


@pytest.mark.parametrize('defect',['mode','budget','source_checkpoint','source_config','source_shape','rate','clock',
                                 'scheduler_base','scheduler_last','trace_lr','shift','checkpoint_model','checkpoint_iter','checkpoint_step'])
def test_run_validation_refuses_actual_source_budget_rate_shift_and_checkpoint_defects(toy_completed_run,defect):
    r,s,run,args,launch,protocol=toy_completed_run
    if defect in ('mode','budget','source_checkpoint','source_config','source_shape'):
        path=run/'provenance.json';value=s.read(path)
        if defect=='mode':value['mode']='density'
        elif defect=='budget':value['end_iteration']=4002
        elif defect=='source_checkpoint':value['source_checkpoint_sha256']='0'*64
        elif defect=='source_config':value['source_config_sha256']='0'*64
        else:value['source_model_shapes_before_restore']['positions']=[3,3]
        path.write_text(json.dumps(value))
    elif defect in ('rate','clock','scheduler_base','scheduler_last'):
        path=run/'geometry_provenance.json';value=s.read(path)
        if defect=='rate':value['final']['groups'][3]['lr']*=2
        elif defect=='clock':value['final']['scheduler']['last_epoch']=4002
        elif defect=='scheduler_base':value['final']['scheduler']['base_lrs'][0]*=2
        else:value['final']['scheduler']['last_lrs'][3]*=2
        path.write_text(json.dumps(value))
    elif defect in ('trace_lr','shift'):
        path=run/'step_trace.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
        if defect=='trace_lr':rows[-1]['learning_rates_used']['rotations']*=2
        else:rows[-1]['wavelet_shift']=[9,9,9]
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    else:
        path=run/'latest.pth';value=torch.load(path,weights_only=True)
        if defect=='checkpoint_iter':value['iter']=4002
        elif defect=='checkpoint_model':value['model_state']['positions'][0,0]+=1
        else:value['optim_state']['state'][0]['step'].fill_(4002)
        torch.save(value,path)
    with pytest.raises(ValueError):r.verify_run(args,launch,protocol)
    validation=s.read(run/'validation.json')
    assert validation['passed'] is False and validation['status']=='validation_failed'


@pytest.mark.parametrize('fallback',[False,True])
def test_verified_prebuilt_module_is_installed_before_constructor_and_fallback_is_rejected(tmp_path,monkeypatch,fallback):
    import hashlib
    r=module('runner');s=r.supervisor();folder=tmp_path/'trainer/paired_gs/gaussian/cuda_ext';folder.mkdir(parents=True)
    for name,value in [('tile_voxelize_cuda.cpp',b'cpp source'),('tile_voxelize_cuda_kernel.cu',b'cuda source')]:
        (folder/name).write_bytes(value)
    pieces=[hashlib.md5((folder/name).read_bytes()).hexdigest()[:8] for name in ('tile_voxelize_cuda.cpp','tile_voxelize_cuda_kernel.cu')]
    binary=folder/'build'/('tile_voxelize_cuda_'+'_'.join(pieces)+'.so');binary.parent.mkdir();binary.write_bytes(b'verified prebuilt')
    source=folder.parent/'tile_voxelizer.py';source.write_text('# fixture source\n')
    tile=SimpleNamespace(__file__=str(source),_cuda_ext=None,_cuda_load_error='previous error')
    extension=SimpleNamespace(__file__=str(binary),forward=lambda:None,backward=lambda:None)
    observed=[]
    def original_factory(cfg,device=None):
        assert tile._cuda_ext is extension and tile._cuda_load_error is None
        observed.append(cfg['mode']['strict_cuda'])
        return SimpleNamespace(voxelizer=SimpleNamespace(backend='reference' if fallback else 'tile_cuda',
                    strict_cuda=True,use_cuda=not fallback,use_cupy=False),writer=None)
    trainer_module=SimpleNamespace(__file__=str(tmp_path/'trainer/paired_gs/trainers/joint_trainer.py'),JointTrainer=original_factory)
    monkeypatch.setattr(r,'supervisor',lambda:s)
    monkeypatch.setattr(r,'load_file',lambda path,name:extension)
    monkeypatch.setattr(r.importlib,'import_module',lambda name:tile if name=='gaussian.tile_voxelizer' else trainer_module)
    base=SimpleNamespace(TRAINER=tmp_path/'trainer')
    record,restore=r.install_strict_backend(base,{'prebuilt_extension':{'path':str(binary),'sha256':s.sha256(binary)}})
    try:
        if fallback:
            with pytest.raises(RuntimeError,match='fallback'):trainer_module.JointTrainer({'mode':{'strict_cuda':True}})
        else:
            trainer_module.JointTrainer({'mode':{'strict_cuda':True}})
            assert record['status']=='constructed'
        assert observed==[True]
    finally:restore()
    assert tile._cuda_ext is None and tile._cuda_load_error=='previous error'
    assert trainer_module.JointTrainer is original_factory
