"""Private CPU artifact fixtures never qualify as scientific gates."""
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


def protocol():
    if importlib.util.find_spec('run_caseprior') is None:
        pytest.fail('case-prior public protocol is missing')
    return importlib.import_module('run_caseprior')


def test_parent_is_stdlib_only_and_cli_is_fixed():
    p=protocol()
    result=subprocess.run([sys.executable,'-B','-c',
        "import run_caseprior,sys; assert 'torch' not in sys.modules; run_caseprior.parser().parse_args(['--help'])"],
        capture_output=True,text=True,cwd=p.ROOT)
    assert result.returncode==0,result.stderr
    for flag in ('--width','--lambda-nl','--provider','--route','--updates'):
        with pytest.raises(SystemExit): p.parser().parse_args(['--stage','zero','--mode','control','--steps','0',
            '--exp','fixture','--review-gate','none','--recipe','none',flag,'1'])


def test_source_closure_keeps_original_helpers_and_new_component_bound():
    p=protocol();old=p.old.code_hashes();current=p.code_hashes()
    assert all(current[path]==digest for path,digest in old.items())
    for name in ('run_caseprior.py','caseprior_runner.py','case_denoiser.py','relocation_runner.py',
                 'relocation_runtime.py','run_relocation.py','run_gpu.py','runner.py'):
        assert current[str(p.ROOT/name)]==p.base.sha256(p.ROOT/name)


@pytest.mark.parametrize('kind',['old_route','missing_report','changed_report','findings'])
def test_review_refuses_old_or_incomplete_independent_evidence(tmp_path,kind):
    p=protocol();hashes={'fixture':'a'*64};invariants={'fixture':'b'*64}
    report=tmp_path/'private_report.txt';report.write_text('private CPU review-validator fixture\n')
    review=dict(schema_version=1,status='passed',passed=True,round=1,actionable_findings=0,unresolved_findings=0,
        code_sha256=hashes,invariants=invariants,route=p.ROUTE,review_report=p.base.descriptor(report))
    if kind=='old_route':review['route']='g2_fixed_count_relocation_v1'
    elif kind=='missing_report':review['review_report']['path']=str(tmp_path/'missing')
    elif kind=='changed_report':report.write_text('changed\n')
    elif kind=='findings':review['actionable_findings']=1
    path=tmp_path/'private_review_fixture.json';path.write_text(json.dumps(review))
    args=p.parser().parse_args(['--stage','zero','--mode','control','--steps','0','--exp','fixture',
        '--review-gate',str(path),'--recipe',str(tmp_path/'absent_recipe')])
    with pytest.raises((ValueError,FileNotFoundError)):p.validate_stage_gates(args,hashes,invariants)


@pytest.mark.parametrize('stage,mode,steps,valid', [('zero','control',0,True),('smoke','control',2,True),
    ('smoke','learned',2,True),('trial','learned',500,True),('trial','control',500,True),
    ('zero','learned',0,False),('smoke','learned',500,False),('trial','learned',200,False),
    ('zero','control',False,False),('smoke','relocate',2,False)])
def test_exact_public_budget(stage,mode,steps,valid):
    p=protocol();args=SimpleNamespace(route=p.ROUTE,stage=stage,mode=mode,steps=steps)
    if valid:p.validate_parameters(args)
    else:
        with pytest.raises(ValueError):p.validate_parameters(args)


def produced(directory,mode='learned',steps=2,scan_names=('plain','contrast')):
    import torch
    import caseprior_runner as child
    from test_caseprior_route import run_tiny,SMALL
    p=protocol()
    trainer,restore,result=run_tiny(directory,mode,steps,scan_names)
    result['arrays']=child.save_exports(result.pop('exports'),
        {s:dict(scale_trainer=1.,scale_gauge=1.,factor_scale=1.) for s in ('pre','post')},directory)
    result.update(restore=restore,data_evidence={s:dict(ground_truth_constructed=False,mask_inferred=False,
        outside_mask_nonzero=0,coils=2,shape=[3,3,3]) for s in ('pre','post')},zero_parity={})
    if not steps: result['zero_parity']={s:dict(passed=True,tolerance=1e-5,complex_relative_l2=0.,array_equal=True) for s in ('pre','post')}
    for name,value in [('config.json',trainer.config),('progress.json',dict(status='complete',counters=result['counters'],final_iteration=4500+steps))]:
        (directory/name).write_text(json.dumps(value))
    (directory/'config.yaml').write_text('private original trainer fixture\n')
    source=directory/'fixture_source.pth';torch.save({'fixture':True},source)
    hashes={'fixture':'0'*64}
    invariants=dict(route=p.ROUTE,policy=p.POLICY,g2_checkpoint=str(source),g2_checkpoint_sha256=p.base.sha256(source))
    args=SimpleNamespace(route=p.ROUTE,stage='zero' if not steps else 'smoke' if steps==2 else 'trial',mode=mode,steps=steps)
    launch=dict(**p.parameters(args),run_dir=str(directory),parent_pid=12345,gpu_uuid='CPU-FIXTURE',code_sha256=hashes,invariants=invariants)
    for name,value in [('launch.json',launch),('child_started.json',dict(pid=23456,gpu_uuid='CPU-FIXTURE')),
        ('childcontext_ready.json',dict(pid=23456,parent_pid=12345,gpu_uuid='CPU-FIXTURE')),
        ('childcontext_ack.json',dict(schema_version=1,child_pid=23456,parent_pid=12345,gpu_uuid='CPU-FIXTURE',context_observed=True))]:
        (directory/name).write_text(json.dumps(value))
    runtime=dict(device='cpu',torch=str(torch.__version__),cuda=torch.version.cuda,python=str(p.base.PYTHON),
        renderer='tiny_cpu_fixture',strict_cuda=False,python_version=list(sys.version_info[:3]),
        default_dtype=str(torch.get_default_dtype()),autocast_enabled=False,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    child.publish_result(launch,directory,result,runtime,source_checkpoint=source,elapsed=0.,peak_allocated=0.,peak_reserved=0.)
    return p,args,hashes,invariants


def validate_fixture(p,args,directory,hashes,invariants):
    from test_caseprior_route import SMALL
    return p._validate_child_artifacts(args,directory,hashes,invariants,point_count=5,device='cpu',backend='tiny_cpu_fixture',coils=2,
        denoiser_settings=SMALL)


def complete_fixture(p,directory,args,hashes,invariants,artifacts):
    """Private protocol metadata: CPU inputs cannot pass the real validator."""
    controls={str(directory/'config.yaml'):p.base.sha256(directory/'config.yaml')}
    proof=dict(status='verified',before=dict(code_sha256=hashes,invariants=invariants),
        after=dict(code_sha256=hashes,invariants=invariants),control_bindings_before=controls,control_bindings_after=controls)
    (directory/'post_run_invariants.json').write_text(json.dumps(proof))
    gate=dict(schema_version=1,passed=True,status='complete',**vars(args),parameters=vars(args),
        run_dir=str(directory),code_sha256=hashes,invariants=invariants,artifacts=artifacts,
        control_bindings=controls,post_run_invariants=p.base.descriptor(directory/'post_run_invariants.json'),
        worker_pid=23456,returncode=0,gpu_claim_policy=p.base.GPU_CLAIM_POLICY)
    path=directory/'completed.json';path.write_text(json.dumps(gate));return path,gate


@pytest.mark.parametrize('defect',[None,'source','inputs','route','mode','false_pass','pid','returncode','claim','proof',
    'control_binding','artifact','ack'])
def test_completed_gates_bind_actual_artifacts_ack_and_postproof(tmp_path,defect):
    from types import FunctionType
    p,args,hashes,invariants=produced(tmp_path,'control')
    artifacts=validate_fixture(p,args,tmp_path,hashes,invariants)
    path,gate=complete_fixture(p,tmp_path,args,hashes,invariants,artifacts)
    if defect=='source':gate['code_sha256']={}
    elif defect=='inputs':gate['invariants']={}
    elif defect=='route':gate['route']=gate['parameters']['route']='g2_fixed_count_relocation_v1'
    elif defect=='mode':gate['mode']=gate['parameters']['mode']='learned'
    elif defect=='false_pass':gate['passed']=False
    elif defect=='pid':gate['worker_pid']+=1
    elif defect=='returncode':gate['returncode']=1
    elif defect=='claim':gate['gpu_claim_policy']='unobserved'
    elif defect=='proof':
        proof=p.base.read(tmp_path/'post_run_invariants.json');proof['after']={}
        (tmp_path/'post_run_invariants.json').write_text(json.dumps(proof))
        gate['post_run_invariants']=p.base.descriptor(tmp_path/'post_run_invariants.json')
    elif defect=='control_binding':gate['control_bindings']={}
    elif defect=='artifact':(tmp_path/'scheduler.json').write_text('{}')
    elif defect=='ack':(tmp_path/'childcontext_ack.json').write_text('{}')
    path.write_text(json.dumps(gate))
    # Only the hardware/N boundary is private; observed gate bytecode and all
    # ACK/postproof/artifact/state checks are the same ones production executes.
    validator=FunctionType(p.validate_observed_gate.__code__,{**p.validate_observed_gate.__globals__,
        'validate_child_artifacts':lambda args,directory,hashes,invariants:validate_fixture(p,args,Path(directory),hashes,invariants)})
    if defect:
        with pytest.raises((ValueError,KeyError)):validator(path,'smoke',hashes,invariants,dict(mode='control'))
    else:
        assert validator(path,'smoke',hashes,invariants,dict(mode='control'))==gate
        with pytest.raises(ValueError):p.validate_observed_gate(path,'smoke',hashes,invariants,dict(mode='control'))


@pytest.mark.parametrize('mode,steps',[('control',0),('control',2),('learned',2),('learned',500)])
def test_actual_producer_to_validator_and_public_refusal(tmp_path,mode,steps):
    p,args,hashes,invariants=produced(tmp_path,mode,steps)
    artifacts=validate_fixture(p,args,tmp_path,hashes,invariants)
    assert str(tmp_path/'scheduler.json') in artifacts
    assert str(tmp_path/'events.jsonl') in artifacts
    with pytest.raises(ValueError):p.validate_child_artifacts(args,tmp_path,hashes,invariants)


def test_original_production_pre_post_history_names_are_not_gaussian_role_names(tmp_path):
    p,args,hashes,invariants=produced(tmp_path,scan_names=('pre','post'))
    validate_fixture(p,args,tmp_path,hashes,invariants)
    m=p.base.read(tmp_path/'metrics.json')
    assert set(m['nonlocal_summary']['refresh_history'])=={'pre','post'}
    assert [r['channel'] for r in m['counters']['render_calls'][:2]]==['plain','contrast']
    assert m['trace'][0]['frequency_use']['loss_call_scans']==['pre','post']


@pytest.mark.parametrize('wholly_inactive',[False,True])
def test_occasional_zero_prior_step_is_allowed_but_wholly_inactive_is_refused(tmp_path,monkeypatch,wholly_inactive):
    import runner
    sys.path[:0]=[str(runner.TRAINER/'paired_gs'),str(runner.TRAINER)]
    import losses.nonlocal_red as red
    original=red.NonLocalRED.__call__
    calls=0
    def prior(self,name,vol):
        nonlocal calls
        value=original(self,name,vol);calls+=1
        # Isolate an admissible zero-residual observation at the prior boundary;
        # all CNN fits, original Gaussian steps/summaries/saves remain real.
        return value*0 if wholly_inactive or calls<=2 else value
    monkeypatch.setattr(red.NonLocalRED,'__call__',prior)
    if wholly_inactive:
        with pytest.raises(RuntimeError,match='no validated active prior'):produced(tmp_path)
    else:
        p,args,hashes,invariants=produced(tmp_path)
        m=p.base.read(tmp_path/'metrics.json')
        assert m['nonlocal_summary']['weighted_R_nl_min']==0.
        assert m['nonlocal_summary']['nonzero_weighted_step_count']==1
        validate_fixture(p,args,tmp_path,hashes,invariants)


@pytest.mark.parametrize('defect',['seed','updates','bool_updates','identity','gradient','changed','hash','settings',
    'call_order','history','refresh','active','weighted','scan_loss','missing_call','target','manifest','checkpoint',
    'ack','gaussian_vjp','gaussian_forward','clock','source','policy','route','parameter_id','scheduler',
    'frequency','physics','event','prior_config','unlisted_artifact','rng_count','rng_cpu','rng_cuda','rng_index',
    'cnn_check_count','cnn_check_index','cnn_check_descriptor','cnn_check_digest','cnn_check_flag','cnn_check_shapes',
    'step_prior_raw','step_refresh_clock','summary_max','summary_last','summary_raw_max'])
def test_refuses_forged_learning_and_inherited_evidence(tmp_path,defect):
    p,args,hashes,invariants=produced(tmp_path)
    m=p.base.read(tmp_path/'metrics.json');r=m['denoiser_calls'][0];s=m['nonlocal_summary']
    if defect=='seed':r['seed']+=1
    elif defect=='updates':r['updates']=0
    elif defect=='bool_updates':r['updates']=True
    elif defect=='identity':r['status']='identity_zero_sigma'
    elif defect=='gradient':r['gradients']['any_nonzero']=False
    elif defect=='changed':r['changed_parameters']=[]
    elif defect=='hash':r['final_parameter_sha256']=r['initial_parameter_sha256']
    elif defect=='settings':r['patch_size']=32
    elif defect=='call_order':m['denoiser_calls'].reverse()
    elif defect=='history':s['refresh_history']['plain'][0]['kernel_stats']={}
    elif defect=='refresh':s['refresh_history']['contrast'][0]['iter']=4502
    elif defect=='active':s['active_step_count']=0
    elif defect=='weighted':s['weighted_R_nl_max']=0.
    elif defect=='scan_loss':s['raw_R_nl_by_channel_max']['contrast']=0.
    elif defect=='missing_call':m['denoiser_calls'].pop()
    elif defect in ('target','manifest','checkpoint'):
        with Path(r[defect]['path']).open('ab') as f:f.write(b'x')
    elif defect=='ack':(tmp_path/'childcontext_ack.json').write_text('{}')
    elif defect=='gaussian_vjp':m['counters']['gaussian_vjps']=0
    elif defect=='gaussian_forward':m['counters']['gaussian_forwards']+=2
    elif defect=='clock':m['after']['optimizer']['states']['positions']['step']['value']=4501
    elif defect=='source':m['source_checkpoint']['sha256']='1'*64
    elif defect=='policy':m['policy']={}
    elif defect=='route':m['model_schema']='g2_fixed_count_relocation_v1'
    elif defect=='parameter_id':m['after']['parameters']['positions']['tensor_id']+=1
    elif defect=='scheduler':m['after']['scheduler']['state']['_step_count']=4502
    elif defect=='frequency':m['frequency']['loss_calls_total']=0
    elif defect=='physics':m['physics_work']['export']['fft_batches']=0
    elif defect=='event':m['events']=[{}]
    elif defect=='prior_config':m['before']['config']['loss']['lambda_nl']=0.
    elif defect=='unlisted_artifact':(tmp_path/'forged.bin').write_bytes(b'x')
    elif defect=='rng_count':m['denoiser_rng']=[]
    elif defect=='rng_cpu':m['denoiser_rng'][0]['cpu']['after_sha256']='1'*64
    elif defect=='rng_cuda':m['denoiser_rng'][0]['cuda']=[dict(device='cuda:0',before_sha256='1'*64,after_sha256='1'*64,preserved=True)]
    elif defect=='rng_index':m['denoiser_rng'][0]['call_index']=1
    elif defect=='cnn_check_count':m['denoiser_checkpoint_checks']=[]
    elif defect=='cnn_check_index':m['denoiser_checkpoint_checks'][0]['call_index']=1
    elif defect=='cnn_check_descriptor':m['denoiser_checkpoint_checks'][0]['checkpoint']['sha256']='1'*64
    elif defect=='cnn_check_digest':m['denoiser_checkpoint_checks'][0]['final_parameter_sha256']='1'*64
    elif defect=='cnn_check_flag':m['denoiser_checkpoint_checks'][0]['metadata_equal']=False
    elif defect=='cnn_check_shapes':m['denoiser_checkpoint_checks'][0]['tensor_shapes']['network.0.weight']=[1]
    elif defect=='step_prior_raw':m['trace'][0]['nonlocal']['raw_by_channel']={}
    elif defect=='step_refresh_clock':m['trace'][0]['nonlocal']['refresh_iterations_by_channel']['plain']=[4502]
    elif defect=='summary_max':m['nonlocal_summary']['weighted_R_nl_max']+=1.
    elif defect=='summary_last':m['nonlocal_summary']['raw_R_nl_by_channel_last']['plain']+=1.
    elif defect=='summary_raw_max':m['nonlocal_summary']['raw_R_nl_by_channel_max']['plain']+=1.
    for filename,key in [('step_trace.jsonl','trace'),('events.jsonl','events')]:
        (tmp_path/filename).write_text(''.join(json.dumps(row)+'\n' for row in m[key]))
        m['artifact_manifest'][filename]=p.base.descriptor(tmp_path/filename)
    for name in ('metrics.json','provenance.json'):(tmp_path/name).write_text(json.dumps(m))
    with pytest.raises((ValueError,KeyError,FileNotFoundError)):
        validate_fixture(p,args,tmp_path,hashes,invariants)


@pytest.mark.parametrize('defect',['empty_losses','extra_field','missing_loss','seed_bool','gradient_steps','clipping_bool',
    'zero_sigma','negative_normalization','output_hash','unchanged_weights','no_changed_weights'])
def test_refuses_self_consistent_forged_task1_records(tmp_path,defect):
    p,args,hashes,invariants=produced(tmp_path)
    m=p.base.read(tmp_path/'metrics.json');r=m['denoiser_calls'][0]
    if defect=='empty_losses':r['losses']={}
    elif defect=='extra_field':r['fabricated_learning']=True
    elif defect=='missing_loss':r['losses'].pop('first')
    elif defect=='seed_bool':r['seed']=True
    elif defect=='gradient_steps':r['gradients']['steps_with_nonzero']=0
    elif defect=='clipping_bool':r['output']['clipped_count']=False
    elif defect=='zero_sigma':r['sigma']=0.;r['normalized_sigma']=0.
    elif defect=='negative_normalization':r['normalization']=-1.
    elif defect=='output_hash':r['output']['sha256']='1'*64
    elif defect=='unchanged_weights':r['final_parameter_sha256']=r['initial_parameter_sha256']
    elif defect=='no_changed_weights':r['changed_parameters']=[]
    # Rewrite every duplicate observation and descriptor so tests exercise the
    # semantic learning checks, rather than just duplicated metadata mismatch.
    target=Path(r['manifest']['path'])
    target.write_text(json.dumps({k:v for k,v in r.items() if k!='manifest'}))
    r['manifest']=p.base.descriptor(target)
    m['artifact_manifest'][str(target.relative_to(tmp_path))]=r['manifest']
    m['nonlocal_summary']['refresh_history']['plain'][0]['kernel_stats']=copy.deepcopy(r)
    m['nonlocal_summary']['refresh_history']['plain'][0]['sigma']=r['sigma']
    for name in ('metrics.json','provenance.json'):(tmp_path/name).write_text(json.dumps(m))
    with pytest.raises((ValueError,KeyError)):
        validate_fixture(p,args,tmp_path,hashes,invariants)


@pytest.mark.parametrize('defect',['source','inputs','parameters','policy','route'])
def test_exact_recipe_refuses_mismatch(tmp_path,monkeypatch,defect):
    p=protocol();args=p.parser().parse_args(['--stage','zero','--mode','control','--steps','0','--exp','fixture',
        '--review-gate',str(tmp_path/'unused'),'--recipe',str(tmp_path/'recipe.json')])
    hashes={'fixture':'a'*64};invariants={'fixture':'b'*64}
    monkeypatch.setattr(p.base,'validate_review',lambda *a:dict(route=p.ROUTE,review_report={'path':str(tmp_path/'unused')}))
    r=dict(schema_version=1,status='selected',route=p.ROUTE,parameters=p.parameters(args),policy=copy.deepcopy(p.POLICY),code_sha256=hashes,invariants=invariants)
    if defect=='source':r['code_sha256']={}
    elif defect=='inputs':r['invariants']={}
    elif defect=='parameters':r['parameters']['steps']=2
    elif defect=='policy':r['policy']['seed']=0
    elif defect=='route':r['route']='g2_fixed_count_relocation_v1'
    args.recipe.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='exact selected'):p.validate_stage_gates(args,hashes,invariants)


def native_fixture(tmp_path):
    """Actual original CPU producers and evaluator's pure objective producer.

    The explicitly private completion metadata below exercises protocol bindings;
    it does not represent an observed GPU claim or authorize public execution.
    """
    from test_relocation_protocol import produced_cpu_artifacts
    import run_relocation as old
    import evaluate
    p=protocol()
    def complete(*args):return complete_fixture(p,*args)
    records=[]
    for name,steps in [('native',500),('old_smoke',2)]:
        directory=tmp_path/name;directory.mkdir()
        _,args,hashes,invariants=produced_cpu_artifacts(directory,'control',steps)
        args.stage='trial' if steps==500 else 'smoke'
        for filename in ('launch.json','metrics.json','provenance.json'):
            value=p.base.read(directory/filename);value['stage']=args.stage
            if records:
                invariants=records[0][1]['invariants']
                value['invariants']=invariants
                if filename!='launch.json':value['source_checkpoint']=dict(path=invariants['g2_checkpoint'],sha256=invariants['g2_checkpoint_sha256'])
            (directory/filename).write_text(json.dumps(value))
        artifacts=old._validate_child_artifacts(args,directory,hashes,invariants,point_count=5,device='cpu',backend='tiny_cpu_fixture',coils=2)
        records.append(complete(directory,args,hashes,invariants,artifacts))
    directory=tmp_path/'new_smoke';directory.mkdir()
    _,args,hashes,invariants=produced(directory,'control',2)
    # Fresh common inputs have the same physical/source checkpoint binding as
    # the old run; only this private fixture's scalar route/policy differ.
    old_inv=p.base.read(records[0][0])['invariants']
    invariants={**old_inv,'route':p.ROUTE,'policy':p.POLICY}
    for filename in ('launch.json','metrics.json','provenance.json'):
        value=p.base.read(directory/filename);value['invariants']=invariants
        value['source_checkpoint']=dict(path=invariants['g2_checkpoint'],sha256=invariants['g2_checkpoint_sha256']) if filename!='launch.json' else None
        if filename=='launch.json':value.pop('source_checkpoint')
        (directory/filename).write_text(json.dumps(value))
    artifacts=validate_fixture(p,args,directory,hashes,invariants)
    new_path,new_gate=complete(directory,args,hashes,invariants,artifacts)
    native_dir=Path(records[0][1]['run_dir'])
    candidate={s:{**p.base.descriptor(native_dir/f'candidate_{s}_complex.npy'),'shape':[3,3,3],'dtype':'complex64'} for s in ('pre','post')}
    inputs=dict(candidate=candidate,files={str(Path(evaluate.__file__).resolve()):p.base.sha256(evaluate.__file__),
        **{r['path']:r['sha256'] for r in candidate.values()}})
    objective=evaluate.build_objective('private_fixture',native_dir,
        dict(score=-.1,strict_success=False,metrics={s:dict(psnr=1.,ssim=.2,nmse=.3) for s in ('pre','post')},comparisons={}),
        dict(eligible_planes=59,planes=[]),inputs)
    objective['all_inputs_unchanged']=True
    score=tmp_path/'fixture_objective.json';score.write_text(json.dumps(objective))
    receipt=dict(schema='native_control_receipt_v1',route=p.ROUTE,status='verified',code_sha256=hashes,invariants=invariants,
        native_code_sha256=records[0][1]['code_sha256'],native_invariants=old_inv,
        native_completion=p.base.descriptor(records[0][0]),native_smoke=p.base.descriptor(records[1][0]),
        ordinary_smoke=p.base.descriptor(new_path),objective=p.base.descriptor(score))
    path=tmp_path/'private_receipt_fixture.json';path.write_text(json.dumps(receipt))
    def validate_old(path,stage,hashes,invariants,matched):
        from types import FunctionType
        def artifacts(args,directory,hashes,invariants):
            return old._validate_child_artifacts(args,directory,hashes,invariants,
                point_count=5,device='cpu',backend='tiny_cpu_fixture',coils=2)
        fn=FunctionType(old.validate_observed_gate.__code__,{**old.validate_observed_gate.__globals__,'validate_child_artifacts':artifacts})
        return fn(path,stage,hashes,invariants,matched)
    return p,path,hashes,invariants,new_gate,validate_old


def test_native_receipt_uses_original_route_closure_and_scored_arrays(tmp_path):
    p,path,hashes,invariants,new_gate,validator=native_fixture(tmp_path)
    receipt=p._validate_native_control_receipt(path,hashes,invariants,new_gate,
        old_hashes={'fixture':'0'*64},old_gate_validator=validator)
    assert str(tmp_path/'fixture_objective.json') in receipt['bindings']
    with pytest.raises(ValueError):p.validate_native_control_receipt(path,hashes,invariants,new_gate)


@pytest.mark.parametrize('defect',['new_source_as_old','native_policy','common_inputs','native_mode','native_steps',
    'old_source','old_proof','old_ack','objective_array','objective_route','objective_score','ordinary_smoke',
    'smoke_parity','old_smoke_mode','artifact_bytes'])
def test_native_receipt_refuses_stale_mismatched_or_unscored_control(tmp_path,defect):
    p,path,hashes,invariants,new_gate,validator=native_fixture(tmp_path)
    r=p.base.read(path)
    if defect=='new_source_as_old':r['native_code_sha256']={'new_route':'1'*64}
    elif defect=='native_policy':r['native_invariants']['policy']=p.POLICY
    elif defect=='common_inputs':r['native_invariants']['g2_checkpoint_sha256']='a'*64
    elif defect=='ordinary_smoke':r['ordinary_smoke']=r['native_smoke']
    elif defect.startswith('objective'):
        target=Path(r['objective']['path']);v=p.base.read(target)
        if defect=='objective_array':v['candidate_sha256']['pre']='1'*64
        elif defect=='objective_route':v['run_dir']=str(tmp_path/'other')
        elif defect=='objective_score':v['score']=True
        target.write_text(json.dumps(v));r['objective']=p.base.descriptor(target)
    elif defect=='smoke_parity':
        import numpy as np
        np.save(tmp_path/'new_smoke'/'candidate_pre_complex.npy',np.zeros((3,3,3),np.complex64))
    elif defect=='artifact_bytes':
        with (tmp_path/'native'/'candidate_pre_complex.npy').open('ab') as f:f.write(b'x')
    else:
        target=Path(r['native_smoke' if defect=='old_smoke_mode' else 'native_completion']['path']);v=p.base.read(target)
        if defect in ('native_mode','old_smoke_mode'):v['mode']=v['parameters']['mode']='relocate'
        elif defect=='native_steps':v['steps']=v['parameters']['steps']=2
        elif defect=='old_source':v['code_sha256']={}
        elif defect=='old_proof':
            proof=Path(v['post_run_invariants']['path']);proof.write_text('{}');v['post_run_invariants']=p.base.descriptor(proof)
        elif defect=='old_ack':(tmp_path/'native'/'childcontext_ack.json').write_text('{}')
        target.write_text(json.dumps(v));r['native_smoke' if defect=='old_smoke_mode' else 'native_completion']=p.base.descriptor(target)
    path.write_text(json.dumps(r))
    with pytest.raises((ValueError,KeyError,FileNotFoundError)):
        p._validate_native_control_receipt(path,hashes,invariants,new_gate,old_hashes={'fixture':'0'*64},old_gate_validator=validator)
