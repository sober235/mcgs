"""Parent is stdlib-only, fixed recipes and real-only gates fail closed."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def protocol():
    if not (ROOT/'run_relocation.py').exists():
        pytest.fail('relocation protocol production behavior is missing')
    import run_relocation
    return run_relocation


def test_parent_import_and_cli_never_import_torch():
    protocol()
    out = subprocess.run([sys.executable, '-B', '-c',
        "import run_relocation,sys; assert 'torch' not in sys.modules; run_relocation.parser().parse_args(['--help'])"],
        cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert '--stage' in out.stdout and '--mode' in out.stdout


@pytest.mark.parametrize('stage,mode,steps,valid', [
    ('zero','control',0,True), ('smoke','control',2,True), ('smoke','relocate',2,True),
    ('trial','control',500,True), ('trial','relocate',500,True),
    ('zero','relocate',0,False), ('smoke','control',0,False), ('trial','relocate',200,False),
    ('trial','explicit',500,False), ('zero','control',False,False)])
def test_exact_budget_route(stage,mode,steps,valid):
    from types import SimpleNamespace
    p = protocol()
    args = SimpleNamespace(route=p.ROUTE,stage=stage,mode=mode,steps=steps,
                           requested_k=18000,event_step=1)
    if valid: p.validate_parameters(args)
    else:
        with pytest.raises(ValueError): p.validate_parameters(args)


def produced_cpu_artifacts(directory,mode='relocate',steps=2):
    """Produce genuine tiny artifacts; fixture runtime can never pass public gates."""
    import json
    from types import SimpleNamespace
    import torch
    import relocation_runner as child
    from test_relocation_runner import tiny_trainer
    p=protocol()
    trainer,restore,frequency,rotation=tiny_trainer(directory,steps)
    result=child.run_continuation(trainer,frequency,directory,mode=mode,steps=steps,
                                  rotation_matrix_fn=rotation,requested_k=1)
    gauges={s:dict(scale_trainer=1.,scale_gauge=1.,factor_scale=1.) for s in ('pre','post')}
    result['arrays']=child.save_exports(result.pop('exports'),gauges,directory)
    result.update(restore=restore,data_evidence={s:dict(ground_truth_constructed=False,mask_inferred=False,
        outside_mask_nonzero=0,coils=2,shape=[3,3,3]) for s in ('pre','post')},zero_parity={})
    if steps==0:
        result['zero_parity']={s:dict(passed=True,tolerance=1e-5,complex_relative_l2=0.,array_equal=True)
                               for s in ('pre','post')}
    for name,value in [('config.json',trainer.config),('progress.json',dict(status='complete',counters=result['counters'],final_iteration=4500+steps))]:
        (directory/name).write_text(json.dumps(value))
    (directory/'config.yaml').write_text('tiny CPU fixture only\n')
    source=directory/'fixture_source.pth';torch.save({'fixture':True},source)
    hashes={'fixture':'0'*64}
    invariants=dict(route=p.ROUTE,policy=p.POLICY,g2_checkpoint=str(source),g2_checkpoint_sha256=p.base.sha256(source))
    args=SimpleNamespace(route=p.ROUTE,stage='smoke' if steps else 'zero',mode=mode,steps=steps,requested_k=18000,event_step=1)
    launch=dict(**p.parameters(args),run_dir=str(directory),parent_pid=12345,gpu_uuid='CPU-FIXTURE',
                code_sha256=hashes,invariants=invariants)
    for name,value in [('launch.json',launch),('child_started.json',dict(pid=23456,gpu_uuid='CPU-FIXTURE')),
        ('childcontext_ready.json',dict(pid=23456,parent_pid=12345,gpu_uuid='CPU-FIXTURE')),
        ('childcontext_ack.json',dict(schema_version=1,child_pid=23456,parent_pid=12345,
                                     gpu_uuid='CPU-FIXTURE',context_observed=True))]:
        (directory/name).write_text(json.dumps(value))
    runtime=dict(device='cpu',torch=str(torch.__version__),cuda=torch.version.cuda,python=str(p.base.PYTHON),
                 renderer='tiny_cpu_fixture',strict_cuda=False,python_version=list(sys.version_info[:3]),
                 default_dtype=str(torch.get_default_dtype()),autocast_enabled=torch.is_autocast_enabled(),
                 cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    child.publish_result(launch,directory,result,runtime,source_checkpoint=source,
                         elapsed=0.,peak_allocated=0.,peak_reserved=0.)
    return p,args,hashes,invariants


@pytest.mark.parametrize('mode,steps',[('control',0),('control',2),('relocate',2)])
def test_real_tiny_producer_to_parent_validator(tmp_path,mode,steps):
    p,args,hashes,invariants=produced_cpu_artifacts(tmp_path,mode,steps)
    artifacts=p._validate_child_artifacts(args,tmp_path,hashes,invariants,
        point_count=5,device='cpu',requested_k=1,backend='tiny_cpu_fixture',coils=2)
    assert str(tmp_path/'scheduler.json') in artifacts
    assert str(tmp_path/'events.jsonl') in artifacts
    with pytest.raises(ValueError):
        p.validate_child_artifacts(args,tmp_path,hashes,invariants)


@pytest.mark.parametrize('defect',[
    'adam_clock','scheduler_clock','parameter_id','moment_id','point_growth','extra_network',
    'missing_gradient','counter_bool','counter_missing','forward_budget','vjp_budget','render_order',
    'event_count','event_iteration','event_overlap','event_untouched','event_zero_hash','event_clock',
    'event_missing_state','restore_false','source_checkpoint','checkpoint_hash','export_bytes',
    'ack','runtime','config','false_gradients','frequency','scheduler_bytes',
    'physics_missing','physics_event','physics_export','frequency_count','frequency_arrays',
    'event_count_bool','event_missing_steps','checkpoint_bytes','progress_bytes','nonfinite_export',
    'event_foreign_moment_id','event_foreign_model_id','precision','telemetry'])
def test_tiny_parent_rejects_changed_or_incomplete_actual_evidence(tmp_path,defect):
    import copy
    import json
    p,args,hashes,invariants=produced_cpu_artifacts(tmp_path)
    metrics=p.base.read(tmp_path/'metrics.json')
    event=metrics['events'][0];tx=event['transaction']
    if defect=='adam_clock': metrics['after']['optimizer']['states']['positions']['step']['value']=4501
    elif defect=='scheduler_clock': metrics['after']['scheduler']['state']['_step_count']=4502
    elif defect=='parameter_id': metrics['after']['parameters']['positions']['tensor_id']+=1
    elif defect=='moment_id': metrics['after']['optimizer']['states']['positions']['exp_avg']['tensor_id']+=1
    elif defect=='point_growth': metrics['after']['parameters']['positions']['shape'][0]+=1
    elif defect=='extra_network': metrics['after']['parameters']['network.weight']=copy.deepcopy(metrics['after']['parameters']['positions'])
    elif defect=='missing_gradient': metrics['trace'][0]['gradient_norms'].pop('scales')
    elif defect=='counter_bool': metrics['counters']['event_calls']=True
    elif defect=='counter_missing': metrics['counters'].pop('gaussian_vjps')
    elif defect=='forward_budget': metrics['counters']['gaussian_forwards']+=2
    elif defect=='vjp_budget': metrics['counters']['gaussian_vjps']-=1
    elif defect=='render_order': metrics['counters']['render_calls'][0]['role']='event'
    elif defect=='event_count': metrics['events'].append(copy.deepcopy(event))
    elif defect=='event_iteration': event['iteration']=4502
    elif defect=='event_overlap': event['selection']['donors']=event['selection']['victims'][:]
    elif defect=='event_untouched': tx['after']['parameters']['positions']['untouched_sha256']='0'*64
    elif defect=='event_zero_hash': tx['after']['moments']['scales']['exp_avg']['changed_sha256']='0'*64
    elif defect=='event_clock': tx['after']['steps']['positions']=0
    elif defect=='event_missing_state': tx['before']['moments'].pop('scales')
    elif defect=='restore_false': metrics['restore']['after']['optimizer']['states']['positions']['step']['value']=0
    elif defect=='source_checkpoint': metrics['source_checkpoint']['sha256']='0'*64
    elif defect=='checkpoint_hash': metrics['final_checkpoint_evidence']['adam_sha256']['positions']['step']='0'*64
    elif defect=='export_bytes':
        with (tmp_path/'candidate_pre_complex.npy').open('ab') as stream: stream.write(b'x')
    elif defect=='ack':
        (tmp_path/'childcontext_ack.json').write_text(json.dumps(dict(context_observed=True)))
    elif defect=='runtime': metrics['runtime']['renderer']='cuda_fallback'
    elif defect=='config': metrics['before']['config']['adaptive_control']['enable']=True
    elif defect=='false_gradients': metrics['trace'][0]['gradients_finite']=False
    elif defect=='frequency': metrics['trace'][0]['frequency_use']['source_offset']=4000
    elif defect=='scheduler_bytes': (tmp_path/'scheduler.json').write_text('{}')
    elif defect=='physics_missing': metrics.pop('physics_work')
    elif defect=='physics_event': metrics['physics_work']['event']['fft_batches']=0
    elif defect=='physics_export': metrics['canonical_work']['canonical_ifft_batches']=3
    elif defect=='frequency_count': metrics['frequency']['loss_calls_total']=0
    elif defect=='frequency_arrays': metrics['frequency']['frequency_evidence']['scans']['pre']['effective_weight']['file_sha256']='0'*64
    elif defect=='event_count_bool': tx['k_eff']=True
    elif defect=='event_missing_steps': tx['before']['step_ids'].pop('positions')
    elif defect=='checkpoint_bytes':
        with Path(metrics['final_checkpoint']).open('ab') as stream: stream.write(b'x')
    elif defect=='progress_bytes': (tmp_path/'progress.json').write_text('{}')
    elif defect=='nonfinite_export':
        import numpy as np
        np.save(tmp_path/'candidate_pre_complex.npy',np.full((3,3,3),complex(float('nan')),dtype=np.complex64))
        metrics['artifact_manifest']['candidate_pre_complex.npy']=p.base.descriptor(tmp_path/'candidate_pre_complex.npy')
    elif defect=='event_foreign_moment_id':
        for state in (tx['before'],tx['after']): state['moments']['positions']['exp_avg']['id']+=1
    elif defect=='event_foreign_model_id':
        for state in (tx['before'],tx['after']): state['model_id']+=1
    elif defect=='precision': metrics['runtime']['autocast_enabled']=True
    elif defect=='telemetry': metrics['training_seconds']=True
    # Keep duplicate observations synchronized so the semantic validator, rather
    # than a trivial metrics/provenance disagreement, catches altered metadata.
    for name,key in [('events.jsonl','events'),('step_trace.jsonl','trace')]:
        (tmp_path/name).write_text(''.join(json.dumps(row)+'\n' for row in metrics[key]))
        metrics['artifact_manifest'][name]=p.base.descriptor(tmp_path/name)
    for name in ('metrics.json','provenance.json'): (tmp_path/name).write_text(json.dumps(metrics))
    with pytest.raises((ValueError,KeyError)):
        p._validate_child_artifacts(args,tmp_path,hashes,invariants,
            point_count=5,device='cpu',requested_k=1,backend='tiny_cpu_fixture',coils=2)


def test_foreign_old_and_stale_gates_never_authorize_route(tmp_path):
    import json
    p=protocol()
    for route in ('g2_raw_proxdc_v1',p.ROUTE):
        gate=tmp_path/'gate.json';gate.write_text(json.dumps(dict(schema_version=1,route=route,
            passed=True,status='complete',stage='zero',parameters={})))
        with pytest.raises(ValueError):p.validate_observed_gate(gate,'zero',{}, {})


@pytest.mark.parametrize('defect', ['source','inputs','parameters','policy','route'])
def test_selected_recipe_must_bind_exact_cli_source_inputs_policy(tmp_path,monkeypatch,defect):
    import copy
    import json
    p=protocol()
    args=p.parser().parse_args(['--stage','zero','--mode','control','--steps','0','--exp','fixture',
        '--review-gate',str(tmp_path/'unused'), '--recipe',str(tmp_path/'fixture-recipe.json')])
    hashes={'fixture_source':'a'*64};invariants={'fixture_input':'b'*64}
    # Isolate selected-recipe validation from the separately tested real-review
    # descriptor validator. This test never writes a passed review or control.
    monkeypatch.setattr(p.base,'validate_review',lambda *a:dict(route=p.ROUTE,review_report={'path':str(tmp_path/'unused')}))
    recipe=dict(schema_version=1,route=p.ROUTE,status='selected',parameters=p.parameters(args),
                policy=copy.deepcopy(p.POLICY),code_sha256=hashes,invariants=invariants)
    if defect=='source': recipe['code_sha256']={}
    elif defect=='inputs': recipe['invariants']={}
    elif defect=='parameters': recipe['parameters']['steps']=2
    elif defect=='policy': recipe['policy']['event_step']=2
    elif defect=='route': recipe['route']='g2_raw_proxdc_v1'
    args.recipe.write_text(json.dumps(recipe))
    with pytest.raises(ValueError,match='exact selected'):p.validate_stage_gates(args,hashes,invariants)


@pytest.mark.parametrize('defect', ['stale_source','stale_inputs','foreign_route','unmatched_smoke','changed_artifacts','false_pass'])
def test_control_gate_rejects_stale_foreign_or_unmatched_metadata(tmp_path,defect):
    import json
    p=protocol()
    params=dict(route=p.ROUTE,stage='smoke',mode='control',steps=2,requested_k=18000,event_step=1)
    hashes={'fixture':'a'*64};invariants={'fixture':'b'*64}
    gate=dict(schema_version=1,passed=True,status='complete',**params,parameters=params,
              code_sha256=hashes,invariants=invariants,artifacts={str(tmp_path/'missing'):'c'*64})
    if defect=='stale_source': gate['code_sha256']={}
    elif defect=='stale_inputs': gate['invariants']={}
    elif defect=='foreign_route': gate['route']='g2_raw_proxdc_v1'
    elif defect=='false_pass': gate['passed']=False
    path=tmp_path/'fixture-rejected-gate.json';path.write_text(json.dumps(gate))
    with pytest.raises((ValueError,FileNotFoundError)):
        p.validate_observed_gate(path,'smoke',hashes,invariants,
                                dict(mode='relocate') if defect=='unmatched_smoke' else None)


def test_parent_rejects_torch_import_before_any_resource_action():
    import torch
    p=protocol()
    with pytest.raises(ValueError,match='must not import Torch'):
        p.base.execute(object(),protocol=p)


@pytest.mark.parametrize('argument',['--provider','--requested-k','--event-step','--route'])
def test_public_cli_cannot_select_test_provider_or_unreviewed_policy(argument):
    p=protocol()
    with pytest.raises(SystemExit):p.parser().parse_args(['--stage','zero','--mode','control','--steps','0',
        '--exp','fixture','--review-gate','missing','--recipe','missing',argument,'1'])
