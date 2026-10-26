"""Independent G2/three-metric protocol; these are temporary CPU fixtures."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest


def module():
    import run_proxdc
    return run_proxdc


def save(path, value):
    path.write_text(json.dumps(value))
    return path


def arguments(tmp_path, stage='tiny'):
    s = module()
    return s.parser().parse_args(['--stage', stage, '--exp', 'new', '--review-gate',
        str(tmp_path/'review.json'), '--recipe', str(tmp_path/'recipe.json')])


def reviewed(tmp_path, stage='tiny'):
    s = module(); a = arguments(tmp_path, stage)
    hashes, invariant = {'source':'hash'}, {'route':s.ROUTE, 'metric_contract':'AF16_PSNR_SSIM_NMSE'}
    report = tmp_path/'review.md'; report.write_text('Independent route review; no actionable findings.')
    save(a.review_gate, dict(schema_version=1,status='passed',passed=True,round=1,
        actionable_findings=0,unresolved_findings=0,route=s.ROUTE,
        review_report=s.base.descriptor(report),code_sha256=hashes,invariants=invariant))
    save(a.recipe, dict(schema_version=1,status='selected',route=s.ROUTE,
        parameters=s.parameters(a),code_sha256=hashes,invariants=invariant))
    return a, hashes, invariant


def test_new_route_requires_a_real_g2_three_metric_review(tmp_path):
    s = module(); args, hashes, invariant = reviewed(tmp_path)
    assert s.validate_stage_gates(args,hashes,invariant)['control_bindings']
    review = s.base.read(args.review_gate); review.pop('route'); save(args.review_gate,review)
    with pytest.raises(ValueError,match='route'):
        s.validate_stage_gates(args,hashes,invariant)


@pytest.mark.parametrize('name,value', [('lambda_ratio',.2),('cg_max_matvecs',33),('cg_rtol',1e-4),
                                       ('route','H1_four_metric'),('stage','trial')])
def test_no_lambda_sweep_budget_change_or_old_route_alias(tmp_path,name,value):
    args = arguments(tmp_path); setattr(args,name,value)
    with pytest.raises(ValueError): module().validate_parameters(args)


@pytest.mark.parametrize('mutation', [lambda v:v['parameters'].pop('lambda_ratio'),
    lambda v:v['parameters'].update(stage='prox'),lambda v:v.update(route='H1_four_metric'),
    lambda v:v.update(code_sha256={'stale':'hash'})])
def test_selected_recipe_binds_exact_new_route_and_current_sources(tmp_path,mutation):
    s=module();args,hashes,invariant=reviewed(tmp_path)
    recipe=s.base.read(args.recipe);mutation(recipe);save(args.recipe,recipe)
    with pytest.raises(ValueError):s.validate_stage_gates(args,hashes,invariant)


def test_canonical_and_prox_require_fresh_observed_prior_stages(tmp_path):
    s=module()
    for stage in ('canonical','prox'):
        args,hashes,invariant=reviewed(tmp_path,stage)
        with pytest.raises(ValueError,match='tiny'):s.validate_stage_gates(args,hashes,invariant)


def test_public_import_and_help_are_stdlib_only():
    result=subprocess.run([sys.executable,'-B','-c',
        "import run_proxdc,sys; assert 'torch' not in sys.modules;run_proxdc.parser().parse_args(['--help'])"],
        capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert '--canonical-gate' in result.stdout and '--tiny-gate' in result.stdout


@pytest.mark.parametrize('name,value',[('lambda_effective',.6),('true_relative_residual',.2),
    ('J_after',1e9),('directional_derivative',1.),('initial_residual_normal_evaluations',0),
    ('true_residual_tolerance',.1)])
def test_scalar_booleans_cannot_hide_changed_fixed_physics_or_failed_endpoints(name,value):
    import torch
    import proxdc_runner
    row=copy.deepcopy(proxdc_runner.tiny_stage(torch.device('cpu'))['scans']['pre'])
    row['record'][name]=value
    with pytest.raises(ValueError):module().validate_scan(row)


def child_artifacts(tmp_path,stage,hashes,invariant):
    import torch
    import proxdc_runner as child
    s=module();directory=tmp_path/stage;directory.mkdir()
    args=arguments(tmp_path,stage);params=s.parameters(args)
    if stage=='tiny':
        result=child.tiny_stage(torch.device('cpu'))
    else:
        from test_proxdc_runner import fake_trainer
        trainer,gauges,references=fake_trainer(coils=42)
        result=child.reconstruct(trainer,references,gauges,directory,stage,child.progress_state())
    value={**params,**result,'status':'complete','model_schema':s.ROUTE,
        'metric_contract':s.METRIC_CONTRACT,'steps_observed':0,'training_reference_access':False,
        'no_hard_dc_after_solve':True,'g2_restore_performed':stage!='tiny',
        'source_iteration':None if stage=='tiny' else 4500,'anchor_source':'same_call_raw_Gaussian_render',
        'fixed_model_unchanged':True,'code_sha256':hashes,'invariants':invariant}
    if stage!='tiny':
        net=torch.nn.Module()
        for name in child.g2.NAMES:net.register_parameter(name,torch.nn.Parameter(torch.ones(2),requires_grad=False))
        optimizer=torch.optim.Adam(net.parameters())
        trainer=type('Frozen',(),dict(model=net,optimizer=optimizer,scheduler=None,config={}))()
        state=child.helpers().snapshot(trainer)
        value.update(source_state_before=state,source_state_after=copy.deepcopy(state),
            forbidden_calls=dict(trainer_train=0,optimizer_step=0,scheduler_step=0),
            data_evidence={scan:dict(ground_truth_constructed=False,outside_mask_nonzero=0,coils=42)
                           for scan in ('pre','post')})
        (directory/'config.yaml').write_text('fixed: true\n')
    save(directory/'launch.json',dict(run_dir=str(directory),**params,code_sha256=hashes,invariants=invariant,
        parent_pid=456,gpu_uuid='GPU-fixture'))
    save(directory/'metrics.json',value);save(directory/'provenance.json',copy.deepcopy(value))
    save(directory/'progress.json',{})
    save(directory/'childcontext_ready.json',dict(pid=123,parent_pid=456,gpu_uuid='GPU-fixture'))
    save(directory/'childcontext_ack.json',dict(schema_version=1,child_pid=123,parent_pid=456,
        gpu_uuid='GPU-fixture',context_observed=True))
    return args,directory


@pytest.mark.parametrize('stage',['tiny','canonical','prox'])
def test_observed_child_uses_independent_route_schema_and_exact_artifact_set(tmp_path,stage):
    s=module();args,directory=child_artifacts(tmp_path,stage,{'source':'hash'},{'fixed':'input'})
    artifacts=s.validate_child_artifacts(args,directory,{'source':'hash'},{'fixed':'input'})
    assert str(directory/'progress.json') in artifacts
    if stage!='tiny':assert str(directory/'raw_post_complex.npy') in artifacts


@pytest.mark.parametrize('mutation',[
    lambda v:v.pop('source_state_before'),lambda v:v['source_state_after']['requires_grad'].update(positions=True),
    lambda v:v['forbidden_calls'].update(optimizer_step=1),
    lambda v:v['data_evidence']['post'].update(ground_truth_constructed=True),
])
def test_complete_flags_cannot_replace_export_only_state_and_acquisition_proof(tmp_path,mutation):
    s=module();args,directory=child_artifacts(tmp_path,'prox',{'source':'hash'},{'fixed':'input'})
    for name in ('metrics.json','provenance.json'):
        value=s.base.read(directory/name);mutation(value);save(directory/name,value)
    with pytest.raises(ValueError):s.validate_child_artifacts(args,directory,{'source':'hash'},{'fixed':'input'})


def observed_gate(tmp_path,stage,hashes,invariant):
    s=module();args,directory=child_artifacts(tmp_path,stage,hashes,invariant)
    artifacts=s.validate_child_artifacts(args,directory,hashes,invariant)
    control=directory/'control.json';control.write_text('reviewed controls')
    bindings={str(control):s.base.sha256(control)}
    expected=dict(code_sha256=hashes,invariants=invariant)
    proof=save(directory/'post_run_invariants.json',dict(status='verified',before=expected,after=expected,
        control_bindings_before=bindings,control_bindings_after=bindings))
    gate=save(directory/'completed.json',dict(schema_version=1,status='complete',passed=True,
        **s.parameters(args),parameters=s.parameters(args),run_dir=str(directory),code_sha256=hashes,
        invariants=invariant,artifacts=artifacts,control_bindings=bindings,post_run_invariants=s.base.descriptor(proof)))
    return gate,directory


def test_prox_requires_both_real_prior_stages_and_dereferences_their_proofs(tmp_path):
    s=module();args,hashes,invariant=reviewed(tmp_path,'prox')
    args.tiny_gate,_=observed_gate(tmp_path,'tiny',hashes,invariant)
    args.canonical_gate,directory=observed_gate(tmp_path,'canonical',hashes,invariant)
    accepted=s.validate_stage_gates(args,hashes,invariant)
    assert accepted['tiny']['stage']=='tiny' and accepted['canonical']['stage']=='canonical'
    (directory/'raw_pre_complex.npy').write_bytes(b'modified raw anchor evidence')
    with pytest.raises(ValueError,match='artifact'):s.validate_stage_gates(args,hashes,invariant)


@pytest.mark.parametrize('mutation',[lambda v:v.update(route='H1_four_metric'),
    lambda v:v.update(invariants={'stale':'input'}),lambda v:v.pop('post_run_invariants')])
def test_old_route_stale_input_or_missing_before_after_proof_cannot_be_reused(tmp_path,mutation):
    s=module();hashes,invariant={'source':'hash'},{'fixed':'input'}
    gate,_=observed_gate(tmp_path,'tiny',hashes,invariant)
    value=s.base.read(gate);mutation(value);save(gate,value)
    with pytest.raises(ValueError):s.validate_observed_gate(gate,'tiny',hashes,invariant)


def test_canonical_control_cannot_hide_solver_work_behind_a_false_executed_flag(tmp_path):
    s=module();args,directory=child_artifacts(tmp_path,'canonical',{'source':'hash'},{'fixed':'input'})
    for name in ('metrics.json','provenance.json'):
        value=s.base.read(directory/name);value['scans']['pre']['work']['normal_calls']=1;save(directory/name,value)
    with pytest.raises(ValueError):s.validate_child_artifacts(args,directory,{'source':'hash'},{'fixed':'input'})


@pytest.mark.parametrize('stage,mutation',[
    ('tiny',lambda v:v.pop('tiny_identity')),
    ('tiny',lambda v:v['tiny_identity']['pre'].update(forward_adjoint_relative=1.)),
    ('tiny',lambda v:v.pop('work_total')),
    ('tiny',lambda v:v['work_total'].update(total_fft_batches=0)),
    ('prox',lambda v:v.pop('work_total')),
    ('prox',lambda v:v.update(work_total=dict(total_fft_batches=0,gaussian_renders=0))),
    ('tiny',lambda v:v['tiny_identity']['post'].update(normal_curvature_real=float('nan'))),
    ('tiny',lambda v:v['tiny_identity']['pre']['work'].update(normal_calls=0)),
    ('prox',lambda v:v['work_total']['physics'].update(terminal_normal_calls=0)),
    ('canonical',lambda v:v['work_total'].update(gaussian_renders=0)),
])
def test_actual_route_results_reject_missing_or_corrupt_identity_and_total_work(tmp_path,stage,mutation):
    s=module();args,directory=child_artifacts(tmp_path,stage,{'source':'hash'},{'fixed':'input'})
    for name in ('metrics.json','provenance.json'):
        value=s.base.read(directory/name);mutation(value);save(directory/name,value)
    with pytest.raises(ValueError):s.validate_child_artifacts(args,directory,{'source':'hash'},{'fixed':'input'})


@pytest.mark.parametrize('change',[None,dict(child_pid=999),dict(context_observed=False)])
def test_route_artifacts_require_the_matching_observed_context_ack(tmp_path,change):
    s=module();args,directory=child_artifacts(tmp_path,'tiny',{'source':'hash'},{'fixed':'input'})
    path=directory/'childcontext_ack.json'
    if change is None:path.unlink()
    else:
        ack=s.base.read(path);ack.update(change);save(path,ack)
    with pytest.raises((OSError,ValueError)):
        s.validate_child_artifacts(args,directory,{'source':'hash'},{'fixed':'input'})
