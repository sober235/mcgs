"""CPU supervisor contracts; never select a GPU or publish an observed gate."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import os

import pytest

ROOT=Path(__file__).resolve().parents[1]


def supervisor():
    path=ROOT/'run_gpu.py'
    assert path.is_file(),'proximal stdlib supervisor is absent'
    spec=importlib.util.spec_from_file_location('_proximal_supervisor_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


def args(m,tmp_path,stage='tiny'):
    return m.parser().parse_args(['--stage',stage,'--exp','unit_fixture','--review-gate',str(tmp_path/'review.json')])


def review_fixture(m,tmp_path,monkeypatch):
    hashes={'own_source':'abc'};I={'input_manifest':{'files':{'frozen':'old'}},'accepted_h1':{'definition':{'version':1}}}
    monkeypatch.setattr(m,'source_hashes',lambda:hashes)
    monkeypatch.setattr(m,'snapshot_invariants',lambda *a,**k:I)
    path=tmp_path/'report.md';path.write_text('Independent review: zero actionable findings.\n')
    gate=m.build_review_gate(path,actionable_findings=0,unresolved_findings=0,round=1)
    return hashes,I,put(tmp_path/'review.json',gate)


def test_profile_has_fixed_native_lambda_budget_and_zero_training():
    m=supervisor();p=m.stage_profile('prox')
    assert p['source_iteration']==4500 and p['source_model']=='adamflat_g1_v1'
    assert p['lambda_ratio']==.1 and p['cg_max_matvecs']==32 and p['terminal_normal_evaluations']==1
    assert p['working_dtype']=='complex64' and p['training_updates']==0
    assert p['true_residual_tolerance']==2e-5 and p['J_relative_tolerance']==2e-6 and p['J_absolute_tolerance']==1e-10
    assert m.stage_profile('tiny')==m.stage_profile('canonical')==p
    with pytest.raises(ValueError):m.stage_profile('solve')


def test_review_builder_returns_full_observed_I_and_does_not_publish(tmp_path,monkeypatch):
    m=supervisor();hashes,I,path=review_fixture(m,tmp_path,monkeypatch);gate=m.read(path)
    assert gate['invariants']==I and gate['code_sha256']==hashes and gate['round']==1
    assert gate['review_report_sha256']==m.sha256(tmp_path/'report.md')
    assert sorted(p.name for p in tmp_path.iterdir())==['report.md','review.json']
    gate['invariants']={'path':'snapshot.json','sha256':'descriptor_only'};put(path,gate)
    with pytest.raises(ValueError):m.validate_stage_gates(args(m,tmp_path),hashes,I)


@pytest.mark.parametrize('field,value',[('actionable_findings',1),('unresolved_findings',1),('round',4)])
def test_review_builder_refuses_actionable_or_unbounded_review(tmp_path,monkeypatch,field,value):
    m=supervisor();review_fixture(m,tmp_path,monkeypatch)
    options=dict(actionable_findings=0,unresolved_findings=0,round=1);options[field]=value
    with pytest.raises(ValueError):m.build_review_gate(tmp_path/'report.md',**options)


@pytest.mark.parametrize('stage,missing',[('canonical','tiny'),('prox','tiny')])
def test_clinical_stages_cannot_bypass_actual_tiny_gate(tmp_path,monkeypatch,stage,missing):
    m=supervisor();hashes,I,_=review_fixture(m,tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='tiny'):m.validate_stage_gates(args(m,tmp_path,stage),hashes,I)


def test_tiny_review_is_validated_against_fresh_full_I(tmp_path,monkeypatch):
    m=supervisor();hashes,I,_=review_fixture(m,tmp_path,monkeypatch)
    assert m.validate_stage_gates(args(m,tmp_path),hashes,I)['invariants']==I
    with pytest.raises(ValueError):m.validate_stage_gates(args(m,tmp_path),hashes,{**I,'new_input':'changed'})


@pytest.mark.parametrize('exp',['..','.','a/b',''])
def test_exp_never_escapes_owned_fresh_output(tmp_path,monkeypatch,exp):
    m=supervisor();hashes,I,_=review_fixture(m,tmp_path,monkeypatch);a=args(m,tmp_path);a.exp=exp
    with pytest.raises(ValueError):m.validate_stage_gates(a,hashes,I)


def test_child_command_and_request_bind_exact_output_and_H1_definition(tmp_path):
    m=supervisor();a=args(m,tmp_path);request=m.request(a);command=m.command(a,tmp_path/'launch.json')
    assert request['output']==str(m.ROOT/'outputs'/a.exp)
    assert request['source_iteration']==4500 and request['training_updates']==0
    assert command[:5]==[str(m.PYTHON),'-X','utf8','-B','-u']
    assert command[5]==str(m.ROOT/'runner.py') and command[command.index('--output')+1]==str(m.ROOT/'outputs')
    assert command[command.index('--model-definition')+1]==str(a.model_definition.resolve())
    assert '--stage' in command and not any(x in command for x in ('--steps','--resume','--outer-steps'))


def test_parent_actual_nvgen_is_stdlib_only_and_never_selects_GPU():
    m=supervisor()
    result=subprocess.run([str(m.CPU_PYTHON),'-X','utf8','-B',str(ROOT/'run_gpu.py'),'--describe-runtime'],
        capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'})
    assert result.returncode==0,result.stderr
    value=json.loads(result.stdout)
    assert value['torch_imported'] is False and value['numpy_imported'] is False


def model_fixture(m,tmp_path,monkeypatch):
    value=m.read(ROOT/'evidence/h1_definition_v1.json')
    path=put(tmp_path/'definition.json',value);monkeypatch.setattr(m,'MODEL_DEFINITION',path)
    actual=m.sha256;gate=m.read(value['lineage_completed']['path'])
    frozen={**value['files'],**gate['invariants']['numerical_reference_artifacts']}
    monkeypatch.setattr(m,'sha256',lambda p:frozen[str(p)] if str(p) in frozen else actual(p))
    return path,value


def test_model_manifest_is_loaded_and_original4000_is_not_substituted(tmp_path,monkeypatch):
    m=supervisor();path,value=model_fixture(m,tmp_path,monkeypatch)
    old=m.read(ROOT.parent/'phase_firstorder/scoring/invariants_v1.json');result=m.model_definition(path,old_invariants=old)
    assert result['definition']==value and result['files']==value['files']
    assert result['definition']['checkpoint']['sha256']!=m.read(m._frozen.INPUT_DEFINITION)['files'][m.read(m._frozen.INPUT_DEFINITION)['source_checkpoint']]
    assert result['definition_manifest']==m.descriptor(path)
    assert result['lineage_reference_artifacts']==m.read(value['lineage_completed']['path'])['invariants']['numerical_reference_artifacts']


@pytest.mark.parametrize('attack',['iteration','model','frequency','recipe','adam','scheduler','file','array','gauge','accepted'])
def test_altered_H1_definition_or_published_control_is_refused(tmp_path,monkeypatch,attack):
    m=supervisor();path,value=model_fixture(m,tmp_path,monkeypatch)
    if attack=='iteration':value['source_iteration']=4000
    elif attack=='model':value['source_model']='other'
    elif attack=='frequency':value['frequency_t']=0.
    elif attack=='recipe':value['frequency_adam_recipe']['requires_same_recipe_on_resume']=False
    elif attack=='adam':value['expected_model_state']['adam']['positions']['step']=4000
    elif attack=='scheduler':value['expected_optimizer_actual']['scheduler']['last_epoch']=4000
    elif attack=='file':value['files'][value['checkpoint']['path']]='changed'
    elif attack=='array':value['scans']['pre']['canonical_array']['sha256']='changed'
    elif attack=='gauge':value['scans']['post']['factor_scale']*=2
    else:value['accepted_history_anchor']['model_trial']['accepted']=False
    put(path,value)
    with pytest.raises(ValueError):m.model_definition(path)


def scoring_fixture():
    spec=importlib.util.spec_from_file_location('_prox_supervisor_scoring_tests',ROOT/'scoring/test_evaluate.py')
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def test_stdlib_guard_checks_actual_physics_initial_normal_and_all_FFT_roles():
    m=supervisor();t=scoring_fixture();record,work,_=t.scan_fixture(t.utility())
    assert m.validate_scan(record,work)==work
    work['roles']['terminal']['fft_batches']+=1
    with pytest.raises(ValueError):m.validate_scan(record,work)


def test_parent_recipe_validates_dynamic_families_without_literal_trial_count(tmp_path,monkeypatch):
    m=supervisor();t=scoring_fixture();e=t.utility();recipe,_,_=t.anchor_fixture(tmp_path,e)
    monkeypatch.setattr(m,'R',tmp_path);monkeypatch.setattr(m,'CURRENT_INCUMBENT',e.CURRENT_INCUMBENT)
    value=m.read(recipe);hashes={'source':'actual'};I={'accepted_h1':{'definition_manifest':{'path':'fixed','sha256':'bytes'}}}
    value.update(code_sha256=hashes,invariants=I,model_definition=I['accepted_h1']['definition_manifest']);put(recipe,value)
    assert m.validate_recipe(recipe,hashes,I)==value
    summary=Path(value['completed_prior_summary']['path']);history=m.read(summary)
    history['phase_firstorder_family']['completed_candidates']=[];put(summary,history)
    value['completed_prior_summary']['sha256']=m.sha256(summary)
    current=m.read(e.CURRENT_INCUMBENT);current['validated_summary_sha256']=m.sha256(summary);put(e.CURRENT_INCUMBENT,current)
    value['incumbent']['sha256']=m.sha256(e.CURRENT_INCUMBENT);put(recipe,value)
    with pytest.raises(ValueError,match='PF'):m.validate_recipe(recipe,hashes,I)


def tiny_state():
    m=supervisor();names=m.GROUPS
    return dict(parameter_sha256=dict.fromkeys(names,'source'),parameter_identities={n:i+1 for i,n in enumerate(names)},
        requires_grad=dict.fromkeys(names,True),gradient_sha256=dict.fromkeys(names,None),gradient_identities=dict.fromkeys(names,None),
        adam={},adam_identities={},optimizer_groups=[],scheduler_state=None,config_sha256='config',
        model_sha256='model',geometry_sha256='geometry',density_sha256='density')


@pytest.mark.parametrize('field',['parameter_sha256','parameter_identities','requires_grad','gradient_sha256','gradient_identities',
    'adam','adam_identities','optimizer_groups','scheduler_state','config_sha256','model_sha256','geometry_sha256','density_sha256'])
def test_measured_source_state_cannot_change_even_if_boolean_checks_claim_success(field):
    m=supervisor();before=tiny_state()
    value=dict(source_state_before=before,source_state_after=copy.deepcopy(before),source_restore_performed=False,restored_iteration=None,
        checks=dict.fromkeys(('model_parameters_unchanged','parameter_identities_unchanged','gradients_unchanged','adam_unchanged',
            'adam_identities_unchanged','scheduler_unchanged','config_unchanged','no_training_steps','strict_cuda_prebuilt'),True),
        forbidden_calls=dict(trainer_train=0,optimizer_step=0,scheduler_step=0))
    m.validate_states(value,'tiny',{})
    value['source_state_after'][field]='mutated'
    with pytest.raises(ValueError):m.validate_states(value,'tiny',{})


def test_canonical_control_has_no_invented_CG_and_work_totals_count_actual_exports():
    m=supervisor();work=dict.fromkeys(m.PHYSICS_COUNTERS,0)
    rows={s:dict(n_coils=n,work=work) for s,n in [('pre',3),('post',4)]}
    total=dict(gaussian_renders=2,gaussian_cuda_renders=2,gaussian_reference_renders=0,gaussian_vjps=0,
        canonical_export_calls=2,canonical_fft_batches=2,canonical_ifft_batches=4,canonical_coil_fft_transforms=7,
        canonical_coil_ifft_transforms=14,total_fft_batches=2,total_ifft_batches=4,total_coil_fft_transforms=7,
        total_coil_ifft_transforms=14,physics=work,tiny_identity_physics=work)
    value=dict(scans=rows,work_total=total);m.validate_work_total(value,'canonical')
    total['canonical_coil_ifft_transforms']-=1
    with pytest.raises(ValueError):m.validate_work_total(value,'canonical')


def test_canonical_execute_calibrates_only_after_actual_raw_marker_before_final_gate(tmp_path,monkeypatch):
    m=supervisor();monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'VERIFY',tmp_path/'verification')
    a=args(m,tmp_path,'canonical');a.reserved_gpu=0;a.reservation_token='unit';a.claim_fd=os.open(tmp_path/'lock',os.O_CREAT|os.O_RDWR,0o600)
    hashes={'source':'frozen'};I={'full':'object'};bound={'control':'frozen'};events=[]
    for name,value in [('parent_runtime',{}),('source_hashes',hashes),('snapshot_invariants',I),('control_bindings',bound),
                       ('load_campaign',None),('select_resource',{'gpu':0,'free':True})]:
        monkeypatch.setattr(m,name,lambda *a,_v=value,**k:_v)
    monkeypatch.setattr(m,'validate_stage_gates',lambda *a:None);monkeypatch.setattr(m,'validate_reservation',lambda *a:None)
    monkeypatch.setattr(m,'initial_visibility',lambda:'0');put(m.VERIFY/'unit_fixture_selection.json',{'resource':{'free':True}})
    monkeypatch.setattr(m,'invariant_proof',lambda *a:dict(status='verified',before={},after={}))
    def validate(stage,*a,**k):events.append(('validate',k.get('require_calibration',True)));return {}
    monkeypatch.setattr(m,'validate_child_artifacts',validate)
    class Worker:
        pid=123;returncode=0
        def poll(self):return 0
    def popen(cmd,**kwargs):
        if 'calibrate' in cmd:
            assert (m.VERIFY/'unit_fixture_rendered.json').exists()
            assert not (m.VERIFY/'unit_fixture_completed.json').exists()
            events.append(('calibrate',True));put(tmp_path/'scoring/calibration_unit_fixture/calibration.json',{'unit':'proof'})
        else:events.append(('numeric',True))
        return Worker()
    monkeypatch.setattr(m.subprocess,'Popen',popen);monkeypatch.setattr(m,'wait_owned_worker',lambda worker:0)
    m.execute(a,[])
    assert events==[('numeric',True),('validate',False),('calibrate',True),('validate',True)]
    gate=m.read(m.VERIFY/'unit_fixture_completed.json')
    assert gate['calibration_proof']==m.descriptor(tmp_path/'scoring/calibration_unit_fixture/calibration.json')


def test_raw_rendered_validator_needs_no_completed_gate_and_binds_full_I(tmp_path,monkeypatch):
    m=supervisor();monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'VERIFY',tmp_path/'verification')
    a=args(m,tmp_path,'canonical');hashes={'source':'actual'};I={'full':'object'};bound={};launch={'request':m.request(a)}
    launch_path=put(m.VERIFY/'unit_fixture_launch.json',launch)
    monkeypatch.setattr(m,'validate_launch',lambda *a:(launch,globals_args));globals_args=a
    monkeypatch.setattr(m,'control_bindings',lambda *a:bound)
    calls=[]
    def artifacts(*a,**k):calls.append(k);return {'actual_child':'bytes'}
    monkeypatch.setattr(m,'validate_child_artifacts',artifacts)
    value=dict(status='rendered',stage='canonical',exp='unit_fixture',**m.stage_profile('canonical'),returncode=0,
        worker_pid=123,code_sha256=hashes,invariants=I,before=dict(code_sha256=hashes,invariants=I),
        after=dict(code_sha256=hashes,invariants=I),launch=m.descriptor(launch_path),request=launch['request'],
        control_bindings=bound,artifacts={'actual_child':'bytes'})
    path=put(m.VERIFY/'unit_fixture_rendered.json',value)
    assert m.validate_rendered('unit_fixture',hashes,I)[0]==value and calls==[{'require_calibration':False}]
    assert not (m.VERIFY/'unit_fixture_completed.json').exists()
    value['invariants']={'path':'only_descriptor','sha256':'not_full_I'};put(path,value)
    with pytest.raises(ValueError):m.validate_rendered('unit_fixture',hashes,I)


@pytest.mark.parametrize('defect',['head','complex','fit','artifact','rendered','I'])
def test_calibration_proof_cannot_hide_failed_metric_or_changed_binding(tmp_path,monkeypatch,defect):
    m=supervisor();monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'VERIFY',tmp_path/'verification')
    hashes={'source':'actual'};I={'full':'object'};artifacts={'child':'bytes'}
    rendered=put(m.VERIFY/'unit_rendered.json',{'control_bindings':{}})
    row=dict(complex_relative_l2=0.,complex_relative_l2_tolerance=1e-5,amplitude_or_phase_fit=False,
        head_psnr_absolute_difference_db=0.,head_psnr_tolerance_db=.001)
    value=dict(status='verified',passed=True,stage='canonical',exp='unit',**m.stage_profile('canonical'),code_sha256=hashes,invariants=I,
        numerical_runtime=dict(python=str(m.PYTHON),python_version=[3,11,13],torch_version='2.7.1+cu118',cuda_version='11.8'),
        bound_artifacts_sha256=artifacts,rendered_proof=m.descriptor(rendered),control_bindings={},scans={s:copy.deepcopy(row) for s in ('pre','post')})
    path=put(tmp_path/'scoring/calibration_unit/calibration.json',value)
    m.validate_calibration(m.descriptor(path),'unit',hashes,I,artifacts)
    if defect=='head':value['scans']['post']['head_psnr_absolute_difference_db']=.002
    elif defect=='complex':value['scans']['pre']['complex_relative_l2']=2e-5
    elif defect=='fit':value['scans']['pre']['amplitude_or_phase_fit']=True
    elif defect=='artifact':value['bound_artifacts_sha256']={}
    elif defect=='rendered':value['rendered_proof']['sha256']='changed'
    else:value['invariants']={'path':'descriptor','sha256':'not_full_I'}
    put(path,value)
    with pytest.raises(ValueError):m.validate_calibration(m.descriptor(path),'unit',hashes,I,artifacts)

