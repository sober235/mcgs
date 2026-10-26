"""Fixed-budget adapter boundaries; every numerical artifact is a tiny CPU fixture."""
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import FunctionType, SimpleNamespace
from unittest.mock import patch

import pytest

ROUTE = 'g2_case_internal_magnitude_prior_fit512_v1'
SMALL = dict(patch_size=2, batch_size=1, width=2, tile_size=2)


def loaders():
    assert importlib.util.find_spec('caseprior_budget_variant') is not None, 'fixed 512 adapter is missing'
    return importlib.import_module('caseprior_budget_variant')


def protocol():
    return loaders().load_protocol()


def child():
    return loaders().load_child()


def cli(p, stage='zero', mode='control', steps=0):
    return p.parser().parse_args(['--stage', stage, '--mode', mode, '--steps', str(steps),
        '--exp', 'private_fixture', '--review-gate', 'unused', '--recipe', 'unused'])


def test_repeated_loading_preserves_old_modules_functions_constants_and_bytes():
    import run_caseprior as old
    import caseprior_runner as old_child
    parent_state = dict(vars(old)); child_state = dict(vars(old_child))
    before = old.code_hashes()
    settings = copy.deepcopy(old.SETTINGS); policy = copy.deepcopy(old.POLICY)
    p = protocol(); c = child()
    assert p is protocol() and c is child() and c.route is p
    assert p is not old and c is not old_child
    assert sys.modules[p.__name__] is p and sys.modules[c.__name__] is c
    assert vars(old) == parent_state and vars(old_child) == child_state
    assert old.SETTINGS == settings and old.POLICY == policy and old.SETTINGS['updates'] == 128
    assert p.SETTINGS is not old.SETTINGS and p.POLICY['cnn'] is p.SETTINGS
    assert old.code_hashes() == before
    assert p.main.__globals__ is vars(p)
    assert c._RNGObservedDenoiser.__call__.__globals__ is vars(c)
    assert c.run_case_continuation.__globals__ is vars(c)


def test_parent_import_and_help_are_stdlib_only_and_public_budget_is_fixed():
    p = protocol()
    result = subprocess.run([sys.executable, '-B', '-c',
        "import run_caseprior512,sys; from caseprior_budget_variant import load_protocol; "
        "p=load_protocol(); assert not any(n in sys.modules for n in ('torch','numpy','runner','caseprior_runner')); "
        "assert p.parser().get_default('route')=='g2_case_internal_magnitude_prior_fit512_v1'; "
        "run_caseprior512.main(['--help'])"], cwd=p.ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for flag in ('--updates', '--route', '--width', '--provider', '--backend'):
        with pytest.raises(SystemExit):
            p.parser().parse_args(['--stage','zero','--mode','control','--steps','0',
                '--exp','fixture','--review-gate','unused','--recipe','unused',flag,'1'])


def test_original_main_dispatches_registered_variant_protocol(monkeypatch, capsys):
    p = protocol()
    def observe(args, *, protocol):
        assert args.route == ROUTE
        assert protocol is p and protocol.CHILD_RUNNER == 'caseprior512_runner.py'
        return 'private_dispatch_observed'
    monkeypatch.setattr(p.base, 'execute', observe)
    import run_caseprior512
    assert run_caseprior512.main(['--stage','zero','--mode','control','--steps','0',
        '--exp','fixture','--review-gate','unused','--recipe','unused']) == 0
    assert 'private_dispatch_observed' in capsys.readouterr().out


def test_child_execute_reaches_variant_launch_validation_before_hardware(monkeypatch):
    c = child(); seen = []
    def stop(path):
        seen.append((path, c.route.ROUTE))
        raise ValueError('private launch boundary')
    monkeypatch.setattr(c, 'validate_launch', stop)
    import caseprior512_runner
    with pytest.raises(ValueError, match='private launch boundary'):
        caseprior512_runner.execute(Path('private-launch.json'))
    assert seen == [(Path('private-launch.json'), ROUTE)]


def test_source_closure_retains_221_and_native_218_and_binds_all_adapters():
    import run_caseprior as old
    p = protocol(); inherited = old.code_hashes(); hashes = p.code_hashes()
    assert len(inherited) == 221 and len(p.old.code_hashes()) == 218
    assert all(hashes[path] == digest for path,digest in inherited.items())
    assert set(hashes) - set(inherited) == {str(p.ROOT/name) for name in
        ('caseprior_budget_variant.py','run_caseprior512.py','caseprior512_runner.py')}
    assert all(p.base.sha256(path) == digest for path,digest in hashes.items())


@pytest.mark.parametrize('stage,mode,steps,valid', [('zero','control',0,True),
    ('smoke','control',2,True),('smoke','learned',2,True),('trial','control',500,True),
    ('trial','learned',500,True),('zero','learned',0,False),('smoke','learned',500,False),
    ('trial','learned',2,False)])
def test_fixed_stage_contract(stage, mode, steps, valid):
    p = protocol(); args = SimpleNamespace(route=ROUTE,stage=stage,mode=mode,steps=steps)
    if valid: p.validate_parameters(args)
    else:
        with pytest.raises(ValueError): p.validate_parameters(args)


def test_factory_default_reaches_actual_constructor_and_unwinds_on_error(tmp_path):
    c = child()
    import runner
    sys.path[:0] = [str(runner.TRAINER/'paired_gs'),str(runner.TRAINER)]
    import losses.nonlocal_red as red
    original = red.make_denoiser
    for fail in (False, True):
        try:
            with c.case_denoiser_factory(tmp_path,device='cpu') as instances:
                real = red.make_denoiser('case_internal_cnn_v1')
                assert instances == [real]
                assert real.producer.updates == 512
                assert (real.producer.patch_size,real.producer.batch_size,real.producer.width,
                    real.producer.tile_size,real.producer.lr,real.producer.seed) == (32,4,32,32,.001,42)
                assert red.make_denoiser('fake_identity') is not None
                with pytest.raises(ValueError): red.make_denoiser('case_internal_cnn_v1')
                if fail: raise RuntimeError('private fitting failure')
        except RuntimeError:
            assert fail
        assert red.make_denoiser is original


def produced(directory, mode='learned', steps=2, updates=None, common=None):
    """Reuse the real restore/trainer fixture; metadata is private, never a gate."""
    import torch
    import relocation_runtime as runtime
    from test_relocation_runner import tiny_trainer
    p = protocol(); c = child(); actual = runtime.restore_training_state
    observed = []
    def configured(trainer, checkpoint):
        c.configure_prior(trainer.config, mode)
        trainer.name_plain,trainer.name_contrast = 'pre','post'
        trainer.config['data'] = {'plain':dict(name='pre'),'contrast':dict(name='post')}
        record = actual(trainer, checkpoint)
        observed.append(copy.deepcopy(record))
        return record
    with patch.object(runtime,'restore_training_state',configured):
        trainer, restore, frequency, rotation = tiny_trainer(directory,steps)
    settings = dict(SMALL)
    if updates is not None: settings['updates'] = updates
    result = c.run_case_continuation(trainer,frequency,directory,mode=mode,steps=steps,
        rotation_matrix_fn=rotation,_denoiser_settings=settings)
    assert restore == observed[0] and restore['after'] == result['before']
    result['arrays'] = c.save_exports(result.pop('exports'),
        {s:dict(scale_trainer=1.,scale_gauge=1.,factor_scale=1.) for s in ('pre','post')},directory)
    result.update(restore=restore,data_evidence={s:dict(ground_truth_constructed=False,
        mask_inferred=False,outside_mask_nonzero=0,coils=2,shape=[3,3,3]) for s in ('pre','post')},zero_parity={})
    if not steps:
        result['zero_parity'] = {s:dict(passed=True,tolerance=1e-5,complex_relative_l2=0.,array_equal=True) for s in ('pre','post')}
    for name,value in [('config.json',trainer.config),('progress.json',
        dict(status='complete',counters=result['counters'],final_iteration=4500+steps))]:
        (directory/name).write_text(json.dumps(value))
    (directory/'config.yaml').write_text('private actual original trainer fixture\n')
    source = directory/'fixture_source.pth'; torch.save({'fixture':True},source)
    invariants = dict(route=ROUTE,policy=p.POLICY,g2_checkpoint=str(source),g2_checkpoint_sha256=p.base.sha256(source))
    if common:
        invariants = {**common,'route':ROUTE,'policy':p.POLICY}; source = Path(common['g2_checkpoint'])
    hashes = {'fixture':'0'*64}
    args = SimpleNamespace(route=ROUTE,stage='zero' if not steps else 'smoke' if steps==2 else 'trial',mode=mode,steps=steps)
    launch = dict(**p.parameters(args),run_dir=str(directory),parent_pid=12345,gpu_uuid='CPU-FIXTURE',code_sha256=hashes,invariants=invariants)
    for name,value in [('launch.json',launch),('child_started.json',dict(pid=23456,gpu_uuid='CPU-FIXTURE')),
        ('childcontext_ready.json',dict(pid=23456,parent_pid=12345,gpu_uuid='CPU-FIXTURE')),
        ('childcontext_ack.json',dict(schema_version=1,child_pid=23456,parent_pid=12345,gpu_uuid='CPU-FIXTURE',context_observed=True))]:
        (directory/name).write_text(json.dumps(value))
    environment = dict(device='cpu',torch=str(torch.__version__),cuda=torch.version.cuda,python=str(p.base.PYTHON),
        renderer='tiny_cpu_fixture',strict_cuda=False,python_version=list(sys.version_info[:3]),
        default_dtype=str(torch.get_default_dtype()),autocast_enabled=False,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    c.publish_result(launch,directory,result,environment,source_checkpoint=source,
        elapsed=0.,peak_allocated=0.,peak_reserved=0.)
    return p,args,hashes,invariants


def validate(p, args, directory, hashes, invariants, updates=None):
    settings = dict(SMALL)
    if updates is not None: settings['updates'] = updates
    return p._validate_child_artifacts(args,directory,hashes,invariants,point_count=5,
        device='cpu',backend='tiny_cpu_fixture',coils=2,denoiser_settings=settings)


@pytest.mark.parametrize('mode,steps,calls', [('control',0,0),('control',2,0),
    ('learned',2,2),('learned',500,10)])
def test_actual_original_producer_runs_512_fits_and_validates(tmp_path,mode,steps,calls):
    p,args,hashes,invariants = produced(tmp_path,mode,steps)
    bindings = validate(p,args,tmp_path,hashes,invariants)
    assert str(tmp_path/'scheduler.json') in bindings
    m = p.base.read(tmp_path/'metrics.json')
    assert m['route'] == m['model_schema'] == ROUTE
    assert m['policy']['cnn']['updates'] == 512
    assert len(m['denoiser_calls']) == len(m['denoiser_rng']) == len(m['denoiser_checkpoint_checks']) == calls
    assert sum(row['updates'] for row in m['denoiser_calls']) == calls*512
    assert m['counters']['gaussian_forwards'] == 2*steps+2
    assert m['counters']['gaussian_vjps'] == 2*steps
    assert m['after']['scheduler']['state']['_step_count'] == 4501+steps
    for j,row in enumerate(m['denoiser_calls']):
        assert row['updates'] == 512 and row['seed'] == 42+1009*j
        assert row['initial_parameter_sha256'] != row['final_parameter_sha256']
        assert m['denoiser_checkpoint_checks'][j]['metadata_equal'] is True
        assert m['denoiser_rng'][j]['cpu']['preserved'] is True
    if not steps: assert m['nonlocal_summary'] is None
    with pytest.raises(ValueError): p.validate_child_artifacts(args,tmp_path,hashes,invariants)


@pytest.mark.parametrize('steps',[0,2])
def test_control_bypasses_cnn_constructor(tmp_path,monkeypatch,steps):
    import case_denoiser
    def forbidden(*a,**k): raise AssertionError('control constructed CNN')
    monkeypatch.setattr(case_denoiser,'CaseMagnitudeDenoiser',forbidden)
    p,args,h,i = produced(tmp_path,'control',steps)
    validate(p,args,tmp_path,h,i)


def test_actual_128_evidence_is_refused_and_512_is_refused_by_128_policy(tmp_path):
    old_dir = tmp_path/'old'; old_dir.mkdir()
    p,args,h,i = produced(old_dir,updates=128)
    with pytest.raises(ValueError,match='budget|setting'): validate(p,args,old_dir,h,i)
    # Same route isolates the budget check: original validator globals remain128.
    import run_caseprior as old
    m = p.base.read(old_dir/'metrics.json')
    old._validate_denoisers(m,old_dir,args,settings=SMALL,cuda_devices=[])
    new_dir = tmp_path/'new'; new_dir.mkdir()
    p,args,h,i = produced(new_dir)
    m = p.base.read(new_dir/'metrics.json')
    with pytest.raises(ValueError,match='budget|setting'):
        old._validate_denoisers(m,new_dir,args,settings=SMALL,cuda_devices=[])


def test_old_route_is_refused_even_when_budget_matches(tmp_path):
    from test_caseprior_protocol import produced as old_produced
    _,args,h,i = old_produced(tmp_path,'control',2)
    with pytest.raises(ValueError): validate(protocol(),args,tmp_path,h,i)


@pytest.mark.parametrize('defect',['rng','readback','ack','restore'])
def test_inherited_safeguards_refuse_actual_producer_corruption(tmp_path,defect):
    p,args,h,i = produced(tmp_path)
    m = p.base.read(tmp_path/'metrics.json')
    if defect=='rng': m['denoiser_rng'][0]['cpu']['preserved'] = False
    elif defect=='readback': m['denoiser_checkpoint_checks'][0]['metadata_equal'] = False
    elif defect=='ack': (tmp_path/'childcontext_ack.json').write_text('{}')
    elif defect=='restore': m['restore']['after']['config']['loss']['lambda_nl'] = 0.
    for name in ('metrics.json','provenance.json'): (tmp_path/name).write_text(json.dumps(m))
    with pytest.raises((ValueError,KeyError)): validate(p,args,tmp_path,h,i)


def test_new_gate_binding_refuses_old_route_and_accepts_actual_private_completion(tmp_path):
    from test_caseprior_protocol import complete_fixture
    p,args,h,i = produced(tmp_path,'control',0)
    path,gate = complete_fixture(p,tmp_path,args,h,i,validate(p,args,tmp_path,h,i))
    fn = FunctionType(p.validate_observed_gate.__code__,{**p.validate_observed_gate.__globals__,
        'validate_child_artifacts':lambda a,d,h,i:validate(p,a,Path(d),h,i)})
    assert fn(path,'zero',h,i,None) == gate
    gate['route'] = gate['parameters']['route'] = 'g2_case_internal_magnitude_prior_v1'
    path.write_text(json.dumps(gate))
    with pytest.raises(ValueError): fn(path,'zero',h,i,None)


def test_stage_gates_enforce_new_recipe_zero_both_smokes_and_one_control(tmp_path,monkeypatch):
    p = protocol(); args = cli(p,'trial','learned',500)
    report = tmp_path/'report'; report.write_text('private fixture report')
    review = tmp_path/'review'; review.write_text('private fixture review')
    args.review_gate = review; args.recipe = tmp_path/'recipe'
    h = {'fixture':'a'*64}; i = {'fixture':'b'*64}
    monkeypatch.setattr(p.base,'validate_review',lambda *a:dict(route=ROUTE,review_report=p.base.descriptor(report)))
    recipe = dict(schema_version=1,status='selected',route=ROUTE,parameters=p.parameters(args),policy=p.POLICY,code_sha256=h,invariants=i)
    args.recipe.write_text(json.dumps(recipe))
    with pytest.raises(ValueError,match='zero gate'): p.validate_stage_gates(args,h,i)
    seen = []
    def observed(path,stage,*a):
        seen.append((stage,a[-1]))
        return dict(artifacts={},control_bindings={},post_run_invariants=p.base.descriptor(report))
    monkeypatch.setattr(p,'validate_observed_gate',observed)
    args.zero_gate = review
    with pytest.raises(ValueError,match='control smoke'): p.validate_stage_gates(args,h,i)
    args.control_smoke_gate = review
    with pytest.raises(ValueError,match='learned smoke'): p.validate_stage_gates(args,h,i)
    args.learned_smoke_gate = review
    with pytest.raises(ValueError,match='one native receipt'): p.validate_stage_gates(args,h,i)
    args.control_gate = review
    p.validate_stage_gates(args,h,i)
    assert seen[-4:] == [('zero',None),('smoke',dict(mode='control')),('smoke',dict(mode='learned')),('trial',dict(mode='control'))]
    recipe['route'] = 'g2_case_internal_magnitude_prior_v1'; args.recipe.write_text(json.dumps(recipe))
    with pytest.raises(ValueError,match='exact selected'): p.validate_stage_gates(args,h,i)


def test_native_receipt_keeps_real_old_closure_and_new_actual_smoke(tmp_path):
    from test_caseprior_protocol import native_fixture, complete_fixture
    old_p,path,h,_,old_new_smoke,validator = native_fixture(tmp_path)
    previous = old_p.base.read(path)
    directory = tmp_path/'fit512_smoke'; directory.mkdir()
    p,args,h,i = produced(directory,'control',2,common=previous['native_invariants'])
    new_path,new_gate = complete_fixture(p,directory,args,h,i,validate(p,args,directory,h,i))
    receipt = {**previous,'route':ROUTE,'invariants':i,'ordinary_smoke':p.base.descriptor(new_path)}
    new_receipt = tmp_path/'private_fit512_receipt.json'; new_receipt.write_text(json.dumps(receipt))
    accepted = p._validate_native_control_receipt(new_receipt,h,i,new_gate,
        old_hashes={'fixture':'0'*64},old_gate_validator=validator)
    assert accepted['ordinary_smoke_relative_l2'] == {'pre':0.,'post':0.}
    assert accepted['native_invariants']['route'] == 'g2_fixed_count_relocation_v1'
    with pytest.raises(ValueError):
        p._validate_native_control_receipt(path,h,i,old_new_smoke,old_hashes={'fixture':'0'*64},old_gate_validator=validator)
    # Actual public entry must retain real-only source/input/hardware checking.
    with pytest.raises(ValueError): p.validate_native_control_receipt(new_receipt,h,i,new_gate)
