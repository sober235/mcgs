"""Protocol checks consume actual tiny CPU producer output, never a passing real gate."""
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
from test_representation_runner import run_tiny

ROOT = Path(__file__).resolve().parents[1]


def api():
    assert (ROOT / 'run_representation.py').is_file(), 'stdlib representation protocol is missing'
    return importlib.import_module('run_representation')


def test_public_cli_parent_stays_stdlib_and_rejects_mutable_budget():
    module = api()
    command = [sys.executable, '-B', '-c', "import run_representation as r,sys; a=r.parser().parse_args(['--stage','fit','--encoder','mlp','--exp','x','--review-gate','r','--recipe','p']); r.validate_parameters(a); assert 'torch' not in sys.modules and 'numpy' not in sys.modules; print(r.parameters(a))"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = module.parser().parse_args(['--stage','fit','--encoder','mlp','--exp','x','--review-gate','r','--recipe','p'])
    args.coefficient_updates = 1999
    with pytest.raises(ValueError): module.validate_parameters(args)
    args.coefficient_updates = 2000; args.encoder = 'none'
    with pytest.raises(ValueError): module.validate_parameters(args)


@pytest.fixture(scope='module')
def produced(tmp_path_factory):
    module = api()
    directory = tmp_path_factory.mktemp('representation_real_cpu')
    result = run_tiny(directory, full=True, duplicate=True)
    args = module.parser().parse_args(['--stage','fit','--encoder','mlp','--exp','fixture','--review-gate','r','--recipe','p'])
    hashes = {'/cpu/source.py': 'a' * 64}
    invariants = {'route': module.ROUTE, 'source_definition': {'anchors': {scan: f'/tiny/{scan}.npy' for scan in ('pre','post')}, 'expected_sha256': {f'/tiny/{scan}.npy': '0' * 64 for scan in ('pre','post')}, 'gauges': {'pre': .9997583823636317, 'post': 1.0003600786949414}, 'num_points': 2}}
    invariants['runtime_contract'] = result['source_runtime']
    recipe_path = directory / 'cpu_fixture_recipe.json'
    recipe_path.write_text(json.dumps(dict(schema_version=1, status='selected', route=module.ROUTE,
        parameters=module.parameters(args), code_sha256=hashes, invariants=invariants, cpu_fixture_only=True)))
    recipe = module.base.descriptor(recipe_path)
    launch = {**module.parameters(args), 'run_dir': str(directory), 'code_sha256': hashes, 'invariants': invariants,
              'parent_pid': 123, 'gpu_uuid': 'GPU-cpu-fixture', 'recipe': recipe,
              'recipe_path': recipe['path'], 'recipe_sha256':recipe['sha256'],
              'control_bindings': {recipe['path']:recipe['sha256']}}
    for name, value in [('launch.json', launch), ('childcontext_ready.json', {'pid': 456, 'parent_pid':123, 'gpu_uuid':'GPU-cpu-fixture'}),
                        ('childcontext_ack.json', {'schema_version':1, 'child_pid':456, 'parent_pid':123, 'gpu_uuid':'GPU-cpu-fixture', 'context_observed':True})]:
        (directory / name).write_text(json.dumps(value))
    evidence = {**launch, **result, 'status':'complete', 'model_schema':module.ROUTE,
                'training_reference_access':False, 'formal_mri_experiments':0}
    for name in ('metrics.json','provenance.json'):
        (directory / name).write_text(json.dumps(evidence))
    return module, args, directory, hashes, invariants


def test_actual_fixed_cpu_producer_is_accepted_with_negative_admission(produced):
    module, args, directory, hashes, invariants = produced
    artifacts = module.validate_child_artifacts(args, directory, hashes, invariants)
    assert str(directory / 'final_field.pth') in artifacts
    assert module.base.read(directory / 'metrics.json')['representation_admitted'] is False


@pytest.mark.parametrize('mutation', ['admission','fake_pass','count','checkpoint','ack','missing_ack','source','recipe','trace','phase','anchor','encoder','route','teacher','parameter_count','optimizer','invariant','nonfinite_error','precision','progress'])
def test_rejects_forged_or_incomplete_evidence(produced, tmp_path, mutation):
    import shutil
    module, args, original, hashes, invariants = produced
    directory = tmp_path / 'run'; shutil.copytree(original, directory)
    # Relocate actual artifact descriptors; this does not alter their bytes.
    for name in ('metrics.json','provenance.json','launch.json'):
        value = module.base.read(directory / name)
        value['run_dir'] = str(directory)
        if value.get('final_checkpoint'):
            value['final_checkpoint']['path'] = str(directory / 'final_field.pth')
        (directory / name).write_text(json.dumps(value))
    for name in ('metrics.json','provenance.json'):
        value = module.base.read(directory / name)
        if mutation == 'admission': value['representation_admitted'] = True
        if mutation == 'fake_pass': value['raw_anchor']['pre']['relative_l2'] = .1
        if mutation == 'count': value['work']['gaussian_vjps'] -= 1
        if mutation == 'source': value['code_sha256'] = {}
        if mutation == 'phase': value['phase_delta_zero'] = False
        if mutation == 'anchor': value['raw_anchor']['pre']['anchor_sha256'] = 'f' * 64
        if mutation == 'encoder': value['encoder'] = 'grid'
        if mutation == 'route': value['route'] = 'g2_raw_proxdc_v1'
        if mutation == 'teacher': value['teacher_state_before'] = {'pre': 'e' * 64, 'post': 'f' * 64}
        if mutation == 'parameter_count': value['amplitude_parameter_count'] = 1
        if mutation == 'optimizer': value['optimizer_phases']['raw']['state_entries_initial'] = 1
        if mutation == 'invariant': value['field_invariants_before'] = value['field_invariants_after'] = {'fake': 'a' * 64}
        if mutation == 'nonfinite_error': value['raw_relative_l2']['final']['pre'] = -1
        if mutation == 'precision': value['precision']['cuda_matmul_allow_tf32'] = not value['precision']['cuda_matmul_allow_tf32']
        (directory / name).write_text(json.dumps(value))
    if mutation == 'checkpoint': (directory / 'final_field.pth').write_bytes(b'forged')
    if mutation == 'ack': (directory / 'childcontext_ack.json').write_text('{}')
    if mutation == 'missing_ack': (directory / 'childcontext_ack.json').unlink()
    if mutation == 'trace': (directory / 'step_trace.jsonl').write_text('')
    if mutation == 'progress': (directory / 'progress.json').write_text('{}')
    if mutation == 'recipe':
        value = module.base.read(directory / 'launch.json'); value['coefficient_batch'] = 1
        (directory / 'launch.json').write_text(json.dumps(value))
    with pytest.raises((ValueError, FileNotFoundError)):
        module.validate_child_artifacts(args, directory, hashes, invariants)


def test_supervisor_refuses_torch_before_any_stage_or_gpu_access():
    module = api()
    script = "import sys,types; import run_representation as r; sys.modules['torch']=types.ModuleType('torch'); r.base.execute(None,protocol=r)"
    result = subprocess.run([sys.executable, '-B', '-c', script], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0 and 'must not import Torch' in result.stderr


@pytest.mark.parametrize('route,encoder', [('g2_raw_proxdc_v1','none'), ('g2_full_amplitude_representation_v1','baseline'), ('g2_full_amplitude_representation_v1','mlp')])
def test_render_gate_rejects_foreign_route_baseline_or_fit_encoder(tmp_path, route, encoder):
    module = api()
    path = tmp_path / 'not_real_gate.json'
    path.write_text(json.dumps({'schema_version':1,'route':route,'stage':'render_control','encoder':encoder,
                               'status':'complete','passed':True, 'parameters':{}}))
    with pytest.raises(ValueError): module.validate_observed_gate(path, 'render_control', {}, {})


def test_launch_recipe_descriptor_is_checked_against_bytes_and_parameters(produced, tmp_path):
    import shutil
    module, args, original, hashes, invariants = produced
    directory = tmp_path / 'run'; shutil.copytree(original, directory)
    launch = module.base.read(directory / 'launch.json')
    launch['run_dir'] = str(directory)
    launch['recipe'] = {'path': str(tmp_path / 'missing_recipe.json'), 'sha256': 'e' * 64}
    (directory / 'launch.json').write_text(json.dumps(launch))
    for name in ('metrics.json', 'provenance.json'):
        evidence = module.base.read(directory / name)
        evidence['run_dir'] = str(directory)
        evidence['final_checkpoint']['path'] = str(directory / 'final_field.pth')
        (directory / name).write_text(json.dumps(evidence))
    with pytest.raises((ValueError, FileNotFoundError)):
        module.validate_child_artifacts(args, directory, hashes, invariants)


def test_runtime_configuration_is_not_replaced_by_an_unchanged_source_boolean(produced, tmp_path):
    import shutil
    module, args, original, hashes, invariants = produced
    directory = tmp_path / 'run'; shutil.copytree(original, directory)
    launch = module.base.read(directory / 'launch.json'); launch['run_dir'] = str(directory)
    (directory / 'launch.json').write_text(json.dumps(launch))
    for name in ('metrics.json', 'provenance.json'):
        evidence = module.base.read(directory / name)
        evidence['run_dir'] = str(directory)
        evidence['final_checkpoint']['path'] = str(directory / 'final_field.pth')
        evidence['source_runtime'] = {'backend': 'fallback'}
        (directory / name).write_text(json.dumps(evidence))
    with pytest.raises(ValueError):
        module.validate_child_artifacts(args, directory, hashes, invariants)


def test_fresh_input_snapshot_reads_only_declared_g2_and_raw_whitelist():
    module = api()
    value = module.snapshot_invariants()
    definition = value['source_definition']
    assert value['verified_input_sha256'] == definition['expected_sha256']
    assert value['mri_arrays_loaded'] is False and value['formal_mri_experiments'] == 0
    assert {p for p in value['verified_input_sha256'] if Path(p).suffix == '.npy'} == set(definition['anchors'].values())


@pytest.mark.parametrize('change', ['foreign_review', 'foreign_recipe', 'baseline_recipe', 'stale_inputs', 'missing_render', 'timeout'])
def test_stage_gates_bind_route_exact_cli_inputs_and_wall_limit(tmp_path, change):
    module = api()
    args = module.parser().parse_args(['--stage','fit','--encoder','mlp','--exp','fixture',
                                      '--review-gate',str(tmp_path/'review.json'),'--recipe',str(tmp_path/'recipe.json')])
    hashes = {'cpu_fixture': 'a' * 64}; invariants = {'cpu_fixture': True}
    report = tmp_path/'report.txt'; report.write_text('CPU fixture only, never actual independent review')
    review = dict(schema_version=1, status='passed', passed=True, round=1, actionable_findings=0, unresolved_findings=0,
                  review_report=module.base.descriptor(report), route=module.ROUTE, code_sha256=hashes, invariants=invariants)
    recipe = dict(schema_version=1, status='selected', route=module.ROUTE, parameters=module.parameters(args),
                  code_sha256=hashes, invariants=invariants)
    if change == 'foreign_review': review['route'] = 'g2_raw_proxdc_v1'
    if change == 'foreign_recipe': recipe['route'] = 'g2_raw_proxdc_v1'
    if change == 'baseline_recipe': recipe['parameters']['encoder'] = 'baseline'
    if change == 'stale_inputs': recipe['invariants'] = {}
    if change == 'timeout': args.timeout = 1801
    Path(args.review_gate).write_text(json.dumps(review)); Path(args.recipe).write_text(json.dumps(recipe))
    with pytest.raises(ValueError): module.validate_stage_gates(args, hashes, invariants)
