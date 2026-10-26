#!/usr/bin/env python3
"""Stdlib-only CPU reservation supervisor for one gated geometry-LR candidate.

Only the explicit GSTMI child imports Torch. This process uses the installed
NVGEN Python for orchestration and remains alive with its initial CVD lease
until training, numerical checks, export and immutable source proofs finish.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'nonlocal_autoresearch'
VERIFY = ROOT / 'verification'
PYTHON = Path('/home/congcongliu/anaconda3/envs/GSTMI/bin/python')
CPU_PYTHON = Path('/home/congcongliu/anaconda3/envs/nvgen/bin/python')
SOURCE_DEFINITION = ROOT.parent / 'optimizer_probe/verification/source_integrity_before.json'
INPUT_DEFINITION = BASE / 'verification/loop_input_freeze.json'
CLAIM_LOCK = Path(f'/tmp/mcgs_ce_gpu_claim_{os.getuid()}.lock')
PRODUCTION = ('overrides.py', 'runner.py', 'run_gpu.py', 'frequency_weights.py')
SOURCE_TRAINER = Path('/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois/trainer/_trainer_e778bf2_ar')
NOOP_REFERENCES = (BASE / 'outputs/smoke_zero_nlr_v1', ROOT.parent / 'optimizer_probe/outputs/noop_default_2')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def invalid(value):
        raise ValueError(f'nonfinite JSON value {value} in {path}')
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}.{time.time_ns()}')
    try:
        with temporary.open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def code_hashes():
    return {str(ROOT / name): sha256(ROOT / name) for name in PRODUCTION}


def source_files(root):
    return {p.relative_to(root).as_posix(): p for p in root.rglob('*')
            if p.is_file() and not any(part in ('results_arc', '__pycache__', 'build')
                                      for part in p.relative_to(root).parts)
            and p.suffix not in ('.pyc', '.so', '.o')}


def snapshot_invariants():
    """Read existing definitions; never create or refreeze an original input."""
    definitions = {str(p): sha256(p) for p in (SOURCE_DEFINITION, INPUT_DEFINITION)}
    source, inputs = read(SOURCE_DEFINITION), read(INPUT_DEFINITION)
    observed = {}
    for name, count in (('gated_code_sha256', 13), ('production_code_sha256', 103)):
        expected = source[name]
        require(isinstance(expected, dict) and len(expected) == count, f'expected frozen C source count {count}')
        observed[name] = {path: sha256(path) for path in expected}
        require(observed[name] == expected, f'frozen C {name} changed')
    require(isinstance(inputs.get('files'), dict) and len(inputs['files']) == 35, 'expected exactly 35 frozen inputs')
    actual_inputs = {path: sha256(path) for path in inputs['files']}
    require(actual_inputs == inputs['files'], 'frozen input/checkpoint/config bytes changed')
    require(inputs['source_checkpoint'] in actual_inputs and inputs['source_config'] in actual_inputs,
            'source checkpoint and config must be frozen inputs')
    original, copied = source_files(SOURCE_TRAINER), source_files(BASE / 'trainer')
    require(bool(original) and bool(copied), 'original and copied trainer source definitions must be nonempty')
    # New C helpers have no original counterpart. List them explicitly rather
    # than ever passing a missing/None source path to the hashing function.
    source_definition = {
        'original_root': str(SOURCE_TRAINER), 'copy_root': str(BASE / 'trainer'),
        'original_sha256': {name: sha256(path) for name, path in sorted(original.items())},
        'copy_sha256': {name: sha256(path) for name, path in sorted(copied.items())},
        'missing_original_sources': sorted(set(copied) - set(original)),
        'missing_copy_sources': sorted(set(original) - set(copied)),
        'copy_files_outside_production_subset': sorted(name for name, path in copied.items()
                                                      if str(path) not in observed['production_code_sha256']),
    }
    require(not source_definition['missing_copy_sources'], 'copied trainer is missing original source files')
    folder = BASE / 'trainer/paired_gs/gaussian/cuda_ext'
    pieces = [hashlib.md5((folder / name).read_bytes()).hexdigest()[:8]
              for name in ('tile_voxelize_cuda.cpp', 'tile_voxelize_cuda_kernel.cu')]
    binary = folder / 'build' / ('tile_voxelize_cuda_' + '_'.join(pieces) + '.so')
    require(binary.is_file(), 'source-hash-named prebuilt C extension is missing; compilation is forbidden')
    reference_files = {}
    for directory in NOOP_REFERENCES:
        metrics = read(directory / 'metrics.json')
        require(metrics.get('source_iteration') == 4000 and metrics.get('end_iteration') == 4002
                and metrics.get('forward_backward_evaluations') == 2
                and read(directory / 'provenance.json').get('status') == 'complete',
                'numerical reference is not the actual completed original two-step control')
        files = [directory / name for name in ('metrics.json', 'provenance.json', 'step_trace.jsonl',
                  'candidate_pre_complex.npy', 'candidate_post_complex.npy')] + [Path(metrics['final_checkpoint'])]
        reference_files.update({str(path): sha256(path) for path in files})
    require(definitions == {str(p): sha256(p) for p in (SOURCE_DEFINITION, INPUT_DEFINITION)},
            'frozen definitions changed during validation')
    return {'frozen_definitions_sha256': definitions, 'campaign_sources': observed,
            'source_definition': source_definition, 'input_manifest': inputs,
            'source_checkpoint_sha256': actual_inputs[inputs['source_checkpoint']],
            'source_config_sha256': actual_inputs[inputs['source_config']],
            'prebuilt_extension': {'path': str(binary), 'sha256': sha256(binary)},
            'numerical_reference_artifacts': reference_files}


def stage_profile(stage, factor, frequency_t=None):
    require(not isinstance(factor, bool) and isinstance(factor, (int, float)) and math.isfinite(factor),
            'geometry factor must be finite')
    if stage in ('audit0', 'noop2'):
        require(factor == 1., 'audit0/noop2 require the exact factor-1 control')
        expected_t=1. if stage == 'audit0' else 0.
        require(frequency_t is None or type(frequency_t) in (int,float) and frequency_t == expected_t,
                'audit0 requires t1; source noop2 requires t0')
        return {'steps': 0 if stage == 'audit0' else 2, 'mode': 'audit' if stage == 'audit0' else 'all',
                'geometry_factor': float(factor),'frequency_t':expected_t}
    require(stage == 'trial500' and factor in (1., 2.) and
            (frequency_t is None or type(frequency_t) in (int,float) and frequency_t == 1.),
            'trial500 permits only t1 geometry factors1/2')
    return {'steps': 500, 'mode': 'all', 'geometry_factor': float(factor),'frequency_t':1.}


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--stage', required=True, choices=('audit0', 'noop2', 'trial500'))
    p.add_argument('--exp', required=True)
    p.add_argument('--geometry-factor', type=float, default=1.)
    p.add_argument('--frequency-t',type=float,choices=(0.,1.))
    p.add_argument('--review-gate', type=Path, required=True)
    p.add_argument('--zero-audit-gate', type=Path)
    p.add_argument('--noop-gate', type=Path)
    p.add_argument('--selected-recipe', type=Path)
    p.add_argument('--preferred-gpu', type=int, default=5)
    p.add_argument('--reserved-gpu', type=int, help=argparse.SUPPRESS)
    p.add_argument('--reservation-token', help=argparse.SUPPRESS)
    p.add_argument('--claim-fd', type=int, help=argparse.SUPPRESS)
    return p


def validate_review(path, hashes, invariants):
    value = read(path)
    require(value.get('status') == 'pass' and value.get('round') in (1, 2, 3)
            and value.get('actionable_findings') == value.get('unresolved_findings') == 0,
            'fresh independent clean review is required')
    require(value.get('code_sha256') == hashes and value.get('invariants') == invariants,
            'review source/input/original-copy/binary bindings are stale or missing')
    return value


def validate_frequency_files(directory, evidence, frequency_t):
    """Parent binds bytes only; the GSTMI child independently reads all arrays."""
    directory=Path(directory).resolve();folder=directory/'frequency_weights'
    require(evidence.get('version') == 1 and evidence.get('frequency_t') == frequency_t
            and evidence.get('mass_relative_tolerance') == 2e-6 and set(evidence.get('scans',{})) == {'pre','post'}
            and read(folder/'manifest.json') == evidence,'actual frequency manifest differs')
    files={str(folder/'manifest.json'):sha256(folder/'manifest.json')}
    for scan,row in evidence['scans'].items():
        shapes=[]
        for name in ('source_weight','effective_weight','acquired_mask'):
            value=row.get(name,{});path=folder/f'{scan}_{name}.npy';shape=value.get('shape',[])
            require(value.get('path') == str(path) and path.resolve().parent == folder
                    and sha256(path) == value.get('file_sha256') and len(value.get('byte_sha256','')) == 64
                    and len(shape) == 3 and all(type(v) is int and v > 0 for v in shape)
                    and value.get('dtype') == 'torch.float32' and value.get('device') in ('cuda:0','cpu'),
                    'actual frequency six-array file/metadata differs')
            shapes.append(shape);files[str(path)]=value['file_sha256']
        require(shapes[0] == shapes[1] == shapes[2] and type(row.get('acquired_count')) is int
                and 0 < row['acquired_count'] <= math.prod(shapes[0]),'frequency source/mask shape/count differs')
        require(all(type(row.get(k)) in (int,float) and math.isfinite(row[k]) and row[k] > 0
                    for k in ('source_mean_squared_weight','effective_mean_squared_weight','c'))
                and math.isfinite(row.get('relative_mass_error',float('nan')))
                and 0 <= row['relative_mass_error'] <= 2e-6,'frequency acquired RMS metadata differs')
        require(row.get('t0_exact_equal') is (True if frequency_t == 0. else None),'frequency t0 identity claim differs')
    return files


def validate_frequency_check(check, evidence, frequency_t):
    runtime=check.get('numerical_runtime',{})
    require(check.get('passed') is True and check.get('frequency_t') == frequency_t
            and check.get('mass_relative_tolerance') == 2e-6 and set(check.get('scans',{})) == {'pre','post'}
            and Path(runtime.get('python','')).resolve() == PYTHON.resolve()
            and runtime.get('torch_version') == '2.7.1+cu118' and runtime.get('cuda_version') == '11.8',
            'independent saved-array GSTMI check missing')
    for scan,row in check['scans'].items():
        saved=evidence['scans'][scan]
        require(row.get('original_acquired_mask_exact') is row.get('original_source_weight_exact') is True
                and row.get('effective_identity') is (frequency_t == 0.)
                and all(row.get(k) == saved[k] for k in ('acquired_count','c','source_mean_squared_weight','effective_mean_squared_weight','relative_mass_error'))
                and row.get('interpolation_relative_tolerance') == 8.*2.**-23
                and math.isfinite(row.get('interpolation_max_relative_error',float('nan')))
                and 0 <= row['interpolation_max_relative_error'] <= row['interpolation_relative_tolerance'],
                'independent original-source interpolation/RMS check failed')


def validate_child_artifacts(args, directory, hashes, invariants):
    directory = Path(directory)
    profile = stage_profile(args.stage, args.geometry_factor,getattr(args,'frequency_t',None))
    provenance, metrics, geometry, validation = [read(directory / name) for name in
        ('provenance.json', 'metrics.json', 'geometry_provenance.json', 'validation.json')]
    require(provenance.get('status') == geometry.get('status') == 'complete'
            and validation.get('passed') is True and validation.get('status') == 'complete',
            'child run or numerical validation is incomplete/failed')
    require(validation.get('stage') == args.stage and validation.get('geometry_factor') == args.geometry_factor
            and validation.get('code_sha256') == geometry.get('code_sha256') == hashes
            and validation.get('invariants') == invariants, 'child request/source/binary bindings differ')
    require(validation.get('checks', {}).get('ordinary_adam_checkpoint') is True
            and validation.get('checks', {}).get('wavelet_replay') is True
            and validation.get('checks', {}).get('actual_optimizer_rates') is True,
            'child lacks checkpoint, wavelet and actual LR checks')
    frequency=read(directory/'frequency_provenance.json')
    require(all(value.get('method_variant') == 'frequency_adam' and value.get('frequency_t') == profile['frequency_t']
                and value.get('geometry_factor') == args.geometry_factor for value in (validation,geometry,provenance,metrics))
            and all(value.get('frequency_adam') == frequency for value in (geometry,provenance,metrics))
            and frequency.get('status') == 'complete' and frequency.get('completed_steps') == profile['steps']
            and frequency.get('loss_calls_total') == 2*profile['steps']
            and validation['checks'].get('actual_frequency_weights') is True
            and validation['checks'].get('independent_frequency_arrays') is True,
            'actual frequency setup/loss-used evidence missing')
    frequency_files=validate_frequency_files(directory,frequency['frequency_evidence'],profile['frequency_t'])
    require(all(info.get('device') == 'cuda:0' for row in frequency['frequency_evidence']['scans'].values()
                for name,info in row.items() if name in ('source_weight','effective_weight','acquired_mask')),
            'clinical frequency arrays were not installed on the actual CUDA trainer')
    validate_frequency_check(frequency['frequency_array_check'],frequency['frequency_evidence'],profile['frequency_t'])
    require(validation.get('frequency_files') == frequency_files,'numerical child frequency file bindings differ')
    require(provenance.get('torch_version') == '2.7.1+cu118' and provenance.get('cuda_version') == '11.8'
            and geometry.get('runtime', {}).get('torch_version') == '2.7.1+cu118',
            'numerical child did not use the fixed GSTMI Torch/CUDA')
    for value in (metrics, provenance):
        require(value.get('source_iteration') == 4000 and value.get('end_iteration') == 4000 + profile['steps']
                and value.get('forward_backward_evaluations') == profile['steps'], 'observed iteration budget differs')
        require(value.get('source_checkpoint_sha256') == invariants['source_checkpoint_sha256'],
                'child source checkpoint differs')
    if args.stage == 'audit0':
        require(validation.get('zero_audit', {}).get('passed') is True, 'clinical zero audit is missing')
    if args.stage == 'noop2':
        require(validation.get('noop', {}).get('passed') is True, 'observed numerical no-op proof is missing')
    files = ['provenance.json', 'metrics.json', 'geometry_provenance.json', 'frequency_provenance.json','validation.json',
             'actual_config.yaml', 'step_trace.jsonl', 'source_to_copy_manifest.json', 'source_to_copy.diff',
             'candidate_pre_complex.npy', 'candidate_post_complex.npy']
    checkpoint = Path(metrics['final_checkpoint']).resolve()
    require(checkpoint.is_relative_to(directory.resolve()), 'final checkpoint is outside the owned run')
    paths = [directory / name for name in files] + [checkpoint]
    return {**{str(path): sha256(path) for path in paths},**frequency_files}


def validate_observed_gate(path, stage, hashes, invariants):
    value = read(path)
    require(value.get('passed') is True and value.get('status') == 'complete' and value.get('stage') == stage
            and value.get('code_sha256') == hashes and value.get('invariants') == invariants,
            f'passing observed {stage} gate with current bindings is required')
    artifacts = value.get('artifacts', {})
    require(bool(artifacts) and all(sha256(p) == h for p, h in artifacts.items()), f'{stage} artifacts changed/missing')
    proof = Path(value['post_run_invariants'])
    require(sha256(proof) == value.get('post_run_invariants_sha256'), f'{stage} source proof changed')
    expected = {'code_sha256': hashes, 'invariants': invariants}
    observed = read(proof)
    require(observed.get('status') == 'verified' and observed.get('before') == observed.get('after') == expected,
            f'{stage} lacks a matching before/after source proof')
    bindings=value.get('control_bindings',{})
    require(bool(bindings) and observed.get('control_bindings_before') == observed.get('control_bindings_after') == bindings
            and all(sha256(p) == h for p,h in bindings.items()),f'{stage} review/control bindings changed')
    files = [p for p in artifacts if Path(p).name == 'validation.json']
    require(len(files) == 1, f'{stage} needs exactly one actual validation artifact')
    profile=stage_profile(stage,1.)
    require(value.get('frequency_t') == profile['frequency_t'] and value.get('geometry_factor') == 1.,
            'observed control frequency/geometry protocol differs')
    args = argparse.Namespace(stage=stage, geometry_factor=1.,frequency_t=profile['frequency_t'])
    require(validate_child_artifacts(args, Path(files[0]).parent, hashes, invariants) == artifacts,
            f'{stage} artifact set differs from observed run')
    return value


def validate_stage_gates(args, hashes, invariants):
    stage_profile(args.stage, args.geometry_factor,args.frequency_t)
    require(args.exp not in ('', '.', '..') and Path(args.exp).name == args.exp, 'exp must be one fresh directory name')
    require(args.preferred_gpu >= 0 and (args.reserved_gpu is None or args.reserved_gpu >= 0), 'invalid GPU index')
    validate_review(args.review_gate, hashes, invariants)
    if args.stage != 'audit0':
        require(args.zero_audit_gate is not None, 'noop2/trial500 require an observed clinical audit0 gate')
        validate_observed_gate(args.zero_audit_gate, 'audit0', hashes, invariants)
    if args.stage == 'trial500':
        require(args.noop_gate is not None and args.selected_recipe is not None,
                'formal trial requires actual noop2 and root-selected recipe gates')
        validate_observed_gate(args.noop_gate, 'noop2', hashes, invariants)
        recipe = read(args.selected_recipe)
        require(recipe.get('status') == 'selected' and recipe.get('selected_by') == 'root'
                and recipe.get('code_sha256') == hashes and recipe.get('invariants') == invariants
                and recipe.get('geometry_factors') == [1., 2.] and recipe.get('frequency_t') == 1.
                and recipe.get('source_iteration') == 4000 and recipe.get('steps') == 500
                and recipe.get('local_factor') == 1. and recipe.get('nlambda') == 0.
                and recipe.get('sigma_rel') == .015 and recipe.get('density_lr_factor') == 1.
                and recipe.get('betas') == [.9, .999], 'selected recipe differs from the bounded flat-frequency Adam grid')
        incumbent = recipe.get('incumbent', {})
        path = ROOT.parent / 'development_incumbent.json'
        require(incumbent.get('path') == str(path) and incumbent.get('sha256') == sha256(path)
                and incumbent.get('score') == read(path).get('score')
                and recipe.get('minimum_score_improvement') == 1e-4,
                'selected recipe must freeze the latest development incumbent, not an older control')
        actual=read(path);prior=recipe.get('completed_prior_summary',{});summary=Path(actual['validated_summary'])
        require(actual.get('strict_success') is False,'strict-win confirmation takes priority over further H trials')
        require(prior.get('path') == str(summary) and prior.get('sha256') == actual['validated_summary_sha256']
                and summary.resolve().is_relative_to(ROOT.parent.resolve()) and sha256(summary) == prior['sha256'],
                'selected recipe must freeze the latest completed campaign summary')
        previous=read(summary);family=previous.get('inner_precision_family',{});names=family.get('completed_candidates',[])
        trials=previous.get('trials',[]);candidates=[row for row in trials if row.get('name') in names]
        require(previous.get('status') == 'completed_real_trial_snapshot' and family.get('frequency_t') == 1.
                and family.get('cg_steps') == 96 and family.get('true_residual_diagnostics') is True
                and type(names) is list and len(names) == len(candidates) == 1
                and candidates[0].get('method_variant') == 'inner_precision_mm'
                and candidates[0].get('cg_steps') == 96 and candidates[0].get('true_residual_diagnostics') is True,
                'formal H trial requires the completed single-candidate Q family, not pending Q')
        require(candidates[0].get('strict_success') is False,
                'provisional Q strict win requires confirmation before H formal trials')


def control_bindings(args):
    paths=[args.review_gate]
    if args.stage != 'audit0':paths.append(args.zero_audit_gate)
    if args.stage == 'trial500':
        paths.extend((args.noop_gate,args.selected_recipe))
        recipe=read(args.selected_recipe)
        paths.extend(Path(recipe[key]['path']) for key in ('incumbent','completed_prior_summary'))
    return {str(Path(path).resolve()):sha256(path) for path in paths}


def load_campaign():
    """Load unchanged C scheduling definitions, all stdlib at import time."""
    original_path = list(sys.path); old_objective = sys.modules.get('objective')
    try:
        sys.path.insert(0, str(BASE))
        spec = importlib.util.spec_from_file_location('objective', BASE / 'objective.py')
        objective = importlib.util.module_from_spec(spec); sys.modules['objective'] = objective
        spec.loader.exec_module(objective)
        spec = importlib.util.spec_from_file_location('_geometry_campaign_selector', BASE / 'autoresearch.py')
        campaign = importlib.util.module_from_spec(spec); spec.loader.exec_module(campaign)
        require(Path(campaign.__file__).resolve() == BASE / 'autoresearch.py', 'unexpected C selector')
        return campaign
    finally:
        sys.path[:] = original_path
        if old_objective is None: sys.modules.pop('objective', None)
        else: sys.modules['objective'] = old_objective


@contextmanager
def claim_lock(path):
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        require(os.fstat(descriptor).st_uid == os.getuid(), 'claim lock has another owner')
        fcntl.flock(descriptor, fcntl.LOCK_EX); os.set_inheritable(descriptor, True)
        yield descriptor
    finally:
        os.close(descriptor)


def initial_visibility():
    for item in Path('/proc/self/environ').read_bytes().split(b'\0'):
        if item.startswith(b'CUDA_VISIBLE_DEVICES='):
            return item.split(b'=', 1)[1].decode()
    return None


def process_start():
    return Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]


def request(args):
    return {'stage': args.stage, 'exp': args.exp, 'geometry_factor': args.geometry_factor,
            'frequency_t':stage_profile(args.stage,args.geometry_factor,args.frequency_t)['frequency_t'],
            **{name: str(getattr(args, name).resolve()) if getattr(args, name) is not None else None
               for name in ('review_gate', 'zero_audit_gate', 'noop_gate', 'selected_recipe')}}


def validate_reservation(args, selection, hashes, invariants):
    require(args.reservation_token and selection.get('reservation_token') == args.reservation_token
            and selection.get('parent_pid') == os.getpid() and selection.get('process_start') == process_start(),
            'reservation is not owned by this exact CPU parent')
    require(selection.get('code_sha256') == hashes and selection.get('invariants') == invariants
            and selection.get('request') == request(args), 'reservation source/input/request changed')
    require(selection.get('control_bindings') == control_bindings(args),'reservation review/control/anchor changed')
    require(selection.get('resource', {}).get('gpu') == args.reserved_gpu
            and initial_visibility() == os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.reserved_gpu),
            'reservation must be in this CPU parent INITIAL CVD')
    require(args.claim_fd is not None and args.claim_fd >= 3, 'inherited common claim lock is required')
    actual, expected = os.fstat(args.claim_fd), CLAIM_LOCK.stat()
    require((actual.st_dev, actual.st_ino, actual.st_uid) == (expected.st_dev, expected.st_ino, os.getuid()),
            'inherited common claim descriptor differs')
    fcntl.flock(args.claim_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def child_command(args, launch_path, gpu):
    return [str(PYTHON), '-X', 'utf8', '-B', '-u', str(ROOT / 'runner.py'),
            '--stage', args.stage, '--exp', args.exp, '--geometry-factor', str(args.geometry_factor),
            '--frequency-t',str(stage_profile(args.stage,args.geometry_factor,args.frequency_t)['frequency_t']),
            '--gpu', str(gpu), '--launch-manifest', str(launch_path)]


def wait_owned_worker(worker):
    interrupted = None
    while worker.poll() is None:
        try: worker.wait()
        except KeyboardInterrupt as exc: interrupted = exc
    if interrupted is not None:
        interrupted.worker_returncode, interrupted.worker_pid = worker.returncode, worker.pid
        raise interrupted
    return worker.returncode


def parent_runtime():
    require('torch' not in sys.modules and 'frequency_weights' not in sys.modules,
            'CPU supervisor must never import Torch/frequency utility')
    require(Path(sys.executable).resolve() == CPU_PYTHON.resolve()
            and sys.version_info[:3] == (3, 11, 15), 'CPU supervisor must use NVGEN Python 3.11.15')
    return {'python': sys.executable, 'python_version': list(sys.version_info[:3]), 'torch_imported': False}


def execute(args, full_argv):
    runtime = parent_runtime(); hashes = code_hashes(); invariants = snapshot_invariants()
    validate_stage_gates(args, hashes, invariants)
    bindings=control_bindings(args)
    require(not (ROOT / 'outputs' / args.exp).exists(), 'fresh output is required; preserve previous artifacts')
    selection_path = VERIFY / f'{args.exp}_selection.json'; campaign = load_campaign()
    if args.reserved_gpu is None:
        require(args.claim_fd is None and args.reservation_token is None, 'internal reservation flags cannot be supplied initially')
        with claim_lock(CLAIM_LOCK) as descriptor:
            selected = campaign.select_gpu(campaign.gpu_snapshot(), args.preferred_gpu, os.environ)
            args.reservation_token = uuid.uuid4().hex
            selection = {'timestamp': time.time(), 'parent_pid': os.getpid(), 'process_start': process_start(),
                         'reservation_token': args.reservation_token, 'request': request(args), 'resource': selected,
                         'code_sha256': hashes, 'invariants': invariants}
            selection['control_bindings']=bindings
            write_new(selection_path, selection)
            environment = {**os.environ, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': str(selected['gpu'])}
            # Same stdlib-only interpreter and PID; no NVGEN Torch numerical work.
            os.execve(sys.executable, [sys.executable, '-X', 'utf8', '-B', '-u', str(Path(__file__).resolve()),
                      *full_argv, '--reserved-gpu', str(selected['gpu']), '--reservation-token', args.reservation_token,
                      '--claim-fd', str(descriptor)], environment)
        raise RuntimeError('reservation reexec unexpectedly returned')
    selection = read(selection_path)
    validate_reservation(args, selection, hashes, invariants)
    try:
        selected = campaign.select_gpu(campaign.gpu_snapshot(), args.reserved_gpu, os.environ)
        require(selected['gpu'] == args.reserved_gpu, 'reserved GPU changed')
        if selection['resource']['free']:
            require(selected['free'], 'previously idle GPU gained a workload; leave it untouched')
    finally:
        os.close(args.claim_fd); args.claim_fd = None
    launch_path = VERIFY / f'{args.exp}_launch.json'
    command = child_command(args, launch_path, args.reserved_gpu)
    launch = {'timestamp': time.time(), 'parent_pid': os.getpid(), 'process_start': process_start(),
              'method_variant':'frequency_adam','frequency_t':stage_profile(args.stage,args.geometry_factor,args.frequency_t)['frequency_t'],
              'geometry_factor':args.geometry_factor,
              'reservation_token': args.reservation_token, 'request': request(args), 'command': command,
              'initial_visibility': initial_visibility(), 'resource': selected, 'initial_resource': selection['resource'],
              'code_sha256': hashes, 'invariants': invariants, 'parent_runtime': runtime}
    launch['control_bindings']=bindings
    require(code_hashes() == hashes, 'geometry source changed before launch')
    write_new(launch_path, launch)
    environment = {**os.environ, 'PYTHONNOUSERSITE': '1', 'PYTHONPATH': '', 'PYTHONDONTWRITEBYTECODE': '1',
                   'CUDA_CACHE_DISABLE': '1'}
    worker, returncode, error = None, None, None
    with (VERIFY / f'{args.exp}_launch.log').open('x') as stream:
        try:
            worker = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            write_new(VERIFY / f'{args.exp}_started.json', {'parent_pid': os.getpid(), 'worker_pid': worker.pid,
                      'reservation_token': args.reservation_token})
            returncode = wait_owned_worker(worker)
            if returncode: raise subprocess.CalledProcessError(returncode, command)
        except BaseException as exc:
            error = exc
        finally:
            if worker is not None and worker.poll() is None:
                try: wait_owned_worker(worker)
                except BaseException as exc:
                    if error is None: error = exc
            if worker is not None: returncode = worker.returncode
    proof_path = VERIFY / f'{args.exp}_post_run_invariants.json'
    proof = {'status': 'checking', 'before': {'code_sha256': hashes, 'invariants': invariants}, 'after': {},
             'control_bindings_before':bindings}
    try:
        proof['after'] = {'code_sha256': code_hashes(), 'invariants': snapshot_invariants()}
        require(proof['after'] == proof['before'], 'source/input/checkpoint/binary bytes changed after child')
        proof['control_bindings_after']=control_bindings(args)
        require(proof['control_bindings_after'] == bindings,'review/control/quality anchor bytes changed after child')
        proof['status'] = 'verified'
    except BaseException as exc:
        proof.update(status='failed', error=repr(exc))
        if error is None: error = exc
    try:
        write_new(proof_path, proof)
        if error is None:
            artifacts = validate_child_artifacts(args, ROOT / 'outputs' / args.exp, hashes, invariants)
        parent_runtime()
    except BaseException as exc:
        if error is None: error = exc
    if error is not None:
        error.worker_returncode, error.worker_pid = returncode, worker.pid if worker is not None else None
        raise error
    try:
        write_new(VERIFY / f'{args.exp}_completed.json', {
            'passed': True, 'status': 'complete', 'stage': args.stage, 'geometry_factor': args.geometry_factor,
            'method_variant':'frequency_adam','frequency_t':stage_profile(args.stage,args.geometry_factor,args.frequency_t)['frequency_t'],
            'timestamp': time.time(), 'parent_pid': os.getpid(), 'worker_pid': worker.pid, 'returncode': returncode,
            'code_sha256': hashes, 'invariants': invariants, 'artifacts': artifacts,
            'control_bindings':bindings,
            'post_run_invariants': str(proof_path), 'post_run_invariants_sha256': sha256(proof_path)})
    except BaseException as exc:
        exc.worker_returncode, exc.worker_pid = returncode, worker.pid
        raise


def record_failure(args, error):
    if (not args.reservation_token or args.exp in ('', '.', '..') or Path(args.exp).name != args.exp): return
    try: selection = read(VERIFY / f'{args.exp}_selection.json')
    except (OSError, ValueError): return
    if (selection.get('parent_pid'), selection.get('process_start'), selection.get('reservation_token')) != \
            (os.getpid(), process_start(), args.reservation_token): return
    destination = VERIFY / f'{args.exp}_failure.json'
    if destination.exists(): return
    path = ROOT / 'outputs' / args.exp / 'provenance.json'; artifact, artifact_error = {}, None
    try:
        if path.exists():
            artifact = read(path); require(isinstance(artifact, dict), 'partial child provenance is not an object')
    except (OSError, ValueError) as exc: artifact, artifact_error = {}, repr(exc)
    write_new(destination, {'status': 'failed', 'stage': args.stage, 'timestamp': time.time(),
        'parent_pid': os.getpid(), 'reservation_token': args.reservation_token, 'error': repr(error),
        'returncode': getattr(error, 'worker_returncode', getattr(error, 'returncode', None)),
        'worker_pid': getattr(error, 'worker_pid', None), 'child_artifact': str(path),
        'artifact_status': artifact.get('status'), 'artifact_read_error': artifact_error})


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ['--describe-runtime']:
        print(json.dumps(parent_runtime())); return
    args = parser().parse_args(argv)
    try: execute(args, argv)
    except BaseException as exc:
        try: record_failure(args, exc)
        except Exception as audit_error: print(f'Failure evidence could not be written: {audit_error!r}', file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
