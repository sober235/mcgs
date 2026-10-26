#!/usr/bin/env python3
"""Stdlib source-bound supervisor for the case-internal magnitude prior."""
from __future__ import annotations

import argparse
import ast
import hashlib
import math
from pathlib import Path
import re
import struct
import sys
from types import FunctionType

import run_relocation as old

base = old.base
ROOT = old.ROOT
ROUTE = 'g2_case_internal_magnitude_prior_v1'
CHILD_RUNNER = 'caseprior_runner.py'
CONTEXT_ACK = True
PARAMETERS = ('route', 'stage', 'mode', 'steps')
SETTINGS = dict(updates=128, patch_size=32, batch_size=4, width=32, lr=.001, tile_size=32)
POLICY = dict(source_iteration=4500, seed=42, point_count=3600000,
    optimizer='original_seven_group_Adam_full_state', gamma=.9995,
    restoration='original_resume_protocol_no_geometry_reapply',
    frequency='original_flat_t1_4500_observer', event='none',
    denoiser='case_internal_cnn_v1', lambda_nl=.1, nl_sigma_rel=.015,
    nl_start=4501, nl_refresh=100, cnn=SETTINGS,
    cnn_seed_stride=1009, normalization='max_P99_crop_1e-6',
    inference_halo=5, target='detached_independent_per_scan_magnitude')
require = base.require


def code_hashes():
    # Old closure includes the original trainer and all numerical/common code.
    return {**old.code_hashes(), **{str((ROOT/name).resolve()): base.sha256(ROOT/name)
        for name in ('run_caseprior.py', 'caseprior_runner.py', 'case_denoiser.py')}}


def snapshot_invariants():
    return {**old.snapshot_invariants(), 'route': ROUTE, 'policy': POLICY}


def parameters(args):
    return {key: getattr(args, key) for key in PARAMETERS}


def validate_parameters(args):
    require(args.route == ROUTE and args.stage in ('zero', 'smoke', 'trial')
            and args.mode in ('control', 'learned'), 'invalid case-prior route/stage/mode')
    require(type(args.steps) is int and args.steps == {'zero': 0, 'smoke': 2, 'trial': 500}[args.stage],
            'fixed zero0/smoke2/trial500 required')
    require(args.stage != 'zero' or args.mode == 'control', 'zero is control only')


def _prior_config(config, mode):
    loss = config.get('loss', {})
    coefficient = .1 if mode == 'learned' else 0.
    for key, value in dict(lambda_nl=coefficient, lambda_nl_plain=coefficient,
        lambda_nl_contrast=coefficient, nl_denoiser='case_internal_cnn_v1',
        nl_sigma_rel=.015, nl_start=4501, nl_refresh=100, nl_denoiser_kwargs={}).items():
        require(loss.get(key) == value and type(loss.get(key)) is type(value), 'frozen prior configuration differs')


def _real_target(path):
    """Check canonical float32 NPY target and its raw-array hash without Torch."""
    with Path(path).open('rb') as stream:
        require(stream.read(6) == b'\x93NUMPY', 'invalid target NPY')
        major, minor = stream.read(2)
        require((major, minor) in ((1, 0), (2, 0)), 'unsupported target NPY')
        size = struct.unpack('<H' if major == 1 else '<I', stream.read(2 if major == 1 else 4))[0]
        require(size <= 65536, 'target header too large')
        header = ast.literal_eval(stream.read(size).decode('latin1'))
        shape = header.get('shape')
        require(header.get('descr') == '<f4' and header.get('fortran_order') is False
            and isinstance(shape, tuple) and len(shape) == 3 and all(type(x) is int and x > 0 for x in shape),
            'target must be C-order float32 3D')
        digest = hashlib.sha256(); remaining = math.prod(shape)*4; positive = False
        while remaining:
            chunk = stream.read(min(1 << 20, remaining))
            require(chunk and len(chunk) % 4 == 0, 'truncated target')
            for (value,) in struct.iter_unpack('<f', chunk):
                require(math.isfinite(value) and value >= 0, 'nonfinite/negative target')
                positive |= value > 0
            digest.update(chunk); remaining -= len(chunk)
        require(stream.read(1) == b'', 'extra target bytes')
    return dict(shape=list(shape), dtype='float32', sha256=digest.hexdigest()), positive


def _cnn_shapes(width):
    channels = [1, width, width, width, width, 1]
    return {f'network.{2*i}.{kind}': ([channels[i+1], channels[i], 3, 3, 3] if kind == 'weight' else [channels[i+1]])
            for i in range(5) for kind in ('weight','bias')}


def _validate_denoisers(metrics, directory, args, *, settings, cuda_devices):
    settings = {**SETTINGS, **settings}
    for state in ('before', 'after'):
        _prior_config(metrics[state]['config'], args.mode)
    calls = metrics.get('denoiser_calls')
    require(isinstance(calls, list), 'ordered denoiser calls required')
    checks = metrics.get('denoiser_checkpoint_checks')
    require(isinstance(checks, list) and len(checks) == len(calls), 'actual child checkpoint readback count differs')
    for index, (record, check) in enumerate(zip(calls, checks)):
        require(check.get('schema') == 'case_denoiser_checkpoint_check_v1'
            and type(check.get('call_index')) is int and check['call_index'] == index
            and check.get('checkpoint') == record.get('checkpoint')
            and check.get('final_parameter_sha256') == record.get('final_parameter_sha256')
            and check.get('metadata_equal') is True and check.get('finite_float32_cpu') is True
            and check.get('tensor_shapes') == _cnn_shapes(settings['width']),
            'trusted child checkpoint semantic readback binding differs')
    rng = metrics.get('denoiser_rng')
    require(isinstance(rng, list) and len(rng) == len(calls), 'actual per-call RNG evidence required')
    for index, row in enumerate(rng):
        require(row.get('schema') == 'case_denoiser_rng_v1' and type(row.get('call_index')) is int
                and row['call_index'] == index, 'per-call RNG identity differs')
        require(isinstance(row.get('cuda'), list) and [r.get('device') for r in row['cuda']] == cuda_devices,
                'all visible CUDA default RNG observations required')
        for state in [row.get('cpu', {}), *row['cuda']]:
            old._sha(state.get('before_sha256')); old._sha(state.get('after_sha256'))
            require(state['before_sha256'] == state['after_sha256'] and state.get('preserved') is True,
                    'caller CPU/visible CUDA default RNG changed')
    summary = metrics.get('nonlocal_summary')
    data_config = metrics['before']['config'].get('data', {})
    scans = tuple(data_config.get(channel, {}).get('name', channel) for channel in ('plain','contrast'))
    require(all(isinstance(s, str) and s for s in scans) and len(set(scans)) == 2,
            'distinct original scan names required')
    if not args.steps:
        require(calls == [] and summary is None, 'zero must bypass trainer/CNN work')
        return
    require(isinstance(summary, dict), 'original nonlocal summary required')
    learned = args.mode == 'learned'
    require(summary.get('enabled') is learned and summary.get('denoiser') == 'case_internal_cnn_v1'
        and summary.get('coefficients') == {s: .1 if learned else 0. for s in scans}
        and summary.get('start') == 4501 and summary.get('refresh_every') == 100
        and summary.get('sigma_rel') == .015 and summary.get('all_gradients_finite') is True,
        'original RED policy differs')
    iterations = list(range(4501, 4501+args.steps, 100)) if learned else []
    require(summary.get('expected_refresh_iterations') == iterations, 'actual expected refresh schedule differs')
    history = summary.get('refresh_history', {})
    require(set(history) == set(scans), 'two independent scan histories required')
    count = 2*len(iterations)
    require(len(calls) == count and sum(old._integer(r.get('updates'), 'CNN updates') for r in calls)
            == count*settings['updates'], 'actual CNN calls/update budget differs')
    require(summary.get('validated_active') is learned and type(summary.get('active_step_count')) is int
            and summary['active_step_count'] == (args.steps if learned else 0), 'validated active RED steps differ')
    if learned:
        require(0 < old._integer(summary.get('nonzero_weighted_step_count'), 'nonzero RED steps') <= args.steps,
                'positive prior work required')
        for key in ('weighted_R_nl_min', 'weighted_R_nl_max', 'weighted_R_nl_last'):
            value = old._number(summary.get(key), key, 0)
            if key == 'weighted_R_nl_max': require(value > 0, 'positive original prior loss required')
    else:
        require(summary.get('nonzero_weighted_step_count') == 0 and summary.get('weighted_R_nl_max') == 0.
                and summary.get('weighted_R_nl_last') == 0. and summary.get('weighted_R_nl_min') is None,
                'control reports learned work')
    for scan_index, scan in enumerate(scans):
        rows = history[scan]
        require(isinstance(rows, list) and [r.get('iter') for r in rows] == iterations, 'actual refresh order differs')
        for key in ('raw_R_nl_by_channel_last', 'raw_R_nl_by_channel_max'):
            value = old._number(summary.get(key, {}).get(scan), 'per-scan prior loss', 0)
            require((value > 0 if key.endswith('_max') else value >= 0) if learned else value == 0,
                    'independent per-scan positive prior required')
        for index, row in enumerate(rows):
            record = calls[2*index + scan_index]
            require(row.get('kernel_stats') == record, 'original refresh/Task1 record linkage differs')
            require(row.get('sigma') == record.get('sigma') and list(row.get('crop', [])) == record.get('input', {}).get('shape'),
                    'actual source crop/sigma differs')
            old._number(row.get('seconds'), 'refresh time', 0)
            old._number(row.get('target_rms_difference'), 'target difference', 0)
            old._number(row.get('target_relative_norm'), 'relative target difference', 0)
        if learned: require(any(row['target_relative_norm'] > 0 for row in rows), 'unchanged learned targets refused')
    for index, record in enumerate(calls):
        require(set(record) == {'schema','call_index','seed','status','input','sigma','normalization',
            'normalized_sigma','updates','patch_size','batch_size','width','lr','tile_size','losses','gradients',
            'changed_parameters','initial_parameter_sha256','final_parameter_sha256','elapsed_seconds',
            'output','checkpoint','target','manifest'}, 'exact frozen Task1 record schema required')
        require(record.get('schema') == 'case_magnitude_denoiser_call_v1'
            and type(record.get('call_index')) is int and record['call_index'] == index
            and type(record.get('seed')) is int and record['seed'] == 42+1009*index
            and record.get('status') == 'trained', 'fresh trained CNN order/seed required')
        for key, value in settings.items():
            require(type(record.get(key)) is type(value) and record[key] == value, 'frozen CNN setting/work differs')
        require(old._number(record.get('sigma'), 'sigma', 0) > 0, 'positive sigma required')
        normalization = old._number(record.get('normalization'), 'normalization', 1e-6)
        require(record.get('normalized_sigma') == record['sigma']/normalization, 'normalized sigma differs')
        old._number(record.get('elapsed_seconds'), 'CNN seconds', 0)
        require(isinstance(record.get('losses'), dict) and set(record['losses']) == {'first','last'},
                'actual first/last CNN losses required')
        for value in record['losses'].values(): old._number(value, 'CNN loss', 0)
        gradients = record.get('gradients', {})
        require(gradients.get('all_finite') is True and gradients.get('any_nonzero') is True
            and 0 < old._integer(gradients.get('steps_with_nonzero'), 'nonzero CNN gradients') <= settings['updates']
            and old._number(gradients.get('max_abs'), 'CNN gradients', 0) > 0, 'actual finite nonzero CNN gradients required')
        names = {f'network.{i}.{kind}' for i in (0,2,4,6,8) for kind in ('weight','bias')}
        changed = record.get('changed_parameters')
        require(isinstance(changed, list) and changed and len(changed) == len(set(changed))
                and set(changed) <= names, 'changed CNN parameters required')
        for key in ('initial_parameter_sha256', 'final_parameter_sha256'): old._sha(record.get(key))
        require(record['initial_parameter_sha256'] != record['final_parameter_sha256'], 'unchanged CNN weights refused')
        root = directory/'denoiser'/f'call_{index:04d}'
        for key, filename in (('manifest','manifest.json'),('target','target.npy'),('checkpoint','checkpoint.pt')):
            path = base.verify_descriptor(record.get(key), 'CNN '+key)
            require(Path(path).resolve() == (root/filename).resolve() and Path(path).resolve().is_relative_to(directory),
                    'CNN artifact escapes owned call directory')
        manifest = base.read(record['manifest']['path'])
        require(manifest == {k:v for k,v in record.items() if k != 'manifest'}, 'Task1 manifest/record differs')
        output, positive = _real_target(record['target']['path'])
        require(positive and all(record.get('output', {}).get(k) == v for k,v in output.items())
                and output['shape'] == record.get('input', {}).get('shape'), 'actual positive target/output hash differs')
        old._integer(record['output'].get('clipped_count'), 'clipped target voxels')
        require(record['output']['clipped_count'] <= math.prod(output['shape']), 'invalid clipped count')
        old._sha(record['input'].get('sha256'))
        require(record['input'].get('dtype') == 'float32', 'original RED float32 crop required')
    for row in metrics['trace']:
        prior = row.get('nonlocal', {})
        require(prior.get('enabled') is learned, 'actual per-step RED enable differs')
        weighted = old._number(prior.get('weighted'), 'per-step prior', 0)
        require(learned or weighted == 0, 'actual per-step prior loss differs')
        raw = prior.get('raw_by_channel', {})
        require(set(raw) == set(scans), 'per-step independent scan prior losses required')
        for value in raw.values():
            old._number(value, 'per-step scan prior', 0)
            require(learned or value == 0, 'per-step scan prior loss differs')
        expected = .1*sum(raw.values()) if learned else 0.
        require(math.isclose(weighted, expected, rel_tol=1e-6, abs_tol=1e-12), 'weighted original prior loss differs')
        require(prior.get('refresh_iterations_by_channel') ==
            {s: [i for i in iterations if i <= row['iteration']] for s in scans}, 'per-step actual refresh clock differs')
    weighted = [row['nonlocal']['weighted'] for row in metrics['trace']]
    require(summary.get('weighted_R_nl_last') == weighted[-1]
        and summary.get('weighted_R_nl_max') == max(weighted)
        and summary.get('weighted_R_nl_min') == (min(weighted) if learned else None)
        and summary.get('nonzero_weighted_step_count') == sum(v > 0 for v in weighted)
        and summary.get('raw_R_nl_by_channel_max') ==
            {s: max(row['nonlocal']['raw_by_channel'][s] for row in metrics['trace']) for s in scans}
        and summary.get('raw_R_nl_by_channel_last') == metrics['trace'][-1]['nonlocal']['raw_by_channel'],
        'actual per-step prior/summary observations disagree')


def _validate_child_artifacts(args, directory, hashes, invariants, *, point_count=3600000,
        device='cuda:0', backend='tile_cuda', coils=42, denoiser_settings=None):
    # Fixed private adapter over immutable common validator bytecode. It only
    # changes route/policy/CLI dispatch; all original state/clock/ACK/frequency/
    # physics/checkpoint/canonical-export checks still run. No module is patched.
    validator = FunctionType(old._validate_child_artifacts.__code__,
        {**old._validate_child_artifacts.__globals__, 'ROUTE': ROUTE, 'POLICY': POLICY,
         'validate_parameters': validate_parameters, 'parameters': parameters}, 'caseprior_common_artifacts')
    validator.__kwdefaults__ = old._validate_child_artifacts.__kwdefaults__
    bindings = validator(args, directory, hashes, invariants, point_count=point_count,
                         device=device, backend=backend, coils=coils)
    directory = Path(directory).resolve(strict=True)
    metrics = base.read(directory/'metrics.json')
    _validate_denoisers(metrics, directory, args, settings=denoiser_settings or {},
                        cuda_devices=['cuda:0'] if device == 'cuda:0' else [])
    excluded = {'launch.json','metrics.json','provenance.json','launch.log','child_started.json',
                'childcontext_ready.json','childcontext_ack.json'}
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p.name not in excluded}
    # post_run_invariants.json is parent-owned and appears after child publishing.
    actual.difference_update(('post_run_invariants.json','completed.json'))
    require(actual == set(metrics['artifact_manifest']), 'unlisted child artifact refused')
    return bindings


def validate_child_artifacts(args, directory, hashes, invariants):
    require(invariants.get('route') == ROUTE and invariants.get('policy') == POLICY
            and invariants.get('g2_checkpoint_sha256') == base.G2_SHA256, 'real frozen case-prior inputs required')
    metrics = base.read(Path(directory)/'metrics.json')
    data = metrics.get('before', {}).get('config', {}).get('data', {})
    require(data.get('plain', {}).get('name') == 'pre' and data.get('contrast', {}).get('name') == 'post',
            'frozen original production pre/post scan names required')
    return _validate_child_artifacts(args, directory, hashes, invariants)


def validate_observed_gate(path, stage, hashes, invariants, matched=None):
    validator = FunctionType(old.validate_observed_gate.__code__,
        {**old.validate_observed_gate.__globals__, 'ROUTE': ROUTE, 'PARAMETERS': PARAMETERS,
         'validate_parameters': validate_parameters, 'validate_child_artifacts': validate_child_artifacts},
        'caseprior_observed_gate')
    return validator(path, stage, hashes, invariants, matched)


def _smoke_start(metrics):
    before = metrics['before']
    return dict(parameters={n: {k: before['parameters'][n][k] for k in ('shape','dtype','sha256')}
                    for n in old.NAMES},
        adam={n: {k: {f: r[f] for f in ('shape','dtype','sha256')}
                  for k,r in before['optimizer']['states'][n].items()} for n in old.NAMES},
        groups=[g['options'] for g in before['optimizer']['groups']],
        scheduler=before['scheduler']['state'])


def _relative_complex_npy(left, right):
    # The common stdlib parser first validates shape/dtype/finiteness/exact EOF.
    require(old._npy_record(left) == old._npy_record(right), 'ordinary smoke export shape/dtype differs')
    def skip_header(stream):
        stream.read(6); major, minor = stream.read(2)
        size = struct.unpack('<H' if major == 1 else '<I', stream.read(2 if major == 1 else 4))[0]
        stream.read(size)
    difference = norm = 0.
    with Path(left).open('rb') as a, Path(right).open('rb') as b:
        skip_header(a); skip_header(b)
        while True:
            x, y = a.read(1 << 20), b.read(1 << 20)
            require(len(x) == len(y), 'ordinary smoke export length differs')
            if not x: break
            for (v,), (w,) in zip(struct.iter_unpack('<f', x), struct.iter_unpack('<f', y)):
                difference += (v-w)**2; norm += w*w
    relative = math.sqrt(difference/max(norm, 1e-24))
    require(relative <= 1e-5, 'new ordinary smoke canonical parity failed')
    return relative


def _validate_native_control_receipt(path, hashes, invariants, ordinary_smoke, *,
                                     old_hashes, old_gate_validator):
    """Keep OLD semantics/closure while deriving common inputs from one fresh snapshot.

    Private CPU tests supply only the old tiny-artifact validator; the public
    entry below always uses the unmodified old real-only validator/closure.
    No call takes another complete MRI input snapshot to change route policy.
    """
    receipt = base.read(path)
    require(receipt.get('schema') == 'native_control_receipt_v1' and receipt.get('route') == ROUTE
        and receipt.get('status') == 'verified' and receipt.get('code_sha256') == hashes
        and receipt.get('invariants') == invariants, 'fresh source/input-bound native receipt required')
    old_invariants = {**invariants, 'route': old.ROUTE, 'policy': old.POLICY}
    require(receipt.get('native_code_sha256') == old_hashes
        and receipt.get('native_invariants') == old_invariants, 'original native source closure/common inputs differ')
    paths = {key: base.verify_descriptor(receipt.get(key), key) for key in
             ('native_completion','native_smoke','ordinary_smoke','objective')}
    native = old_gate_validator(paths['native_completion'], 'trial', old_hashes, old_invariants, dict(mode='control'))
    smoke = old_gate_validator(paths['native_smoke'], 'smoke', old_hashes, old_invariants, dict(mode='control'))
    for gate, stage, steps in ((native,'trial',500),(smoke,'smoke',2)):
        require(gate.get('route') == old.ROUTE and gate.get('stage') == stage and gate.get('mode') == 'control'
            and type(gate.get('steps')) is int and gate['steps'] == steps,
            'actual old native500/control and ordinary smoke required')
    require(base.read(paths['ordinary_smoke']) == ordinary_smoke and ordinary_smoke.get('route') == ROUTE
        and ordinary_smoke.get('stage') == 'smoke' and ordinary_smoke.get('mode') == 'control'
        and ordinary_smoke.get('steps') == 2 and ordinary_smoke.get('code_sha256') == hashes
        and ordinary_smoke.get('invariants') == invariants, 'matching new ordinary smoke required')
    old_metrics = base.read(Path(smoke['run_dir'])/'metrics.json')
    new_metrics = base.read(Path(ordinary_smoke['run_dir'])/'metrics.json')
    require(_smoke_start(old_metrics) == _smoke_start(new_metrics), 'ordinary smoke restored start/state differs')
    require(old_metrics['counters'] == new_metrics['counters'] and old_metrics['physics_work'] == new_metrics['physics_work']
        and old_metrics['after']['scheduler']['state'] == new_metrics['after']['scheduler']['state'],
        'ordinary smoke clock/work differs')
    parity = {s: _relative_complex_npy(Path(ordinary_smoke['run_dir'])/f'candidate_{s}_complex.npy',
                                     Path(smoke['run_dir'])/f'candidate_{s}_complex.npy') for s in ('pre','post')}
    objective = base.read(paths['objective']); directory = Path(native['run_dir']).resolve(strict=True)
    require(objective.get('schema_version') == 1 and objective.get('run_dir') == str(directory)
        and objective.get('metrics') == ['psnr','ssim','nmse']
        and objective.get('primary_objective') == 'whole_head_min_six_normalized_margin'
        and objective.get('all_inputs_unchanged') is True and type(objective.get('strict_success')) is bool,
        'actual scored native objective required')
    old._number(objective.get('score'), 'native objective score')
    candidates = objective.get('input_hashes', {}).get('candidate', {})
    require(set(candidates) == {'pre','post'} and set(objective.get('candidate_sha256', {})) == {'pre','post'},
            'both scored native arrays required')
    native_metrics = base.read(directory/'metrics.json')
    for scan in ('pre','post'):
        array = directory/f'candidate_{scan}_complex.npy'
        expected = {**base.descriptor(array), **old._npy_record(array)}
        require(candidates[scan] == expected and objective['candidate_sha256'][scan] == expected['sha256']
            and native_metrics['arrays'][scan]['candidate'] == expected, 'scored native candidate descriptors differ')
        require(objective.get('input_hashes', {}).get('files', {}).get(str(array)) == expected['sha256'],
                'native score file binding differs')
        values = objective.get('whole_head_metrics', {}).get(scan, {})
        require(set(values) == {'psnr','ssim','nmse'}, 'six native score metrics required')
        for key, value in values.items(): old._number(value, 'native '+key)
    require(objective.get('lesion_planes', {}).get('eligible_planes') == 59, 'native 59-plane score required')
    require(objective.get('input_hashes', {}).get('files', {}).get(str(ROOT/'evaluate.py')) == base.sha256(ROOT/'evaluate.py'),
            'native frozen evaluator source differs')
    # Preserve transitive source/control/proof/artifact descriptors in the new
    # launch's before/after binding rather than merely hashing the receipt text.
    bindings = {str(Path(path).resolve()): base.sha256(path) for path in paths.values()}
    for gate in (native,smoke,ordinary_smoke):
        bindings.update(gate['artifacts']); bindings.update(gate['control_bindings'])
        proof = base.verify_descriptor(gate['post_run_invariants'], 'native control proof')
        bindings[str(proof)] = base.sha256(proof)
    return {**receipt, 'bindings': bindings, 'ordinary_smoke_relative_l2': parity}


def validate_native_control_receipt(path, hashes, invariants, ordinary_smoke):
    receipt = base.read(path)
    actual_smoke = validate_observed_gate(base.verify_descriptor(receipt.get('ordinary_smoke'), 'ordinary smoke'),
        'smoke', hashes, invariants, dict(mode='control'))
    require(actual_smoke == ordinary_smoke, 'native receipt ordinary smoke differs')
    return _validate_native_control_receipt(path, hashes, invariants, ordinary_smoke,
        old_hashes=old.code_hashes(), old_gate_validator=old.validate_observed_gate)


def validate_stage_gates(args, hashes, invariants):
    validate_parameters(args)
    require(isinstance(args.exp, str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]*', args.exp), 'safe unique exp required')
    old._integer(args.preferred_gpu, 'preferred GPU')
    require(0 < old._number(args.timeout, 'timeout') <= 3600, 'owned-child timeout at most3600 required')
    review = base.validate_review(args.review_gate, hashes, invariants)
    require(review.get('route') == ROUTE, 'review must cover case-prior route')
    recipe = base.read(args.recipe)
    require(recipe.get('route') == ROUTE and recipe.get('schema_version') == 1 and recipe.get('status') == 'selected'
        and recipe.get('parameters') == parameters(args) and recipe.get('policy') == POLICY
        and recipe.get('code_sha256') == hashes and recipe.get('invariants') == invariants,
        'exact selected case-prior CLI/policy/source/input recipe required')
    paths = [args.review_gate, args.recipe, Path(review['review_report']['path'])]
    result = dict(review=review, recipe=recipe)
    required = []
    if args.stage in ('smoke','trial'): required.append(('zero', args.zero_gate, None))
    if args.stage == 'trial':
        required.extend([('smoke', args.control_smoke_gate, 'control'), ('smoke', args.learned_smoke_gate, 'learned')])
    for stage, path, mode in required:
        require(path is not None, f'{args.stage} requires actual {mode or "control"} {stage} gate')
        gate = validate_observed_gate(path, stage, hashes, invariants, dict(mode=mode) if mode else None)
        result[mode or stage] = gate; paths.append(path)
        paths.extend(gate['artifacts']); paths.extend(gate['control_bindings'])
        paths.append(base.verify_descriptor(gate['post_run_invariants'], 'prerequisite postproof'))
    if args.stage == 'trial' and args.mode == 'learned':
        require((args.native_control_receipt is None) != (args.control_gate is None), 'one native receipt or new control500 required')
        if args.native_control_receipt is not None:
            receipt = validate_native_control_receipt(args.native_control_receipt, hashes, invariants, result['control'])
            paths.append(args.native_control_receipt)
            paths.extend(receipt['bindings'])
        else:
            control = validate_observed_gate(args.control_gate, 'trial', hashes, invariants, dict(mode='control'))
            paths.append(args.control_gate)
            paths.extend(control['artifacts']); paths.extend(control['control_bindings'])
            paths.append(base.verify_descriptor(control['post_run_invariants'], 'control500 postproof'))
    result['control_bindings'] = {str(Path(p).resolve(strict=True)): base.sha256(p) for p in paths}
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.set_defaults(route=ROUTE)
    p.add_argument('--stage', choices=('zero','smoke','trial'), required=True)
    p.add_argument('--mode', choices=('control','learned'), required=True)
    p.add_argument('--steps', type=int, required=True); p.add_argument('--exp', required=True)
    p.add_argument('--review-gate', type=Path, required=True); p.add_argument('--recipe', type=Path, required=True)
    p.add_argument('--zero-gate', type=Path)
    p.add_argument('--control-smoke-gate', type=Path); p.add_argument('--learned-smoke-gate', type=Path)
    p.add_argument('--native-control-receipt', type=Path); p.add_argument('--control-gate', type=Path)
    p.add_argument('--preferred-gpu', type=int, default=0); p.add_argument('--timeout', type=float, default=3600)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    with base.cancellation_signals():
        try: print(base.execute(args, protocol=sys.modules[__name__]), flush=True)
        except BaseException as error:
            print(base.record_failure(args, error), file=sys.stderr, flush=True)
            return 128+error.signum if isinstance(error, base.SupervisorCancelled) else 1
    return 0


if __name__ == '__main__': raise SystemExit(main())
