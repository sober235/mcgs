#!/usr/bin/env python3
"""Stdlib-only supervisor for the bounded G2 amplitude representation diagnostic."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys

import raw_g2
import run_gpu as base

ROOT = Path(__file__).resolve().parent
ROUTE = 'g2_full_amplitude_representation_v1'
CHILD_RUNNER = 'representation_runner.py'
CONTEXT_ACK = True
FIT_CONFIG = dict(seed=42, bands=8, width=64, chunk_size=131072,
                  coefficient_updates=2000, coefficient_batch=32768, raw_updates=128,
                  lr=.001, betas=[.9, .999], epsilon=1e-8, weight_decay=0.,
                  clip_max_norm=1., log_floor=1e-12, std_floor=1e-6,
                  admission_relative_l2=.01, anchor_relative_l2=1e-5)
PARAMETERS = ('route', 'stage', 'encoder', *FIT_CONFIG)
LOCAL_SOURCES = ('run_representation.py', CHILD_RUNNER, 'full_amplitude.py', 'raw_g2.py',
                 'run_gpu.py', 'proxdc_runner.py', 'run_proxdc.py', 'runner.py')


def code_hashes():
    definition = raw_g2.source_definition()
    paths = [ROOT / name for name in LOCAL_SOURCES]
    paths += [base.BASE / name for name in ('autoresearch.py', 'objective.py')]
    external = {p: h for p, h in definition['expected_sha256'].items()
                if Path(p).suffix in ('.py', '.so', '.cpp', '.cu')}
    actual = {p: base.sha256(p) for p in external}
    base.require(actual == external, 'original Gaussian source/backend changed')
    return {**{str(p): base.sha256(p) for p in paths}, **actual}


def snapshot_invariants():
    # Deliberately do not call the inherited MRI snapshot: this route reads only
    # the G2 model/metadata and G2-derived raw teachers listed here.
    definition = raw_g2.source_definition()
    actual = {p: base.sha256(p) for p in definition['expected_sha256']}
    base.require(actual == definition['expected_sha256'], 'frozen G2/raw input bytes changed')
    return dict(route=ROUTE, source_definition=definition, verified_input_sha256=actual,
                runtime_contract=dict(model_class='gaussian.joint_gaussian_model.JointGaussianModel3D',
                    model_source=str(Path(definition['source_dir']) / 'joint_gaussian_model.py'),
                    voxelizer_class='gaussian.tile_voxelizer.TileVoxelizer',
                    voxelizer_source=str(Path(definition['source_dir']) / 'tile_voxelizer.py'),
                    binary=definition['binary'], backend='tile_cuda', use_cuda=True, use_cupy=False,
                    strict_cuda=True, tile_size=8, max_radius=20, volume_shape=definition['volume_shape'],
                    num_points=definition['num_points'], device_type='cuda', dtype='torch.float32'),
                input_policy='G2_model_metadata_original_Gaussian_and_raw_anchors_only',
                formal_mri_experiments=0, mri_arrays_loaded=False)


def parameters(args):
    return {key: getattr(args, key) for key in PARAMETERS}


def validate_parameters(args):
    base.require(args.route == ROUTE and args.stage in ('render_control', 'fit'), 'foreign representation route/stage')
    base.require(args.encoder in (('none',) if args.stage == 'render_control' else ('mlp', 'grid')),
                 'stage requires none for control or mlp/grid for fitting')
    for key, expected in FIT_CONFIG.items():
        value = getattr(args, key, None)
        base.require(type(value) is type(expected) and value == expected, f'fixed representation {key} differs')


def _finite(value, *, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def _hashes(value, keys=None):
    return isinstance(value, dict) and bool(value) and (keys is None or set(value) == set(keys)) \
        and all(isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v) for v in value.values())


def validate_child_artifacts(args, directory, hashes, invariants):
    validate_parameters(args)
    directory = Path(directory).resolve(strict=True)
    launch, metrics, provenance = [base.read(directory / name) for name in ('launch.json', 'metrics.json', 'provenance.json')]
    for value in (launch, metrics, provenance):
        base.require(all(value.get(k) == v for k, v in parameters(args).items()), 'child route/recipe parameters differ')
        base.require(value.get('code_sha256') == hashes and value.get('invariants') == invariants,
                     'child source/input bindings differ')
    base.require(launch.get('run_dir') == str(directory), 'child output ownership differs')
    recipe_path = base.verify_descriptor(launch.get('recipe'), 'actual selected recipe')
    recipe = base.read(recipe_path)
    base.require(launch.get('recipe_path') == str(recipe_path)
                 and launch.get('recipe_sha256') == launch['recipe']['sha256']
                 and recipe.get('schema_version') == 1 and recipe.get('status') == 'selected'
                 and recipe.get('route') == ROUTE and recipe.get('parameters') == parameters(args)
                 and recipe.get('code_sha256') == hashes and recipe.get('invariants') == invariants,
                 'actual selected recipe/launch bindings differ')
    controls = launch.get('control_bindings')
    base.require(isinstance(controls, dict) and controls and controls.get(str(recipe_path)) == base.sha256(recipe_path)
                 and all(base.sha256(p) == h for p, h in controls.items()), 'actual launch control bytes changed')
    ready = base.read(directory / 'childcontext_ready.json')
    ack = base.read(directory / 'childcontext_ack.json')
    base.require(type(ready.get('pid')) is int and ready['pid'] > 0
                 and type(launch.get('parent_pid')) is int and launch['parent_pid'] > 0
                 and ready.get('parent_pid') == launch['parent_pid']
                 and ready.get('gpu_uuid') == launch.get('gpu_uuid')
                 and ack == dict(schema_version=1, child_pid=ready['pid'], parent_pid=launch['parent_pid'],
                                 gpu_uuid=launch['gpu_uuid'], context_observed=True), 'observed context ACK differs')
    base.require(metrics == provenance, 'metrics/provenance evidence disagrees')
    value = metrics
    base.require(value.get('status') == 'complete' and value.get('software_complete') is True
                 and value.get('model_schema') == ROUTE and value.get('training_reference_access') is False
                 and type(value.get('formal_mri_experiments')) is int and value['formal_mri_experiments'] == 0,
                 'representation software completion/policy differs')
    base.require(value.get('source_runtime') == invariants.get('runtime_contract')
                 and isinstance(value.get('source_runtime'), dict), 'original source/backend runtime configuration differs')
    source = value.get('source_state_before')
    base.require(_hashes(source, raw_g2.source_definition()['source_names'])
                 and source == value.get('source_state_after') and value.get('source_requires_grad') ==
                 {name: False for name in source} and value.get('source_gradients_absent') is True,
                 'immutable seven-tensor source evidence differs')
    base.require(_hashes(value.get('teacher_state_before'), ('pre', 'post'))
                 and value['teacher_state_before'] == value.get('teacher_state_after'), 'fresh trainer-gauge teacher bytes changed')
    for scan in ('pre', 'post'):
        parity = value.get('raw_anchor', {}).get(scan, {})
        definition = invariants['source_definition']
        anchor = definition['anchors'][scan]
        base.require(parity.get('scan') == scan and parity.get('anchor') == anchor
                     and parity.get('anchor_sha256') == definition['expected_sha256'][anchor]
                     and parity.get('factor') == definition['gauges'][scan]
                     and type(parity.get('array_equal')) is bool and parity.get('passed') is True
                     and parity.get('tolerance') == 1e-5 and _finite(parity.get('relative_l2'))
                     and parity['relative_l2'] <= 1e-5, 'fresh raw anchor binding/numeric parity differs')
    expected_updates = dict(coefficient=2000, raw=128) if args.stage == 'fit' else dict(coefficient=0, raw=0)
    observed = value.get('updates_observed')
    base.require(isinstance(observed, dict) and set(observed) == set(expected_updates)
                 and all(type(observed[k]) is int and observed[k] == n for k, n in expected_updates.items()),
                 'actual update counts differ')
    work = value.get('work')
    expected_work = dict(gaussian_forwards=264, gaussian_vjps=256) if args.stage == 'fit' else dict(gaussian_forwards=2, gaussian_vjps=0)
    base.require(isinstance(work, dict) and set(work) == set(expected_work)
                 and all(type(work[k]) is int and work[k] == n for k, n in expected_work.items()), 'observed Gaussian calls/VJPs differ')
    progress = base.read(directory / 'progress.json')
    base.require(progress.get('status') == 'complete' and progress.get('active_phase') == 'complete'
                 and progress.get('work') == work and progress.get('updates_observed') == observed,
                 'final actual progress/count evidence differs')
    trace = [json.loads(line) for line in (directory / 'step_trace.jsonl').read_text().splitlines()]
    base.require(len(trace) == sum(expected_updates.values()), 'incomplete actual update trace')
    offset = 0
    for phase, count in expected_updates.items():
        rows = trace[offset:offset + count]; offset += count
        for step, row in enumerate(rows, 1):
            base.require(row.get('phase') == phase and type(row.get('step')) is int and row['step'] == step
                         and row.get('optimizer_step') == step and _finite(row.get('loss'))
                         and row.get('gradients_finite') is True
                         and type(row.get('amplitude_gradients_observed')) is int and row['amplitude_gradients_observed'] > 0
                         and _finite(row.get('gradient_norm')) and row.get('clip_max_norm') == 1., 'invalid step/gradient/Adam trace')
            base.require(row.get('gaussian_forward_delta') == (2 if phase == 'raw' else 0)
                         and row.get('gaussian_vjp_delta') == (2 if phase == 'raw' else 0), 'per-update render/VJP work differs')
            if phase == 'coefficient':
                base.require(type(row.get('sample_count')) is int and row['sample_count'] == 32768,
                             'replacement batch count differs')
    if args.stage == 'fit':
        base.require(value.get('gradients_finite') is True and value.get('phase_delta_zero') is True
                     and _hashes(value.get('field_invariants_before'), ('positions', 'log_scales', 'log_scale_mean',
                        'log_scale_std', 'amplitude_scale', 'unit_phase', 'frequencies', 'phase_delta'))
                     and value['field_invariants_before'] == value.get('field_invariants_after'), 'field geometry/source phase drift')
        metadata = value.get('field_metadata', {})
        base.require(metadata.get('encoder') == args.encoder and metadata.get('model') == 'FullAmplitudeField'
                     and metadata.get('bands') == 8 and metadata.get('width') == 64 and metadata.get('chunk_size') == 131072
                     and metadata.get('phase_delta_requires_grad') is False and metadata.get('geometry_frozen') is True
                     and metadata.get('source_phases_frozen') is True, 'full-amplitude field architecture differs')
        base.require(metadata.get('num_points') == invariants['source_definition']['num_points']
                     and value.get('amplitude_parameter_count') == (7810 if args.encoder == 'mlp' else 604290)
                     and value.get('frozen_phase_parameter_count') == 2 * metadata['num_points'],
                     'actual shared amplitude/frozen phase parameter counts differ')
        for phase, count in expected_updates.items():
            actual = value.get('optimizer_phases', {}).get(phase, {})
            base.require(actual.get('state_entries_initial') == 0 and type(actual.get('step_final')) is int
                         and actual['step_final'] == count, 'Adam reset or completed phase count differs')
        errors = value.get('raw_relative_l2', {})
        base.require(set(errors) == {'initial', 'after_coefficient', 'final'}, 'missing three raw error endpoints')
        for row in errors.values():
            base.require(isinstance(row, dict) and set(row) == {'pre', 'post'}
                         and all(_finite(v) for v in row.values()), 'nonfinite/incomplete raw errors')
        admitted = all(v <= .01 for v in errors['final'].values())
        base.require(type(value.get('representation_admitted')) is bool and value['representation_admitted'] == admitted,
                     'claimed admission disagrees with actual final errors')
        final = base.verify_descriptor(value.get('final_checkpoint'), 'final checkpoint')
        base.require(final == directory / 'final_field.pth' and value.get('checkpoint_kind') == 'final_fixed_budget',
                     'final checkpoint ownership/selection differs')
    else:
        base.require(value.get('representation_admitted') is None and value.get('final_checkpoint') is None
                     and value.get('field_metadata') is None and value.get('raw_relative_l2') == {},
                     'render control contains undeclared fit or admission')
    for key in ('training_seconds', 'peak_memory_gb', 'peak_reserved_gb'):
        base.require(_finite(value.get(key)), f'invalid actual {key}')
    precision = value.get('precision', {})
    base.require(value.get('precision_before') == precision, 'original TF32/dtype/autocast flags changed')
    base.require(precision.get('default_dtype') == 'torch.float32' and precision.get('autocast_enabled') is False
                 and type(precision.get('cuda_matmul_allow_tf32')) is bool
                 and type(precision.get('cudnn_allow_tf32')) is bool, 'original precision policy differs')
    names = ['launch.json', 'metrics.json', 'provenance.json', 'step_trace.jsonl', 'progress.json',
             'childcontext_ready.json', 'childcontext_ack.json']
    if args.stage == 'fit': names.append('final_field.pth')
    return {str(directory / name): base.sha256(directory / name) for name in names}


def validate_observed_gate(path, stage, hashes, invariants):
    gate = base.read(path)
    base.require(gate.get('schema_version') == 1 and gate.get('route') == ROUTE and gate.get('stage') == stage
                 and gate.get('status') == 'complete' and gate.get('passed') is True, 'real completed representation gate required')
    params = gate.get('parameters', {})
    base.require(isinstance(params, dict) and set(params) == set(PARAMETERS)
                 and all(gate.get(k) == v for k, v in params.items()), 'render gate parameter binding differs')
    args = argparse.Namespace(**params); validate_parameters(args)
    base.require(gate.get('code_sha256') == hashes and gate.get('invariants') == invariants, 'stale render gate source/input bindings')
    artifacts = gate.get('artifacts', {})
    base.require(artifacts and all(base.sha256(p) == h for p, h in artifacts.items()), 'render gate artifact bytes changed')
    base.require(validate_child_artifacts(args, gate['run_dir'], hashes, invariants) == artifacts, 'render gate artifact set differs')
    proof = base.read(base.verify_descriptor(gate.get('post_run_invariants'), 'render gate source proof'))
    expected = dict(code_sha256=hashes, invariants=invariants)
    controls = gate.get('control_bindings')
    base.require(proof.get('status') == 'verified' and proof.get('before') == proof.get('after') == expected,
                 'render gate before/after proof differs')
    base.require(isinstance(controls, dict) and controls and proof.get('control_bindings_before') ==
                 proof.get('control_bindings_after') == controls and all(base.sha256(p) == h for p, h in controls.items()),
                 'render gate controls changed')
    return gate


def validate_stage_gates(args, hashes, invariants):
    validate_parameters(args)
    base.require(isinstance(args.exp, str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]*', args.exp), 'invalid unique exp name')
    base.require(type(args.preferred_gpu) is int and args.preferred_gpu >= 0, 'invalid preferred GPU')
    base.require(_finite(args.timeout, minimum=0) and 0 < args.timeout <= 1800, 'maximum owned-child timeout is 1800 seconds')
    review = base.validate_review(args.review_gate, hashes, invariants)
    base.require(review.get('route') == ROUTE, 'review does not cover representation route')
    recipe = base.read(args.recipe)
    base.require(recipe.get('schema_version') == 1 and recipe.get('route') == ROUTE and recipe.get('status') == 'selected'
                 and recipe.get('parameters') == parameters(args) and recipe.get('code_sha256') == hashes
                 and recipe.get('invariants') == invariants, 'selected recipe differs from exact representation CLI/source/input')
    paths = [args.review_gate, args.recipe, Path(review['review_report']['path'])]
    result = dict(review=review, recipe=recipe)
    if args.stage == 'fit':
        base.require(args.render_gate is not None, 'fit requires actual matching render-control gate')
        result['render_control'] = validate_observed_gate(args.render_gate, 'render_control', hashes, invariants)
        paths.append(args.render_gate)
    else:
        base.require(args.render_gate is None, 'render control must not consume a foreign gate')
    result['control_bindings'] = {str(Path(p).resolve(strict=True)): base.sha256(p) for p in paths}
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument('--stage', choices=('render_control', 'fit'), required=True)
    result.add_argument('--encoder', choices=('none', 'mlp', 'grid'), required=True)
    result.add_argument('--exp', required=True)
    result.add_argument('--review-gate', type=Path, required=True)
    result.add_argument('--recipe', type=Path, required=True)
    result.add_argument('--render-gate', type=Path)
    result.add_argument('--preferred-gpu', type=int, default=0)
    result.add_argument('--timeout', type=float, default=1800)
    result.set_defaults(route=ROUTE, **FIT_CONFIG)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    with base.cancellation_signals():
        try:
            print(base.execute(args, protocol=sys.modules[__name__]), flush=True)
            return 0
        except BaseException as error:
            print(base.record_failure(args, error), file=sys.stderr, flush=True)
            return 128 + error.signum if isinstance(error, base.SupervisorCancelled) else 1


if __name__ == '__main__':
    raise SystemExit(main())
