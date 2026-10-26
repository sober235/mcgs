#!/usr/bin/env python3
"""G2 raw-render proximal DC: independent three-metric route and evidence."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import struct
import sys

import run_gpu as base

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'mcgs_ce_autoresearch_20261017'
ROUTE = 'g2_raw_proxdc_v1'
METRIC_CONTRACT = 'AF16_PSNR_SSIM_NMSE'
G2_NAMES = ('positions','scales','rotations','density_real','density_imag','density_real_contrast','density_imag_contrast')
CHILD_RUNNER = 'proxdc_runner.py'
CONTEXT_ACK = True
PARAMETERS = ('route', 'stage', 'cg_max_matvecs', 'cg_rtol', 'lambda_ratio')
PHYSICS_COUNTERS = ('rhs_adjoint_calls','normal_calls','terminal_normal_calls','objective_calls',
    'forward_calls','adjoint_calls','fft_batches','ifft_batches','coil_fft_transforms','coil_ifft_transforms',
    'gaussian_renders','gaussian_vjps')
CONSTRUCTION_COUNTERS = ('fft_batches','ifft_batches','scalar_fft_transforms','scalar_ifft_transforms',
    'coil_fft_transforms','coil_ifft_transforms')
CANONICAL_COUNTERS = ('gaussian_renders','canonical_export_calls','canonical_fft_batches',
    'canonical_ifft_batches','canonical_coil_fft_transforms','canonical_coil_ifft_transforms')
EXTERNAL = {
    OLD/'proximal_dc/physics.py': '7d85e86bce70ce2498b233acf7312dbcc9ed527038393a724f42cebb2f8fd59e',
    OLD/'proximal_dc/runner.py': 'da270d1ccddd2406bf5130723e2bfbcae0f1ea7c71b6bf0c3b147ac599c05502',
    OLD/'phase_firstorder/pcg.py': '224a1552873dd0486f3a820d986aee3e76e0376d8c2006077588788666800897',
}


def code_hashes():
    actual = {str(path): base.sha256(path) for path in EXTERNAL}
    base.require(actual == {str(p): h for p, h in EXTERNAL.items()}, 'reviewed proximal sources changed')
    return {**base.code_hashes(), **actual,
        **{str(ROOT/name): base.sha256(ROOT/name) for name in ('run_proxdc.py', CHILD_RUNNER)}}


def snapshot_invariants():
    return {**base.snapshot_invariants(), 'route': ROUTE, 'metric_contract': METRIC_CONTRACT,
            'decision_metrics': ['psnr', 'ssim', 'nmse'], 'anchor': 'G2_same_call_raw_Gaussian_render'}


def parameters(args):
    return {name: getattr(args, name) for name in PARAMETERS}


def validate_parameters(args):
    base.require(args.route == ROUTE and args.stage in ('tiny', 'canonical', 'prox'), 'invalid G2 route/stage')
    base.require(type(args.cg_max_matvecs) is int and args.cg_max_matvecs == 32,
                 'fixed32 CG matvecs including initial residual required')
    base.require(type(args.cg_rtol) in (int,float) and args.cg_rtol == 1e-5
        and type(args.lambda_ratio) in (int,float) and args.lambda_ratio == .1,
        'fixed rtol1e-5 and CSM lambda ratio0.1 required')


def validate_scan(row):
    """Retain the frozen complex64 scalar, endpoint and complete work checks."""
    record, work = row.get('record'), row.get('work')
    base.require(row.get('algorithm_executed') is True and isinstance(record,dict)
        and isinstance(work,dict), 'actual proximal scan is missing')
    def finite(key, nonnegative=False):
        value=record.get(key)
        base.require(type(value) in (int,float) and math.isfinite(value) and (not nonnegative or value >= 0),
                     f'invalid solver {key}')
        return value
    def integer(key,minimum=0):
        value=record.get(key);base.require(type(value) is int and value >= minimum,f'invalid solver {key}')
        return value
    def close(a,b):return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-30)
    L, requested, effective = [finite(k) for k in ('L','lambda_requested','lambda_effective')]
    base.require(record.get('dtype') == 'torch.complex64' and record.get('lambda_ratio') == .1
        and L > 0 and requested == .1*L and effective == struct.unpack('f',struct.pack('f',requested))[0]
        and effective > 0, 'fixed CSM-only lambda differs')
    integer('support_count',1); coils=integer('n_coils',1); chunk=integer('chunk_size',1)
    base.require(chunk <= 4 and type(record.get('initial_anchor_nonzero')) is bool,'invalid coil/anchor evidence')
    initial=integer('initial_residual_normal_evaluations');iterations=integer('cg_iterations');count=integer('cg_matvecs')
    base.require(initial == int(record['initial_anchor_nonzero']) and count == iterations+initial and count <= 32
        and record.get('cg_max_matvecs') == 32 and record.get('cg_rtol') == 1e-5, 'actual CG budget/recipe differs')
    base.require(record.get('cg_converged') is True and record.get('cg_status') == 'converged'
        and record.get('status') == ('stationary' if iterations == 0 else 'converged'), 'proximal solve did not verify')
    rhs=finite('rhs_norm',True);norm=finite('true_residual_norm',True);relative=finite('true_relative_residual',True)
    history=record.get('cg_residual_history')
    base.require(isinstance(history,list) and len(history) == iterations+1
        and all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in history)
        and history[-1] <= 1e-5*rhs, 'CG stopping/history differs')
    base.require(close(relative,norm/max(rhs,1e-30)) and relative <= 2e-5
        and record.get('true_residual_tolerance') == 2e-5 and record.get('true_endpoint_passed') is True,
        'true endpoint residual failed')
    ja,jf,da,df,p=[finite(k,True) for k in ('J_anchor','J_after','data_anchor','data_after','anchor_penalty')]
    tolerance=finite('J_check_tolerance',True);slope=finite('directional_derivative')
    base.require(record.get('J_relative_tolerance') == 2e-6 and record.get('J_absolute_tolerance') == 1e-10
        and close(tolerance,1e-10+2e-6*max(abs(ja),1e-30)) and close(ja,da) and close(jf,df+p)
        and jf <= ja+tolerance and record.get('J_check_passed') is True, 'fixed J/nonincrease differs')
    base.require((iterations == 0 and slope == 0) or (iterations > 0 and slope < 0), 'solve direction is not descent')
    base.require(record.get('outside_support_exact_zero') is True and record.get('terminal_normal_evaluations') == 1
        and record.get('training_updates') == 0, 'support/terminal/zero-training evidence differs')
    batches = (coils+chunk-1)//chunk
    expected=dict(rhs_adjoint_calls=1,normal_calls=count,terminal_normal_calls=1,objective_calls=2,
        forward_calls=0,adjoint_calls=1,fft_batches=batches*(count+3),ifft_batches=batches*(count+2),
        coil_fft_transforms=coils*(count+3),coil_ifft_transforms=coils*(count+2),gaussian_renders=0,gaussian_vjps=0)
    base.require(set(work) == set(expected)|{'roles'} and
        all(type(work[k]) is int and work[k] == v for k,v in expected.items()), 'actual solver work differs')
    roles=dict(rhs=dict(fft_batches=0,ifft_batches=batches),terminal=dict(fft_batches=batches,ifft_batches=batches),
        objective_anchor=dict(fft_batches=batches,ifft_batches=0),objective_candidate=dict(fft_batches=batches,ifft_batches=0))
    if initial: roles['initial_normal']=dict(fft_batches=batches,ifft_batches=batches)
    if iterations: roles['cg_normal']=dict(fft_batches=batches*iterations,ifft_batches=batches*iterations)
    base.require(isinstance(work['roles'],dict) and {k:v for k,v in work['roles'].items() if any(v.values())} == roles,
                 'actual FFT role work differs')


def validate_work(value, stage):
    """Check independent identities and recompute every disclosed work total."""
    def counters(actual, expected, label):
        base.require(isinstance(actual,dict) and set(actual) == set(expected)
            and all(type(actual[k]) is int and actual[k] == v for k,v in expected.items()),label)
    identities=value.get('tiny_identity',{})
    if stage == 'tiny':
        base.require(isinstance(identities,dict) and set(identities) == {'pre','post'}, 'missing tiny identities')
        for scan,identity in identities.items():
            base.require(isinstance(identity,dict), 'invalid independent tiny identity')
            for key in ('forward_adjoint_relative','normal_hermitian_relative','normal_linearity_relative'):
                number=identity.get(key)
                base.require(type(number) in (int,float) and math.isfinite(number) and 0 <= number <= 2e-5,
                    f'tiny {key} failed')
            curvature=identity.get('normal_curvature_real')
            base.require(type(curvature) in (int,float) and math.isfinite(curvature) and curvature > 0,
                         'tiny curvature must be finite and positive')
            record=value['scans'][scan]['record'];coils=record['n_coils'];chunk=record['chunk_size']
            base.require(coils == 2 and chunk == 4, 'fixed synthetic tiny coil/chunk recipe differs')
            batches=(coils+chunk-1)//chunk
            expected=dict.fromkeys(PHYSICS_COUNTERS,0)
            expected.update(normal_calls=3,forward_calls=1,adjoint_calls=1,fft_batches=4*batches,
                ifft_batches=4*batches,coil_fft_transforms=4*coils,coil_ifft_transforms=4*coils)
            work=identity.get('work')
            base.require(isinstance(work,dict), 'missing independent tiny identity work')
            counters({k:v for k,v in work.items() if k != 'roles'},expected,'independent tiny identity work differs')
            roles=dict(forward=dict(fft_batches=batches,ifft_batches=0),
                adjoint=dict(fft_batches=0,ifft_batches=batches),normal=dict(fft_batches=3*batches,ifft_batches=3*batches))
            base.require(work.get('roles') == roles,'independent tiny identity FFT roles differ')
    else:
        base.require(identities == {}, 'clinical route contains undeclared tiny identity work')
        if stage == 'prox':
            base.require(all(row['record']['n_coils'] == 42 for row in value['scans'].values()),
                         'actual acquired coil count differs')
    total=value.get('work_total')
    required=set(CANONICAL_COUNTERS)|{'gaussian_cuda_renders','gaussian_reference_renders','gaussian_vjps',
        'physics','tiny_identity_physics','construction','total_scalar_fft_transforms','total_scalar_ifft_transforms',
        'total_fft_batches','total_ifft_batches','total_coil_fft_transforms','total_coil_ifft_transforms'}
    base.require(isinstance(total,dict) and set(total) == required,'complete work total is missing')
    construction=total['construction']
    base.require(isinstance(construction,dict) and set(construction) == set(CONSTRUCTION_COUNTERS)
        and all(type(v) is int and v >= 0 for v in construction.values()), 'invalid constructor work')
    for kind in ('fft','ifft'):
        batches=construction[kind+'_batches']-construction['scalar_'+kind+'_transforms']
        coils=construction['coil_'+kind+'_transforms']
        base.require(batches >= 0 and (coils == 0 if batches == 0 else batches <= coils <= 42*batches),
                     'constructor scalar/coil work differs')
    if stage == 'tiny':
        base.require(not any(construction.values()),'synthetic tiny performed undeclared constructor work')
    canonical=dict.fromkeys(CANONICAL_COUNTERS,0)
    if stage != 'tiny':
        canonical.update(gaussian_renders=2,canonical_export_calls=2,canonical_fft_batches=2,
            canonical_ifft_batches=4,canonical_coil_fft_transforms=84,canonical_coil_ifft_transforms=168)
    counters({k:total[k] for k in CANONICAL_COUNTERS},canonical,'canonical render/export work differs')
    physics={k:sum(row['work'][k] for row in value['scans'].values()) for k in PHYSICS_COUNTERS}
    identity={k:sum(row['work'][k] for row in identities.values()) for k in PHYSICS_COUNTERS}
    counters(total['physics'],physics,'solver work aggregate differs')
    counters(total['tiny_identity_physics'],identity,'identity work aggregate differs')
    expected=dict(gaussian_cuda_renders=canonical['gaussian_renders'],gaussian_reference_renders=0,gaussian_vjps=0,
        total_scalar_fft_transforms=construction['scalar_fft_transforms'],
        total_scalar_ifft_transforms=construction['scalar_ifft_transforms'])
    for kind in ('fft','ifft'):
        expected['total_'+kind+'_batches']=construction[kind+'_batches']+canonical['canonical_'+kind+'_batches']+physics[kind+'_batches']+identity[kind+'_batches']
        expected['total_coil_'+kind+'_transforms']=construction['coil_'+kind+'_transforms']+canonical['canonical_coil_'+kind+'_transforms']+physics['coil_'+kind+'_transforms']+identity['coil_'+kind+'_transforms']
    counters({k:total[k] for k in expected},expected,'complete actual work total differs')


def validate_child_artifacts(args, directory, hashes, invariants):
    validate_parameters(args)
    directory = Path(directory).resolve(strict=True)
    launch, metrics, provenance = [base.read(directory/name) for name in
                                  ('launch.json','metrics.json','provenance.json')]
    for value in (launch,metrics,provenance):
        base.require(all(value.get(k) == v for k,v in parameters(args).items()), 'child route/parameters differ')
        base.require(value.get('code_sha256') == hashes and value.get('invariants') == invariants,
                     'child source/input bindings differ')
    base.require(launch.get('run_dir') == str(directory), 'child output directory differs')
    ready=base.read(directory/'childcontext_ready.json');ack=base.read(directory/'childcontext_ack.json')
    base.require(type(ready.get('pid')) is int and ready['pid'] > 0
        and ready.get('parent_pid') == launch.get('parent_pid') and ready.get('gpu_uuid') == launch.get('gpu_uuid')
        and ack == dict(schema_version=1,child_pid=ready['pid'],parent_pid=launch['parent_pid'],
            gpu_uuid=launch['gpu_uuid'],context_observed=True),'observed parent context acknowledgment differs')
    for value in (metrics,provenance):
        base.require(value.get('status') == 'complete' and value.get('model_schema') == ROUTE
            and value.get('metric_contract') == METRIC_CONTRACT and value.get('steps_observed') == 0
            and value.get('training_reference_access') is False and value.get('no_hard_dc_after_solve') is True,
            'child route schema/training policy differs')
        base.require(set(value.get('scans',{})) == {'pre','post'}, 'both scans required')
        if args.stage != 'tiny':
            base.require(value.get('source_iteration') == 4500 and value.get('g2_restore_performed') is True
                and value.get('fixed_model_unchanged') is True
                and value.get('anchor_source') == 'same_call_raw_Gaussian_render', 'raw G2 restore evidence differs')
            before,after=value.get('source_state_before'),value.get('source_state_after')
            fields=('parameter_sha256','parameter_identities','requires_grad','gradient_sha256','gradient_identities',
                'adam','adam_identities','optimizer_groups','scheduler_state','config_sha256','model_sha256',
                'geometry_sha256','density_sha256')
            base.require(isinstance(before,dict) and isinstance(after,dict) and before == after
                and all(k in before for k in fields) and set(before.get('parameter_sha256',{})) == set(G2_NAMES)
                and set(before.get('parameter_identities',{})) == set(G2_NAMES)
                and before.get('requires_grad') == {k:False for k in G2_NAMES}
                and value.get('forbidden_calls') == dict(trainer_train=0,optimizer_step=0,scheduler_step=0),
                'export-only model/optimizer/gradient/config state proof differs')
            evidence=value.get('data_evidence',{})
            base.require(set(evidence) == {'pre','post'} and all(
                evidence[scan].get('ground_truth_constructed') is False
                and evidence[scan].get('outside_mask_nonzero') == 0 and evidence[scan].get('coils') == 42
                for scan in ('pre','post')), 'acquired-only source evidence differs')
            for scan in ('pre','post'):
                parity = value.get('canonical_parity',{}).get(scan,{})
                relative = parity.get('complex_relative_l2')
                base.require(parity.get('passed') is True and parity.get('tolerance') == 1e-5
                    and type(relative) in (int,float) and math.isfinite(relative) and 0 <= relative <= 1e-5,
                    'fresh G2 canonical parity failed')
        else:
            base.require(value.get('g2_restore_performed') is False and value.get('source_iteration') is None
                and value.get('tiny_checks_passed') is True, 'tiny control evidence differs')
        for row in value['scans'].values():
            if args.stage == 'canonical':
                base.require(row.get('algorithm_executed') is False and row.get('record') is None,
                             'canonical control performed a proximal solve')
                work=row.get('work')
                base.require(isinstance(work,dict) and set(work) == set(PHYSICS_COUNTERS)|{'roles'} and work.get('roles') == {}
                    and all(type(v) is int and v == 0 for k,v in work.items() if k != 'roles'),
                    'canonical control has nonzero solver work')
            else:
                validate_scan(row)
        validate_work(value,args.stage)
    base.require(metrics == {k:provenance[k] for k in metrics}, 'metrics/provenance evidence disagrees')
    names = ['launch.json','metrics.json','provenance.json','progress.json','childcontext_ready.json','childcontext_ack.json']
    if args.stage != 'tiny':
        names += [f'{kind}_{scan}_complex.npy' for scan in ('pre','post') for kind in ('raw','canonical','candidate')]
        names += ['config.yaml']
    return {str(directory/name):base.sha256(directory/name) for name in names}


def validate_observed_gate(path, stage, hashes, invariants):
    gate = base.read(path)
    params = gate.get('parameters',{})
    base.require(gate.get('schema_version') == 1 and gate.get('route') == ROUTE
        and gate.get('stage') == stage and gate.get('status') == 'complete' and gate.get('passed') is True
        and isinstance(params,dict) and set(params) == set(PARAMETERS)
        and all(gate.get(k) == v for k,v in params.items()), 'real completed G2 route gate required')
    args = argparse.Namespace(**params); validate_parameters(args)
    base.require(gate.get('code_sha256') == hashes and gate.get('invariants') == invariants,
                 'route gate source/input bindings are stale')
    artifacts = gate.get('artifacts',{})
    base.require(artifacts and all(base.sha256(p) == h for p,h in artifacts.items()), 'route artifact bytes changed')
    base.require(validate_child_artifacts(args,gate['run_dir'],hashes,invariants) == artifacts,
                 'route artifact set differs')
    proof = base.read(base.verify_descriptor(gate.get('post_run_invariants'),'route source proof'))
    expected = {'code_sha256':hashes,'invariants':invariants}
    controls = gate.get('control_bindings')
    base.require(proof.get('status') == 'verified' and proof.get('before') == proof.get('after') == expected,
                 'route before/after proof differs')
    base.require(isinstance(controls,dict) and controls and proof.get('control_bindings_before') ==
        proof.get('control_bindings_after') == controls and all(base.sha256(p) == h for p,h in controls.items()),
        'route controls changed')
    return gate


def validate_stage_gates(args, hashes, invariants):
    validate_parameters(args)
    base.require(isinstance(args.exp,str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',args.exp), 'invalid exp')
    base.require(type(args.preferred_gpu) is int and args.preferred_gpu >= 0, 'invalid preferred GPU')
    base.require(type(args.timeout) in (int,float) and math.isfinite(args.timeout) and 0 < args.timeout <= 3600,
                 'invalid timeout')
    review = base.validate_review(args.review_gate,hashes,invariants)
    base.require(review.get('route') == ROUTE, 'review does not cover this G2 route')
    recipe = base.read(args.recipe)
    base.require(recipe.get('route') == ROUTE and recipe.get('schema_version') == 1
        and recipe.get('status') == 'selected' and recipe.get('parameters') == parameters(args)
        and recipe.get('code_sha256') == hashes and recipe.get('invariants') == invariants,
        'selected recipe differs from exact G2 route/CLI/source/input bindings')
    paths = [args.review_gate,args.recipe,Path(review['review_report']['path'])]
    result = {'review':review,'recipe':recipe}
    for stage, path in [('tiny',args.tiny_gate)] if args.stage == 'canonical' else (
            [('tiny',args.tiny_gate),('canonical',args.canonical_gate)] if args.stage == 'prox' else []):
        base.require(path is not None, f'{args.stage} requires an observed {stage} gate')
        result[stage] = validate_observed_gate(path,stage,hashes,invariants); paths.append(path)
    result['control_bindings'] = {str(Path(p).resolve(strict=True)):base.sha256(p) for p in paths}
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--route',choices=(ROUTE,),default=ROUTE)
    p.add_argument('--stage',choices=('tiny','canonical','prox'),required=True)
    p.add_argument('--exp',required=True)
    p.add_argument('--review-gate',type=Path,required=True); p.add_argument('--recipe',type=Path,required=True)
    p.add_argument('--tiny-gate',type=Path); p.add_argument('--canonical-gate',type=Path)
    p.add_argument('--cg-max-matvecs',type=int,default=32)
    p.add_argument('--cg-rtol',type=float,default=1e-5); p.add_argument('--lambda-ratio',type=float,default=.1)
    p.add_argument('--preferred-gpu',type=int,default=0); p.add_argument('--timeout',type=float,default=3600)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    with base.cancellation_signals():
        try:
            print(base.execute(args,protocol=sys.modules[__name__]),flush=True)
            return 0
        except BaseException as error:
            print(base.record_failure(args,error),file=sys.stderr,flush=True)
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
