#!/usr/bin/env python3
"""Owned numerical child: frozen G2 teacher, coefficient warm fit, full raw VJP fit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import raw_g2
import run_representation as route

FIT_CONFIG = route.FIT_CONFIG
CHANNELS = (('pre', 'plain'), ('post', 'contrast'))


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def relative_squared_error(prediction, teacher):
    """Trainer-gauge complex error with every square and reduction in float64."""
    import torch
    if prediction.shape != teacher.shape or not prediction.is_complex() or not teacher.is_complex():
        raise ValueError('raw complex teacher and prediction must have identical shapes')
    denominator = (teacher.real.double().square() + teacher.imag.double().square()).sum()
    difference = prediction - teacher
    numerator = (difference.real.double().square() + difference.imag.double().square()).sum()
    if not torch.isfinite(denominator) or denominator <= 0 or not torch.isfinite(numerator):
        raise ValueError('raw relative error has nonfinite arithmetic or zero teacher energy')
    return numerator / denominator


def _write_progress(directory, value):
    temporary = directory / 'progress.json.tmp'
    temporary.write_text(json.dumps(value, allow_nan=False) + '\n')
    temporary.replace(directory / 'progress.json')


def _field_invariants(field):
    return {**{name: tensor_hash(value) for name, value in field.named_buffers()},
            'phase_delta': tensor_hash(field.phase_delta)}


def _precision():
    import torch
    return dict(default_dtype=str(torch.get_default_dtype()), autocast_enabled=torch.is_autocast_enabled(),
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)


def _source_runtime(model, voxelizer):
    def identity(value):
        cls = type(value)
        return cls.__module__ + '.' + cls.__qualname__, getattr(sys.modules.get(cls.__module__), '__file__', None)
    model_class, model_source = identity(model)
    voxelizer_class, voxelizer_source = identity(voxelizer)
    tile = sys.modules.get('gaussian.tile_voxelizer')
    extension = getattr(tile, '_cuda_ext', None)
    return dict(model_class=model_class, model_source=model_source, voxelizer_class=voxelizer_class,
                voxelizer_source=voxelizer_source, binary=getattr(extension, '__file__', None),
                **{k: getattr(voxelizer, k, None) for k in ('backend', 'use_cuda', 'use_cupy', 'strict_cuda', 'tile_size', 'max_radius')},
                volume_shape=list(voxelizer.volume_shape) if voxelizer is not None else None,
                num_points=model.positions.shape[0], device_type=model.positions.device.type, dtype=str(model.positions.dtype))


def _adam_step(optimizer):
    steps = {int(state['step'].item()) for state in optimizer.state.values()}
    if len(steps) != 1:
        raise ValueError('Adam parameter update counts disagree')
    return steps.pop()


def run_diagnostic(model, voxelizer, directory, *, stage, encoder, config=None,
                   render=None, verify_anchor=None):
    """Fit against fresh raw renders. Internal adapters support small CPU checks.

    The supervised entry point supplies no adapters or overrides and independently
    enforces the exact fixed public recipe before this function is reached.
    """
    import torch
    from full_amplitude import FullAmplitudeField
    config = dict(FIT_CONFIG if config is None else config)
    render = raw_g2.render_density if render is None else render
    verify_anchor = raw_g2.verify_raw_anchor if verify_anchor is None else verify_anchor
    if stage not in ('render_control', 'fit') or encoder not in (('none',) if stage == 'render_control' else ('mlp', 'grid')):
        raise ValueError('invalid representation stage/encoder')
    if torch.get_default_dtype() != torch.float32 or torch.is_autocast_enabled():
        raise ValueError('original float32 and disabled autocast required')
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config['seed'])
    started = time.perf_counter()
    work = dict(gaussian_forwards=0, gaussian_vjps=0)
    updates = dict(coefficient=0, raw=0)
    progress = dict(status='started', active_phase='teacher', work=work, updates_observed=updates)
    before = raw_g2.source_hashes(model)
    precision_before = _precision()
    runtime_before = _source_runtime(model, voxelizer)
    requires_grad = {name: getattr(model, name).requires_grad for name in before}
    if any(requires_grad.values()) or any(getattr(model, name).grad is not None for name in before):
        raise ValueError('source tensors and source gradients must be frozen before fitting')
    teachers = {}; parity = {}; errors = {}; optimizers = {}
    teacher_before = None
    field = None; invariant_before = None; initial_parameters = None

    def observed_render(density, *, backward=False):
        if backward:
            if not density.requires_grad:
                raise ValueError('original density VJP is disconnected')
            def vjp(gradient):
                if not torch.isfinite(gradient).all():
                    raise ValueError('nonfinite original density VJP')
                work['gaussian_vjps'] += 1
                return gradient
            density.register_hook(vjp)
        image = render(model, voxelizer, density)
        work['gaussian_forwards'] += 1
        if not image.is_complex() or not torch.isfinite(image).all():
            raise ValueError('nonfinite or noncomplex original raw render')
        return image

    def unchanged():
        if _precision() != precision_before or _source_runtime(model, voxelizer) != runtime_before:
            raise ValueError('original runtime or TF32/dtype/autocast policy changed')
        if raw_g2.source_hashes(model) != before or any(getattr(model, n).requires_grad or
                getattr(model, n).grad is not None for n in before):
            raise ValueError('frozen source geometry/phase/coefficient tensors changed')
        if field is not None and (_field_invariants(field) != invariant_before or field.phase_delta.requires_grad
                                  or torch.count_nonzero(field.phase_delta).item()):
            raise ValueError('frozen field geometry/source phase changed')

    def raw_errors():
        with torch.no_grad():
            return {scan: math.sqrt(relative_squared_error(observed_render(field.get_density(channel)),
                                                           teachers[scan]).item()) for scan, channel in CHANNELS}

    try:
        with (directory / 'step_trace.jsonl').open('x') as trace:
            with torch.no_grad():
                for scan, channel in CHANNELS:
                    teachers[scan] = observed_render(model.get_density(channel)).detach()
                    parity[scan] = verify_anchor(teachers[scan], scan)
                    row = parity[scan]
                    if row.get('passed') is not True or type(row.get('relative_l2')) not in (int, float) \
                            or not math.isfinite(row['relative_l2']) or not 0 <= row['relative_l2'] <= config['anchor_relative_l2']:
                        raise ValueError('fresh raw anchor failed before field initialization')
            teacher_before = {scan: tensor_hash(image) for scan, image in teachers.items()}
            unchanged()
            if stage == 'fit':
                field = FullAmplitudeField(model.positions, model.scales, model.get_density('plain'),
                                           model.get_density('contrast'), encoder=encoder, bands=config['bands'],
                                           width=config['width'], chunk_size=config['chunk_size'])
                field.phase_delta.requires_grad_(False)
                parameters = field.amplitude_parameters()
                if {id(p) for p in parameters} != {id(p) for p in field.parameters() if p.requires_grad}:
                    raise ValueError('only shared amplitude parameters may update')
                invariant_before = _field_invariants(field)
                initial_parameters = [tensor_hash(p) for p in parameters]
                errors['initial'] = raw_errors()
                target = torch.stack([model.get_density(channel).abs() for _, channel in CHANNELS], dim=1).clamp_min(config['log_floor']).log()
                target_std = target.std(dim=0, unbiased=False).clamp_min(config['std_floor'])
                generator = torch.Generator(device=model.positions.device).manual_seed(config['seed'])
                for phase, count in (('coefficient', config['coefficient_updates']), ('raw', config['raw_updates'])):
                    # A new optimizer is mandatory: coefficient Adam moments never
                    # enter raw fitting, including when the parameter list matches.
                    optimizer = torch.optim.Adam(parameters, lr=config['lr'], betas=tuple(config['betas']),
                                                 eps=config['epsilon'], weight_decay=config['weight_decay'])
                    optimizers[phase] = dict(state_entries_initial=len(optimizer.state))
                    progress['active_phase'] = phase
                    for step in range(1, count + 1):
                        start_work = dict(work)
                        optimizer.zero_grad(set_to_none=True)
                        sample_count = None
                        if phase == 'coefficient':
                            indices = torch.randint(field.num_points, (config['coefficient_batch'],), device=model.positions.device, generator=generator)
                            sample_count = indices.numel()
                            loss = ((field.magnitudes(indices).clamp_min(config['log_floor']).log() - target[indices]) / target_std).square().mean()
                            if not torch.isfinite(loss): raise ValueError('nonfinite coefficient loss')
                            loss.backward()
                        else:
                            loss_value = 0.
                            for scan, channel in CHANNELS:
                                prediction = observed_render(field.get_density(channel), backward=True)
                                half_loss = .5 * relative_squared_error(prediction, teachers[scan])
                                loss_value += half_loss.item()
                                half_loss.backward()
                                del prediction, half_loss
                            loss = torch.tensor(loss_value, dtype=torch.float64)
                        gradients = [p.grad for p in parameters]
                        if any(g is None or not torch.isfinite(g).all() for g in gradients):
                            raise ValueError('missing/nonfinite shared amplitude gradients')
                        norm = torch.nn.utils.clip_grad_norm_(parameters, config['clip_max_norm'], error_if_nonfinite=True)
                        optimizer.step()
                        if any(not torch.isfinite(p).all() for p in parameters):
                            raise ValueError('nonfinite amplitude parameters after update')
                        updates[phase] += 1
                        row = dict(phase=phase, step=updates[phase], optimizer_step=_adam_step(optimizer), loss=loss.item(),
                                   gradients_finite=True, amplitude_gradients_observed=len(gradients), gradient_norm=norm.item(),
                                   clip_max_norm=config['clip_max_norm'],
                                   gaussian_forward_delta=work['gaussian_forwards'] - start_work['gaussian_forwards'],
                                   gaussian_vjp_delta=work['gaussian_vjps'] - start_work['gaussian_vjps'])
                        if sample_count is not None: row['sample_count'] = sample_count
                        if phase == 'raw' and (row['gaussian_forward_delta'] != 2 or row['gaussian_vjp_delta'] != 2):
                            raise ValueError('original two-channel Gaussian VJP did not execute')
                        trace.write(json.dumps(row, allow_nan=False) + '\n'); trace.flush()
                        _write_progress(directory, progress)
                    optimizers[phase]['step_final'] = _adam_step(optimizer)
                    unchanged()
                    if phase == 'coefficient': errors['after_coefficient'] = raw_errors()
                errors['final'] = raw_errors()
                unchanged()
                checkpoint = directory / 'final_field.pth'
                with checkpoint.open('xb') as stream:
                    torch.save(dict(checkpoint_kind='final_fixed_budget', field_state=field.state_dict(),
                                    field_metadata=field.metadata(), config=config, source_state_sha256=before,
                                    updates_observed=dict(updates)), stream)
                final = route.base.descriptor(checkpoint)
            else:
                final = None
            unchanged()
            teacher_after = {scan: tensor_hash(image) for scan, image in teachers.items()}
            if teacher_before != teacher_after:
                raise ValueError('fresh trainer-gauge raw teachers changed during fitting')
        progress.update(status='complete', active_phase='complete')
        result = dict(software_complete=True, representation_admitted=(all(v <= config['admission_relative_l2'] for v in errors['final'].values()) if field is not None else None),
                      work=dict(work), updates_observed=dict(updates), raw_anchor=parity, raw_relative_l2=errors,
                      teacher_state_before=teacher_before, teacher_state_after=teacher_after,
                      optimizer_phases=optimizers, source_state_before=before, source_state_after=raw_g2.source_hashes(model),
                      source_requires_grad=requires_grad, source_gradients_absent=all(getattr(model, n).grad is None for n in before),
                      field_invariants_before=invariant_before, field_invariants_after=_field_invariants(field) if field is not None else None,
                      phase_delta_zero=bool(torch.count_nonzero(field.phase_delta).item() == 0) if field is not None else None,
                      amplitude_parameters_changed=([tensor_hash(p) for p in parameters] != initial_parameters) if field is not None else None,
                      gradients_finite=True, field_metadata=field.metadata() if field is not None else None,
                      amplitude_parameter_count=sum(p.numel() for p in parameters) if field is not None else 0,
                      frozen_phase_parameter_count=field.phase_delta.numel() if field is not None else 0,
                      final_checkpoint=final, checkpoint_kind='final_fixed_budget' if field is not None else None,
                      training_seconds=time.perf_counter() - started,
                      peak_memory_gb=torch.cuda.max_memory_allocated() / 1024 ** 3 if model.positions.is_cuda else 0.,
                      peak_reserved_gb=torch.cuda.max_memory_reserved() / 1024 ** 3 if model.positions.is_cuda else 0.,
                      source_runtime=_source_runtime(model, voxelizer), precision_before=precision_before, precision=_precision())
        return result
    except BaseException as error:
        progress.update(status='failed', error=repr(error))
        raise
    finally:
        _write_progress(directory, progress)


def validate_launch(path):
    launch = route.base.read(path)
    route.validate_parameters(argparse.Namespace(**{k: launch.get(k) for k in route.PARAMETERS}))
    directory = Path(launch['run_dir']).resolve()
    if directory.parent != route.ROOT / 'outputs' or Path(path).resolve() != directory / 'launch.json':
        raise ValueError('child launch/output ownership differs')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != launch.get('gpu_uuid') or launch.get('parent_pid') != os.getppid():
        raise ValueError('child parent/device visibility differs')
    if route.code_hashes() != launch.get('code_sha256') or route.snapshot_invariants() != launch.get('invariants'):
        raise ValueError('reviewed source/input bindings changed before child initialization')
    recipe_path = route.base.verify_descriptor(launch.get('recipe'), 'selected representation recipe')
    recipe = route.base.read(recipe_path)
    if recipe.get('route') != route.ROUTE or recipe.get('status') != 'selected' or recipe.get('schema_version') != 1 \
            or recipe.get('parameters') != {k: launch[k] for k in route.PARAMETERS} \
            or recipe.get('code_sha256') != launch['code_sha256'] or recipe.get('invariants') != launch['invariants']:
        raise ValueError('selected recipe/launch binding differs')
    controls = launch.get('control_bindings', {})
    if not controls or any(route.base.sha256(p) != h for p, h in controls.items()):
        raise ValueError('review/control bytes changed before initialization')
    return launch, directory


def execute(path):
    if Path(sys.executable).resolve() != route.base.PYTHON.resolve():
        raise ValueError('numerical child requires actual GSTMI Python')
    launch, directory = validate_launch(path)
    import torch
    from proxdc_runner import wait_context_ack
    if torch.__version__ != '2.7.1+cu118' or torch.version.cuda != '11.8' or torch.cuda.device_count() != 1:
        raise ValueError('exact GSTMI Torch/CUDA and one reserved visible GPU required')
    hold = torch.zeros(1, device='cuda:0')
    route.base.write_new(directory / 'childcontext_ready.json', dict(pid=os.getpid(), parent_pid=os.getppid(), gpu_uuid=launch['gpu_uuid']))
    wait_context_ack(directory, launch)
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    model, voxelizer = raw_g2.construct_source('cuda:0')
    # Preserve original TF32 flags/default dtype; autocast is explicitly disabled.
    with torch.autocast(device_type='cuda', enabled=False):
        result = run_diagnostic(model, voxelizer, directory, stage=launch['stage'], encoder=launch['encoder'])
    torch.cuda.synchronize()
    if route.code_hashes() != launch['code_sha256'] or route.snapshot_invariants() != launch['invariants']:
        raise ValueError('source/raw anchor/model inputs changed after fitting')
    metrics = {**{key: launch[key] for key in route.PARAMETERS}, **result, 'status': 'complete',
               'model_schema': route.ROUTE, 'training_reference_access': False, 'formal_mri_experiments': 0,
               'code_sha256': launch['code_sha256'], 'invariants': launch['invariants']}
    route.base.write_new(directory / 'metrics.json', metrics)
    route.base.write_new(directory / 'provenance.json', metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--launch', type=Path, required=True)
    args = parser.parse_args()
    try:
        execute(args.launch)
    except BaseException as error:
        try:
            directory = Path(route.base.read(args.launch)['run_dir'])
            route.base.write_new(directory / 'numeric_failure.json', dict(status='failed', error=repr(error), pid=os.getpid()))
        except Exception:
            pass
        raise
