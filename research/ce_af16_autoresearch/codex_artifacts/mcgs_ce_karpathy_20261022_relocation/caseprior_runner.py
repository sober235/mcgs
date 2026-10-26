#!/usr/bin/env python3
"""Original G2 child with a scoped, case-only magnitude-denoiser factory."""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

import runner as g2
import relocation_runner as original
import run_caseprior as route

save_exports = original.save_exports


def _readback_checkpoint(record):
    """Trusted child semantic readback; no extra model, randomness or fitting."""
    import torch
    from case_denoiser import _parameter_hash
    path = route.base.verify_descriptor(record.get('checkpoint'), 'trained CNN checkpoint')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    metadata = ('call_index','seed','input','output','updates','sigma','normalization','normalized_sigma',
                'initial_parameter_sha256','final_parameter_sha256','width','patch_size','batch_size','lr','tile_size')
    if not isinstance(saved, dict) or set(saved) != {'schema','model_state_dict',*metadata} \
            or saved.get('schema') != 'case_magnitude_denoiser_checkpoint_v1' \
            or any(saved.get(k) != record.get(k) for k in metadata):
        raise ValueError('saved Task1 checkpoint schema/record metadata differs')
    shapes = route._cnn_shapes(record['width'])
    tensors = saved['model_state_dict']
    if not isinstance(tensors, dict) or set(tensors) != set(shapes):
        raise ValueError('saved Task1 checkpoint tensor names differ')
    for name, shape in shapes.items():
        value = tensors[name]
        if not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.dtype != torch.float32 \
                or list(value.shape) != shape or not bool(torch.isfinite(value).all()):
            raise ValueError('saved Task1 weights must be finite float32 CPU tensors of fixed shape')
    digest = _parameter_hash(tensors)
    if digest != record['final_parameter_sha256']:
        raise ValueError('saved Task1 checkpoint final parameter digest differs')
    return dict(schema='case_denoiser_checkpoint_check_v1', call_index=record['call_index'],
                checkpoint=copy.deepcopy(record['checkpoint']), final_parameter_sha256=digest,
                metadata_equal=True, finite_float32_cpu=True, tensor_shapes=shapes)


class _RNGObservedDenoiser:
    """Keep caller-default RNG observations outside the frozen Task1 schema."""
    def __init__(self, producer):
        self.producer = producer
        self.rng_records = []
        self.checkpoint_checks = []

    @property
    def records(self):
        return self.producer.records

    @property
    def last_stats(self):
        return self.producer.last_stats

    def __call__(self, z, sigma):
        import torch
        cpu = torch.get_rng_state()
        cuda = [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
        result = self.producer(z, sigma)
        record = self.producer.last_stats
        check = _readback_checkpoint(record) if record['status'] == 'trained' else None
        cpu_after = torch.get_rng_state()
        cuda_after = [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
        def evidence(before, after):
            return dict(before_sha256=g2.tensor_hash(before), after_sha256=g2.tensor_hash(after),
                        preserved=bool(torch.equal(before, after)))
        row = dict(schema='case_denoiser_rng_v1', call_index=len(self.rng_records),
            cpu=evidence(cpu, cpu_after), cuda=[dict(device=f'cuda:{i}', **evidence(a,b))
                for i,(a,b) in enumerate(zip(cuda, cuda_after))])
        if len(cuda_after) != len(cuda) or not row['cpu']['preserved'] or any(not r['preserved'] for r in row['cuda']):
            raise ValueError('case denoiser changed caller CPU/visible CUDA default RNG')
        self.rng_records.append(row)
        if check is not None:
            self.checkpoint_checks.append(check)
        return result


def configure_prior(config, mode):
    if mode not in ('control', 'learned'):
        raise ValueError('case-prior mode must be control or learned')
    loss = config['loss']
    coefficient = .1 if mode == 'learned' else 0.
    loss.update(lambda_nl=coefficient, lambda_nl_plain=coefficient,
                lambda_nl_contrast=coefficient, nl_denoiser='case_internal_cnn_v1',
                nl_sigma_rel=.015, nl_start=4501, nl_refresh=100,
                nl_denoiser_kwargs={})
    return config


@contextmanager
def case_denoiser_factory(directory, *, device, settings=None):
    """Replace only the original named factory binding, and always unwind it.

    The trainer imports this binding inside train(), not at module load. Other
    names retain the original factory. Settings are private small CPU fixtures;
    the production driver calls this with the exact frozen Task1 defaults.
    """
    sys.path[:0] = [str(g2.TRAINER/'paired_gs'), str(g2.TRAINER)]
    import losses.nonlocal_red as red
    from case_denoiser import CaseMagnitudeDenoiser
    previous = red.make_denoiser
    instances = []

    def factory(name, *args, **kwargs):
        if name != 'case_internal_cnn_v1':
            return previous(name, *args, **kwargs)
        if kwargs or instances:
            raise ValueError('one fresh case denoiser with frozen settings required')
        denoiser = _RNGObservedDenoiser(CaseMagnitudeDenoiser(directory, device=device, **(settings or {})))
        instances.append(denoiser)
        return denoiser

    red.make_denoiser = factory
    try:
        yield instances
    finally:
        red.make_denoiser = previous


def run_case_continuation(trainer, frequency, directory, *, mode, steps,
                          rotation_matrix_fn, _denoiser_settings=None):
    """Delegate every Gaussian update/export to the unchanged original helper."""
    route.validate_parameters(SimpleNamespace(route=route.ROUTE, mode=mode, steps=steps,
        stage='zero' if steps == 0 else 'smoke' if steps == 2 else 'trial'))
    configure_prior(trainer.config, mode)
    if mode == 'learned':
        with case_denoiser_factory(Path(directory)/'denoiser', device=trainer.device,
                                   settings=_denoiser_settings) as instances:
            result = original.run_continuation(trainer, frequency, directory, mode='control',
                steps=steps, rotation_matrix_fn=rotation_matrix_fn)
        if len(instances) != 1:
            raise ValueError('original trainer did not construct the learned denoiser')
        calls = copy.deepcopy(instances[0].records)
        rng = copy.deepcopy(instances[0].rng_records)
        checks = copy.deepcopy(instances[0].checkpoint_checks)
    else:
        result = original.run_continuation(trainer, frequency, directory, mode='control',
            steps=steps, rotation_matrix_fn=rotation_matrix_fn)
        calls = []
        rng = []
        checks = []
    result.update(nonlocal_summary=copy.deepcopy(getattr(trainer, 'nlr_summary', None)),
                  denoiser_calls=calls, denoiser_rng=rng, denoiser_checkpoint_checks=checks)
    return result


def _build_config(source, directory, steps, mode):
    return configure_prior(g2.build_config(source, directory, steps), mode)


def clinical_stage(launch, directory):
    # Fixed private adapter: only g2.build_config and the continuation dispatch
    # change. The original clinical driver's N=3.6M/CUDA/acquired-only loading,
    # restore, parity, save and writer cleanup code executes unchanged.
    facade = SimpleNamespace(**{**vars(g2), 'build_config':
        lambda source, directory, steps: _build_config(source, directory, steps, launch['mode'])})
    implementation = FunctionType(original.clinical_stage.__code__,
        {**original.clinical_stage.__globals__, 'route': route, 'g2': facade,
         'run_continuation': run_case_continuation}, 'caseprior_clinical_stage')
    return implementation(launch, directory)


def validate_launch(path):
    implementation = FunctionType(original.validate_launch.__code__,
        {**original.validate_launch.__globals__, 'route': route}, 'caseprior_validate_launch')
    return implementation(path)


def publish_result(launch, directory, result, runtime, *, source_checkpoint, elapsed,
                   peak_allocated, peak_reserved):
    implementation = FunctionType(original.publish_result.__code__,
        {**original.publish_result.__globals__, 'route': route}, 'caseprior_publish_result')
    return implementation(launch, directory, result, runtime, source_checkpoint=source_checkpoint,
        elapsed=elapsed, peak_allocated=peak_allocated, peak_reserved=peak_reserved)


def execute(path):
    implementation = FunctionType(original.execute.__code__,
        {**original.execute.__globals__, 'route': route, 'validate_launch': validate_launch,
         'clinical_stage': clinical_stage, 'publish_result': publish_result}, 'caseprior_execute')
    return implementation(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--launch', type=Path, required=True)
    args = parser.parse_args()
    try:
        execute(args.launch)
    except BaseException as error:
        try:
            directory = Path(route.base.read(args.launch)['run_dir'])
            import os
            g2.write_json(directory/'numeric_failure.json',
                          dict(status='failed', error=repr(error), pid=os.getpid()))
        except Exception:
            pass
        raise
