"""CPU contract tests and opt-in, source-bound CUDA microgate (no MRI inputs)."""
import importlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def operator():
    try:
        return importlib.import_module('carrier_operator')
    except ModuleNotFoundError:
        pytest.fail('carrier_operator is not implemented')


def inputs(k=1, dtype=torch.float32, device='cpu'):
    return [torch.zeros(k, 3, dtype=dtype, device=device),
            torch.full((k, 3), -1., dtype=dtype, device=device),
            torch.tensor([[1., .2, -.1, .3]], dtype=dtype, device=device).repeat(k, 1),
            torch.ones(k, dtype=torch.complex128 if dtype == torch.float64 else torch.complex64, device=device),
            torch.full((k, 3), .08, dtype=dtype, device=device)]


def fixture_inputs(case):
    # A length-one density view is considered contiguous regardless of stride.
    count = 0 if case == 'empty' else (2 if case in ('overlap', 'nondefault_stream', 'noncontiguous') else 1)
    return inputs(count, torch.float64)


def strided_inputs(values):
    return [torch.stack((x, x), -1)[..., 0] for x in values]


def test_noncontiguous_fixture_preserves_values_and_strides_for_all_inputs():
    reference = fixture_inputs('noncontiguous')
    values = [x.to(dtype=torch.complex64 if i == 3 else torch.float32) for i, x in enumerate(reference)]
    views = strided_inputs(values)
    assert all(not x.is_contiguous() for x in views)
    for value, view in zip(values, views):
        assert view.shape == value.shape
        assert torch.equal(value, view)
        # Strided leaf inputs must retain the same gradients through contiguous
        # conversion used by the real wrapper, including complex density.
        view.requires_grad_()
        gradient, = torch.autograd.grad(view.contiguous().real.sum(), view)
        assert torch.equal(gradient, torch.ones_like(view))


def test_import_has_no_cuda_or_build_side_effects():
    code = 'import sys,torch; import carrier_operator; assert not torch.cuda.is_initialized(); assert "torch.utils.cpp_extension" not in sys.modules'
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_explicit_load_required():
    with pytest.raises(RuntimeError, match='load_carrier_extension'):
        operator().carrier_correction(*inputs(), (3, 4, 5))


@pytest.mark.parametrize('shape,exception', [((0, 2, 3), ValueError), ((True, 2, 3), TypeError), ((2, 3), ValueError), ((2., 3, 4), TypeError)])
def test_invalid_volume_shape(shape, exception):
    with pytest.raises(exception):
        operator().carrier_correction(*inputs(), shape)


@pytest.mark.parametrize('index', range(5))
def test_rejects_wrong_dtype_before_loading(index):
    values = inputs()
    values[index] = values[index].to(torch.complex128 if index == 3 else torch.float64)
    with pytest.raises(TypeError):
        operator().carrier_correction(*values, (3, 4, 5))


@pytest.mark.parametrize('index', range(5))
def test_rejects_shape_and_nonfinite_before_loading(index):
    values = inputs()
    values[index] = values[index].repeat(2, *([1] * (values[index].ndim - 1)))
    with pytest.raises(ValueError):
        operator().carrier_correction(*values, (3, 4, 5))
    values = inputs()
    values[index].flatten()[0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        operator().carrier_correction(*values, (3, 4, 5))


@pytest.mark.parametrize('frequency', [-.5, .5, float('inf')])
def test_rejects_frequency_bounds_before_loading(frequency):
    values = inputs()
    values[-1][0, 0] = frequency
    with pytest.raises(ValueError):
        operator().carrier_correction(*values, (3, 4, 5))


def test_loader_rejects_historical_and_unowned_destinations(tmp_path):
    for path in (ROOT, ROOT / 'carrier_cuda', tmp_path / 'build'):
        with pytest.raises(ValueError, match='carrier_builds'):
            operator().load_carrier_extension(path)
    assert not (tmp_path / 'build').exists()


def test_geometry_and_tiles_cover_supported_voxels():
    op = operator()
    values = inputs(2)
    values[0][1] = torch.tensor([.7, -.6, .2])
    values[1][1] = torch.tensor([-2., -.5, -1.])
    shape = (9, 5, 11)
    c, m = op._geometry(*values[:3], shape)
    ranges, ids = op._tile_pairs(c, m, shape)
    assert ranges.dtype == ids.dtype == torch.int32
    assert ranges.shape == (4, 2)
    assert not ranges.requires_grad and not ids.requires_grad
    for d in range(shape[0]):
        for h in range(shape[1]):
            for w in range(shape[2]):
                delta = (torch.tensor([d, h, w]) - c) / (torch.tensor(shape) / 2)
                supported = ((delta.unsqueeze(1) @ m @ delta.unsqueeze(-1)).flatten() <= 9).nonzero().flatten().tolist()
                tile = (d // 8) * 2 + w // 8
                start, end = ranges[tile].tolist()
                assert set(supported).issubset(set(ids[start:end].tolist()))


def test_degenerate_normalized_rotation_retains_exact_support():
    # q/(norm+eps) can make R singular even for finite nonzero q. Historical
    # R diag(s^2) R^T bounds would lose this supported distant voxel.
    op = operator()
    values = inputs()
    shape = (17, 9, 11)
    values[0][0] = (torch.tensor([4., 4., 4.])+.5)/(torch.tensor(shape)/2)-1
    values[1].fill_(-4)
    values[2][0] = torch.tensor([0., 1e-8/(2**.5-1), 0., 0.])
    c, m = op._geometry(*values[:3], shape)
    target = torch.tensor([4., 8., 10.])
    d = (target-c[0])/(torch.tensor(shape)/2)
    assert (d @ m[0] @ d).item() < 9
    ranges, ids = op._tile_pairs(c, m, shape)
    start, end = ranges[3].tolist()  # tile (0,1,1)
    assert ids[start:end].tolist() == [0]
    # Every tile contains the atom in this unbounded-support case.
    assert torch.equal(ranges[:, 1]-ranges[:, 0], torch.ones(12, dtype=torch.int32))


def test_empty_tile_membership_has_no_pairs():
    op = operator()
    values = inputs(0)
    c, m = op._geometry(*values[:3], (9, 5, 11))
    ranges, ids = op._tile_pairs(c, m, (9, 5, 11))
    assert ranges.shape == (4, 2) and torch.count_nonzero(ranges) == 0
    assert ids.shape == (0,)


def test_geometry_is_symmetric_at_cuda_boundary():
    values = inputs()
    values[1][:] = torch.tensor([-.7, -1.1, -.9])
    _, m = operator()._geometry(*values[:3], (3, 4, 5))
    assert torch.equal(m, m.transpose(-1, -2))


@pytest.mark.parametrize('bad_intermediate', ['exp_scale', 'rotation_norm', 'centers'])
def test_nonfinite_geometry_intermediates_refused(bad_intermediate):
    values = inputs()
    if bad_intermediate == 'exp_scale':
        values[1].fill_(100)
    elif bad_intermediate == 'rotation_norm':
        values[2].fill_(1e30)
    else:
        values[0].fill_(torch.finfo(torch.float32).max)
    assert all(torch.isfinite(t).all() for t in values)
    with pytest.raises(ValueError, match='derived'):
        operator()._geometry(*values[:3], (3, 4, 5))


def test_finite_geometry_zero_local_vjp_preserves_exact_zero_gradients():
    values = inputs()
    geometry_inputs = [v.requires_grad_() for v in values[:3]]
    c, m = operator()._geometry(*geometry_inputs, (3, 4, 5))
    gradients = torch.autograd.grad((c, m), geometry_inputs, (torch.zeros_like(c), torch.zeros_like(m)))
    assert all(torch.isfinite(g).all() and torch.count_nonzero(g) == 0 for g in gradients)


def test_loader_requires_pinned_toolchain_before_importing_builder(monkeypatch):
    monkeypatch.delenv('CUDA_HOME', raising=False)
    with pytest.raises(RuntimeError, match='CUDA_HOME'):
        operator().load_carrier_extension(ROOT / 'carrier_builds/contract-test-unused')
    assert not (ROOT / 'carrier_builds/contract-test-unused').exists()


# Reject selector/flag/include/linker overrides at the loader boundary before
# invoking tools, importing the builder, creating directories, or cache reuse.
BUILD_OVERRIDE_VARIABLES = [
    'PYTORCH_NVCC', 'CC', 'NVCC_CCBIN', 'NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS',
    'CUDACXX', 'CUDAHOSTCXX', 'CUDA_PATH', 'CUDA_INC_PATH', 'CUDNN_HOME', 'CUDNN_PATH',
    'ROCM_HOME', 'ROCM_PATH', 'CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'LDFLAGS',
    'GCC_EXEC_PREFIX', 'COMPILER_PATH', 'CPATH', 'CPLUS_INCLUDE_PATH',
    'C_INCLUDE_PATH', 'OBJC_INCLUDE_PATH', 'LIBRARY_PATH', 'LD_PRELOAD',
    'LD_LIBRARY_PATH', 'TORCH_DONT_CHECK_COMPILER_ABI',
    'TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES', 'TORCH_EXTENSIONS_DIR',
]


def build_environment(monkeypatch):
    for variable in BUILD_OVERRIDE_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv('CUDA_HOME', '/usr/local/cuda-11.8')
    monkeypatch.setenv('CXX', '/usr/bin/c++')
    monkeypatch.setenv('PATH', '/usr/local/cuda-11.8/bin:/usr/bin:/bin')
    monkeypatch.setenv('TORCH_CUDA_ARCH_LIST', '8.6')


def refuse_tool_invocation(*args, **kwargs):
    pytest.fail('invalid environment reached tool invocation instead of public loader refusal')


@pytest.mark.parametrize('variable', BUILD_OVERRIDE_VARIABLES)
def test_loader_refuses_unbound_toolchain_overrides(monkeypatch, variable):
    build_environment(monkeypatch)
    monkeypatch.setenv(variable, 'unreviewed-override')
    op = operator()
    # Guard the external tool boundary so a regressed refusal cannot accidentally
    # proceed to compilation during a CPU-only contract run.
    monkeypatch.setattr(op.subprocess, 'check_output', refuse_tool_invocation)
    with pytest.raises(RuntimeError, match=variable):
        op.load_carrier_extension(ROOT / 'carrier_builds/round1-refusal-unused')
    assert not (ROOT / 'carrier_builds/round1-refusal-unused').exists()


@pytest.mark.parametrize('variable,value', [('CXX', None), ('CXX', '/tmp/c++'), ('PATH', '/tmp:/usr/bin')])
def test_loader_refuses_unpinned_effective_executables(monkeypatch, variable, value):
    build_environment(monkeypatch)
    if value is None:
        monkeypatch.delenv(variable)
    else:
        monkeypatch.setenv(variable, value)
    op = operator()
    monkeypatch.setattr(op.subprocess, 'check_output', refuse_tool_invocation)
    with pytest.raises(RuntimeError, match=variable):
        op.load_carrier_extension(ROOT / 'carrier_builds/round1-refusal-unused')


def test_loader_effective_toolchain_identity_changes_with_architecture(monkeypatch):
    build_environment(monkeypatch)
    validate_environment = getattr(operator(), '_validated_build_environment', None)
    assert callable(validate_environment), 'loader must bind effective build environment'
    first = validate_environment()
    assert first['CXX'] == '/usr/bin/c++'
    assert first['CUDA_HOME'] == '/usr/local/cuda-11.8'
    assert first['cuda_host_compiler'] == '/usr/bin/gcc'
    monkeypatch.setenv('TORCH_CUDA_ARCH_LIST', '8.9')
    second = validate_environment()
    assert first != second and first['TORCH_CUDA_ARCH_LIST'] == '8.6'


GPU_CASES = ['ragged', 'single', 'overlap', 'outside', 'empty', 'noncontiguous',
             'zero', 'quarter_d', 'quarter_h', 'quarter_w', 'tiny_phase', 'near_bound', 'degenerate', 'nondefault_stream']
GPU_ENABLED = os.environ.get('CARRIER_RUN_GPU') == '1'


def test_gpu_report_refuses_missing_stream_case():
    # A complete report of the original default-stream cases must not pass.
    report = {'fixtures': {case: {'passed': True} for case in [
        'ragged', 'single', 'overlap', 'outside', 'empty', 'noncontiguous',
        'zero', 'quarter_d', 'quarter_h', 'quarter_w', 'tiny_phase', 'near_bound', 'degenerate']},
        'invalid_inputs': True, 'historical_binary_sha256_before': 'unchanged',
        'historical_binary_sha256_after': 'unchanged'}
    complete = globals().get('_gpu_report_complete')
    assert callable(complete), 'microgate needs a CPU-checkable completeness boundary'
    assert not complete(report)
    report['fixtures']['nondefault_stream'] = {'passed': True}
    assert complete(report)


@pytest.mark.parametrize('shape,expected', [((1, 1, 2147483647), (1, 1, 268435456)),
                                         ((1, 2147483647, 1), (1, 268435456, 1)),
                                         ((2147483647, 1, 1), (268435456, 1, 1)),
                                         ((9, 5, 11), (2, 1, 2))])
def test_accepted_shape_tile_arithmetic_without_volume_allocation(shape, expected):
    op = operator()
    assert op._validate(inputs(), shape) == shape
    tile_shape = getattr(op, '_tile_shape', None)
    assert callable(tile_shape), 'validated shape needs overflow-safe shared tile arithmetic'
    assert tile_shape(shape) == expected
    # Last tile may include a padded coordinate equal to INT32_MAX, never more.
    assert all((n-1)*8+7 <= 2147483647 for n in expected)


def _gpu_report_complete(report):
    return (all(report['fixtures'].get(case, {}).get('passed') for case in GPU_CASES)
            and report['invalid_inputs']
            and report['historical_binary_sha256_before'] == report['historical_binary_sha256_after'])


@pytest.fixture(scope='module')
def gpu_gate():
    if not GPU_ENABLED:
        pytest.skip('CUDA numerical gate requires reviewed source and CARRIER_RUN_GPU=1')
    assert torch.cuda.is_available(), 'GPU gate requested without an available GPU'
    path = Path(os.environ['CARRIER_REPORT_PATH']).resolve()
    assert not path.exists(), 'microgate report must have a unique unused path'
    historical = Path(os.environ['CARRIER_HISTORICAL_BINARY']).resolve()
    historical_hash = hashlib.sha256(historical.read_bytes()).hexdigest()
    assert historical_hash == '3973daeef5f55ebfed931c38538a37d83921a9287b96b69a431da238057194b5'
    module, provenance = operator().load_carrier_extension(os.environ['CARRIER_BUILD_DIRECTORY'])
    report = {'provenance': provenance, 'fixtures': {}, 'invalid_inputs': False,
              'historical_binary': str(historical), 'historical_binary_sha256_before': historical_hash,
              'gpu': {'name': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(),
                      'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')},
              'test_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'tolerances': {'rtol': 2e-4, 'atol': 2e-5}, 'status': 'incomplete'}
    try:
        yield module, report
    finally:
        report['historical_binary_sha256_after'] = hashlib.sha256(historical.read_bytes()).hexdigest()
        report['status'] = 'passed' if _gpu_report_complete(report) else 'failed'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
        assert report['status'] == 'passed', f'incomplete/failed GPU gate; see {path}'


def error_stats(actual, expected):
    diff = (actual - expected).abs()
    return {'maxabs': diff.max().item() if diff.numel() else 0.,
            'maxrelative': (diff / expected.abs().clamp_min(2e-5)).max().item() if diff.numel() else 0.}


@pytest.mark.parametrize('case', GPU_CASES)
def test_gpu_oracle_and_all_vjps(gpu_gate, case):
    from carrier_reference import carrier_correction_reference
    _, report = gpu_gate
    shape = (9, 5, 11) if case in ('ragged', 'noncontiguous', 'nondefault_stream') else ((17, 9, 11) if case == 'degenerate' else (3, 4, 5))
    values = fixture_inputs(case)
    if values[0].numel():
        values[0][:] = torch.tensor([-.12, .17, -.23])
        values[1][:] = torch.tensor([-.7, -1.1, -.9])
        values[3][:] = complex(.8, -.35)
    if case in ('overlap', 'nondefault_stream'):
        values[3][1] = complex(-.4, .6)
        values[2][1] *= 2.7
    if case == 'nondefault_stream':
        return check_nondefault_stream(gpu_gate, values, shape)
    if case == 'outside':
        values[0][:] = 100
    if case == 'zero':
        values[-1].zero_()
        values[3].fill_(1)
    if case.startswith('quarter_'):
        values[-1].zero_()
        values[-1][:, {'d': 0, 'h': 1, 'w': 2}[case[-1]]] = .25
    if case == 'tiny_phase':
        values[-1].fill_(1e-7)
        values[3].fill_(1)
    if case == 'near_bound':
        values[-1][:] = torch.tensor([.499, -.499, .49])
    if case == 'degenerate':
        values[0][0] = (torch.tensor([4., 4., 4.])+.5)/(torch.tensor(shape)/2)-1
        values[1].fill_(-4)
        values[2][0] = torch.tensor([0., 1e-8/(2**.5-1), 0., 0.])
        # Pathological parameter conditioning is a support check, not a claim
        # of float32/double parity for the nearly singular quaternion chain.
        # Compare local CUDA VJPs against the same rounded float32 geometry.
        return check_degenerate_geometry(gpu_gate, values, shape)
    cpu = [x.requires_grad_() for x in values]
    cuda = [x.detach().to(device='cuda', dtype=torch.complex64 if i == 3 else torch.float32) for i, x in enumerate(cpu)]
    if case == 'noncontiguous':
        cuda = strided_inputs(cuda)
        assert all(not x.is_contiguous() for x in cuda)
    cuda = [x.requires_grad_() for x in cuda]
    expected = carrier_correction_reference(*cpu, shape)
    actual = operator().carrier_correction(*cuda, shape)
    n = expected.numel()
    upstream = torch.complex(torch.linspace(-.6, .8, n), torch.linspace(.9, -.2, n)).reshape(shape)
    if case == 'zero':
        upstream.zero_()
        upstream[1, 1, 2] = 1j
    cpu_grads = torch.autograd.grad(expected, cpu, upstream.to(torch.complex128))
    repeats = [torch.autograd.grad(actual, cuda, upstream.cuda(), retain_graph=True) for _ in range(3)]
    observed = [g.cpu().to(r.dtype) for g, r in zip(repeats[0], cpu_grads)]
    record = {'output': error_stats(actual.cpu().to(expected.dtype), expected),
              'vjps': {name: error_stats(g, r) for name, g, r in zip(['positions', 'log_scales', 'rotations', 'density', 'frequencies'], observed, cpu_grads)},
              'atomic_vjp_maxvariation': max(((a - b).abs().max().item() if a.numel() else 0.) for repeat in repeats[1:] for a, b in zip(repeats[0], repeat)), 'passed': False}
    report['fixtures'][case] = record
    torch.testing.assert_close(actual.cpu().to(expected.dtype), expected, rtol=2e-4, atol=2e-5)
    for g, ref in zip(observed, cpu_grads):
        torch.testing.assert_close(g, ref, rtol=2e-4, atol=2e-5)
    if case == 'zero':
        assert torch.count_nonzero(actual) == 0
        assert all(torch.count_nonzero(g) == 0 for g in observed[:4])
        assert torch.isfinite(observed[-1]).all() and torch.count_nonzero(observed[-1]) > 0
    if case == 'tiny_phase':
        assert torch.count_nonzero(actual.real) > 0, 'stable cos(theta)-1 must retain the quadratic real term'
    record['passed'] = True


def check_nondefault_stream(gpu_gate, values, shape):
    """Capture side-stream results on pinned host memory before global sync.

    Producer fills leaf inputs; operator stream waits on that producer, runs
    forward and all VJPs; consumer waits on operator completion and captures
    outputs and VJPs. A delayed default stream makes accidental default-stream
    launches observable. Only the consumer event is synchronized before reading
    the host snapshots. Global synchronization is cleanup after capture only.
    """
    from carrier_reference import carrier_correction_reference
    assert os.environ.get('CUDA_LAUNCH_BLOCKING', '0') == '0', 'stream gate requires CUDA_LAUNCH_BLOCKING=0'
    _, report = gpu_gate
    cpu = [v.requires_grad_() for v in values]
    reference = carrier_correction_reference(*cpu, shape)
    factor = complex(.75, -.25)
    upstream = torch.complex(torch.linspace(-.6, .8, reference.numel()),
                             torch.linspace(.9, -.2, reference.numel())).reshape(shape)
    ref_grads = torch.autograd.grad(reference*factor, cpu, upstream.to(torch.complex128))
    source = [v.detach().to('cuda', dtype=torch.complex64 if i == 3 else torch.float32) for i, v in enumerate(cpu)]
    leaves = [torch.zeros_like(v).requires_grad_() for v in source]
    upstream_cuda = upstream.cuda()
    # Allocate pinned snapshots before queuing stream delays, so their eventual
    # contents are fixed by the consumer's asynchronous copies.
    host_output = torch.empty(shape, dtype=torch.complex64, pin_memory=True)
    host_consumed = torch.empty_like(host_output, pin_memory=True)
    host_grads = [[torch.empty(v.shape, dtype=v.dtype, pin_memory=True) for v in source] for _ in range(3)]
    producer, operator_stream, consumer = (torch.cuda.Stream() for _ in range(3))
    produced, operated, captured, default_delay_done = (torch.cuda.Event() for _ in range(4))
    default = torch.cuda.default_stream()
    # Initialize linalg/autograd and populate this stream's allocator cache
    # before the tested sequence; cudaMalloc initialization must not drain the
    # intentionally delayed default stream during the actual ordering check.
    operator_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(operator_stream):
        warm_inputs = [v.detach().clone().requires_grad_() for v in source]
        warm_output = operator().carrier_correction(*warm_inputs, shape)*factor
        for _ in range(3):
            warm_gradients = torch.autograd.grad(warm_output, warm_inputs, upstream_cuda, retain_graph=True)
        del warm_gradients, warm_output, warm_inputs
    torch.cuda.synchronize()
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer), torch.no_grad():
        torch.cuda._sleep(2_000_000)
        for leaf, src in zip(leaves, source):
            leaf.copy_(src)
        produced.record()
    # Do not synchronize this stream before captured.synchronize(). A wrongly
    # launched kernel should remain behind this delay while consumer snapshots
    # are taken, rather than be rescued by a later default-stream .cpu() copy.
    with torch.cuda.stream(default):
        torch.cuda._sleep(2_000_000_000)
        default_delay_done.record()
    actual = consumed = repetitions = None
    try:
        with torch.cuda.stream(operator_stream):
            operator_stream.wait_event(produced)
            actual = operator().carrier_correction(*leaves, shape)
            consumed = actual*factor
            repetitions = [torch.autograd.grad(consumed, leaves, upstream_cuda, retain_graph=True) for _ in range(3)]
            operated.record()
        with torch.cuda.stream(consumer), torch.no_grad():
            consumer.wait_event(operated)
            host_output.copy_(actual, non_blocking=True)
            host_consumed.copy_(consumed, non_blocking=True)
            for destination, gradients in zip(host_grads, repetitions):
                for snapshot, gradient in zip(destination, gradients):
                    snapshot.copy_(gradient, non_blocking=True)
            captured.record()
        captured.synchronize()
        default_finished_at_capture = default_delay_done.query()
        observed = [g.to(r.dtype) for g, r in zip(host_grads[0], ref_grads)]
        record = {'output': error_stats(host_output.to(reference.dtype), reference),
                  'consumer_output': error_stats(host_consumed.to(reference.dtype), reference*factor),
                  'vjps': {name: error_stats(g, r) for name, g, r in zip(
                      ['positions', 'log_scales', 'rotations', 'density', 'frequencies'], observed, ref_grads)},
                  'atomic_vjp_maxvariation': max((a-b).abs().max().item() for repeat in host_grads[1:] for a, b in zip(host_grads[0], repeat)),
                  'default_delay_completed_at_capture': default_finished_at_capture,
                  'cuda_launch_blocking': os.environ.get('CUDA_LAUNCH_BLOCKING', '0'),
                  'ordering': 'producer event -> operator forward/backward event -> consumer pinned-host snapshots; consumer event synchronized before any global sync',
                  'passed': False}
        report['fixtures']['nondefault_stream'] = record
        assert not default_finished_at_capture, 'default-stream delay expired before consumer capture; stream ordering was not isolated'
        torch.testing.assert_close(host_output.to(reference.dtype), reference, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(host_consumed.to(reference.dtype), reference*factor, rtol=2e-4, atol=2e-5)
        for gradient, expected in zip(observed, ref_grads):
            torch.testing.assert_close(gradient, expected, rtol=2e-4, atol=2e-5)
        record['passed'] = True
    finally:
        # Snapshots are already host values; cleanup cannot change them.
        torch.cuda.synchronize()


def check_degenerate_geometry(gpu_gate, values, shape):
    from carrier_reference import _correction_from_geometry, carrier_correction_reference
    _, report = gpu_gate
    op = operator()
    cuda = [v.to('cuda', dtype=torch.complex64 if i == 3 else torch.float32) for i, v in enumerate(values)]
    public_output = op.carrier_correction(*cuda, shape)
    public_reference = carrier_correction_reference(*values, shape)
    c, m = op._geometry(*cuda[:3], shape)
    local = [t.detach().requires_grad_() for t in (c, m, cuda[3], cuda[4])]
    oracle = [t.detach().cpu().to(torch.complex128 if i == 2 else torch.float64).requires_grad_() for i, t in enumerate(local)]
    ranges, ids = op._tile_pairs(c, m, shape)
    actual = op._Carrier.apply(*local, ranges, ids, shape, gpu_gate[0])
    expected = _correction_from_geometry(*oracle, shape)
    upstream = torch.complex(torch.linspace(-.6, .8, expected.numel()), torch.linspace(.9, -.2, expected.numel())).reshape(shape)
    reference_grads = torch.autograd.grad(expected, oracle, upstream.to(torch.complex128))
    repeats = [torch.autograd.grad(actual, local, upstream.cuda(), retain_graph=True) for _ in range(3)]
    observed = [g.cpu().to(r.dtype) for g, r in zip(repeats[0], reference_grads)]
    record = {'scope': 'public output/support and local VJPs with identical rounded M; ill-conditioned quaternion chain excluded',
              'output': error_stats(public_output.cpu().to(torch.complex128), public_reference),
              'vjps': {name: error_stats(g, r) for name, g, r in zip(['centers', 'M', 'density', 'frequencies'], observed, reference_grads)},
              'atomic_vjp_maxvariation': max((a-b).abs().max().item() for repeat in repeats[1:] for a, b in zip(repeats[0], repeat)), 'passed': False}
    report['fixtures']['degenerate'] = record
    torch.testing.assert_close(public_output.cpu().to(torch.complex128), public_reference, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(actual.cpu().to(torch.complex128), expected, rtol=2e-4, atol=2e-5)
    for g, r in zip(observed, reference_grads):
        torch.testing.assert_close(g, r, rtol=2e-4, atol=2e-5)
    record['passed'] = True


def test_gpu_invalid_inputs_and_raw_tile_indices(gpu_gate):
    module, report = gpu_gate
    values = inputs(device='cuda')
    for i in range(5):
        bad = [x.clone() for x in values]
        bad[i].flatten()[0] = float('nan')
        with pytest.raises(ValueError):
            operator().carrier_correction(*bad, (3, 4, 5))
    bad = list(values)
    bad[0] = bad[0].cpu()
    with pytest.raises(ValueError):
        operator().carrier_correction(*bad, (3, 4, 5))
    c, m = operator()._geometry(*values[:3], (3, 4, 5))
    ranges, ids = operator()._tile_pairs(c, m, (3, 4, 5))
    for corrupted_ranges, corrupted_ids in [(ranges, ids + 5), (ranges - 1, ids), (ranges + 10, ids), (ranges.flip(-1).contiguous(), ids), (ranges, torch.cat((ids, ids)))]:
        with pytest.raises(RuntimeError):
            module.forward(c, m, values[3], values[4], corrupted_ranges, corrupted_ids, [3, 4, 5])
    for raw in [(c.double(), m, values[3], values[4], ranges, ids),
                (c, m, values[3], values[4], ranges.long(), ids),
                (c, m, values[3], values[4], ranges, ids.unsqueeze(0)),
                (c, m[:, :2].contiguous(), values[3], values[4], ranges, ids)]:
        with pytest.raises(RuntimeError):
            module.forward(*raw, [3, 4, 5])
    report['invalid_inputs'] = True
