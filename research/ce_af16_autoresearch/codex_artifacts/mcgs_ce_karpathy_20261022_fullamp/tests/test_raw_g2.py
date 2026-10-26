"""CPU contract checks; CUDA doubles do not establish kernel or clinical parity."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def api():
    assert (ROOT / 'raw_g2.py').exists(), 'raw G2 renderer is missing'
    return importlib.import_module('raw_g2')


def test_definition_is_parent_safe_and_binds_exact_frozen_inputs():
    module = api()
    script = "import sys,json,raw_g2; d=raw_g2.source_definition(); assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules; print(json.dumps(d))"
    result = subprocess.run([sys.executable, '-B', '-c', script], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    d = json.loads(result.stdout)
    assert d['volume_shape'] == [256, 232, 176]
    assert d['num_points'] == 3600000
    assert d['gauges'] == {'pre': 0.9997583823636317, 'post': 1.0003600786949414}
    assert d['source_names'] == ['positions', 'scales', 'rotations', 'density_real',
                               'density_imag', 'density_real_contrast', 'density_imag_contrast']
    assert d['expected_sha256'][d['checkpoint']] == '282e0303c46a6c18775d6224c032611072bd465b4ef975ba7a7e2ea636230fd0'
    assert d['expected_sha256'][d['binary']] == '3973daeef5f55ebfed931c38538a37d83921a9287b96b69a431da238057194b5'
    assert Path(d['binary']).name == 'tile_voxelize_cuda_0141c525_7aedd3b0.so'
    assert all(Path(p).is_absolute() for p in d['expected_sha256'])
    # Each call returns a private definition; callers cannot mutate the binding.
    d['gauges']['pre'] = 2
    assert module.source_definition()['gauges']['pre'] == 0.9997583823636317


def state_fixture():
    return {'positions': torch.tensor([[.1, .2, .3], [-.2, .7, -.4]]),
            'scales': torch.tensor([[-8.123456, -4.99999, -3.14159], [-9.32, -1.2, -6.789]]),
            'rotations': torch.tensor([[.9, .1, .2, .3], [.8, -.2, .4, .1]]),
            'density_real': torch.tensor([1., 2.]), 'density_imag': torch.tensor([3., 4.]),
            'density_real_contrast': torch.tensor([5., 6.]),
            'density_imag_contrast': torch.tensor([7., 8.])}


@pytest.fixture
def construction(monkeypatch, tmp_path):
    m = api()
    d = m.source_definition()
    state = state_fixture()
    checkpoint = tmp_path / 'latest.pth'
    torch.save({'iter': 4500, 'frequency_adam_recipe': {'frequency_t': 1., 'geometry_factor': 2.},
                'model_state': state}, checkpoint)
    d.update(checkpoint=str(checkpoint), num_points=2, volume_shape=[1, 1, 2])
    d['state_shapes'] = {k: list(v.shape) for k, v in state.items()}
    d['expected_sha256'] = {str(checkpoint): hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    monkeypatch.setattr(m, 'source_definition', lambda: d)
    # The only device substitution: original model construction/math stay real CPU.
    monkeypatch.setattr(m, '_cuda_device', lambda device: torch.device('cpu'))
    modules = m._original_modules()
    tile = modules['gaussian.tile_voxelizer']
    ext = SimpleNamespace(__file__=d['binary'], forward=lambda *a: None, backward=lambda *a: None)
    monkeypatch.setattr(m, '_load_extension', lambda path: ext)
    monkeypatch.setattr(tile, '_cuda_ext', None)
    monkeypatch.setattr(tile, '_cuda_load_error', None)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    return m, d, state, modules, ext


def test_constructor_uses_exp_scales_then_restores_all_seven_original_tensors(construction, monkeypatch):
    m, d, state, modules, ext = construction
    cls = modules['gaussian.joint_gaussian_model'].JointGaussianModel3D
    original = cls.__init__
    supplied = {}
    def capture(self, *args, **kwargs):
        supplied.update(kwargs)
        original(self, *args, **kwargs)
    monkeypatch.setattr(cls, '__init__', capture)
    model, voxelizer = m.construct_source('cuda:0')
    assert torch.equal(supplied['initial_scales'], torch.exp(state['scales']))
    assert supplied['num_points'] == 2
    assert set(model.state_dict()) == set(state)
    for key, value in state.items():
        assert torch.equal(model.state_dict()[key], value), key
        assert not getattr(model, key).requires_grad
    assert voxelizer.volume_shape == (1, 1, 2)
    assert voxelizer.tile_size == 8 and voxelizer.max_radius == 20
    assert voxelizer.use_cuda and voxelizer.strict_cuda and not voxelizer.use_cupy
    assert voxelizer.backend == 'tile_cuda'
    assert modules['gaussian.tile_voxelizer']._cuda_ext is ext
    before = m.source_hashes(model)
    assert before == {k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k, v in state.items()}
    with torch.no_grad():
        model.rotations[0, 0] += 1
    assert m.source_hashes(model)['rotations'] != before['rotations']


@pytest.mark.parametrize('change', ['extra', 'missing', 'shape', 'dtype', 'nonfinite', 'recipe'])
def test_construct_refuses_malformed_state_before_render(construction, change):
    m, d, state, modules, ext = construction
    checkpoint = torch.load(d['checkpoint'], weights_only=True)
    if change == 'extra': checkpoint['model_state']['extra'] = torch.ones(2)
    if change == 'missing': del checkpoint['model_state']['rotations']
    if change == 'shape': checkpoint['model_state']['scales'] = torch.ones(2)
    if change == 'dtype': checkpoint['model_state']['scales'] = state['scales'].double()
    if change == 'nonfinite': checkpoint['model_state']['scales'][0, 0] = float('nan')
    if change == 'recipe': checkpoint['frequency_adam_recipe']['geometry_factor'] = 1.
    torch.save(checkpoint, d['checkpoint'])
    d['expected_sha256'][d['checkpoint']] = hashlib.sha256(Path(d['checkpoint']).read_bytes()).hexdigest()
    with pytest.raises(ValueError): m.construct_source('cuda:0')


def test_construct_refuses_changed_checkpoint_bytes(construction):
    m, d, *_ = construction
    with Path(d['checkpoint']).open('ab') as handle: handle.write(b'changed')
    with pytest.raises(ValueError, match='SHA256'): m.construct_source('cuda:0')


@pytest.mark.parametrize('kind', ['source', 'binary'])
def test_input_binding_refuses_changed_source_or_binary(tmp_path, monkeypatch, kind):
    m = api()
    path = tmp_path / ('original.py' if kind == 'source' else 'backend.so')
    path.write_bytes(b'original')
    definition = {'expected_sha256': {str(path): hashlib.sha256(b'original').hexdigest()}}
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA256'): m._verify_files(definition)


@pytest.mark.parametrize('change', ['fallback', 'cupy', 'non_strict', 'extension', 'source_identity', 'tile_size', 'volume_shape'])
def test_render_refuses_backend_or_source_substitution(construction, monkeypatch, change):
    m, d, state, modules, ext = construction
    model, voxelizer = m.construct_source('cuda:0')
    monkeypatch.setattr(type(voxelizer), 'forward', lambda self, p, s, r, density: (density.real.reshape(1, 1, 2), density.imag.reshape(1, 1, 2)))
    if change == 'fallback': voxelizer.backend = 'tile'
    if change == 'cupy': voxelizer.use_cupy = True
    if change == 'non_strict': voxelizer.strict_cuda = False
    if change == 'tile_size': voxelizer.tile_size = 4
    if change == 'volume_shape': voxelizer.volume_shape = (2, 1, 1)
    if change == 'extension': modules['gaussian.tile_voxelizer']._cuda_ext = None
    if change == 'source_identity': monkeypatch.setattr(modules['gaussian.joint_gaussian_model'], '__file__', '/tmp/substitute.py')
    with pytest.raises(ValueError): m.render_source(model, voxelizer, 'plain')


def test_original_raw_call_keeps_density_vjp_and_geometry_unscaled(construction, monkeypatch):
    m, d, state, modules, ext = construction
    model, voxelizer = m.construct_source('cuda:0')
    # Substitute just CUDA-dependent forward; exact geometry and real Torch VJP survive.
    def tiny_forward(self, positions, scales, rotations, density):
        assert positions is model.positions and rotations is model.rotations
        assert torch.equal(scales, state['scales'].exp())
        return (2 * density.real).reshape(1, 1, 2), (3 * density.imag).reshape(1, 1, 2)
    monkeypatch.setattr(type(voxelizer), 'forward', tiny_forward)
    density = torch.tensor([1 + 2j, 3 + 4j], requires_grad=True)
    image = m.render_density(model, voxelizer, density)
    assert torch.equal(image, torch.tensor([[[2 + 6j, 6 + 12j]]]))
    (image.real.sum() + image.imag.sum()).backward()
    assert torch.equal(density.grad, torch.tensor([2 + 3j, 2 + 3j]))
    assert torch.equal(m.render_source(model, voxelizer, 'contrast'), torch.tensor([[[10 + 21j, 12 + 24j]]]))
    with pytest.raises(ValueError): m.render_source(model, voxelizer, 'post')


@pytest.mark.parametrize('scan,factor', [('pre', np.float32(0.9997583823636317)), ('post', np.float32(1.0003600786949414))])
def test_reporting_gauge_is_cpu_complex64_times_float32(scan, factor):
    m = api()
    image = torch.tensor([[[1.123456789 + 2.987654321j, -3.23456789 + .123456789j]]], dtype=torch.complex128, requires_grad=True)
    actual = m.report_array(image, scan)
    expected = image.detach().numpy().astype(np.complex64) * factor
    assert actual.dtype == np.complex64
    assert np.array_equal(actual, expected)
    assert image.requires_grad
    with pytest.raises(ValueError): m.report_array(image, 'plain')


def test_anchor_verification_rejects_wrong_and_reversed_gauge(tmp_path, monkeypatch):
    m = api()
    image = torch.tensor([[[1 + 2j, 3 + 4j]]])
    anchor = tmp_path / 'raw_pre_complex.npy'
    np.save(anchor, np.array([[[1 + 2j, 3 + 4j]]], dtype=np.complex64) * np.float32(.9997583823636317))
    descriptor = tmp_path / 'completed.json'; descriptor.write_text('{}')
    d = m.source_definition(); d.update(volume_shape=[1, 1, 2], descriptor=str(descriptor))
    d['anchors']['pre'] = str(anchor)
    d['expected_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [anchor, descriptor]}
    monkeypatch.setattr(m, 'source_definition', lambda: d)
    result = m.verify_raw_anchor(image, 'pre')
    assert result['array_equal'] is True and result['relative_l2'] == 0 and result['passed'] is True
    # A trainer image reconstructed by dividing a reporting gauge twice must fail.
    result = m.verify_raw_anchor(image / float(np.float32(.9997583823636317)) ** 2, 'pre')
    assert not result['passed'] and result['relative_l2'] > 1e-5
    assert not m.verify_raw_anchor(image * 2, 'pre')['passed']
    with anchor.open('ab') as handle: handle.write(b'changed')
    with pytest.raises(ValueError, match='SHA256'): m.verify_raw_anchor(image, 'pre')


def test_non_cuda_device_refused_without_loading_extension():
    with pytest.raises(ValueError, match='CUDA'): api()._cuda_device('cpu')


def test_extension_identity_and_forward_backward_are_required(tmp_path):
    m = api()
    binary = tmp_path / 'backend.so'
    with pytest.raises(ValueError): m._validate_extension(SimpleNamespace(__file__=str(binary), forward=lambda: None), binary)
    with pytest.raises(ValueError): m._validate_extension(SimpleNamespace(__file__='/tmp/wrong.so', forward=lambda: None, backward=lambda: None), binary)


def test_actual_frozen_input_bytes_match_definition():
    # Reads only checkpoint/metadata/code and G2-derived checksum raw arrays.
    m = api()
    d = m.source_definition()
    assert d['expected_sha256'][d['descriptor']] == '9ab09a15fe422df7c188ac6a4951985281cf917d071ff5b7dc2cf95aa663b0ee'
    m._verify_files(d)


def test_prebinding_never_enters_jit_or_cupy(construction, monkeypatch):
    m, d, state, modules, ext = construction
    import torch.utils.cpp_extension
    def forbidden(*args, **kwargs):
        pytest.fail('compilation/fallback is forbidden')
    monkeypatch.setattr(torch.utils.cpp_extension, 'load', forbidden)
    monkeypatch.setattr(modules['gaussian.tile_voxelizer'], '_load_cupy_ext', forbidden)
    model, voxelizer = m.construct_source('cuda:0')
    assert voxelizer.backend == 'tile_cuda'


def test_non_float32_default_is_refused_without_silent_precision_tuning(construction):
    m, *_ = construction
    original = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        with pytest.raises(ValueError, match='float32'): m.construct_source('cuda:0')
        assert torch.get_default_dtype() is torch.float64
    finally:
        torch.set_default_dtype(original)


def test_original_module_collision_is_refused_before_import(monkeypatch):
    m = api()
    monkeypatch.setitem(sys.modules, 'gaussian', SimpleNamespace(__file__='/tmp/substitute/__init__.py'))
    with pytest.raises(ValueError, match='identity'): m._original_modules()
