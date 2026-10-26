"""Frozen G2 source binding and faithful original raw Gaussian rendering.

Importing this module is stdlib-only. Numerical imports belong to the supervised
child; the reporting gauge is applied only on a detached CPU copy.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys

_BASE = Path('/home/congcongliu/codex_artifacts')
_OLD = _BASE / 'mcgs_ce_autoresearch_20261017'
_PAIRED = _OLD / 'nonlocal_autoresearch/trainer/paired_gs'
_G2 = _OLD / 'frequency_adam/outputs/adamflat_g2_v1'
_ANCHOR = _BASE / 'mcgs_ce_karpathy_20261019_proxdc/outputs/g2_proxdc_canonical_v1'
_NAMES = ('positions', 'scales', 'rotations', 'density_real', 'density_imag',
          'density_real_contrast', 'density_imag_contrast')
# Normal gaussian package import also executes these original package modules.
# These are Python definitions only: no datasets are constructed or arrays read.
_PYTHON_HASHES = {
    'gaussian/__init__.py': 'dcf951376a40e6f6830041cd27b5189af8d491b214e54c02d1da34e5bc9f8bfe',
    'gaussian/gaussian_model.py': '9303c79e70dfced8631d10261fd4c06b1f87958a5ddaa4a80d3b93e722ce8622',
    'gaussian/joint_gaussian_model.py': '31eea44eb636113e29b1f99192f297e38ac0c7b937e8dc3b5daff11c349fd966',
    'gaussian/bayesian_joint_model.py': 'b2d8d1275eb28f395f9102d373f3f33904ede4c205b276c4087f20527d6439c2',
    'gaussian/voxelizer.py': '8d63e1c93b7482d9d236c8756d050c5b83ba01f79b3bf14421445579f66b1c93',
    'gaussian/tile_voxelizer.py': '2769454cde8276b11ded79f6729ba62e3ed67b7a2af585de15807d3f7957dc0a',
    'losses/__init__.py': '8ff1e8e2db6dbea65b9a2bc87d724c16952fc8a78ac190eb84b560ab653d1476',
    'losses/kspace_loss.py': '23bdcf9980656592e052b63d9a04f4fffd39477ab644c4815b43c65d04136f90',
    'losses/multiscale_loss.py': 'e4d242df6dec1e31f5634eb9d7fba31103ba615775e49d0ea1e0d2c6e727afee',
    'losses/losses.py': 'afb06a60ada378760051898cbf43fa1ec89d0260b1943429b714a06bb0baee1e',
    'losses/complex_gaussian_kl.py': '852a1f5ee6caccc2ea2310444c1fac39c9b3ac693a44640ed236548807622134',
    'losses/elbo.py': '69ab91ad29f6476cb6daf99e40f0d38f8475f2bd713ccf7a9744e12f19ab3144',
    'losses/cf_safe.py': 'eac4a27e702c789f1af28c5cd00a2d686cdffeaf36303b56b80b293f1ebe8766',
    'losses/cf_damp.py': '783c89a0ae1a7ddc4507e5d681e9e4bea3dbd4bdad0c292a18c5fdbd5429b517',
    'data/__init__.py': '4954a1d4db61951b7fb87ee7e8871835dfae5732eca3286cd1fa3e8173551033',
    'data/transforms.py': '0037b2ee2c085af69ab18637bfaf87d3146f86b0b7e6fdf96a73b7bcd40d7015',
    'data/mask_three_way_split.py': 'a790539f5f349276666db51ca4bc52c44e19a8c9aaf17284dc30c6d41cb5a092',
    'data/dataset.py': '28ae44153e8085bc36636aa210024ab49d882d056ec08602e683051fba2fc909',
}


def source_definition() -> dict:
    """Return immutable-input definitions without importing numerical packages."""
    checkpoint = str(_G2 / 'training/ar_i1_k30p90/checkpoints/latest.pth')
    binary = str(_PAIRED / 'gaussian/cuda_ext/build/tile_voxelize_cuda_0141c525_7aedd3b0.so')
    descriptor = str(_ANCHOR / 'completed.json')
    anchors = {scan: str(_ANCHOR / f'raw_{scan}_complex.npy') for scan in ('pre', 'post')}
    hashes = {str(_PAIRED / path): digest for path, digest in _PYTHON_HASHES.items()}
    hashes.update({
        checkpoint: '282e0303c46a6c18775d6224c032611072bd465b4ef975ba7a7e2ea636230fd0',
        binary: '3973daeef5f55ebfed931c38538a37d83921a9287b96b69a431da238057194b5',
        descriptor: '9ab09a15fe422df7c188ac6a4951985281cf917d071ff5b7dc2cf95aa663b0ee',
        anchors['pre']: '202985d20e2a6554e369e5573668169d30ae73096f985753acd0caa172da92ab',
        anchors['post']: '9830d5bcb22a70d5af5cd56c2d4c01d248e694911d81888295cc796f253dee75',
        str(_G2 / 'actual_config.yaml'): 'c91610d5d4bbd61c333d60f26af1f9319a4cb3413c8ca003378f290f28ab8311',
        str(_G2 / 'metrics.json'): '47831297a2359bca7d1b2f72e75f1fbf14b3fb2ed8f43062a04461650442b556',
        str(_G2 / 'provenance.json'): '74bc739b1498eaf666ffb25881c3bed8de886dbfae69f79b287370d3e4e14461',
        str(_G2 / 'frequency_provenance.json'): '49b85db4a11eb7c8b1932b06833a0962cc3749e27164561019b5de3c0357e46b',
        str(_PAIRED / 'gaussian/cuda_ext/tile_voxelize_cuda.cpp'): 'e412af61cc4f5c75b9988d4f433a73e2e1c2b5910feff48b7beaea8737012d49',
        str(_PAIRED / 'gaussian/cuda_ext/tile_voxelize_cuda_kernel.cu'): '515673bd9bf0af1d23b91679f5d9e177d9b034c9172f75ec0ce937258686da14',
    })
    return dict(checkpoint=checkpoint, source_dir=str(_PAIRED / 'gaussian'),
                binary=binary, descriptor=descriptor, anchors=anchors,
                expected_sha256=hashes, volume_shape=[256, 232, 176], num_points=3600000,
                source_names=list(_NAMES), source_iteration=4500, frequency_t=1.,
                geometry_factor=2., geometry_factor_role='historical optimizer LR policy only',
                state_shapes={name: [3600000, 3 if name != 'rotations' else 4]
                              if name in _NAMES[:3] else [3600000] for name in _NAMES},
                gauges={'pre': .9997583823636317, 'post': 1.0003600786949414})


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _verify_files(definition):
    for path, expected in definition['expected_sha256'].items():
        if _sha256(path) != expected:
            raise ValueError(f'frozen input SHA256 changed: {path}')


def _module_paths():
    return {path[:-3].replace('/', '.').removesuffix('.__init__'): _PAIRED / path
            for path in _PYTHON_HASHES}


def _validate_module_paths(require_all=True):
    modules = {}
    for name, expected in _module_paths().items():
        module = sys.modules.get(name)
        if module is None and not require_all:
            continue
        actual = getattr(module, '__file__', None)
        if actual is None or Path(actual).resolve() != expected.resolve():
            raise ValueError(f'original source module identity differs: {name}')
        modules[name] = module
    return modules


def _original_modules():
    # Reject collisions before any package __init__ executes, not after use.
    _validate_module_paths(require_all=False)
    sys.path.insert(0, str(_PAIRED))
    try:
        importlib.import_module('gaussian')
    finally:
        sys.path.remove(str(_PAIRED))
    return _validate_module_paths()


def _validate_extension(extension, binary):
    actual = getattr(extension, '__file__', None)
    if actual is None or Path(actual).resolve() != Path(binary).resolve() \
            or not all(callable(getattr(extension, name, None)) for name in ('forward', 'backward')):
        raise ValueError('original CUDA binary identity/forward/backward differs; no compilation or fallback')


def _load_extension(binary):
    binary = Path(binary)
    existing = sys.modules.get(binary.stem)
    if existing is not None:
        _validate_extension(existing, binary)
        return existing
    spec = importlib.util.spec_from_file_location(binary.stem, binary)
    if spec is None or spec.loader is None:
        raise ValueError('cannot load frozen CUDA binary; no compilation or fallback')
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    _validate_extension(extension, binary)
    sys.modules[binary.stem] = extension
    return extension


def _cuda_device(device):
    import torch
    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('the original strict renderer requires a CUDA device')
    return device


def _validate_state(checkpoint, definition):
    import torch
    recipe = checkpoint.get('frequency_adam_recipe', {})
    if checkpoint.get('iter') != 4500 or recipe.get('frequency_t') != 1. \
            or recipe.get('geometry_factor') != 2.:
        raise ValueError('expected source iteration4500/frequencyt1/geometryfactor2 optimizer recipe')
    state = checkpoint.get('model_state', {})
    if set(state) != set(_NAMES):
        raise ValueError('expected exactly seven original G2 state tensors')
    for name in _NAMES:
        value = state[name]
        if not isinstance(value, torch.Tensor) or list(value.shape) != definition['state_shapes'][name] \
                or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f'G2 state shape/dtype/finiteness differs: {name}')
    return state


def construct_source(device):
    """Restore all original tensors and bind the prebuilt strict CUDA renderer."""
    import torch
    definition = source_definition()
    if torch.get_default_dtype() != torch.float32:
        raise ValueError('original G2 constructor requires the float32 default dtype')
    device = _cuda_device(device)
    _verify_files(definition)
    modules = _original_modules()
    tile = modules['gaussian.tile_voxelizer']
    extension = _load_extension(definition['binary'])
    _validate_extension(extension, definition['binary'])
    # Prebinding prevents the original loader from entering JIT or CuPy paths.
    if tile._cuda_ext is not None and tile._cuda_ext is not extension:
        raise ValueError('another original renderer extension is already bound')
    tile._cuda_ext = extension
    tile._cuda_load_error = None
    checkpoint = torch.load(definition['checkpoint'], map_location='cpu', weights_only=True)
    state = _validate_state(checkpoint, definition)
    cls = modules['gaussian.joint_gaussian_model'].JointGaussianModel3D
    model = cls(num_points=definition['num_points'], volume_shape=tuple(definition['volume_shape']),
                device=device, initial_positions=state['positions'].to(device),
                initial_scales=state['scales'].exp().to(device),
                initial_densities_plain=torch.complex(state['density_real'], state['density_imag']).to(device),
                initial_densities_contrast=torch.complex(state['density_real_contrast'], state['density_imag_contrast']).to(device))
    model.load_state_dict(state, strict=True)
    for name in _NAMES:
        tensor = getattr(model, name)
        if not torch.equal(tensor.detach().cpu(), state[name]):
            raise ValueError(f'G2 restore drift: {name}')
        tensor.requires_grad_(False)
    voxelizer = tile.TileVoxelizer(volume_shape=tuple(definition['volume_shape']), tile_size=8,
                                  max_radius=20, use_cuda=True, strict_cuda=True,
                                  device=str(device)).to(device)
    _validate_runtime(model, voxelizer)
    return model, voxelizer


def _validate_runtime(model, voxelizer):
    modules = _validate_module_paths()
    tile = modules['gaussian.tile_voxelizer']
    if type(model) is not modules['gaussian.joint_gaussian_model'].JointGaussianModel3D \
            or type(voxelizer) is not tile.TileVoxelizer:
        raise ValueError('original model/voxelizer class identity differs')
    if voxelizer.backend != 'tile_cuda' or not voxelizer.use_cuda or voxelizer.use_cupy \
            or not voxelizer.strict_cuda:
        raise ValueError('original strict tile_cuda backend required; no compilation or fallback')
    definition = source_definition()
    shape = tuple(definition['volume_shape'])
    if voxelizer.tile_size != 8 or voxelizer.max_radius != 20 \
            or tuple(voxelizer.volume_shape) != shape or tuple(model.volume_shape) != shape:
        raise ValueError('original voxelizer geometry/configuration differs')
    _validate_extension(tile._cuda_ext, definition['binary'])
    if tile._cuda_load_error is not None:
        raise ValueError('original CUDA loader contains an error; no compilation or fallback')


def render_density(model, voxelizer, density):
    """Render through the original density VJP without changing geometry or gauge."""
    import torch
    _validate_runtime(model, voxelizer)
    real, imag = voxelizer(model.positions, model.get_scale_values(), model.rotations, density)
    return torch.complex(real, imag)


def render_source(model, voxelizer, channel):
    return render_density(model, voxelizer, model.get_density(channel))


def source_hashes(model) -> dict:
    return {name: hashlib.sha256(getattr(model, name).detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name in _NAMES}


def report_array(image, scan):
    """Copy a fresh trainer-gauge image through the canonical CPU gauge arithmetic."""
    import numpy as np
    definition = source_definition()
    if scan not in definition['gauges']:
        raise ValueError('scan must be pre or post')
    if hasattr(image, 'detach'):
        image = image.detach().cpu().numpy()
    array = np.asarray(image, dtype=np.complex64)
    if not np.isfinite(array).all():
        raise ValueError('raw image is nonfinite')
    return array * np.float32(definition['gauges'][scan])


def verify_raw_anchor(image, scan) -> dict:
    """Check reporting-gauge G2 checksum anchors; never reconstruct a teacher."""
    import numpy as np
    definition = source_definition()
    if scan not in definition['anchors']:
        raise ValueError('scan must be pre or post')
    paths = [definition['descriptor'], definition['anchors'][scan]]
    _verify_files({'expected_sha256': {p: definition['expected_sha256'][p] for p in paths}})
    anchor = np.load(definition['anchors'][scan], allow_pickle=False)
    fresh = report_array(image, scan)
    shape = tuple(definition['volume_shape'])
    if anchor.shape != shape or fresh.shape != shape or anchor.dtype != np.complex64 \
            or not np.isfinite(anchor).all():
        raise ValueError('raw anchor shape/dtype/finiteness differs')
    # Complex128 for error accumulation only; gauge rounding above stays canonical.
    difference = fresh.astype(np.complex128) - anchor
    numerator = np.sum(np.abs(difference) ** 2, dtype=np.float64)
    denominator = np.sum(np.abs(anchor.astype(np.complex128)) ** 2, dtype=np.float64)
    if denominator <= 0:
        raise ValueError('raw anchor has zero energy')
    relative_l2 = float(np.sqrt(numerator / denominator))
    return dict(scan=scan, factor=definition['gauges'][scan],
                array_equal=bool(np.array_equal(fresh, anchor)), relative_l2=relative_l2,
                tolerance=1e-5, passed=bool(np.isfinite(relative_l2) and relative_l2 <= 1e-5),
                anchor=definition['anchors'][scan],
                anchor_sha256=definition['expected_sha256'][definition['anchors'][scan]])
