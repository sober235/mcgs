"""Selected-row sparse CUDA carrier correction; explicit isolated build only.

Import and argument validation do not import cpp_extension or initialize CUDA.
Only first derivatives are supported. No historical renderer is imported.
"""
import hashlib
import json
import math
import os
from numbers import Integral
from pathlib import Path
import subprocess
import shutil
import threading
import sys

import torch
from torch.autograd.function import once_differentiable

_ROOT = Path(__file__).resolve().parent
_SOURCES = (_ROOT / 'carrier_cuda/carrier.cpp', _ROOT / 'carrier_cuda/carrier_kernel.cu')
_FLAGS = {'cxx': ['-O3'], 'cuda': ['-O3', '--fmad=false', '-ccbin', '/usr/bin/gcc']}
_BUILD_PATH = '/usr/local/cuda-11.8/bin:/usr/bin:/bin'
_UNSUPPORTED_BUILD_ENV = (
    'PYTORCH_NVCC', 'CC', 'NVCC_CCBIN', 'NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS',
    'CUDACXX', 'CUDAHOSTCXX', 'CUDA_PATH', 'CUDA_INC_PATH', 'CUDNN_HOME', 'CUDNN_PATH',
    'ROCM_HOME', 'ROCM_PATH', 'CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'LDFLAGS',
    'GCC_EXEC_PREFIX', 'COMPILER_PATH', 'CPATH', 'CPLUS_INCLUDE_PATH',
    'C_INCLUDE_PATH', 'OBJC_INCLUDE_PATH', 'LIBRARY_PATH', 'LD_PRELOAD',
    'LD_LIBRARY_PATH', 'TORCH_DONT_CHECK_COMPILER_ABI',
    'TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES', 'TORCH_EXTENSIONS_DIR',
)
_extension = None
_provenance = None
_load_lock = threading.Lock()
_TILE = 8


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validated_build_environment():
    """Reject effective tool/flag overrides before import, tools or cache reuse.

    Torch JIT honors CC, PYTORCH_NVCC, CUDA include/CUDNN/ROCM paths and ABI /
    dependency selectors; GCC/nvcc also honor implicit flag/search variables.
    Keep the reviewed build process explicit rather than silently clearing any.
    """
    required = {'CUDA_HOME': '/usr/local/cuda-11.8', 'CXX': '/usr/bin/c++', 'PATH': _BUILD_PATH}
    for name, value in required.items():
        if os.environ.get(name) != value:
            raise RuntimeError(f'set {name}={value} before explicit carrier loading')
    for name in _UNSUPPORTED_BUILD_ENV:
        if name in os.environ:
            raise RuntimeError(f'unset unsupported carrier build override {name}')
    if not os.environ.get('TORCH_CUDA_ARCH_LIST'):
        raise RuntimeError('set explicit TORCH_CUDA_ARCH_LIST for provenance')
    return {**required, 'TORCH_CUDA_ARCH_LIST': os.environ['TORCH_CUDA_ARCH_LIST'],
            'MAX_JOBS': os.environ.get('MAX_JOBS'),
            'SOURCE_DATE_EPOCH': os.environ.get('SOURCE_DATE_EPOCH'),
            'cuda_host_compiler': '/usr/bin/gcc',
            'refused_overrides': {name: None for name in _UNSUPPORTED_BUILD_ENV}}


def load_carrier_extension(build_directory):
    """Build/load only a carrier-owned, content-addressed artifact.

    build_directory must be a new or previously carrier-owned directory below
    this component's carrier_builds/. Set CUDA_HOME=/usr/local/cuda-11.8 and an
    explicit TORCH_CUDA_ARCH_LIST before calling. This is the only build entry.
    """
    global _extension, _provenance
    destination = Path(build_directory).expanduser().resolve()
    owned_root = (_ROOT / 'carrier_builds').resolve()
    if owned_root != _ROOT / 'carrier_builds':
        raise ValueError('carrier_builds must not be a symlink outside the component')
    if destination == owned_root or owned_root not in destination.parents:
        raise ValueError('build_directory must be strictly below component carrier_builds/')
    # Must happen before importing cpp_extension: toolkit/include configuration
    # is import-cached. Also validate before the in-process artifact cache.
    build_environment = _validated_build_environment()
    if torch.version.cuda != '11.8':
        raise RuntimeError('this reviewed toolchain requires Torch CUDA 11.8')
    architectures = build_environment['TORCH_CUDA_ARCH_LIST']
    cc = build_environment['CXX']
    host_cc = build_environment['cuda_host_compiler']
    nvcc = Path(build_environment['CUDA_HOME']) / 'bin/nvcc'
    builder_source = Path(torch.__file__).resolve().parent / 'utils/cpp_extension.py'
    identity = {'sources': {str(p.relative_to(_ROOT)): _sha(p) for p in (*_SOURCES, Path(__file__))},
                'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
                'cxx11_abi': torch._C._GLIBCXX_USE_CXX11_ABI,
                'torch_config': torch.__config__.show(), 'cuda_home': os.environ['CUDA_HOME'],
                'build_environment': build_environment, 'builder_source_sha256': _sha(builder_source),
                'python': sys.version, 'python_executable': sys.executable,
                'architectures': architectures, 'flags': _FLAGS,
                'compiler_executable': str(Path(shutil.which(cc) or cc).resolve()),
                'compiler_sha256': _sha(shutil.which(cc) or cc),
                'compiler': subprocess.check_output([cc, '--version'], text=True),
                'cuda_host_compiler_executable': str(Path(host_cc).resolve()),
                'cuda_host_compiler_sha256': _sha(host_cc),
                'cuda_host_compiler': subprocess.check_output([host_cc, '--version'], text=True),
                'nvcc_executable': str(nvcc.resolve()), 'nvcc_sha256': _sha(nvcc),
                'nvcc': subprocess.check_output([str(nvcc), '--version'], text=True),
                'nvcc_profile_sha256': _sha(nvcc.parent / 'nvcc.profile'),
                'ninja_executable': str(Path(shutil.which('ninja')).resolve()),
                'ninja_sha256': _sha(shutil.which('ninja')),
                'ninja_version': subprocess.check_output(['/usr/bin/ninja', '--version'], text=True),
                'support_bounds': 'inverse(actual_M), historical floor/ceil/clamp/stable ordering'}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    name = 'carrier_cuda_' + digest[:24]
    with _load_lock:
        marker = destination / 'carrier-owner.json'
        if destination.exists() and not marker.exists() and any(destination.iterdir()):
            raise ValueError('carrier_builds destination is nonempty and not carrier-owned')
        if marker.exists() and json.loads(marker.read_text()) != {'component': str(_ROOT)}:
            raise ValueError('carrier_builds destination owner mismatch')
        destination.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({'component': str(_ROOT)}))
        artifact = destination / digest
        artifact.mkdir(exist_ok=True)
        manifest = artifact / 'provenance.json'
        if manifest.exists():
            previous = json.loads(manifest.read_text())
            binary = Path(previous['binary'])
            if (previous['identity'] != identity or binary.parent != artifact
                    or _sha(binary) != previous['binary_sha256']
                    or _sha(artifact / 'build.ninja') != previous['build_ninja_sha256']):
                raise RuntimeError('carrier artifact identity/hash mismatch; refusing reuse')
        elif any(artifact.iterdir()):
            raise RuntimeError('unverified partial carrier build; use a new dedicated directory')
        if _provenance is not None and _provenance['identity'] == identity:
            if Path(_provenance['binary']).parent == artifact and _sha(_provenance['binary']) == _provenance['binary_sha256']:
                return _extension, json.loads(json.dumps(_provenance))
        from torch.utils import cpp_extension
        if Path(cpp_extension.CUDA_HOME).resolve() != Path(os.environ['CUDA_HOME']).resolve():
            raise RuntimeError('cpp_extension was imported with a different CUDA_HOME')
        if cpp_extension.CUDNN_HOME is not None or cpp_extension.ROCM_HOME is not None or cpp_extension.IS_HIP_EXTENSION:
            raise RuntimeError('cpp_extension was imported with an unsupported CUDNN/ROCM override')
        module = cpp_extension.load(name=name, sources=[str(p) for p in _SOURCES],
                                    build_directory=str(artifact), extra_cflags=_FLAGS['cxx'],
                                    extra_cuda_cflags=_FLAGS['cuda'], verbose=True)
        binary = Path(module.__file__).resolve()
        if binary.parent != artifact:
            raise RuntimeError('loaded carrier binary escaped dedicated artifact directory')
        provenance = {'module': module.__name__, 'identity_sha256': digest, 'identity': identity,
                      'binary': str(binary), 'binary_sha256': _sha(binary),
                      'build_ninja_sha256': _sha(artifact / 'build.ninja'),
                      'effective_build_ninja': (artifact / 'build.ninja').read_text()}
        if manifest.exists() and json.loads(manifest.read_text()) != provenance:
            raise RuntimeError('carrier artifact changed during load')
        manifest.write_text(json.dumps(provenance, indent=2, sort_keys=True))
        _extension, _provenance = module, provenance
        return module, json.loads(json.dumps(provenance))


def _validate(values, volume_shape):
    if not isinstance(volume_shape, (tuple, list, torch.Size)):
        raise TypeError('volume_shape must contain three positive integers')
    if len(volume_shape) != 3:
        raise ValueError('volume_shape must contain three positive integers')
    if any(isinstance(n, bool) or not isinstance(n, Integral) for n in volume_shape):
        raise TypeError('volume_shape must contain three positive integers')
    if any(n <= 0 for n in volume_shape) or math.prod(volume_shape) > 2**31 - 1:
        raise ValueError('volume_shape must be positive and fit int32 voxel indexing')
    for i, value in enumerate(values):
        if not isinstance(value, torch.Tensor):
            raise TypeError('all inputs must be tensors')
        if value.dtype != (torch.complex64 if i == 3 else torch.float32):
            raise TypeError('inputs require float32 geometry/frequencies and complex64 density')
    if values[0].ndim != 2 or values[0].shape[1] != 3:
        raise ValueError('positions must have shape [K,3]')
    k = values[0].shape[0]
    if k > 2**31 - 1:
        raise ValueError('K must fit int32 indexing')
    for value, shape in zip(values, ((k, 3), (k, 3), (k, 4), (k,), (k, 3))):
        if tuple(value.shape) != shape:
            raise ValueError(f'input must have shape {shape}')
        if value.device != values[0].device:
            raise ValueError('inputs must share a CUDA device')
        if not torch.isfinite(value).all().item():
            raise ValueError('inputs must be finite')
    if not ((values[-1] > -.5) & (values[-1] < .5)).all().item():
        raise ValueError('frequencies must be strictly inside (-0.5,0.5) cycles per voxel')
    return tuple(int(n) for n in volume_shape)


def _geometry(positions, log_scales, rotations, shape):
    half = positions.new_tensor(shape) / 2
    centers = (positions + 1) * half - .5
    scales = log_scales.exp()
    rotation_norm = rotations.norm(dim=-1, keepdim=True)
    # A finite final M can hide exp(scale)=inf or norm(q)=inf. Such
    # intermediates can turn even a zero local VJP into NaN during the chain.
    for name, value in (('centers', centers), ('scales', scales), ('rotation norm', rotation_norm)):
        if not torch.isfinite(value).all().item():
            raise ValueError(f'derived {name} must be finite')
    q = rotations / (rotation_norm + 1e-8)
    w, x, y, z = q.unbind(-1)
    r = torch.stack((torch.stack((1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w), -1),
                     torch.stack((2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w), -1),
                     torch.stack((2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y), -1)), -2)
    m = (r * (scales + 1e-8).pow(-2).unsqueeze(1)) @ r.transpose(-1, -2)
    # Float32 products need not be bitwise symmetric. The local center VJP
    # assumes symmetry; averaging retains the quadratic form and its chain.
    return centers, .5*(m + m.transpose(-1, -2))


def _tile_shape(shape):
    # Positive ceil-div avoids n+7 overflow in the matching native arithmetic.
    return tuple(1+(n-1)//_TILE for n in shape)


@torch.no_grad()
def _tile_pairs(centers, m, shape):
    # Conservative support of the actual M, since epsilon-normalized rotations
    # need not be orthogonal. Never clamp a radius to historical max_radius.
    if not torch.isfinite(centers).all() or not torch.isfinite(m).all():
        raise ValueError('derived geometry must be finite')
    eigenvalues = torch.linalg.eigvalsh(m.double())
    reliable = (eigenvalues[:, 0] > 0) & (eigenvalues[:, 0] > eigenvalues[:, -1]*1e-8)
    radius = torch.zeros_like(centers, dtype=torch.float64)
    radius[reliable] = 3 * torch.linalg.inv(m[reliable].double()).diagonal(dim1=-2, dim2=-1).sqrt() * centers.new_tensor(shape).double() / 2
    reliable &= torch.isfinite(radius).all(-1)
    # Clamp in float64 before integer conversion to avoid overflow for far atoms.
    limit = centers.new_tensor(shape, dtype=torch.float64) - 1
    low = torch.minimum((centers.double()-radius).floor().clamp_min(0), limit).long() // _TILE
    high = torch.minimum((centers.double()+radius).ceil().clamp_min(0), limit).long() // _TILE
    nt = _tile_shape(shape)
    # Finite singular/poorly conditioned M can have unbounded support. Cover
    # every tile for these atoms, with the same exact mahal test in the kernel.
    low[~reliable] = 0
    high[~reliable] = high.new_tensor(nt)-1
    ranges = torch.zeros((math.prod(nt), 2), dtype=torch.int32, device=centers.device)
    count = high-low+1
    per = count.prod(-1)
    total = int(per.sum().item())
    if total > 2**31-1:
        raise ValueError('tile pair count exceeds int32')
    if total == 0:
        return ranges, torch.empty(0, dtype=torch.int32, device=centers.device)
    ids = torch.repeat_interleave(torch.arange(len(centers), device=centers.device, dtype=torch.int32), per)
    starts = per.cumsum(0)-per
    local = torch.arange(total, device=centers.device)-torch.repeat_interleave(starts, per)
    dims = torch.repeat_interleave(count, per, dim=0)
    origin = torch.repeat_interleave(low, per, dim=0)
    td = origin[:, 0]+local//(dims[:, 1]*dims[:, 2])
    th = origin[:, 1]+(local//dims[:, 2])%dims[:, 1]
    tw = origin[:, 2]+local%dims[:, 2]
    tiles = (td*nt[1]+th)*nt[2]+tw
    order = tiles.argsort(stable=True)
    tiles, ids = tiles[order], ids[order].contiguous()
    starts = torch.cat((tiles.new_zeros(1), (tiles[1:] != tiles[:-1]).nonzero().flatten()+1))
    ends = torch.cat((starts[1:], starts.new_tensor([total])))
    ranges[tiles[starts], 0] = starts.int()
    ranges[tiles[starts], 1] = ends.int()
    return ranges, ids


class _Carrier(torch.autograd.Function):
    @staticmethod
    def forward(ctx, centers, m, density, frequencies, ranges, ids, shape, extension):
        ctx.save_for_backward(centers, m, density, frequencies, ranges, ids)
        ctx.shape, ctx.extension = shape, extension
        return extension.forward(centers, m, density, frequencies, ranges, ids, list(shape))

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        gradients = ctx.extension.backward(grad_output.contiguous(), *ctx.saved_tensors, list(ctx.shape))
        return (*gradients, None, None, None, None)


def carrier_correction(positions, log_scales, rotations, density, frequencies, volume_shape):
    """CUDA complex64 correction; selected K rows, frequencies cycles/voxel."""
    values = (positions, log_scales, rotations, density, frequencies)
    shape = _validate(values, volume_shape)
    if _extension is None:
        raise RuntimeError('call load_carrier_extension explicitly before carrier_correction')
    if positions.device.type != 'cuda':
        raise ValueError('all inputs must be CUDA tensors')
    positions, log_scales, rotations, density, frequencies = (x.contiguous() for x in values)
    centers, m = _geometry(positions, log_scales, rotations, shape)
    ranges, ids = _tile_pairs(centers, m, shape)
    # Even K=0 follows the derivative-capable custom boundary; no zero-f shortcut.
    return _Carrier.apply(centers.contiguous(), m.contiguous(), density, frequencies, ranges, ids, shape, _extension)
