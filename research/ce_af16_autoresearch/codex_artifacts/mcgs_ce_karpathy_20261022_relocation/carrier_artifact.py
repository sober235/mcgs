"""Verify and load the frozen existing carrier binary, without a build API.

Importing this module uses only the standard library. The context is entered in
the numerical child before clinical data loading; runtime and CUDA checks are
deliberately lazy. Compiler environment variables have no role in this loader.
"""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
_CONTRACT = _ROOT / 'carrier_mri_contract.json'
_active = None


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _descriptor(record):
    path = Path(record['path'])
    if not path.is_file() or _sha(path) != record['sha256']:
        raise RuntimeError('carrier artifact file/hash differs: ' + str(path))
    return path


def _validated_files():
    contract = json.loads(_CONTRACT.read_text())
    artifact = contract['artifact']
    for key in ('binary', 'manifest', 'accepted_microgate'):
        _descriptor(artifact[key])
    provenance = json.loads(Path(artifact['manifest']['path']).read_text())
    identity = provenance['identity']
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if (digest != artifact['identity_sha256'] or provenance['identity_sha256'] != digest
            or provenance['module'] != artifact['module']
            or provenance['binary'] != artifact['binary']['path']
            or provenance['binary_sha256'] != artifact['binary']['sha256']
            or identity['sources'] != artifact['sources'] or identity['architectures'] != '8.0'):
        raise RuntimeError('carrier artifact identity differs')
    for key, digest in artifact['sources'].items():
        if _sha(_ROOT / key) != digest:
            raise RuntimeError('carrier artifact source hash differs: ' + key)
    ninja = Path(provenance['binary']).parent / 'build.ninja'
    if (_sha(ninja) != provenance['build_ninja_sha256']
            or ninja.read_text() != provenance['effective_build_ninja']):
        raise RuntimeError('carrier artifact build manifest hash differs')
    for key, value in artifact['runtime'].items():
        if key != 'capability' and identity[key] != value:
            raise RuntimeError('carrier manifest runtime identity differs: ' + key)
    return contract, provenance


def _runtime(device):
    import torch
    device = torch.device(device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('verified carrier requires a visible CUDA device')
    return dict(torch=str(torch.__version__), torch_cuda=torch.version.cuda,
                cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,
                torch_config=torch.__config__.show(), python=sys.version,
                python_executable=sys.executable,
                capability=list(torch.cuda.get_device_capability(device)))


def _check_runtime(observed, contract):
    if observed != contract['artifact']['runtime']:
        raise RuntimeError('carrier active runtime/ABI/capability differs from frozen sm80 artifact')


def _import_binary(provenance):
    name, path = provenance['module'], provenance['binary']
    if name in sys.modules:
        module = sys.modules[name]
        if str(Path(module.__file__).resolve()) != path:
            raise RuntimeError('conflicting carrier native module already imported')
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError('carrier native extension loader is unavailable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def active_provenance():
    """Return the actual active context binding; refuse unverified use."""
    if _active is None:
        raise RuntimeError('no verified carrier artifact context is active')
    return json.loads(json.dumps(_active))


@contextmanager
def verified_carrier_extension(device):
    """Install only the frozen existing extension, then restore prior state."""
    global _active
    contract, provenance = _validated_files()
    observed = _runtime(device)
    _check_runtime(observed, contract)
    import carrier_operator as operator
    with operator._load_lock:
        if _active is not None or operator._extension is not None or operator._provenance is not None:
            raise RuntimeError('conflicting active carrier extension state')
        module = _import_binary(provenance)
        if (module.__name__ != provenance['module']
                or str(Path(module.__file__).resolve()) != provenance['binary']
                or not callable(getattr(module, 'forward', None))
                or not callable(getattr(module, 'backward', None))):
            raise RuntimeError('carrier native module interface/path differs')
        # Detect a file change between verification and native import as well.
        _validated_files()
        record = dict(provenance, runtime=observed, load_kind='existing_binary_only',
                      manifest=contract['artifact']['manifest'],
                      accepted_microgate=contract['artifact']['accepted_microgate'])
        previous = operator._extension, operator._provenance
        operator._extension, operator._provenance = module, record
        _active = record
    try:
        yield json.loads(json.dumps(record))
    finally:
        with operator._load_lock:
            operator._extension, operator._provenance = previous
            _active = None
