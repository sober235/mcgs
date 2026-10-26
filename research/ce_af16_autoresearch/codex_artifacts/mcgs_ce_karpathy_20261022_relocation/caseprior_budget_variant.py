"""Fixed private copies of the reviewed case-prior parent and child sources.

Only CNN fitting budget, route/child dispatch and source closure change. Loading
the parent is stdlib-only; the numerical child remains lazy. No original module
is patched, and no source, code object or denoiser implementation is rewritten.
"""
from contextlib import contextmanager
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def load_protocol():
    """Return the registered, isolated512 protocol with original public APIs."""
    name = '_caseprior_fit512_protocol_v1'
    spec = importlib.util.spec_from_file_location(name, ROOT/'run_caseprior.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        module.ROUTE = 'g2_case_internal_magnitude_prior_fit512_v1'
        module.CHILD_RUNNER = 'caseprior512_runner.py'
        module.SETTINGS = {**module.SETTINGS, 'updates': 512}
        module.POLICY = {**module.POLICY, 'cnn': module.SETTINGS}
        inherited_hashes = module.code_hashes

        def code_hashes():
            return {**inherited_hashes(), **{str(ROOT/name): module.base.sha256(ROOT/name)
                for name in ('caseprior_budget_variant.py','run_caseprior512.py','caseprior512_runner.py')}}

        module.code_hashes = code_hashes
    except BaseException:
        del sys.modules[name]
        raise
    return module


@lru_cache(maxsize=1)
def load_child():
    """Return the reviewed child copy whose function/class globals use512."""
    protocol = load_protocol()
    name = '_caseprior_fit512_child_v1'
    spec = importlib.util.spec_from_file_location(name, ROOT/'caseprior_runner.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        module.route = protocol
        original_factory = module.case_denoiser_factory

        @contextmanager
        def case_denoiser_factory(directory, *, device, settings=None):
            # Private tiny CPU fixtures retain the inherited settings hook.
            # Production supplies none and therefore reaches the actual Task1
            # constructor with512 rather than its frozen default128.
            with original_factory(directory, device=device,
                    settings={**protocol.SETTINGS, **(settings or {})}) as instances:
                yield instances

        module.case_denoiser_factory = case_denoiser_factory
    except BaseException:
        del sys.modules[name]
        raise
    return module
