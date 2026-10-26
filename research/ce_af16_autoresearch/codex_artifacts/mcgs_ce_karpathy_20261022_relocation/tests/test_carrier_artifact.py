"""Pure-load refusal/compatibility fixtures; no native import or GPU use."""
import ast
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module():
    assert (ROOT/'carrier_artifact.py').exists(), 'pure carrier artifact loader is missing'
    return importlib.import_module('carrier_artifact')


def test_import_is_stdlib_only_and_has_no_build_or_process_api():
    m = module()
    code = "import sys; import carrier_artifact; assert 'torch' not in sys.modules; assert 'carrier_operator' not in sys.modules"
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    tree = ast.parse(Path(m.__file__).read_text())
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any('subprocess' in ast.unparse(n) or 'cpp_extension' in ast.unparse(n) for n in imports)


def test_frozen_files_bind_exact_identity_and_no_tool_environment(monkeypatch):
    m = module()
    monkeypatch.setenv('CXX', '/deliberately/nonexistent/compiler')
    contract, provenance = m._validated_files()
    assert provenance['identity_sha256'] == contract['artifact']['identity_sha256']
    assert provenance['binary_sha256'] == contract['artifact']['binary']['sha256']


@pytest.mark.parametrize('field', ['torch','torch_cuda','cxx11_abi','torch_config','python','python_executable','capability'])
def test_every_active_runtime_binding_is_required(field):
    m = module()
    contract = json.loads((ROOT/'carrier_mri_contract.json').read_text())
    observed = copy.deepcopy(contract['artifact']['runtime'])
    m._check_runtime(observed, contract)
    observed[field] = 'foreign'
    with pytest.raises(RuntimeError, match='runtime'):
        m._check_runtime(observed, contract)


def test_context_installs_then_unwinds_and_refuses_conflict(monkeypatch):
    m = module()
    import carrier_operator as op
    contract, provenance = m._validated_files()
    native = SimpleNamespace(__name__=provenance['module'], __file__=provenance['binary'],
                             forward=lambda: None, backward=lambda: None)
    monkeypatch.setattr(m, '_runtime', lambda device: copy.deepcopy(contract['artifact']['runtime']))
    monkeypatch.setattr(m, '_import_binary', lambda p: native)
    assert op._extension is None
    with m.verified_carrier_extension('cuda:0') as actual:
        assert actual['binary_sha256'] == provenance['binary_sha256']
        assert op._extension is native
        assert m.active_provenance() == actual
        with pytest.raises(RuntimeError, match='active|conflict'):
            with m.verified_carrier_extension('cuda:0'): pass
    assert op._extension is None and op._provenance is None
    with pytest.raises(RuntimeError, match='active'):
        m.active_provenance()
    with pytest.raises(RuntimeError, match='fixture'):
        with m.verified_carrier_extension('cuda:0'):
            raise RuntimeError('fixture')
    assert op._extension is None


def test_descriptor_corruption_refuses_before_runtime(monkeypatch, tmp_path):
    m = module()
    contract = json.loads((ROOT/'carrier_mri_contract.json').read_text())
    contract['artifact']['binary']['sha256'] = '0'*64
    path = tmp_path/'contract.json'; path.write_text(json.dumps(contract))
    monkeypatch.setattr(m, '_CONTRACT', path)
    monkeypatch.setattr(m, '_runtime', lambda d: pytest.fail('runtime reached before file refusal'))
    with pytest.raises(RuntimeError, match='hash'):
        with m.verified_carrier_extension('cuda:0'): pass
