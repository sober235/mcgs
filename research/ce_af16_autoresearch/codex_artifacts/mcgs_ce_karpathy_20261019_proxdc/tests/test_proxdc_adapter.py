"""CPU-only G2 route checks; external numerical sources remain read-only."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_gpu as lifecycle


@pytest.mark.parametrize('context_visible',[True,False])
def test_protocol_hooks_own_the_route_without_replacing_child_lifecycle(tmp_path, monkeypatch, context_visible, request):
    if os.environ.get('PROX_PROTOCOL_TEST_CHILD') != '1':
        completed = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
            request.node.nodeid], env={**os.environ, 'PROX_PROTOCOL_TEST_CHILD': '1'},
            capture_output=True, text=True, timeout=20)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        return
    route = 'g2_raw_proxdc_v1'
    child = tmp_path / 'proxdc_runner.py'
    source_root=str(Path(lifecycle.__file__).parent)
    child.write_text(f'''import argparse,json,pathlib,os,sys
sys.path.insert(0,{source_root!r})
from proxdc_runner import wait_context_ack
p=argparse.ArgumentParser();p.add_argument('--launch');a=p.parse_args()
l=json.loads(pathlib.Path(a.launch).read_text());r=pathlib.Path(l['run_dir'])
assert l['route']=='g2_raw_proxdc_v1'
(r/'childcontext_ready.json').write_text(json.dumps({{'pid':os.getpid(),'parent_pid':os.getppid(),'gpu_uuid':l['gpu_uuid']}}))
wait_context_ack(r,l,timeout=3)
(r/'result.json').write_text(json.dumps({{'route':l['route'],'pid':os.getpid()}}))
''')
    for name in ('recipe.json', 'review.json'):
        (tmp_path / name).write_text('{}')
    args = SimpleNamespace(exp='route_cpu', timeout=5 if context_visible else .3, preferred_gpu=0,
        recipe=tmp_path / 'recipe.json', review_gate=tmp_path / 'review.json')
    hashes = {str(child): lifecycle.sha256(child)}
    invariant = {'g2': 'unchanged', 'route': route}
    calls = []
    def bound_hashes():
        calls.append('source')
        return hashes
    def artifacts(args, directory, observed_hashes, observed_invariants):
        assert observed_hashes == hashes and observed_invariants == invariant
        result = lifecycle.read(directory / 'result.json')
        assert result['route'] == route
        ack=lifecycle.read(directory/'childcontext_ack.json')
        assert ack==dict(schema_version=1,child_pid=result['pid'],parent_pid=os.getpid(),
            gpu_uuid='GPU-cpu',context_observed=True)
        return {str(directory/name): lifecycle.sha256(directory/name)
                for name in ('result.json','childcontext_ack.json')}
    protocol = SimpleNamespace(CHILD_RUNNER='proxdc_runner.py',CONTEXT_ACK=True,
        code_hashes=bound_hashes, snapshot_invariants=lambda: invariant,
        parameters=lambda args: {'route': route, 'stage': 'tiny'},
        validate_stage_gates=lambda *a: {'control_bindings': {}},
        validate_child_artifacts=artifacts)
    observed_pids=[]
    def cards():
        ready = tmp_path / 'outputs/route_cpu/childcontext_ready.json'
        pids = [json.loads(ready.read_text())['pid']] if ready.exists() else []
        if pids:
            assert not (ready.parent/'childcontext_ack.json').exists()
            if context_visible:observed_pids.extend(pids)
        return [dict(uuid='GPU-cpu', gpu=0, free_mib=80*1024, compute_pids=pids if context_visible else [])]
    monkeypatch.setattr(lifecycle, 'ROOT', tmp_path)
    monkeypatch.setattr(lifecycle, 'PYTHON', Path(sys.executable))
    monkeypatch.setattr(lifecycle, 'load_campaign', lambda: SimpleNamespace(
        gpu_snapshot=cards, select_gpu=lambda cards, *args: dict(cards[0])))
    original_lock = lifecycle.claim_lock
    monkeypatch.setattr(lifecycle, 'claim_lock', lambda: original_lock(tmp_path / 'claim.lock'))
    def wrong_route(*args):
        pytest.fail('original amplitude protocol was called')
    for name in ('code_hashes', 'snapshot_invariants', 'validate_stage_gates', 'parameters',
                 'validate_child_artifacts'):
        monkeypatch.setattr(lifecycle, name, wrong_route)
    if not context_visible:
        with pytest.raises(TimeoutError):lifecycle.execute(args,protocol=protocol)
        directory=tmp_path/'outputs/route_cpu'
        assert not (directory/'childcontext_ack.json').exists()
        assert not (directory/'result.json').exists()
        pid=lifecycle.read(directory/'child_started.json')['pid']
        assert not Path(f'/proc/{pid}').exists()
        with original_lock(tmp_path/'claim.lock'):pass
        return
    completed = lifecycle.read(lifecycle.execute(args, protocol=protocol))
    assert completed['route'] == route
    assert completed['parameters'] == {'route': route, 'stage': 'tiny'}
    assert len(calls) == 3
    started = lifecycle.read(tmp_path / 'outputs/route_cpu/child_started.json')
    assert started['command'][-3] == str(child)
    assert completed['worker_pid'] == lifecycle.read(tmp_path / 'outputs/route_cpu/result.json')['pid']
    assert observed_pids == [completed['worker_pid']]
    assert not Path(f"/proc/{completed['worker_pid']}").exists()


@pytest.mark.parametrize('change',[dict(child_pid=-1),dict(parent_pid=-1),dict(gpu_uuid='GPU-wrong'),
                                 dict(context_observed=False)])
def test_child_rejects_ack_with_wrong_process_device_or_observation(tmp_path,change):
    import proxdc_runner as child
    launch=dict(parent_pid=os.getppid(),gpu_uuid='GPU-cpu')
    ack=dict(schema_version=1,child_pid=os.getpid(),parent_pid=os.getppid(),gpu_uuid='GPU-cpu',context_observed=True)
    ack.update(change);(tmp_path/'childcontext_ack.json').write_text(json.dumps(ack))
    with pytest.raises(ValueError):child.wait_context_ack(tmp_path,launch,timeout=.1)


def test_child_ack_wait_is_bounded_and_refuses_parent_loss(tmp_path,monkeypatch):
    import proxdc_runner as child
    launch=dict(parent_pid=os.getppid(),gpu_uuid='GPU-cpu')
    with pytest.raises(TimeoutError):child.wait_context_ack(tmp_path,launch,timeout=.01)
    monkeypatch.setattr(child.os,'getppid',lambda:-1)
    with pytest.raises(ValueError,match='parent'):child.wait_context_ack(tmp_path,launch,timeout=.1)
