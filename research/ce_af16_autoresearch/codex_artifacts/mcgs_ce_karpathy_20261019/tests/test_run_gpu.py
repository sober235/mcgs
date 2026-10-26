"""Gate and scheduling checks: no CUDA initialization or real child launch."""
import argparse
import copy
import json
import sys
from pathlib import Path

import pytest

import run_gpu as supervisor


def save(path, value):
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def setup(tmp_path):
    hashes = {"source.py": "a" * 64}
    invariants = {"fixed": {"input": "b" * 64}}
    report = tmp_path / "review.md"
    report.write_text("0 actionable findings. Independent review of the complete implementation.")
    review = save(tmp_path / "review.json", {
        "schema_version": 1, "status": "passed", "passed": True, "round": 1,
        "actionable_findings": 0, "unresolved_findings": 0,
        "review_report": supervisor.descriptor(report),
        "code_sha256": hashes, "invariants": invariants,
    })
    args = argparse.Namespace(stage="zero", mode="baseline", steps=0, exp="cpu_test",
        review_gate=review, recipe=tmp_path / "recipe.json", zero_gate=None,
        smoke_gate=None, preferred_gpu=0, bands=4, width=64, gain_bound=0.25,
        lr=0.001, timeout=3600)
    save(args.recipe, {"schema_version": 1, "status": "selected",
        "parameters": supervisor.parameters(args), "code_sha256": hashes,
        "invariants": invariants})
    return args, hashes, invariants


def observed_gate(tmp_path, stage, mode, hashes, invariants):
    directory = tmp_path / f"{stage}_{mode}"
    directory.mkdir()
    steps = 0 if stage == "zero" else 2
    params = dict(stage=stage, mode=mode, steps=steps, bands=4, width=64,
                  gain_bound=0.25, lr=0.001)
    launch = {"run_dir": str(directory), **params, "code_sha256": hashes,
              "invariants": invariants}
    save(directory / "launch.json", launch)
    metrics = {"status": "complete", **params, "code_sha256": hashes,
        "invariants": invariants, "steps_observed": steps, "gradients_finite": True,
        "field_metadata": {"mode": mode},
        "zero_parity": {"pre": {"passed": True}, "post": {"passed": True}}}
    save(directory / "metrics.json", metrics)
    save(directory / "provenance.json", copy.deepcopy(metrics))
    (directory / "candidate_pre_complex.npy").write_bytes(b"saved pre")
    (directory / "candidate_post_complex.npy").write_bytes(b"saved post")
    (directory / "step_trace.jsonl").write_text("" if steps == 0 else
        "\n".join(json.dumps({"step": i, "gradients_finite": True, "loss": 1.0})
                  for i in range(1, steps + 1)) + "\n")
    args = argparse.Namespace(**params)
    artifacts = supervisor.validate_child_artifacts(args, directory, hashes, invariants)
    expected = {"code_sha256": hashes, "invariants": invariants}
    control = directory / "reviewed_control.txt"
    control.write_text("Frozen review and recipe bindings for the observed run.")
    bindings = {str(control): supervisor.sha256(control)}
    proof = save(directory / "post_run_invariants.json", {
        "status": "verified", "before": expected, "after": expected,
        "control_bindings_before": bindings, "control_bindings_after": bindings})
    gate = save(directory / "completed.json", {
        "schema_version": 1, "passed": True, "status": "complete", **params,
        "parameters": params, "run_dir": str(directory), "code_sha256": hashes,
        "invariants": invariants, "artifacts": artifacts,
        "post_run_invariants": supervisor.descriptor(proof), "control_bindings": bindings})
    return gate, directory


def select_recipe(args, hashes, invariants):
    save(args.recipe, {"schema_version": 1, "status": "selected",
        "parameters": supervisor.parameters(args), "code_sha256": hashes,
        "invariants": invariants})


def test_reviewed_zero_setup_is_accepted(setup):
    args, hashes, invariants = setup
    result = supervisor.validate_stage_gates(args, hashes, invariants)
    assert str(args.review_gate.resolve()) in result["control_bindings"]
    assert str(args.recipe.resolve()) in result["control_bindings"]


@pytest.mark.parametrize("mutation", [
    lambda v: v.update(passed=False),
    lambda v: v.update(status="pending"),
    lambda v: v.update(unresolved_findings=1),
    lambda v: v.update(code_sha256={"stale": "c" * 64}),
    lambda v: v.update(invariants={"changed": True}),
    lambda v: v.update(review_report={"path": "/missing", "sha256": "d" * 64}),
])
def test_bad_review_gate_is_rejected(setup, mutation):
    args, hashes, invariants = setup
    value = supervisor.read(args.review_gate)
    mutation(value)
    save(args.review_gate, value)
    with pytest.raises((ValueError, OSError)):
        supervisor.validate_stage_gates(args, hashes, invariants)


@pytest.mark.parametrize("name,value", [("mode", "shared"), ("stage", "smoke"),
    ("steps", 2), ("bands", 3), ("width", 32), ("gain_bound", 0.5), ("lr", 0.002)])
def test_recipe_cannot_redefine_actual_cli_parameters(setup, name, value):
    args, hashes, invariants = setup
    recipe = supervisor.read(args.recipe)
    recipe["parameters"][name] = value
    save(args.recipe, recipe)
    with pytest.raises(ValueError, match="recipe"):
        supervisor.validate_stage_gates(args, hashes, invariants)


@pytest.mark.parametrize("change", [dict(stage="smoke", mode="baseline", steps=2),
    dict(stage="zero", steps=2), dict(stage="trial", mode="shared", steps=3),
    dict(width=0), dict(bands=-1), dict(gain_bound=float("nan")), dict(lr=-1),
    dict(exp="../existing"), dict(timeout=0), dict(preferred_gpu=-1)])
def test_invalid_modes_and_parameters_are_rejected(setup, change):
    args, hashes, invariants = setup
    for key, value in change.items():
        setattr(args, key, value)
    with pytest.raises(ValueError):
        supervisor.validate_stage_gates(args, hashes, invariants)


def test_trial_requires_matched_real_smoke_and_current_zero(tmp_path, setup):
    args, hashes, invariants = setup
    args.stage, args.mode, args.steps = "trial", "shared", 200
    args.zero_gate, _ = observed_gate(tmp_path, "zero", "baseline", hashes, invariants)
    args.smoke_gate, _ = observed_gate(tmp_path, "smoke", "shared", hashes, invariants)
    select_recipe(args, hashes, invariants)
    assert supervisor.validate_stage_gates(args, hashes, invariants)["smoke"]["passed"]
    args.mode = "explicit"
    select_recipe(args, hashes, invariants)
    with pytest.raises(ValueError, match="matched"):
        supervisor.validate_stage_gates(args, hashes, invariants)


def test_false_actual_zero_parity_is_rejected_even_if_gate_claims_passed(tmp_path, setup):
    args, hashes, invariants = setup
    args.stage, args.mode, args.steps = "smoke", "shared", 2
    args.zero_gate, directory = observed_gate(tmp_path, "zero", "baseline", hashes, invariants)
    metrics = supervisor.read(directory / "metrics.json")
    metrics["zero_parity"]["post"]["passed"] = False
    save(directory / "metrics.json", metrics)
    gate = supervisor.read(args.zero_gate)
    gate["artifacts"][str(directory / "metrics.json")] = supervisor.sha256(directory / "metrics.json")
    save(args.zero_gate, gate)
    select_recipe(args, hashes, invariants)
    with pytest.raises(ValueError, match="parity"):
        supervisor.validate_stage_gates(args, hashes, invariants)


def test_gate_dereferences_artifacts_and_source_proof(tmp_path, setup):
    args, hashes, invariants = setup
    gate, directory = observed_gate(tmp_path, "zero", "baseline", hashes, invariants)
    (directory / "candidate_pre_complex.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact"):
        supervisor.validate_observed_gate(gate, "zero", hashes, invariants)


def test_nonfinite_or_missing_actual_smoke_gradients_are_rejected(tmp_path, setup):
    args, hashes, invariants = setup
    _, directory = observed_gate(tmp_path, "smoke", "shared", hashes, invariants)
    trace = directory / "step_trace.jsonl"
    trace.write_text('{"step":1,"gradients_finite":true,"loss":1}\n'
                     '{"step":2,"gradients_finite":false,"loss":1}\n')
    args.stage, args.mode, args.steps = "smoke", "shared", 2
    with pytest.raises(ValueError, match="gradient"):
        supervisor.validate_child_artifacts(args, directory, hashes, invariants)


def test_gpu_selection_preserves_visibility_and_requires_65gib():
    campaign = supervisor.load_campaign()
    cards = [dict(gpu=i, uuid=f"GPU-{i}", used_mib=0.0, total_mib=80*1024,
        free_mib=80*1024, utilization=0, compute_pids=[], loading_claim_pids=[], free=True)
        for i in range(8)]
    cards[0]["free_mib"] = 64*1024
    cards[0]["used_mib"] = 16*1024
    for index in (4, 6):
        cards[index]["free_mib"] = 10*1024
        cards[index]["used_mib"] = 70*1024
    assert supervisor.select_gpu(cards, 0, {}, campaign)["gpu"] == 1
    assert supervisor.select_gpu(cards, 4, {"CUDA_VISIBLE_DEVICES": "4,6,7"}, campaign)["gpu"] == 7
    with pytest.raises(ValueError):
        supervisor.select_gpu(cards, 4, {"CUDA_VISIBLE_DEVICES": "4,6"}, campaign)


def test_import_and_help_remain_stdlib_only():
    import subprocess
    command = [sys.executable, "-B", "-c",
        "import run_gpu,sys; assert 'torch' not in sys.modules; run_gpu.parser().parse_args(['--help'])"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0
    assert "--review-gate" in result.stdout


def test_immutable_evidence_publication_refuses_to_overwrite(tmp_path):
    path = tmp_path / "proof.json"
    supervisor.write_new(path, {"status": "original"})
    with pytest.raises(FileExistsError):
        supervisor.write_new(path, {"status": "replacement"})
    assert supervisor.read(path) == {"status": "original"}


def test_invalid_nonfinite_request_still_leaves_failure_evidence(tmp_path, monkeypatch, setup):
    args, _, _ = setup
    args.gain_bound = float("nan")
    monkeypatch.setattr(supervisor, "ROOT", tmp_path)
    path = supervisor.record_failure(args, ValueError("invalid gain_bound"))
    failure = supervisor.read(path)
    assert failure["status"] == "refused_setup"
    assert failure["scientific_rejection"] is False
    assert failure["request"]["gain_bound"] == "nan"


def test_context_claim_requires_correct_child_uuid_and_visible_pid(tmp_path):
    import time
    from types import SimpleNamespace
    worker = SimpleNamespace(pid=12345, poll=lambda: None)
    selected = {"uuid": "GPU-allocated"}
    campaign = SimpleNamespace(gpu_snapshot=lambda: [
        {"uuid": "GPU-allocated", "compute_pids": [12345]}])
    save(tmp_path / "childcontext_ready.json", {"pid": 12345, "gpu_uuid": "GPU-wrong"})
    with pytest.raises(ValueError, match="handshake"):
        supervisor.wait_context(worker, tmp_path, selected, campaign, time.monotonic() + 1)
    save(tmp_path / "childcontext_ready.json", {"pid": 12345, "gpu_uuid": "GPU-allocated"})
    assert supervisor.wait_context(worker, tmp_path, selected, campaign,
                                   time.monotonic() + 1)["pid"] == worker.pid
    with pytest.raises(TimeoutError):
        supervisor.wait_context(worker, tmp_path, selected, campaign, time.monotonic() - 1)


def test_timeout_cleanup_signals_only_its_own_subprocess_group():
    import subprocess
    command = [sys.executable, "-B", "-c", "import time; time.sleep(30)"]
    worker = subprocess.Popen(command, start_new_session=True)
    unrelated = subprocess.Popen(command, start_new_session=True)
    try:
        supervisor.terminate_owned(worker)
        assert worker.poll() is not None
        assert unrelated.poll() is None
    finally:
        supervisor.terminate_owned(worker)
        supervisor.terminate_owned(unrelated)


def cpu_supervisor_process(tmp_path, *, timeout=60, ignore_term=False, signal_after_spawn=False):
    """Run real supervisor/process/lock code with CPU-only acquisition fixtures."""
    import os
    import subprocess
    root = tmp_path / "supervised"
    root.mkdir()
    (root / "recipe.json").write_text("{}")
    (root / "review.json").write_text("{}")
    if ignore_term:
        (root / "ignore_term").touch()
    if signal_after_spawn:
        (root / "signal_after_spawn").touch()
    (root / "runner.py").write_text('''import argparse,json,os,pathlib,signal,time
p=argparse.ArgumentParser();p.add_argument('--launch');a=p.parse_args()
v=json.loads(pathlib.Path(a.launch).read_text());r=pathlib.Path(v['run_dir'])
if (r.parent.parent/'ignore_term').exists():
    signal.signal(signal.SIGTERM,lambda *a:(r/'term_seen').write_text('received'))
(r/'childcontext_ready.json').write_text(json.dumps({'pid':os.getpid(),'gpu_uuid':v['gpu_uuid']}))
time.sleep(60)
''')
    harness = tmp_path / "cpu_supervisor.py"
    harness.write_text('''import json,os,pathlib,signal,sys
sys.path.insert(0,sys.argv[1])
import run_gpu as s
r=pathlib.Path(sys.argv[2]);s.ROOT=r;s.PYTHON=pathlib.Path(sys.executable)
s.code_hashes=lambda:{'cpu_fixture':'bound'}
s.snapshot_invariants=lambda:{'cpu_fixture':'unchanged'}
s.validate_stage_gates=lambda *a:{'control_bindings':{}}
original_campaign=s.load_campaign()
class CPUCampaign:
    def gpu_snapshot(self):
        p=r/'outputs/cpu_case/child_started.json'
        pids=[json.loads(p.read_text())['pid']] if p.exists() else []
        return [dict(gpu=0,uuid='GPU-cpu-fixture',used_mib=1 if pids else 0,total_mib=80*1024,
            free_mib=80*1024-1,utilization=0,compute_pids=pids,loading_claim_pids=[],free=not pids)]
    def select_gpu(self,*a):return original_campaign.select_gpu(*a)
s.load_campaign=lambda:CPUCampaign()
lock=s.claim_lock;s.claim_lock=lambda:lock(r/'claim.lock')
original_popen=s.subprocess.Popen
class ObservedWorker:
    def __init__(self,*a,**k):
        self.worker=original_popen(*a,**k)
        (r/'spawned_pid').write_text(str(self.worker.pid))
        if (r/'signal_after_spawn').exists():os.kill(os.getpid(),signal.SIGTERM)
    def __getattr__(self,k):return getattr(self.worker,k)
    def wait(self,timeout=None):
        if timeout is not None and (timeout>5 or timeout<=.25):(r/'waiting_after_context').write_text('waiting')
        return self.worker.wait(timeout=timeout)
s.subprocess.Popen=ObservedWorker
raise SystemExit(s.main(['--stage','zero','--mode','baseline','--steps','0','--exp','cpu_case',
    '--review-gate',str(r/'review.json'),'--recipe',str(r/'recipe.json'),'--timeout',sys.argv[3]]))
''')
    process = subprocess.Popen([sys.executable, "-B", str(harness), str(supervisor.ROOT), str(root), str(timeout)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
    return process, root


def wait_cpu_file(path, process, seconds=5):
    import time
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if path.exists():
            return
        assert process.poll() is None, "CPU supervisor exited before the expected lifecycle event"
        time.sleep(0.01)
    pytest.fail(f"CPU fixture did not create {path}")


def cleanup_cpu_supervisor(process, root):
    import os
    import signal
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except __import__("subprocess").TimeoutExpired:
            process.kill()
            process.wait()
    started = root / "outputs/cpu_case/child_started.json"
    spawned = root / "spawned_pid"
    if started.exists() or spawned.exists():
        child_pid = supervisor.read(started)["pid"] if started.exists() else int(spawned.read_text())
        try:
            os.killpg(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_successful_tiny_context_keeps_common_claim_until_numeric_child_exits(tmp_path):
    import fcntl
    process, root = cpu_supervisor_process(tmp_path)
    try:
        wait_cpu_file(root / "waiting_after_context", process)
        # The simulated GPU uses only1MiB, so the legacy cohosting selector
        # would still consider it eligible. The common claim must block a
        # second cooperating launcher for the entire numerical lifetime.
        with (root / "claim.lock").open("r+") as stream:
            with pytest.raises(BlockingIOError):
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        cleanup_cpu_supervisor(process, root)


def test_parent_sigterm_reaps_owned_numeric_group_and_records_cancellation(tmp_path):
    import subprocess
    process, root = cpu_supervisor_process(tmp_path)
    unrelated = subprocess.Popen([sys.executable, "-B", "-c", "import time;time.sleep(30)"],
                                 start_new_session=True)
    try:
        wait_cpu_file(root / "waiting_after_context", process)
        child_pid = supervisor.read(root / "outputs/cpu_case/child_started.json")["pid"]
        process.terminate()
        process.wait(timeout=5)
        assert not Path(f"/proc/{child_pid}").exists(), "numeric child survived supervisor SIGTERM"
        assert unrelated.poll() is None
        failures = list((root / "verification").glob("failure_*.json"))
        assert len(failures) == 1
        failure = supervisor.read(failures[0])
        assert failure["status"] == "cancelled"
        assert failure["scientific_rejection"] is False
        assert failure["signal"] == __import__("signal").SIGTERM
        assert failure["gpu_claim_policy"] == "common_lock_until_child_exit_serialized"
        assert process.returncode == 128 + __import__("signal").SIGTERM
        # Cancellation releases the claim only after the owned child is reaped.
        import fcntl
        with (root / "claim.lock").open("r+") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        cleanup_cpu_supervisor(process, root)
        supervisor.terminate_owned(unrelated)


def test_first_sigterm_during_timeout_cleanup_keeps_claim_until_reap(tmp_path):
    import fcntl
    import time
    process, root = cpu_supervisor_process(tmp_path, timeout=1, ignore_term=True)
    try:
        run = root / "outputs/cpu_case"
        wait_cpu_file(run / "term_seen", process)
        child_pid = supervisor.read(run / "child_started.json")["pid"]
        process.terminate()
        # The child deliberately ignores TERM. The first parent cancellation
        # must not interrupt its5-second grace wait or unwind the common claim.
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            assert process.poll() is None
            assert Path(f"/proc/{child_pid}").exists()
            with (root / "claim.lock").open("r+") as stream:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            time.sleep(0.01)
        process.wait(timeout=7)
        assert not Path(f"/proc/{child_pid}").exists()
        failure = supervisor.read(next((root / "verification").glob("failure_*.json")))
        assert failure["status"] == "cancelled"
        assert failure["scientific_rejection"] is False
    finally:
        cleanup_cpu_supervisor(process, root)


def test_sigterm_after_spawn_before_worker_assignment_still_reaps_child(tmp_path):
    process, root = cpu_supervisor_process(tmp_path, signal_after_spawn=True)
    try:
        wait_cpu_file(root / "spawned_pid", process)
        child_pid = int((root / "spawned_pid").read_text())
        process.wait(timeout=5)
        assert not Path(f"/proc/{child_pid}").exists(), "spawn-window cancellation orphaned its child"
        failure = supervisor.read(next((root / "verification").glob("failure_*.json")))
        assert failure["status"] == "cancelled"
        assert failure["worker_pid"] == child_pid
        assert failure["scientific_rejection"] is False
    finally:
        cleanup_cpu_supervisor(process, root)


def test_snapshot_hashes_every_external_numerical_adapter(tmp_path, monkeypatch):
    """An adapter edit must invalidate review even if the legacy freeze omits it."""
    old = tmp_path / "old"
    base = old / "nonlocal_autoresearch"
    g2 = old / "frequency_adam/outputs/g2"
    g2.mkdir(parents=True)
    base.mkdir(parents=True)
    contract = old / "metric_contract.json"
    contract.write_text("three metric contract")
    summary = old / "summary.json"
    summary.write_text("accepted summary")
    legacy = old / "frequency_adam/run_gpu.py"
    legacy.write_text("def snapshot_invariants(): return {'legacy_verified': True}\n")
    for path in (old / "frequency_adam/runner.py", old / "frequency_adam/frequency_weights.py",
                 base / "runner.py", base / "autoresearch.py", base / "objective.py"):
        path.write_text("# frozen dependency\n")
    checkpoint = g2 / "checkpoint.pth"
    checkpoint.write_bytes(b"accepted G2 checkpoint")
    save(g2 / "metrics.json", {"end_iteration": 4500, "final_checkpoint": str(checkpoint)})
    save(g2 / "provenance.json", {"status": "complete"})
    for name in ("actual_config.yaml", "candidate_pre_complex.npy", "candidate_post_complex.npy",
                 "frequency_provenance.json"):
        (g2 / name).write_text("frozen baseline file")
    frequency = g2 / "frequency_weights"
    frequency.mkdir()
    (frequency / "manifest.json").write_text("frozen frequency manifest")
    for scan in ("pre", "post"):
        for kind in ("source_weight", "effective_weight", "acquired_mask"):
            (frequency / f"{scan}_{kind}.npy").write_bytes(b"frozen array")
    reference = old / "reference.bin"
    reference.write_bytes(b"frozen reference")
    input_freeze = old / "input.json"
    input_freeze.write_text("frozen original input definition")
    reference_freeze = save(old / "reference_freeze.json", {
        "files": {str(reference): supervisor.sha256(reference)},
        "loop_input_freeze_path": str(input_freeze),
        "loop_input_freeze_sha256": supervisor.sha256(input_freeze)})
    replacements = dict(OLD=old, BASE=base, G2=g2, CONTRACT=contract, SUMMARY=summary,
        LEGACY_SUPERVISOR=legacy, REFERENCE_FREEZE=reference_freeze,
        G2_SHA256=supervisor.sha256(checkpoint), CONTRACT_SHA256=supervisor.sha256(contract))
    for key, value in replacements.items():
        monkeypatch.setattr(supervisor, key, value)
    before = supervisor.snapshot_invariants()
    for adapter in (old / "frequency_adam/runner.py", old / "frequency_adam/frequency_weights.py",
                    base / "runner.py"):
        assert before["g2_and_contract_files"][str(adapter)] == supervisor.sha256(adapter)
    adapter.write_text("# modified numerical adapter\n")
    assert supervisor.snapshot_invariants() != before
