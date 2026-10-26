"""Exercise durable ledger and failure/resume behavior without CUDA."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("loop_under_test", ROOT / "autoresearch.py")
if spec is not None and spec.loader is not None and (ROOT / "autoresearch.py").exists():
    import sys
    sys.path.insert(0, str(ROOT))
    loop = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loop)
else:
    loop = None


def ledger(tmp_path):
    assert loop is not None, "loop implementation is missing"
    return loop.Ledger(tmp_path / "results.jsonl")


def test_ledger_keeps_failed_and_rejected_runs_and_resumes(tmp_path):
    book = ledger(tmp_path)
    book.append({"event": "trial_started", "exp": "B", "candidate": {"local_factor": .5, "nlambda": 0., "sigma_rel": .015}})
    book.append({"event": "trial_failed", "exp": "B", "error": "runner exited 1"})
    book.append({"event": "trial_scored", "exp": "C", "score": -.2, "accepted": False})
    resumed = loop.Ledger(book.path)
    assert resumed.attempted() == {"B", "C"}
    assert len(resumed.records) == 3
    assert resumed.records[-1]["accepted"] is False
    assert json.loads(book.path.read_text().splitlines()[1])["error"] == "runner exited 1"


def test_incomplete_ledger_tail_is_error_not_silent_loss(tmp_path):
    book = ledger(tmp_path)
    book.append({"event": "trial_failed", "exp": "B"})
    with book.path.open("a") as stream:
        stream.write('{"event":')
    with pytest.raises(ValueError, match="ledger"):
        loop.Ledger(book.path)


def test_failed_subprocess_is_recorded_and_never_overwrites(tmp_path):
    book = ledger(tmp_path)
    candidate = {"name": "B", "local_factor": .5, "nlambda": 0., "sigma_rel": .015}
    run_dir = tmp_path / "outputs" / "B"
    import sys
    result = loop.execute_trial(book, candidate, run_dir, [sys.executable, "-c", "raise SystemExit(7)"], {}, lambda p, c: None)
    assert result is False
    assert book.records[-1]["event"] == "trial_failed"
    assert book.records[-1]["returncode"] == 7
    assert run_dir.exists() is False  # Runner alone reserves training output.
    with pytest.raises(ValueError, match="already"):
        loop.execute_trial(book, candidate, run_dir, [sys.executable, "-c", "pass"], {}, lambda p, c: None)


def test_acceptance_requires_more_than_threshold():
    assert loop is not None, "loop implementation is missing"
    assert loop.is_improvement(.0001, 0) is False
    assert loop.is_improvement(.00010001, 0) is True
    assert loop.is_improvement(-.2, -.1) is False


def test_quiet_guard_rejects_short_stale_or_gapped_history():
    assert loop is not None, "loop implementation is missing"
    rows = [{"timestamp": t, "gpu": 3, "free": True} for t in range(0, 601, 30)]
    assert loop.quiet_ready(rows, 3, 600)
    assert not loop.quiet_ready(rows[1:], 3, 600)
    assert not loop.quiet_ready(rows, 3, 700)
    assert not loop.quiet_ready([rows[0], rows[-1]], 3, 600)
    rows[10]["free"] = False
    assert not loop.quiet_ready(rows, 3, 600)


def run_fixture(tmp_path):
    """A completed runner's public JSON boundary, including observed NLR use."""
    assert loop is not None
    run = tmp_path / "run"
    run.mkdir()
    checkpoint = tmp_path / "checkpoint.pth"
    config = tmp_path / "source.yaml"
    config.write_text("loss: {tv_weight: 0.1, lambda_wav: 0.1, lambda_wav_plain: 0.12}\n")
    candidate = {"name": "C", "local_factor": 1., "nlambda": .01, "sigma_rel": .015}
    overrides = {key: candidate[key] for key in ("local_factor", "nlambda", "sigma_rel")}
    nlr = {"enabled": True, "validated_active": True, "all_gradients_finite": True,
           "start": 4001, "refresh_every": 250, "sigma_rel": .015, "denoiser": "patch_lowrank",
           "expected_refresh_iterations": [4001, 4251], "coefficients": {"pre": .01, "post": .01},
           "active_step_count": 500, "nonzero_weighted_step_count": 500,
           "weighted_R_nl_min": .00001, "weighted_R_nl_max": .001, "weighted_R_nl_last": .0001,
           "raw_R_nl_by_channel_max": {"pre": .1, "post": .1},
           "raw_R_nl_by_channel_last": {"pre": .01, "post": .01},
           "refresh_history": {s: [{"iter": i, "target_rms_difference": .1} for i in (4001, 4251)] for s in ("pre", "post")}}
    rows = {s: {"head_psnr": 35., "head_ssim": .96, "head_nmse": .01, "head_hfen": .1} for s in ("pre", "post")}
    metrics = {"mode": "all", "source_iteration": 4000, "end_iteration": 4500,
               "source_checkpoint_sha256": "frozen", "forward_backward_evaluations": 500,
               "rows": rows, "nlr": nlr, "overrides": overrides}
    provenance = {"status": "complete", "source_iteration": 4000, "end_iteration": 4500,
                  "source_checkpoint": str(checkpoint), "source_checkpoint_sha256": "frozen",
                  "overrides": overrides, "nlr": nlr}
    actual = {"loss": {"tv_weight": .1, "lambda_wav": .1, "lambda_wav_plain": .12,
                       "lambda_nl": .01, "lambda_nl_plain": .01, "lambda_nl_contrast": .01,
                       "nl_start": 4001, "nl_refresh": 250, "nl_sigma_rel": .015,
                       "nl_denoiser": "patch_lowrank"},
              "training": {"resume_offset": 4000, "max_iterations": 4500}}
    import yaml
    (run / "actual_config.yaml").write_text(yaml.safe_dump(actual))
    for name, value in (("metrics.json", metrics), ("provenance.json", provenance)):
        (run / name).write_text(json.dumps(value))
    for scan in ("pre", "post"):
        (run / f"candidate_{scan}_complex.npy").touch()
    manifest = {"source_checkpoint": str(checkpoint), "source_config": str(config), "files": {str(checkpoint): "frozen"}}
    return run, candidate, manifest, metrics, provenance


def test_observed_active_completed_run_is_validated(tmp_path):
    run, c, manifest, _, _ = run_fixture(tmp_path)
    head, provenance = loop.validate_run(run, c, manifest)
    assert set(head) == {"pre", "post"}
    assert provenance["source_checkpoint_sha256"] == "frozen"


@pytest.mark.parametrize("mutation", ["checkpoint", "steps", "local", "sigma", "refresh", "identity", "inactive", "nonfinite", "raw_inactive", "config"])
def test_run_cannot_be_mislabeled_or_pass_without_observed_prior(tmp_path, mutation):
    run, c, manifest, metrics, provenance = run_fixture(tmp_path)
    if mutation == "checkpoint":
        provenance["source_checkpoint"] = "/promoted_weights/latest.pth"
    elif mutation == "steps":
        metrics["forward_backward_evaluations"] = 499
    elif mutation == "local":
        c["local_factor"] = .5
    elif mutation == "sigma":
        metrics["nlr"]["sigma_rel"] = .03
    elif mutation == "refresh":
        metrics["nlr"]["refresh_history"]["pre"].pop()
    elif mutation == "identity":
        metrics["nlr"]["refresh_history"]["post"][0]["target_rms_difference"] = 0
    elif mutation == "inactive":
        metrics["nlr"]["weighted_R_nl_max"] = 0
    elif mutation == "nonfinite":
        metrics["nlr"]["weighted_R_nl_max"] = float("nan")
    elif mutation == "raw_inactive":
        metrics["nlr"]["raw_R_nl_by_channel_max"]["post"] = 0
    else:
        import yaml
        path = run / "actual_config.yaml"
        actual = yaml.safe_load(path.read_text())
        actual["loss"]["lambda_wav"] = .05
        path.write_text(yaml.safe_dump(actual))
    (run / "metrics.json").write_text(json.dumps(metrics))
    (run / "provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError):
        loop.validate_run(run, c, manifest)


def test_adaptive_candidates_follow_measured_best_and_freeze_parent(tmp_path):
    book = ledger(tmp_path)
    grid = json.loads((ROOT / "candidate_grid.json").read_text())
    for c in grid["initial"]:
        book.append({"event": "trial_started", "exp": c["name"], "candidate": c})
        book.append({"event": "trial_scored", "exp": c["name"], "candidate": c,
                     "score": -.1 if c["name"] == "D" else -.2})
    c = loop.next_candidate(grid, book)
    assert c["name"] == "E"
    assert c["local_factor"] == .5
    assert c["nlambda"] == pytest.approx(grid["initial"][2]["nlambda"] / 3)
    book.append({"event": "trial_started", "exp": "E", "candidate": c})
    book.append({"event": "trial_scored", "exp": "E", "candidate": c, "score": 0.1})
    assert loop.next_candidate(grid, book)["nlambda"] == pytest.approx(grid["initial"][2]["nlambda"] * 3)


def test_actual_smokes_validate_but_bare_or_stale_gate_does_not(tmp_path):
    manifest = json.loads((ROOT / "verification/loop_input_freeze.json").read_text())
    parity = ROOT / "verification/parity_gate_v2.json"
    active = ROOT / "verification/active_gate_v2.json"
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"passed": True, "round": 2, "actionable_findings": 0,
                                 "code_sha256": {str(p): loop.sha256_file(p) for p in loop.review_code_paths()}}))
    loop.validate_training_gates(parity, active, review, manifest)
    failed = tmp_path / "failed.json"
    failed.write_text(json.dumps({"passed": True}))
    with pytest.raises(ValueError):
        loop.validate_training_gates(parity, failed, review, manifest)
    stale = json.loads(active.read_text())
    stale["code_sha256"][str(ROOT / "runner.py")] = "stale"
    failed.write_text(json.dumps(stale))
    with pytest.raises(ValueError, match="hash"):
        loop.validate_training_gates(parity, failed, review, manifest)


@pytest.mark.parametrize("gate", ["missing", "failed", "stale_review"])
def test_direct_training_rejects_gates_before_resource_probe_or_dispatch(tmp_path, monkeypatch, gate):
    import sys
    manifest = json.loads((ROOT / "verification/loop_input_freeze.json").read_text())
    monkeypatch.setattr(loop, "VERIFY", tmp_path)
    monkeypatch.setattr(loop, "input_freeze", lambda: manifest)
    def forbidden(*args, **kwargs):
        pytest.fail("resource probe/training reached before valid gates")
    monkeypatch.setattr(loop, "wait_for_gpu", forbidden)
    monkeypatch.setattr(loop, "execute_trial", forbidden)
    active = ROOT / "verification/active_gate_v2.json"
    if gate == "failed":
        active = tmp_path / "failed.json"
        active.write_text('{"passed":false}')
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"passed": True, "round": 2, "actionable_findings": 0,
                                 "code_sha256": {str(p): loop.sha256_file(p) for p in loop.review_code_paths()}}))
    if gate == "stale_review":
        d = json.loads(review.read_text())
        d["code_sha256"][str(ROOT / "objective.py")] = "stale"
        review.write_text(json.dumps(d))
    argv = ["autoresearch.py", "--gpu", "2", "--max-trials", "1", "--active-smoke", str(active), "--review-gate", str(review)]
    if gate != "missing":
        argv += ["--control-parity", str(ROOT / "verification/parity_gate_v2.json")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises((ValueError, SystemExit, FileNotFoundError)):
        loop.main()


def test_valid_gates_allow_controlled_dispatch_without_quiet_wait(tmp_path, monkeypatch):
    import sys
    manifest = json.loads((ROOT / "verification/loop_input_freeze.json").read_text())
    monkeypatch.setattr(loop, "VERIFY", tmp_path)
    monkeypatch.setattr(loop, "input_freeze", lambda: manifest)
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"passed": True, "round": 2, "actionable_findings": 0,
                                 "code_sha256": {str(p): loop.sha256_file(p) for p in loop.review_code_paths()}}))
    monkeypatch.setattr(loop, "score_run", lambda *a, **k: {"score": -.1, "accepted": False, "strict_success": False})
    monkeypatch.setattr(loop, "wait_for_gpu", lambda *a: {"gpu": 5, "timestamp": 123., "free": True})
    calls = []
    def dispatch(book, candidate, run_dir, command, environment, callback):
        calls.append((command, environment))
        book.append({"event": "trial_failed", "exp": candidate["name"], "stage": "training"})
        return False
    monkeypatch.setattr(loop, "execute_trial", dispatch)
    monkeypatch.setattr(sys, "argv", ["autoresearch.py", "--gpu", "2", "--max-trials", "1",
                                    "--control-parity", str(ROOT / "verification/parity_gate_v2.json"),
                                    "--active-smoke", str(ROOT / "verification/active_gate_v2.json"),
                                    "--review-gate", str(review)])
    loop.main()
    assert len(calls) == 1
    assert calls[0][0][calls[0][0].index("--gpu") + 1] == "5"
    assert calls[0][1]["CUDA_VISIBLE_DEVICES"] == "5"


def cards():
    return [{"gpu": g, "uuid": f"GPU-abc{g}-long", "used_mib": 10., "total_mib": 81920.,
             "utilization": 0., "compute_pids": [], "loading_claim_pids": []} for g in (2, 5)]


def test_uuid_loading_claim_full_and_unique_prefix_blocks_card():
    snapshot = cards()
    assert loop.resolve_device_tokens("GPU-abc2", snapshot) == {2}
    assert loop.resolve_device_tokens("GPU-abc2-long", snapshot) == {2}
    assert loop.resolve_device_tokens("GPU-abc", snapshot) == set()
    claims = loop.loading_claims(snapshot, [(123, "GPU-abc2")])
    assert claims == {2: [123]}
    snapshot[0]["loading_claim_pids"] = [123]
    selected = loop.select_gpu(snapshot, 2, {})
    assert selected["gpu"] == 5


def test_parent_allocation_is_usable_but_unrelated_loader_still_blocks():
    import os
    snapshot = cards()
    claims = loop.loading_claims(snapshot, [(os.getppid(), "GPU-abc5"), (2147483647, "GPU-abc2")])
    assert claims == {2: [2147483647]}
    snapshot[0]["loading_claim_pids"] = claims[2]
    assert loop.select_gpu(snapshot, 5, {"CUDA_VISIBLE_DEVICES": "5", "SLURM_JOB_GPUS": "5"})["gpu"] == 5
    unrelated = loop.loading_claims(snapshot, [(2147483647, "GPU-abc5-long")])
    snapshot[1]["loading_claim_pids"] = unrelated[5]
    with pytest.raises(ValueError):
        loop.select_gpu(snapshot, 5, {"CUDA_VISIBLE_DEVICES": "5", "SLURM_JOB_GPUS": "5"})


def test_loading_claims_exclude_full_launch_chain_but_preserve_sibling(monkeypatch):
    monkeypatch.setattr(loop.os, "getpid", lambda: 1000)
    monkeypatch.setattr(loop.os, "getppid", lambda: 2000)
    statuses = {"/proc/2000/status": "PPid:\t3000\n", "/proc/3000/status": "PPid:\t1\n", "/proc/1/status": "PPid:\t0\n"}
    original = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda p, *a, **kw: statuses[str(p)] if str(p) in statuses else original(p, *a, **kw))
    processes = [(pid, "GPU-abc5") for pid in (1000, 2000, 3000, 4000)]
    assert loop.loading_claims(cards(), processes) == {5: [4000]}


def test_selection_prefers_idle_then_lightest_sufficient_cohost_and_visibility():
    snapshot = cards()
    snapshot[0].update(used_mib=20000., utilization=30., compute_pids=[321])
    assert loop.select_gpu(snapshot, 2, {})["gpu"] == 5
    assert loop.select_gpu(snapshot, 2, {"CUDA_VISIBLE_DEVICES": "2"})["gpu"] == 2
    snapshot[1].update(used_mib=10000., utilization=10., compute_pids=[555])
    assert loop.select_gpu(snapshot, 2, {})["gpu"] == 5
    assert loop.select_gpu(snapshot, 2, {"SLURM_JOB_GPUS": "2", "CUDA_VISIBLE_DEVICES": "2,5"})["gpu"] == 2
    with pytest.raises(ValueError):
        loop.select_gpu(snapshot, 2, {"CUDA_VISIBLE_DEVICES": ""})
    snapshot[0]["used_mib"] = 40000.
    with pytest.raises(ValueError):
        loop.select_gpu(snapshot, 2, {"CUDA_VISIBLE_DEVICES": "2"})


@pytest.mark.parametrize("visibility", ["2", "GPU-abc2", "GPU-abc2-long"])
def test_snapshot_reads_loading_environment_claim_before_cuda_context(tmp_path, monkeypatch, visibility):
    from types import SimpleNamespace
    path = tmp_path / "123" / "environ"
    path.parent.mkdir()
    path.write_bytes(f"CUDA_VISIBLE_DEVICES={visibility}\0".encode())
    original_glob = Path.glob
    monkeypatch.setattr(Path, "glob", lambda p, pattern: [path] if p == Path("/proc") else original_glob(p, pattern))
    def nvidia(command, **kwargs):
        if command[1].startswith("--query-gpu="):
            return SimpleNamespace(stdout="2, GPU-abc2-long, 10, 81920, 0\n5, GPU-abc5-long, 10, 81920, 0\n")
        return SimpleNamespace(stdout="")
    monkeypatch.setattr(loop.subprocess, "run", nvidia)
    snapshot = loop.gpu_snapshot()
    assert snapshot[0]["loading_claim_pids"] == [123]
    assert snapshot[0]["free"] is False
    assert loop.select_gpu(snapshot, 2, {})["gpu"] == 5


def test_current_resource_selection_persists_snapshot_without_quiet_delay(tmp_path, monkeypatch):
    for key in ("CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "NVIDIA_VISIBLE_DEVICES"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(loop, "gpu_snapshot", cards)
    def forbidden(*args):
        pytest.fail("old quiet wait used under current user policy")
    monkeypatch.setattr(loop, "quiet_ready", forbidden)
    monkeypatch.setattr(loop.time, "sleep", forbidden)
    evidence = tmp_path / "resources.jsonl"
    assert loop.wait_for_gpu(5, evidence)["gpu"] == 5
    saved = json.loads(evidence.read_text())
    assert saved["selection_reason"] == "idle compatible GPU"
    assert len(saved["snapshot"]) == 2


def drift_fixture(tmp_path):
    names = ("positions", "scales", "rotations", "density_real", "density_imag", "density_real_contrast", "density_imag_contrast")
    zero, reference = tmp_path / "zero", tmp_path / "reference"
    for run in (zero, reference):
        run.mkdir()
        checkpoint = run / "final.pth"
        checkpoint.write_bytes(b"final checkpoint evidence")
        (run / "metrics.json").write_text(json.dumps({"final_checkpoint": str(checkpoint)}))
        for scan in ("pre", "post"):
            (run / f"candidate_{scan}_complex.npy").write_bytes(b"candidate evidence")
    row = {"max_absolute": 1e-7, "relative_l2": 1e-8}
    data = {"zero_run_dir": str(zero), "reference_run_dir": str(reference), "passed": True,
            "checkpoint_sha256": {k: loop.sha256_file(r / "final.pth") for k, r in (("zero", zero), ("reference", reference))},
            "candidate_sha256": {k: {s: loop.sha256_file(r / f"candidate_{s}_complex.npy") for s in ("pre", "post")} for k, r in (("zero", zero), ("reference", reference))},
            "model": {n: dict(row) for n in names},
            "adam": {str(i): {s: dict(row) for s in ("exp_avg", "exp_avg_sq")} for i in range(7)},
            "reconstruction": {s: dict(row) for s in ("pre", "post")},
            "tolerances": {"model_relative_l2": 1e-6, "adam_relative_l2": 1e-4, "reconstruction_relative_l2": 1e-5}}
    path = tmp_path / "drift.json"
    path.write_text(json.dumps(data))
    parity = {"zero_smoke_run_dir": str(zero), "reference_smoke_run_dir": str(reference),
              "final_drift_file": str(path), "final_drift_sha256": loop.sha256_file(path)}
    return parity, data, path


@pytest.mark.parametrize("mutation", ["none", "hash", "model", "adam", "missing", "nonfinite", "candidate"])
def test_final_drift_evidence_binds_actual_artifacts_and_fixed_tolerances(tmp_path, mutation):
    parity, data, path = drift_fixture(tmp_path)
    if mutation == "hash":
        parity["final_drift_sha256"] = "stale"
    elif mutation == "model":
        data["model"]["positions"]["relative_l2"] = .001
    elif mutation == "adam":
        data["adam"]["0"]["exp_avg"]["relative_l2"] = .001
    elif mutation == "missing":
        del data["adam"]["1"]["exp_avg_sq"]
    elif mutation == "nonfinite":
        data["model"]["positions"]["relative_l2"] = float("inf")
    elif mutation == "candidate":
        (Path(parity["zero_smoke_run_dir"]) / "candidate_pre_complex.npy").write_bytes(b"changed")
    if mutation in ("model", "adam", "missing", "nonfinite"):
        path.write_text(json.dumps(data))
        parity["final_drift_sha256"] = loop.sha256_file(path)
    if mutation == "none":
        loop.validate_final_drift(parity)
    else:
        with pytest.raises(ValueError):
            loop.validate_final_drift(parity)


@pytest.mark.parametrize("relative,accepted", [(1e-5, True), (1.000001e-5, False)])
def test_reconstruction_parity_uses_calibrated_fixed_boundary(tmp_path, relative, accepted):
    parity, data, path = drift_fixture(tmp_path)
    data["reconstruction"]["pre"]["relative_l2"] = relative
    path.write_text(json.dumps(data))
    parity["final_drift_sha256"] = loop.sha256_file(path)
    if accepted:
        loop.validate_final_drift(parity)
    else:
        with pytest.raises(ValueError):
            loop.validate_final_drift(parity)
