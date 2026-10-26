"""Small CPU fixtures for the first-family queue and durable journal."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

import autoresearch_loop as loop


def objective(score=-0.1, win=False):
    comparisons = {}
    heads = {scan: {} for scan in ("pre", "post")}
    for scan in heads:
        for metric in loop.METRICS:
            candidate = 30.0 if metric == "psnr" else 0.9 if metric == "ssim" else 0.01
            best = (candidate - score * 10 / math.log(10) if metric == "psnr" else
                    1 - (1 - candidate) * math.exp(score) if metric == "ssim" else
                    candidate * math.exp(score))
            heads[scan][metric] = candidate
            comparisons[f"{scan}:{metric}"] = {
                "candidate": candidate, "best_opponent": best,
                "strict_win": win, "margin": score,
            }
    return {"metrics": list(loop.METRICS), "primary_objective": "whole_head_min_six_normalized_margin",
        "score": score, "strict_success": win, "whole_head_comparisons": comparisons,
        "whole_head_metrics": heads, "lesion_planes": {"strict_success": True}}


def record(key="baseline"):
    return {"record_id": key, "event": "scored", "exp": key,
        "result_row": dict(commit="abc123", whole_head_margin=-0.1, memory_gb=1.5,
                           status="baseline", description=key)}


def test_journal_appends_once_and_refuses_duplicate_ids(tmp_path):
    ledger = loop.Ledger(tmp_path)
    ledger.append(record())
    first_jsonl = (tmp_path / "autoresearch.jsonl").read_bytes()
    first_tsv = (tmp_path / "results.tsv").read_bytes()
    with pytest.raises(ValueError, match="duplicate"):
        ledger.append(record())
    assert (tmp_path / "autoresearch.jsonl").read_bytes() == first_jsonl
    assert (tmp_path / "results.tsv").read_bytes() == first_tsv
    assert len(first_tsv.decode().splitlines()) == 2
    assert first_tsv.decode().splitlines()[0].split("\t") == list(loop.RESULT_COLUMNS)


def test_tsv_projection_recovers_jsonl_commit_without_duplicate_rows(tmp_path):
    ledger = loop.Ledger(tmp_path)
    ledger.append(record())
    (tmp_path / "results.tsv").unlink()
    restored = loop.Ledger(tmp_path)
    assert len(restored.records) == 1
    assert len((tmp_path / "results.tsv").read_text().splitlines()) == 2
    restored.append(record("trial"))
    assert len((tmp_path / "results.tsv").read_text().splitlines()) == 3


def test_foreign_tsv_is_preserved_and_refused_before_append(tmp_path):
    ledger = loop.Ledger(tmp_path)
    ledger.append(record())
    path = tmp_path / "results.tsv"
    path.write_text(path.read_text() + "foreign\trow\tthat\tmust\tremain\n")
    before = (tmp_path / "autoresearch.jsonl").read_bytes()
    with pytest.raises(ValueError, match="canonical"):
        ledger.append(record("trial"))
    assert (tmp_path / "autoresearch.jsonl").read_bytes() == before
    assert path.read_text().endswith("foreign\trow\tthat\tmust\tremain\n")


def test_partial_journal_and_foreign_tsv_are_preserved_and_refused(tmp_path):
    (tmp_path / "autoresearch.jsonl").write_text('{"partial":')
    with pytest.raises(ValueError, match="incomplete"):
        loop.Ledger(tmp_path)
    assert (tmp_path / "autoresearch.jsonl").read_text() == '{"partial":'


def test_incomplete_existing_run_is_refused_without_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "ROOT", tmp_path)
    run = tmp_path / "outputs/family1_shared_smoke_v1"
    run.mkdir(parents=True)
    (run / "launch.log").write_text("preserved failed output")
    with pytest.raises(ValueError, match="incomplete"):
        loop.existing_run("family1_shared_smoke_v1", "smoke", {}, {}, {})
    assert (run / "launch.log").read_text() == "preserved failed output"


def test_completed_run_reuse_still_calls_full_frozen_gate_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "ROOT", tmp_path)
    run = tmp_path / "outputs/complete"
    run.mkdir(parents=True)
    (run / "completed.json").write_text("{}")
    params = {"stage": "smoke", "mode": "shared", "steps": 2}
    called = []

    def validate(path, stage, hashes, invariants, matched=None):
        called.append((path, stage, hashes, invariants))
        return {"parameters": params, "run_dir": str(run)}

    monkeypatch.setattr(loop.s, "validate_observed_gate", validate)
    assert loop.existing_run("complete", "smoke", params, {"source": "sha"}, {"frozen": True})
    assert len(called) == 1


def test_keep_is_separate_from_plane_or_whole_head_strict_success():
    partial = objective(-0.10)
    assert loop.classify_objective(partial, -0.11118128532897047) == "keep"
    assert loop.classify_objective(objective(-0.12), -0.11118128532897047) == "discard"
    assert loop.classify_objective(objective(-0.11108128532897047), -0.11118128532897047) == "discard"
    assert loop.classify_objective(objective(0.01, True), -0.11118128532897047) == "provisional_win"


def test_resume_keeps_frozen_prior_despite_mutable_campaign_score_and_status(tmp_path):
    source = tmp_path / "completed_prior_metrics.json"
    source.write_text('{"completed": true}')
    prior = {"name": "original_prior", "score": -0.11118128532897047,
             "source": loop.s.descriptor(source)}
    saved = {"prior_incumbent": prior, "minimum_keep_improvement": .0001}
    campaign = {"status": "root_updated_status", "minimum_keep_improvement": .001,
        "previous_whole_head_incumbent": {**prior, "score": -.12}}
    anchor, threshold = loop.load_prior(campaign, saved)
    assert anchor == prior
    assert threshold == .0001
    assert loop.classify_objective(objective(-.115), anchor["score"], threshold) == "discard"
    source.write_text("changed evidence")
    with pytest.raises(ValueError):
        loop.load_prior(campaign, saved)


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(metrics=["psnr", "ssim", "nmse", "hfen"]),
    lambda value: value["whole_head_metrics"]["pre"].update(hfen=0.2),
    lambda value: value.update(strict_success=True),
    lambda value: value.update(score=float("nan")),
    lambda value: value.update(score=-0.2),
    lambda value: (value.update(score=-0.2),
                   [row.update(margin=-0.2) for row in value["whole_head_comparisons"].values()]),
])
def test_quality_rejects_extra_metrics_false_success_and_invalid_margin(mutation):
    value = objective()
    mutation(value)
    with pytest.raises(ValueError):
        loop.classify_objective(value, -0.111)


def test_objective_hashes_must_equal_completed_candidate_artifacts(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    value = objective()
    value.update(run_dir=str(run), all_inputs_unchanged=True, candidate_sha256={},
                 input_hashes={"candidate": {}, "files": {}})
    artifacts = {}
    for scan in ("pre", "post"):
        path = run / f"candidate_{scan}_complex.npy"
        path.write_bytes(scan.encode())
        digest = loop.s.sha256(path)
        artifacts[str(path)] = digest
        value["candidate_sha256"][scan] = digest
        value["input_hashes"]["candidate"][scan] = {"path": str(path), "sha256": digest}
        value["input_hashes"]["files"][str(path)] = digest
    gate = {"run_dir": str(run), "artifacts": artifacts}
    loop.validate_objective(value, gate, {})
    (run / "candidate_pre_complex.npy").write_bytes(b"modified")
    with pytest.raises(ValueError, match="candidate"):
        loop.validate_objective(value, gate, {})


def test_stop_forwards_term_only_to_owned_supervisor_and_waits_for_numeric_cleanup(tmp_path):
    numeric = tmp_path / "numeric.py"
    numeric.write_text("import time; time.sleep(30)\n")
    program = tmp_path / "supervisor.py"
    program.write_text('''import os,pathlib,signal,subprocess,sys,time
r=pathlib.Path(sys.argv[1]);c=subprocess.Popen([sys.executable,str(r/'numeric.py')],start_new_session=True)
def stop(*a):
    (r/'term_received').write_text('TERM')
    c.terminate();c.wait();sys.exit(143)
signal.signal(signal.SIGTERM,stop)
(r/'numeric_pid').write_text(str(c.pid))
while True:time.sleep(.1)
''')
    worker = subprocess.Popen([sys.executable, "-B", str(program), str(tmp_path)], start_new_session=True)
    unrelated = subprocess.Popen([sys.executable, "-B", "-c", "import time;time.sleep(30)"],
                                 start_new_session=True)
    try:
        deadline = time.monotonic() + 3
        while not (tmp_path / "numeric_pid").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        pid = int((tmp_path / "numeric_pid").read_text())
        stopped = loop.stop_owned(worker, grace=15)
        assert stopped["cleanup_confirmed"] is True
        assert stopped["forced"] is False
        assert (tmp_path / "term_received").read_text() == "TERM"
        assert not Path(f"/proc/{pid}").exists()
        assert unrelated.poll() is None
    finally:
        if worker.poll() is None:
            os.killpg(worker.pid, signal.SIGKILL)
            worker.wait()
        unrelated.terminate()
        unrelated.wait()


def test_family_budget_is_exactly500_and_cli_uses_public_supervisor(tmp_path):
    args = loop.parser().parse_args(["--steps", "500", "--gpu", "0"])
    params = dict(stage="trial", mode="explicit", steps=500, bands=4, width=64, gain_bound=.25, lr=.001)
    command = loop.supervisor_command(args, "family1_explicit_trial_v1", params,
        tmp_path / "recipe.json", tmp_path / "smoke.json")
    assert command[command.index("--steps") + 1] == "500"
    assert "--zero-gate" in command and "--smoke-gate" in command
    with pytest.raises(SystemExit):
        loop.parser().parse_args(["--steps", "200"])
