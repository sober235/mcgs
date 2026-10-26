#!/usr/bin/env python3
"""Run one declared residual-amplitude family, preserving every result."""
from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
import fcntl
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import run_gpu as s

ROOT = Path(__file__).resolve().parent
FAMILY = "residual_amplitude_fixed_geometry_v1"
METRICS = ("psnr", "ssim", "nmse")
RESULT_COLUMNS = ("commit", "whole_head_margin", "memory_gb", "status", "description")


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def atomic_state(value):
    path = ROOT / "loop_state.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class Ledger:
    """JSONL is canonical; a prefix-only TSV projection repairs interrupted appends."""

    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / "autoresearch.jsonl"
        self.records = self._read(self.path.read_text() if self.path.exists() else "")
        if self.records:
            self._project()

    @staticmethod
    def _read(raw):
        s.require(not raw or raw.endswith("\n"), "incomplete ledger tail; preserve it for diagnosis")
        values = [json.loads(line) for line in raw.splitlines()]
        ids = [row["record_id"] for row in values]
        s.require(len(ids) == len(set(ids)), "duplicate record IDs in ledger")
        return values

    def _project(self):
        text = io.StringIO(newline="")
        writer = csv.writer(text, delimiter="\t", lineterminator="\n")
        writer.writerow(RESULT_COLUMNS)
        for value in self.records:
            if "result_row" in value:
                row = value["result_row"]
                s.require(set(row) == set(RESULT_COLUMNS), "result must have exactly five columns")
                writer.writerow([row[key] for key in RESULT_COLUMNS])
        wanted = text.getvalue()
        path = self.root / "results.tsv"
        actual = path.read_text() if path.exists() else ""
        s.require((not actual or actual.endswith("\n")) and wanted.startswith(actual),
                  "existing results.tsv is not the canonical journal prefix; preserve it")
        if wanted != actual:
            with path.open("a") as stream:
                stream.write(wanted[len(actual):])
                stream.flush()
                os.fsync(stream.fileno())

    def get(self, key):
        return next((row for row in self.records if row["record_id"] == key), None)

    def append(self, value):
        self.root.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.seek(0)
            self.records = self._read(stream.read())
            s.require(self.get(value["record_id"]) is None, "duplicate ledger record refused")
            # Check projection consistency before committing another event.
            self._project()
            record = {"timestamp": time.time(), **value}
            line = json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n"
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
            self.records.append(record)
            self._project()


def classify_objective(value, incumbent, threshold=0.0001):
    s.require(value.get("metrics") == list(METRICS) and value.get("primary_objective") ==
              "whole_head_min_six_normalized_margin", "only the fixed PSNR/SSIM/NMSE objective is allowed")
    heads, rows = value.get("whole_head_metrics", {}), value.get("whole_head_comparisons", {})
    keys = {f"{scan}:{metric}" for scan in ("pre", "post") for metric in METRICS}
    s.require(set(heads) == {"pre", "post"} and set(rows) == keys, "whole-head six-condition evidence missing")
    wins, margins = [], []
    for scan in ("pre", "post"):
        s.require(set(heads[scan]) == set(METRICS), "extra or missing quality metric")
        for metric in METRICS:
            row, candidate = rows[f"{scan}:{metric}"], heads[scan][metric]
            best = row.get("best_opponent")
            s.require(finite(candidate) and finite(best) and finite(row.get("margin")) and
                      row.get("candidate") == candidate, "nonfinite or inconsistent whole-head comparison")
            if metric == "psnr":
                expected_margin = (candidate - best) * math.log(10) / 10
            elif metric == "ssim":
                s.require(candidate < 1 and best < 1, "invalid SSIM for the fixed margin")
                expected_margin = math.log((1 - best) / (1 - candidate))
            else:
                s.require(candidate > 0 and best > 0, "invalid NMSE for the fixed margin")
                expected_margin = math.log(best / candidate)
            s.require(abs(row["margin"] - expected_margin) <= 1e-12, "normalized margin formula differs")
            win = candidate < best if metric == "nmse" else candidate > best
            s.require(row.get("strict_win") is win, "false strict-win comparison")
            wins.append(win)
            margins.append(row["margin"])
    score = value.get("score")
    s.require(finite(score) and finite(incumbent) and abs(score - min(margins)) <= 1e-12,
              "primary margin is not the minimum of the six conditions")
    s.require(value.get("strict_success") is all(wins), "whole-head strict_success claim differs")
    if all(wins):
        return "provisional_win"
    return "keep" if score > incumbent + threshold else "discard"


def validate_objective(value, gate, hashes):
    run = Path(gate["run_dir"]).resolve(strict=True)
    s.require(value.get("run_dir") == str(run) and value.get("all_inputs_unchanged") is True,
              "objective is incomplete or belongs to a different run")
    for scan in ("pre", "post"):
        path = str(run / f"candidate_{scan}_complex.npy")
        digest = gate["artifacts"].get(path)
        s.require(digest is not None and s.sha256(path) == digest and
            value.get("candidate_sha256", {}).get(scan) == digest and
            value.get("input_hashes", {}).get("candidate", {}).get(scan, {}).get("path") == path and
            value["input_hashes"]["candidate"][scan].get("sha256") == digest,
            "objective candidate hashes disagree with completed numerical artifacts")
        s.require(value.get("input_hashes", {}).get("files", {}).get(path) == digest,
                  "objective candidate missing from scored input hashes")
    files = value.get("input_hashes", {}).get("files", {})
    scorer = str(ROOT / "evaluate.py")
    if scorer in hashes:
        s.require(files.get(scorer) == hashes[scorer], "objective scorer hash differs from reviewed source")
    s.require(bool(files) and all(s.sha256(path) == digest for path, digest in files.items()),
              "objective input file bytes changed")
    classify_objective(value, value["score"])


def existing_run(exp, stage, params, hashes, invariants):
    directory = ROOT / "outputs" / exp
    if not directory.exists():
        return None
    s.require((directory / "completed.json").is_file(), f"existing incomplete run refused: {directory}")
    gate = s.validate_observed_gate(directory / "completed.json", stage, hashes, invariants)
    s.require(gate["parameters"] == params and Path(gate["run_dir"]).resolve() == directory,
              "existing completed run has different parameters/directory")
    return gate


def stop_owned(process, grace=120):
    """TERM only our supervisor PID; let it reap its separate numerical group."""
    s.require(grace >= 15, "supervisor cleanup needs at least15 seconds of grace")
    if process.poll() is not None:
        return {"cleanup_confirmed": True, "forced": False, "returncode": process.returncode}
    try:
        process.terminate()
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace
    while process.poll() is None and time.monotonic() < deadline:
        try:
            process.wait(timeout=min(.25, max(.001, deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            pass
    forced = process.poll() is None
    if forced:
        # This session is owned by the loop; numerical children use their own
        # session. Their status is unconfirmed after this last-resort action.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    return {"cleanup_confirmed": not forced, "forced": forced, "returncode": process.returncode}


def run_owned(command, log, state, kind, cpu=False):
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1"}
    if cpu:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    process, started = None, time.monotonic()
    runtime = {"command": command, "kind": kind, "log": str(log)}
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open("x") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
            runtime["pid"] = process.pid
            state["active"].update(pid=process.pid, process_kind=kind)
            atomic_state(state)
            while True:
                s.check_cancellation()
                try:
                    process.wait(timeout=.25)
                    break
                except subprocess.TimeoutExpired:
                    pass
            s.check_cancellation()
            s.require(process.returncode == 0, f"{kind} process exited{process.returncode}; retain {log}")
    except BaseException as exc:
        if process is not None:
            runtime["cleanup"] = stop_owned(process)
        runtime.update(wall_seconds=time.monotonic() - started,
                       returncode=process.returncode if process else None)
        exc.loop_runtime = runtime
        raise
    runtime.update(wall_seconds=time.monotonic() - started, returncode=process.returncode,
                   log_sha256=s.sha256(log))
    return runtime


def supervisor_command(args, exp, params, recipe, smoke_gate=None):
    command = [sys.executable, "-B", "-u", str(ROOT / "run_gpu.py"), "--exp", exp,
        "--review-gate", str(args.review_gate.resolve()), "--recipe", str(recipe.resolve()),
        "--zero-gate", str(args.zero_gate.resolve()), "--preferred-gpu", str(args.gpu)]
    for name in s.PARAMETERS:
        command.extend(("--" + name.replace("_", "-"), str(params[name])))
    if smoke_gate is not None:
        command.extend(("--smoke-gate", str(smoke_gate)))
    return command


def parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--family", choices=("coordinates", "paired_anchor"), default="coordinates")
    parser.add_argument("--zero-gate", type=Path, default=Path("outputs/g2_zero_v2/completed.json"))
    parser.add_argument("--review-gate", type=Path, default=Path("verification/review_gate_v1.json"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500, choices=(500,))
    return parser


def score_run(exp, gate, state, hashes):
    out = ROOT / "scores" / exp
    objective_path = out / "objective.json"
    runtime = {"reused_objective": True, "wall_seconds": None}
    if out.exists():
        s.require(objective_path.is_file(), f"incomplete score directory refused: {out}")
    else:
        state["active"].update(process_kind="scorer", pid=None)
        atomic_state(state)
        command = [str(s.PYTHON), "-B", "-u", str(ROOT / "evaluate.py"),
            "--run-dir", gate["run_dir"], "--out-dir", str(out), "--name", exp]
        runtime = run_owned(command, ROOT / "loop_logs" / f"{exp}_score.log", state, "scorer", cpu=True)
    value = s.read(objective_path)
    s.require(value.get("name") == exp, "objective name differs from queue entry")
    validate_objective(value, gate, hashes)
    reports = [out / name for name in ("objective.json", "whole_head_contract.json", "lesion_planes.json",
        "enhancement.json", "input_hashes.json", "whole_head.csv")]
    return value, s.descriptor(objective_path), runtime, {str(path): s.sha256(path) for path in reports}


def source_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def load_prior(campaign, saved_state=None):
    """Resume against the family's original scientific anchor, not mutable status."""
    prior = (saved_state["prior_incumbent"] if saved_state is not None else
             campaign["previous_whole_head_incumbent"])
    threshold = (saved_state["minimum_keep_improvement"] if saved_state is not None else
                 campaign.get("minimum_keep_improvement"))
    s.verify_descriptor(prior["source"], "previous whole-head incumbent")
    s.require(finite(prior.get("score")) and threshold == .0001,
              "missing fixed incumbent or keep threshold")
    return prior, threshold


def execute(args):
    s.require(args.gpu >= 0 and args.steps == 500, "family requires GPU index>=0 and500 formal updates")
    s.require(args.family in ("coordinates", "paired_anchor"), "unknown family")
    paired = args.family == "paired_anchor"
    family = "paired_anchor_residual_amplitude_fixed_geometry_v1" if paired else FAMILY
    modes = ("shared",) if paired else ("shared", "explicit")
    feature_mode = "paired_anchor" if paired else "coordinates"

    def experiment(stage, mode):
        return f"paired_anchor_{stage}_v1" if paired else f"family1_{mode}_{stage}_v1"

    hashes, invariants = s.code_hashes(), s.snapshot_invariants()
    s.check_cancellation()
    s.validate_review(args.review_gate, hashes, invariants)
    zero = s.validate_observed_gate(args.zero_gate, "zero", hashes, invariants)
    s.require(zero["parameters"]["mode"] == ("shared" if paired else "baseline")
        and zero["parameters"].get("feature_mode") == feature_mode,
        "selected family requires its actual matching zero parameterization")
    campaign = s.read(ROOT / "campaign_state.json")
    state_path = ROOT / "loop_state.json"
    saved_state = s.read(state_path) if state_path.exists() else None
    prior, threshold = load_prior(campaign, saved_state)
    recipes = {}
    for stage in ("smoke", "trial"):
        for mode in modes:
            filename = f"paired_{stage}_recipe_v1.json" if paired else f"{stage}_{mode}_recipe_v1.json"
            path = ROOT / "verification" / filename
            value = s.read(path)
            expected = dict(stage=stage, mode=mode, feature_mode=feature_mode, steps=2 if stage == "smoke" else 500,
                            bands=4, width=64, gain_bound=.25, lr=.001)
            s.require(value.get("parameters") == expected and value.get("code_sha256") == hashes
                and value.get("invariants") == invariants, "predeclared family recipe is stale or changed")
            recipes[f"{stage}:{mode}"] = {"path": path, "parameters": expected}
    pins = {"family": family, "loop_sha256": s.sha256(__file__), "code_sha256": hashes,
        "zero_gate": s.descriptor(args.zero_gate), "review_gate": s.descriptor(args.review_gate),
        "recipes": {key: s.descriptor(value["path"]) for key, value in recipes.items()}}
    state = saved_state if saved_state is not None else {
        **pins, "status": "active", "active": None, "stages": {}, "results": {},
        "latest_metrics": None, "failure": None, "source_commit": source_commit(),
        "prior_incumbent": prior, "minimum_keep_improvement": threshold,
        "campaign_state": s.descriptor(ROOT / "campaign_state.json")}
    s.require(all(state.get(key) == value for key, value in pins.items()), "existing loop bindings differ")
    s.require(state.get("status") not in ("crash", "failed_setup", "cancelled"),
              "family stopped after failure/cancellation; root diagnosis is required before any retry")
    ledger = Ledger(ROOT)
    s.require(all(row.get("family") == family and row.get("code_sha256") == hashes and
        row.get("loop_sha256") == pins["loop_sha256"] and row.get("prior_incumbent") == prior
        and row.get("minimum_keep_improvement") == threshold for row in ledger.records
        if row.get("event") == "scored"), "existing quality ledger has different family/source bindings")
    best = max([prior["score"]] + [row["score"] for row in ledger.records
        if row.get("status") in ("keep", "provisional_win")])
    commit = state["source_commit"]

    def common(exp, stage, mode, gate, runtime, recipe=None):
        metrics = s.read(Path(gate["run_dir"]) / "metrics.json")
        return {"exp": exp, "family": family, "stage": stage, "mode": mode,
            "feature_mode": gate["parameters"]["feature_mode"],
            "source_commit": commit, "loop_sha256": pins["loop_sha256"], "code_sha256": hashes,
            "prior_incumbent": prior, "minimum_keep_improvement": threshold,
            "recipe": recipe, "completed_gate": s.descriptor(Path(gate["run_dir"]) / "completed.json"),
            "runtime": {"supervisor": runtime, "numerical_training_seconds": metrics.get("training_seconds"),
                "actual_steps": metrics.get("steps_observed"), "peak_memory_gb": metrics.get("peak_memory_gb"),
                "supervisor_wall_seconds": gate.get("wall_seconds")}}

    def scored(exp, stage, mode, gate, runtime, recipe=None):
        nonlocal best
        key = f"{family}:{exp}:scored"
        previous = ledger.get(key)
        if previous is not None:
            value = s.read(s.verify_descriptor(previous["objective"], "recorded objective"))
            validate_objective(value, gate, hashes)
            s.require(previous["completed_gate"] == s.descriptor(Path(gate["run_dir"]) / "completed.json"),
                      "recorded numerical gate changed")
            s.require(all(s.sha256(path) == digest for path, digest in previous["score_artifacts"].items()),
                      "recorded scoring report bytes changed")
            return previous
        value, objective_file, scoring_runtime, reports = score_run(exp, gate, state, hashes)
        status = "baseline" if stage == "zero" else classify_objective(value, best, threshold)
        record = common(exp, stage, mode, gate, runtime, recipe)
        record.update(record_id=key, event="scored", status=status, score=value["score"],
            strict_success=value["strict_success"], objective=objective_file, score_artifacts=reports,
            whole_head_metrics=value["whole_head_metrics"], incumbent_before=best,
            scientific_rejection=status == "discard",
            result_row=dict(commit=commit, whole_head_margin=value["score"],
                memory_gb=record["runtime"]["peak_memory_gb"], status=status,
                description=f"{exp}; fixed geometry; {mode}; {gate['steps']} updates"))
        record["runtime"]["scoring"] = scoring_runtime
        ledger.append(record)
        if status in ("keep", "provisional_win"):
            best = max(best, value["score"])
        return record

    try:
        baseline = "paired_anchor_g2_baseline_v1" if paired else "family1_g2_baseline_v1"
        zero_mode = zero["parameters"]["mode"]
        state["active"] = dict(exp=baseline, stage="zero", mode=zero_mode, pid=None, process_kind="setup")
        atomic_state(state)
        entry = scored(baseline, "zero", zero_mode, zero, {"reused_completed": True})
        state["results"][baseline] = {key: entry[key] for key in ("status", "score", "objective")}
        for stage in ("smoke", "trial"):
            for mode in modes:
                exp = experiment(stage, mode)
                recipe = recipes[f"{stage}:{mode}"]
                params, recipe_path = recipe["parameters"], recipe["path"]
                state.update(status="active", active=dict(exp=exp, stage=stage, mode=mode,
                    process_kind="setup", pid=None), failure=None)
                atomic_state(state)
                s.check_cancellation()
                gate = existing_run(exp, stage, params, hashes, invariants)
                runtime = {"reused_completed": True}
                if gate is None:
                    smoke_gate = ROOT / "outputs" / experiment("smoke", mode) / "completed.json" if stage == "trial" else None
                    numerical_args = argparse.Namespace(**params, exp=exp, review_gate=args.review_gate,
                        recipe=recipe_path, zero_gate=args.zero_gate, smoke_gate=smoke_gate,
                        preferred_gpu=args.gpu, timeout=3600)
                    s.validate_stage_gates(numerical_args, hashes, invariants)
                    runtime = run_owned(supervisor_command(args, exp, params, recipe_path, smoke_gate),
                        ROOT / "loop_logs" / f"{exp}_supervisor.log", state, "supervisor")
                    gate = existing_run(exp, stage, params, hashes, invariants)
                    s.require(gate is not None, "successful supervisor omitted its completed gate")
                gate_file = s.descriptor(ROOT / "outputs" / exp / "completed.json")
                if stage == "smoke":
                    key = f"{family}:{exp}:complete"
                    if ledger.get(key) is None:
                        ledger.append({"record_id": key, "event": "run_complete", "status": "smoke_complete",
                            **common(exp, stage, mode, gate, runtime, s.descriptor(recipe_path))})
                    else:
                        s.require(ledger.get(key)["completed_gate"] == gate_file, "recorded smoke gate changed")
                else:
                    entry = scored(exp, stage, mode, gate, runtime, s.descriptor(recipe_path))
                    state["results"][exp] = {key: entry[key] for key in ("status", "score", "objective")}
                    state["latest_metrics"] = entry["whole_head_metrics"]
                state["stages"][exp] = dict(status="complete", completed_gate=gate_file,
                    parameters=params, runtime=runtime)
                state["active"] = None
                atomic_state(state)
        provisional = any(row.get("status") == "provisional_win" for row in ledger.records)
        state.update(status="provisional_win_wait_confirmation" if provisional else "family_complete_goal_unmet",
            active=None, best_whole_head_margin=best,
            next_step="root must select and review the next hypothesis" if not provisional else
            "root must run independent repeat and matched GS-single before attribution", success=False)
        atomic_state(state)
        return state
    except BaseException as exc:
        exc.loop_state = state
        raise


def record_stop(args, error):
    state = getattr(error, "loop_state", None)
    if state is None:
        state = s.read(ROOT / "loop_state.json") if (ROOT / "loop_state.json").exists() else {
            "family": ("paired_anchor_residual_amplitude_fixed_geometry_v1" if args.family == "paired_anchor"
                       else FAMILY), "active": None, "stages": {}, "results": {}}
    runtime = getattr(error, "loop_runtime", {})
    status = ("cancelled" if isinstance(error, s.SupervisorCancelled) else "crash" if runtime.get("pid") or
        (state.get("active") or {}).get("pid") else "failed_setup")
    failure = {"status": status, "scientific_rejection": False, "error": repr(error),
        "traceback": "".join(traceback.format_exception(error)), "runtime": runtime,
        "signal": getattr(error, "signum", None), "requires_root_diagnosis": True}
    state.update(status=status, failure=failure)
    atomic_state(state)
    exp = (state.get("active") or {}).get("exp", "setup")
    ledger = Ledger(ROOT)
    key = f"{state['family']}:{exp}:failure"
    if ledger.get(key) is None:
        commit = state.get("source_commit") or source_commit()
        ledger.append({"record_id": key, "event": "family_stopped", "exp": exp, **failure,
            "source_commit": commit, "loop_sha256": s.sha256(__file__),
            "result_row": dict(commit=commit, whole_head_margin="", memory_gb="", status=status,
                               description=f"{exp}; retain outputs; root diagnosis required")})


def main(argv=None):
    os.chdir(ROOT)
    args = parser().parse_args(argv)
    lock_path = ROOT / ".first_family.lock"
    with lock_path.open("a+") as lock, s.cancellation_signals():
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another first-family loop is active; no work started.", file=sys.stderr)
            return 1
        try:
            state = execute(args)
        except BaseException as exc:
            record_stop(args, exc)
            print(f"Family stopped: {exc}; see loop_state.json and autoresearch.jsonl", file=sys.stderr)
            return 128 + exc.signum if isinstance(exc, s.SupervisorCancelled) else 1
    print(state["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
