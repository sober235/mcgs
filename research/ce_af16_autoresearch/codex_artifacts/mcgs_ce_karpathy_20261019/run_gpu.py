#!/usr/bin/env python3
"""Stdlib supervisor: bind reviewed evidence, claim a GPU, supervise one child."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "mcgs_ce_autoresearch_20261017"
BASE = OLD / "nonlocal_autoresearch"
G2 = OLD / "frequency_adam/outputs/adamflat_g2_v1"
LEGACY_SUPERVISOR = OLD / "frequency_adam/run_gpu.py"
REFERENCE_FREEZE = BASE / "verification/report_reference_freeze.json"
CONTRACT = OLD / "reassessment_without_hfen_20261019/metric_contract_v1.json"
SUMMARY = CONTRACT.parent / "summary.json"
G2_SHA256 = "282e0303c46a6c18775d6224c032611072bd465b4ef975ba7a7e2ea636230fd0"
CONTRACT_SHA256 = "7404413f695bce3b59886de802815011615f7473a0a59626542ffd4dbe60cd13"
PYTHON = Path("/home/congcongliu/anaconda3/envs/GSTMI/bin/python")
CLAIM_LOCK = Path(f"/tmp/mcgs_ce_gpu_claim_{os.getuid()}.lock")
PRODUCTION = ("amplitude_model.py", "runtime.py", "runner.py", "evaluate.py", "run_gpu.py")
PARAMETERS = ("stage", "mode", "steps", "bands", "width", "gain_bound", "lr")
GPU_CLAIM_POLICY = "common_lock_until_child_exit_serialized"
_requested_signal = None


class SupervisorCancelled(Exception):
    """A requested stop, handled through the owned-child cleanup path."""

    def __init__(self, signum):
        self.signum = signum
        super().__init__(f"supervisor cancelled by {signal.Signals(signum).name}")


@contextmanager
def cancellation_signals():
    global _requested_signal
    previous_request = _requested_signal
    _requested_signal = None
    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)}

    def cancel(signum, frame):
        global _requested_signal
        # Never raise from an asynchronous handler: it could interrupt Popen
        # before the caller owns its handle, or interrupt lock-protected reap.
        if _requested_signal is None:
            _requested_signal = signum

    try:
        for signum in previous:
            signal.signal(signum, cancel)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        _requested_signal = previous_request


def check_cancellation():
    """Raise only at a safe boundary; never call this from child cleanup."""
    if _requested_signal is not None:
        raise SupervisorCancelled(_requested_signal)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def invalid(value):
        raise ValueError(f"nonfinite JSON value {value} in {path}")
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def descriptor(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha256(path)}


def verify_descriptor(value, label):
    require(isinstance(value, dict) and set(value) == {"path", "sha256"},
            f"{label} needs a path/sha256 descriptor")
    path = Path(value["path"])
    require(path.is_absolute() and descriptor(path) == value, f"{label} bytes/path changed")
    return path


def write_new(path, value):
    """Atomically publish new evidence; an existing destination is never replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def code_hashes():
    return {str(ROOT / name): sha256(ROOT / name) for name in PRODUCTION}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_campaign():
    """Reuse the original stdlib visibility, loading-claim and selection rules."""
    previous = sys.modules.get("objective")
    try:
        sys.modules["objective"] = load_module(BASE / "objective.py", "objective")
        return load_module(BASE / "autoresearch.py", "_mcgs_original_gpu_selector")
    finally:
        if previous is None:
            sys.modules.pop("objective", None)
        else:
            sys.modules["objective"] = previous


def snapshot_invariants():
    """Read and verify existing freezes, including their actual referenced files."""
    legacy = load_module(LEGACY_SUPERVISOR, "_mcgs_frequency_supervisor")
    original = legacy.snapshot_invariants()
    metrics = read(G2 / "metrics.json")
    provenance = read(G2 / "provenance.json")
    require(provenance.get("status") == "complete" and metrics.get("end_iteration") == 4500,
            "G2 is not the completed iteration4500 baseline")
    checkpoint = Path(metrics["final_checkpoint"]).resolve(strict=True)
    require(checkpoint.is_relative_to(G2), "G2 checkpoint escapes its frozen output")
    checkpoint_hash = sha256(checkpoint)
    require(checkpoint_hash == G2_SHA256, "G2 checkpoint hash differs from the accepted baseline")
    require(sha256(CONTRACT) == CONTRACT_SHA256, "active three-metric contract changed")
    reference = read(REFERENCE_FREEZE)
    require(isinstance(reference.get("files"), dict) and bool(reference["files"]),
            "reference/ROI freeze lacks actual paths")
    actual_references = {path: sha256(path) for path in reference["files"]}
    require(actual_references == reference["files"], "frozen reference/ROI bytes changed")
    require(sha256(reference["loop_input_freeze_path"]) == reference["loop_input_freeze_sha256"],
            "reference freeze parent input definition changed")
    files = [G2 / name for name in ("actual_config.yaml", "metrics.json", "provenance.json",
        "candidate_pre_complex.npy", "candidate_post_complex.npy", "frequency_provenance.json")]
    files += [G2 / "frequency_weights/manifest.json"]
    files += [G2 / "frequency_weights" / f"{scan}_{kind}.npy"
              for scan in ("pre", "post")
              for kind in ("source_weight", "effective_weight", "acquired_mask")]
    files += [CONTRACT, SUMMARY, REFERENCE_FREEZE, LEGACY_SUPERVISOR,
              BASE / "autoresearch.py", BASE / "objective.py", BASE / "runner.py",
              OLD / "frequency_adam/runner.py", OLD / "frequency_adam/frequency_weights.py"]
    return {"legacy": original, "g2_checkpoint": str(checkpoint),
        "g2_checkpoint_sha256": checkpoint_hash,
        "g2_and_contract_files": {str(path): sha256(path) for path in files},
        "reference_freeze": descriptor(REFERENCE_FREEZE), "reference_files": actual_references}


def parameters(args):
    return {name: getattr(args, name) for name in PARAMETERS}


def validate_parameters(args):
    require(args.stage in ("zero", "smoke", "trial") and args.mode in ("baseline", "shared", "explicit"),
            "invalid stage or mode")
    require(type(args.steps) is int and args.steps in
            ({0} if args.stage == "zero" else {2} if args.stage == "smoke" else {200, 500}),
            "steps must be zero0, smoke2 or trial200/500")
    require(args.stage == "zero" or args.mode != "baseline", "baseline is only valid for zero stage")
    for name, minimum in (("bands", 0), ("width", 1)):
        require(type(getattr(args, name)) is int and getattr(args, name) >= minimum, f"invalid {name}")
    for name in ("gain_bound", "lr"):
        value = getattr(args, name)
        require(type(value) in (int, float) and math.isfinite(value) and value > 0, f"invalid {name}")


def validate_review(path, hashes, invariants):
    review = read(path)
    require(review.get("schema_version") == 1 and review.get("status") == "passed"
        and review.get("passed") is True and type(review.get("round")) is int
        and review["round"] in (1, 2, 3) and review.get("actionable_findings") == 0
        and review.get("unresolved_findings") == 0, "a real clean independent review gate is required")
    require(review.get("code_sha256") == hashes and review.get("invariants") == invariants,
            "review source/invariant bindings are stale")
    report = verify_descriptor(review.get("review_report"), "review report")
    require(report.stat().st_size > 0, "independent review report is empty")
    return review


def validate_child_artifacts(args, directory, hashes, invariants):
    directory = Path(directory).resolve(strict=True)
    launch, metrics, provenance = [read(directory / name) for name in
                                  ("launch.json", "metrics.json", "provenance.json")]
    expected = parameters(args)
    for value in (launch, metrics, provenance):
        require(all(value.get(key) == item for key, item in expected.items()),
                "actual child stage/mode/parameters differ")
        require(value.get("code_sha256") == hashes and value.get("invariants") == invariants,
                "actual child source/invariants differ")
    require(launch.get("run_dir") == str(directory), "actual launch directory differs")
    require(metrics.get("status") == provenance.get("status") == "complete", "child is incomplete")
    trace = [json.loads(line) for line in (directory / "step_trace.jsonl").read_text().splitlines()]
    if args.stage == "zero":
        require(metrics.get("steps_observed") == provenance.get("steps_observed") == 0
                and len(trace) == 0, "zero stage performed updates")
        require(all(value.get("zero_parity", {}).get(scan, {}).get("passed") is True
                    for value in (metrics, provenance) for scan in ("pre", "post")),
                "both actual zero parity checks must pass")
        require(metrics["zero_parity"] == provenance["zero_parity"], "zero parity evidence disagrees")
    else:
        require(metrics.get("steps_observed") == provenance.get("steps_observed") == args.steps
            and metrics.get("gradients_finite") is provenance.get("gradients_finite") is True,
            "observed steps/gradients are incomplete or nonfinite")
        require(metrics.get("field_metadata", {}).get("mode") == args.mode
            and provenance.get("field_metadata", {}).get("mode") == args.mode, "field mode differs")
        require(len(trace) == args.steps and [row.get("step") for row in trace] == list(range(1, args.steps + 1)),
                "actual trace step count/order differs")
        require(all(row.get("gradients_finite") is True and type(row.get("loss")) in (int, float)
            and math.isfinite(row["loss"]) for row in trace), "actual trace has invalid gradients/loss")
    names = ["launch.json", "metrics.json", "provenance.json", "step_trace.jsonl",
             "candidate_pre_complex.npy", "candidate_post_complex.npy"]
    paths = [directory / name for name in names]
    for optional in ("childcontext_ready.json", "actual_config.json", "actual_config.yaml"):
        if (directory / optional).is_file():
            paths.append(directory / optional)
    if metrics.get("final_checkpoint"):
        final = Path(metrics["final_checkpoint"]).resolve(strict=True)
        require(final.is_relative_to(directory), "child checkpoint escapes its owned output")
        paths.append(final)
    return {str(path): sha256(path) for path in paths}


def validate_observed_gate(path, stage, hashes, invariants, matched=None):
    gate = read(path)
    require(gate.get("schema_version") == 1 and gate.get("passed") is True
        and gate.get("status") == "complete" and gate.get("stage") == stage,
        f"real completed {stage} gate is required")
    require(gate.get("code_sha256") == hashes and gate.get("invariants") == invariants,
            f"{stage} gate source/invariants are stale")
    params = gate.get("parameters", {})
    require(isinstance(params, dict) and set(params) == set(PARAMETERS), "gate parameter keys differ")
    require(all(gate.get(key) == params.get(key) for key in PARAMETERS), "gate parameters disagree")
    observed_args = argparse.Namespace(**params)
    validate_parameters(observed_args)
    if matched is not None:
        require(all(params.get(key) == matched[key] for key in
                    ("mode", "bands", "width", "gain_bound", "lr")), "trial requires a matched-mode smoke")
    artifacts = gate.get("artifacts", {})
    require(bool(artifacts) and all(sha256(file) == digest for file, digest in artifacts.items()),
            f"{stage} artifact bytes changed")
    require(validate_child_artifacts(observed_args, gate["run_dir"], hashes, invariants) == artifacts,
            f"{stage} actual artifact set differs")
    proof = read(verify_descriptor(gate.get("post_run_invariants"), f"{stage} source proof"))
    expected = {"code_sha256": hashes, "invariants": invariants}
    require(proof.get("status") == "verified" and proof.get("before") == proof.get("after") == expected,
            f"{stage} lacks an unchanged before/after proof")
    controls = gate.get("control_bindings")
    require(isinstance(controls, dict) and bool(controls) and proof.get("control_bindings_before") ==
        proof.get("control_bindings_after") == controls
        and all(sha256(file) == digest for file, digest in controls.items()), f"{stage} controls changed")
    return gate


def validate_stage_gates(args, hashes, invariants):
    validate_parameters(args)
    require(isinstance(args.exp, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.exp),
            "exp must be one safe unique directory name")
    require(type(args.preferred_gpu) is int and args.preferred_gpu >= 0, "invalid preferred GPU")
    require(type(args.timeout) in (int, float) and math.isfinite(args.timeout)
        and 0 < args.timeout <= 3600, "timeout must be positive and at most3600 seconds")
    review = validate_review(args.review_gate, hashes, invariants)
    recipe = read(args.recipe)
    require(isinstance(recipe.get("parameters"), dict) and
            set(recipe["parameters"]) == set(PARAMETERS), "recipe parameter keys differ")
    require(recipe.get("schema_version") == 1 and recipe.get("status") == "selected"
        and recipe.get("parameters") == parameters(args), "recipe parameters do not equal the actual CLI")
    validate_parameters(argparse.Namespace(**recipe["parameters"]))
    require(recipe.get("code_sha256") == hashes and recipe.get("invariants") == invariants,
            "recipe source/invariant bindings are stale")
    result = {"review": review, "recipe": recipe}
    paths = [args.review_gate, args.recipe, Path(review["review_report"]["path"])]
    if args.stage != "zero":
        require(args.zero_gate is not None, "nonzero stage requires an observed zero gate")
        result["zero"] = validate_observed_gate(args.zero_gate, "zero", hashes, invariants)
        paths.append(args.zero_gate)
    if args.stage == "trial":
        require(args.smoke_gate is not None, "trial requires a real two-step smoke gate")
        result["smoke"] = validate_observed_gate(args.smoke_gate, "smoke", hashes, invariants, parameters(args))
        paths.append(args.smoke_gate)
    result["control_bindings"] = {str(Path(path).resolve(strict=True)): sha256(path) for path in paths}
    return result


def select_gpu(cards, preferred, environment, campaign=None):
    campaign = campaign or load_campaign()
    eligible = [card for card in cards if card["free_mib"] >= 65 * 1024]
    require(bool(eligible), "no GPU has the required65GiB free memory")
    result = campaign.select_gpu(eligible, preferred, environment)
    result["snapshot"] = cards
    result["minimum_free_mib"] = 65 * 1024
    return result


@contextmanager
def claim_lock(path=CLAIM_LOCK):
    descriptor_fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        require(os.fstat(descriptor_fd).st_uid == os.getuid(), "GPU claim lock belongs to another user")
        while True:
            check_cancellation()
            try:
                fcntl.flock(descriptor_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(0.25)
        check_cancellation()
        yield descriptor_fd
    finally:
        os.close(descriptor_fd)


def terminate_owned(worker):
    """Popen is created with start_new_session; never signal another job/group."""
    if worker is None or worker.poll() is not None:
        return
    try:
        os.killpg(worker.pid, signal.SIGTERM)
    except ProcessLookupError:
        worker.wait()
        return
    try:
        worker.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(worker.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        worker.wait()


def wait_context(worker, directory, selected, campaign, deadline):
    context_deadline = min(deadline, time.monotonic() + 90)
    path = directory / "childcontext_ready.json"
    while time.monotonic() < context_deadline:
        check_cancellation()
        require(worker.poll() is None, "child exited before its CUDA context became visible")
        if path.exists():
            ready = read(path)
            require(ready.get("pid") == worker.pid and ready.get("gpu_uuid") == selected["uuid"],
                    "child CUDA context handshake has the wrong process/device")
            cards = campaign.gpu_snapshot()
            if any(card["uuid"] == selected["uuid"] and worker.pid in card["compute_pids"] for card in cards):
                return ready
        time.sleep(0.25)
    raise TimeoutError("child CUDA context was not visible within90 seconds")


def wait_owned(worker, deadline):
    """Wait in bounded intervals so cancellation is checked outside cleanup."""
    while True:
        check_cancellation()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(worker.args, 0)
        try:
            returncode = worker.wait(timeout=min(0.25, remaining))
        except subprocess.TimeoutExpired:
            continue
        check_cancellation()
        return returncode


def execute(args):
    require("torch" not in sys.modules, "CPU supervisor must not import Torch")
    hashes, invariants = code_hashes(), snapshot_invariants()
    check_cancellation()
    gates = validate_stage_gates(args, hashes, invariants)
    check_cancellation()
    bindings = gates["control_bindings"]
    directory = ROOT / "outputs" / args.exp
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    worker, error, phase = None, None, "claim"
    campaign = load_campaign()
    started = time.time()
    try:
        with (directory / "launch.log").open("x") as stream:
            with claim_lock():
                selected = select_gpu(campaign.gpu_snapshot(), args.preferred_gpu, os.environ, campaign)
                recipe = descriptor(args.recipe)
                launch = {"schema_version": 1, "run_dir": str(directory), **parameters(args),
                    "gpu": selected["gpu"], "gpu_uuid": selected["uuid"], "resource": selected,
                    "gpu_claim_policy": GPU_CLAIM_POLICY,
                    "code_sha256": hashes, "invariants": invariants, "recipe": recipe,
                    "recipe_path": recipe["path"], "recipe_sha256": recipe["sha256"],
                    "control_bindings": bindings, "parent_pid": os.getpid(), "started": started}
                require(code_hashes() == hashes and snapshot_invariants() == invariants,
                        "reviewed source/invariants changed before launch")
                require(all(sha256(file) == digest for file, digest in bindings.items()), "control bytes changed before launch")
                check_cancellation()
                write_new(directory / "launch.json", launch)
                command = [str(PYTHON), "-X", "utf8", "-B", "-u", str(ROOT / "runner.py"),
                           "--launch", str(directory / "launch.json")]
                environment = {**os.environ, "CUDA_VISIBLE_DEVICES": selected["uuid"],
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "OMP_NUM_THREADS": "1",
                    "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "PYTHONPATH": "",
                    "CUDA_CACHE_DISABLE": "1"}
                phase = "launch"
                worker = subprocess.Popen(command, env=environment, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    # The handle is now assigned, so a stop requested during
                    # Popen can safely enter the lock-protected cleanup path.
                    check_cancellation()
                    deadline = time.monotonic() + args.timeout
                    write_new(directory / "child_started.json", {"pid": worker.pid, "command": command,
                        "gpu_uuid": selected["uuid"], "timestamp": time.time()})
                    phase = "context"
                    wait_context(worker, directory, selected, campaign, deadline)
                    # The parent has no initial-CVD lease visible to legacy
                    # loading_claims(). Keep the shared lock through the whole
                    # child lifetime, including the low-memory context phase.
                    # This deliberately serializes cooperating campaign jobs.
                    phase = "numerics"
                    wait_owned(worker, deadline)
                    require(worker.returncode == 0, f"numerical child exited{worker.returncode}")
                except BaseException:
                    # Retain the claim until any failed/cancelled child has
                    # really exited, including timeout and parent SIGTERM.
                    terminate_owned(worker)
                    check_cancellation()
                    raise
    except BaseException as exc:
        error = exc
    finally:
        terminate_owned(worker)
    try:
        check_cancellation()
    except SupervisorCancelled as exc:
        error = exc
    proof = {"status": "checking", "before": {"code_sha256": hashes, "invariants": invariants},
             "control_bindings_before": bindings}
    try:
        proof["after"] = {"code_sha256": code_hashes(), "invariants": snapshot_invariants()}
        proof["control_bindings_after"] = {file: sha256(file) for file in bindings}
        require(proof["before"] == proof["after"] and proof["control_bindings_before"] ==
                proof["control_bindings_after"], "sources/invariants/control bytes changed after run")
        proof["status"] = "verified"
    except BaseException as exc:
        proof.update(status="failed", error=repr(exc))
        error = error or exc
    write_new(directory / "post_run_invariants.json", proof)
    try:
        check_cancellation()
    except SupervisorCancelled as exc:
        error = exc
    if error is None:
        try:
            phase = "validation"
            check_cancellation()
            artifacts = validate_child_artifacts(args, directory, hashes, invariants)
            check_cancellation()
            write_new(directory / "completed.json", {"schema_version": 1,
                "passed": True, "status": "complete", **parameters(args), "parameters": parameters(args),
                "run_dir": str(directory), "code_sha256": hashes, "invariants": invariants,
                "artifacts": artifacts, "control_bindings": bindings,
                "post_run_invariants": descriptor(directory / "post_run_invariants.json"),
                "worker_pid": worker.pid, "returncode": worker.returncode,
                "gpu_claim_policy": GPU_CLAIM_POLICY,
                "wall_seconds": time.time() - started})
        except BaseException as exc:
            error = exc
    if error is not None:
        error.supervisor_phase = phase
        error.worker_pid = worker.pid if worker else None
        error.worker_returncode = worker.returncode if worker else None
        raise error
    return directory / "completed.json"


def record_failure(args, error):
    phase = getattr(error, "supervisor_phase", "setup")
    status = ("cancelled" if isinstance(error, (SupervisorCancelled, KeyboardInterrupt)) else
        "refused_setup" if phase == "setup" else "infrastructure_failure" if phase in
        ("claim", "launch", "context") or isinstance(error, (TimeoutError, subprocess.TimeoutExpired))
        else "numerical_failure")
    value = {"status": status, "passed": False, "scientific_rejection": False,
        "timestamp": time.time(), "request": {key: str(value) if isinstance(value, Path)
        or isinstance(value, float) and not math.isfinite(value) else value
        for key, value in vars(args).items()}, "phase": phase, "error": repr(error),
        "traceback": "".join(traceback.format_exception(error)),
        "worker_pid": getattr(error, "worker_pid", None),
        "worker_returncode": getattr(error, "worker_returncode", None),
        "signal": getattr(error, "signum", None), "gpu_claim_policy": GPU_CLAIM_POLICY}
    path = ROOT / "verification" / f"failure_{time.time_ns()}_{uuid.uuid4().hex}.json"
    write_new(path, value)
    return path


def parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--stage", required=True, choices=("zero", "smoke", "trial"))
    parser.add_argument("--mode", required=True, choices=("baseline", "shared", "explicit"))
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--review-gate", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--zero-gate", type=Path)
    parser.add_argument("--smoke-gate", type=Path)
    parser.add_argument("--preferred-gpu", type=int, default=0)
    parser.add_argument("--bands", type=int, default=4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--gain-bound", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--timeout", type=float, default=3600)
    return parser


def main(argv=None):
    args = parser().parse_args(argv)
    with cancellation_signals():
        try:
            completed = execute(args)
        except BaseException as exc:
            failure = record_failure(args, exc)
            print(f"Supervisor refused/failed: {exc}; evidence: {failure}", file=sys.stderr)
            return 128 + exc.signum if isinstance(exc, SupervisorCancelled) else 1
    print(completed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
