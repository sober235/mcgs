#!/usr/bin/env python3
"""Bounded, resumable CE AF16 development loop with observed preflight gates.

No training in --plan-only / --score-only. Every formal trial independently resumes
the registered iteration4000 state, runs500 steps, and keeps all results in JSONL.
"""
import argparse
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True  # External /data0 modules and inputs are read-only.

from objective import BASELINE_SHA256, LABELS, load_baseline, read_json, score, sha256_file

ROOT = Path(__file__).resolve().parent
VERIFY = ROOT / "verification"
BASELINE = ROOT.parent / "baseline_lesion_allmetrics.json"
PYTHON = Path("/home/congcongliu/anaconda3/envs/GSTMI/bin/python")
SCRIPTS = Path("/data0/congcong/code/GS/mcgs_ce_mask_wt/ce_retro_cssense_af16/scripts")
SCORER = SCRIPTS / "lesion_slice_allmetrics.py"
TOPIC = Path("/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois_cmp")
ROI = Path("/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast/lesion_candidates/enhance_rois_cssense.npz")
SCORER_HASH = "0e62741d39e0ec739c01382ec48f293e2137b76872a6ae751c5d1bbe58f16b2b"
ARMS_HASH = "592a9443ed9d4d5a8ea306cfe844e401d088a6e472112db305c2879c1ccd27c4"


def write_new(path, value):
    """Publish an immutable JSON file without exposing partially written contents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with temp.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)  # atomic and refuses an existing destination
    finally:
        temp.unlink(missing_ok=True)


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records = []
        if self.path.exists():
            raw = self.path.read_text()
            if raw and not raw.endswith("\n"):
                raise ValueError("incomplete ledger tail; preserve it and repair explicitly before resuming")
            for line in raw.splitlines():
                try:
                    self.records.append(json.loads(line))
                except (ValueError, TypeError) as exc:
                    raise ValueError("invalid ledger JSON; no records discarded") from exc

    def append(self, record):
        record = {"timestamp": time.time(), **record}
        line = (json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n").encode()
        fd = os.open(self.path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            if os.write(fd, line) != len(line):
                raise OSError("short ledger write; stop and inspect ledger")
            os.fsync(fd)
        finally:
            os.close(fd)
        self.records.append(record)

    def attempted(self):
        return {r["exp"] for r in self.records if r.get("event") in ("trial_started", "trial_scored", "trial_failed")}


def is_improvement(value, incumbent):
    if not math.isfinite(value) or not math.isfinite(incumbent):
        raise ValueError("nonfinite incumbent/objective")
    return value > incumbent + 1e-4


def execute_trial(book, candidate, run_dir, command, environment, score_callback):
    """Runner owns atomic training-directory reservation; ledger precedes launch."""
    name = candidate["name"]
    if name in book.attempted() or Path(run_dir).exists():
        raise ValueError(f"{name} already attempted or output already exists")
    journal = book.path.parent / "loop_trials" / name
    journal.mkdir(parents=True, exist_ok=False)
    book.append({"event": "trial_started", "exp": name, "candidate": candidate,
                 "run_dir": str(run_dir), "command": command,
                 "CUDA_VISIBLE_DEVICES": environment.get("CUDA_VISIBLE_DEVICES")})
    rc = None
    stage = "training"
    try:
        with (journal / "launch.log").open("x") as stream:
            rc = subprocess.run(command, env={**os.environ, **environment}, stdout=stream, stderr=subprocess.STDOUT).returncode
        if rc:
            book.append({"event": "trial_failed", "exp": name, "returncode": rc, "stage": stage,
                         "error": f"runner exited {rc}", "log": str(journal / "launch.log")})
            return False
        stage = "scoring"
        result = score_callback(Path(run_dir), candidate)
        book.append({"event": "trial_scored", "exp": name, "candidate": candidate, "run_dir": str(run_dir), **result})
        return True
    except Exception as exc:
        book.append({"event": "trial_failed", "exp": name, "returncode": rc, "stage": stage,
                     "error": repr(exc), "log": str(journal / "launch.log")})
        return False


def quiet_ready(rows, gpu, now):
    selected = [r for r in rows if r.get("gpu") == gpu]
    if not selected or now - selected[-1]["timestamp"] > 65 or now < selected[-1]["timestamp"]:
        return False
    tail = []
    for row in reversed(selected):
        if not row.get("free"):
            break
        if tail and tail[-1]["timestamp"] - row["timestamp"] > 65:
            break
        tail.append(row)
    return bool(tail) and tail[0]["timestamp"] - tail[-1]["timestamp"] >= 600


def resolve_device_tokens(value, cards):
    """Resolve physical indices and unambiguous full/prefix GPU UUIDs."""
    found = set()
    for token in value.split(","):
        token = token.strip()
        if token.isdigit():
            found.update(c["gpu"] for c in cards if c["gpu"] == int(token))
        elif re.fullmatch(r"\d+-\d+", token):
            first, last = map(int, token.split("-"))
            found.update(c["gpu"] for c in cards if first <= c["gpu"] <= last)
        elif token.lower().startswith("gpu-"):
            matches = [c["gpu"] for c in cards if c["uuid"].lower().startswith(token.lower())]
            if len(matches) == 1:
                found.update(matches)
    return found


def loading_claims(cards, process_environments):
    # Scheduler/launch shells carry our allocation while waiting for this job.
    # Their inherited visibility is not a competing loader reservation.
    launch_chain = {os.getpid()}
    ancestor = os.getppid()
    while ancestor > 0 and ancestor not in launch_chain:
        launch_chain.add(ancestor)
        try:
            status = Path(f"/proc/{ancestor}/status").read_text()
            ancestor = int(next(line.split()[1] for line in status.splitlines() if line.startswith("PPid:")))
        except (OSError, StopIteration, ValueError):
            break
    claims = {}
    for pid, visibility in process_environments:
        if pid in launch_chain:
            continue
        for gpu in resolve_device_tokens(visibility, cards):
            card = next(c for c in cards if c["gpu"] == gpu)
            if pid not in card["compute_pids"]:
                claims.setdefault(gpu, []).append(pid)
    return claims


def gpu_snapshot():
    env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    def query(fields):
        return subprocess.run(["nvidia-smi", fields, "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, check=True, env=env).stdout
    cards = []
    for line in query("--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu").splitlines():
        index, uuid, used, total, utilization = [s.strip() for s in line.split(",")]
        cards.append({"gpu": int(index), "uuid": uuid, "used_mib": float(used), "total_mib": float(total),
                      "utilization": float(utilization), "compute_pids": [], "loading_claim_pids": []})
    for line in query("--query-compute-apps=gpu_uuid,pid").splitlines():
        uuid, pid = [s.strip() for s in line.split(",")]
        for card in cards:
            if card["uuid"] == uuid:
                card["compute_pids"].append(int(pid))
    processes = []
    for path in Path("/proc").glob("[0-9]*/environ"):
        try:
            for item in path.read_bytes().split(b"\0"):
                if item.startswith(b"CUDA_VISIBLE_DEVICES="):
                    processes.append((int(path.parent.name), item.split(b"=", 1)[1].decode(errors="replace")))
        except OSError:
            continue
    claims = loading_claims(cards, processes)
    now = time.time()
    for card in cards:
        card.update(timestamp=now, loading_claim_pids=claims.get(card["gpu"], []),
                    free_mib=card["total_mib"] - card["used_mib"],
                    compute_process_present=bool(card["compute_pids"]))
        card["free"] = (card["used_mib"] < 100 and card["utilization"] == 0
                        and not card["compute_pids"] and not card["loading_claim_pids"])
        card["helper_free_and_unclaimed"] = card["free"]  # compatibility for preflight/monitor readers
    return cards


def gpu_probe(gpu):
    """Compatibility probe; UUID reservations are included before CUDA initialization."""
    return next(card for card in gpu_snapshot() if card["gpu"] == gpu)


def select_gpu(cards, preferred, environment):
    visible = {card["gpu"] for card in cards}
    for key in ("CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "NVIDIA_VISIBLE_DEVICES"):
        if key in environment and environment[key] != "all":
            visible &= resolve_device_tokens(environment[key], cards)
    eligible = [c for c in cards if c["gpu"] in visible and c["total_mib"] >= 75 * 1024
                and c["total_mib"] - c["used_mib"] >= 50 * 1024 and not c["loading_claim_pids"]]
    if not eligible:
        raise ValueError("no visible compatible GPU has50GiB free and no loading reservation")
    idle = [c for c in eligible if c["used_mib"] < 100 and c["utilization"] == 0 and not c["compute_pids"]]
    pool = idle or eligible
    selected = min(pool, key=lambda c: (0 if idle and c["gpu"] == preferred else 1,
                                        c["utilization"], c["used_mib"], c["gpu"]))
    return {**selected, "selection_reason": "idle compatible GPU" if idle else "lightest sufficient-memory workload; cohosting",
            "preferred_gpu": preferred, "visible_gpus": sorted(visible), "snapshot": cards}


def wait_for_gpu(gpu, evidence):
    """Select from a fresh snapshot. No quiet-time delay under current user policy."""
    selected = select_gpu(gpu_snapshot(), gpu, os.environ)
    Ledger(evidence).append(selected)
    return selected


def smoke_code_paths():
    return [ROOT / "runner.py", ROOT / "trainer/paired_gs/trainers/joint_trainer.py",
            ROOT / "trainer/paired_gs/trainers/resume_protocol.py", ROOT / "trainer/paired_gs/losses/nonlocal_red.py",
            ROOT / "trainer/paired_gs/losses/nonlocal_lowrank.py"]


def review_code_paths():
    return smoke_code_paths() + [ROOT / name for name in ("autoresearch.py", "objective.py", "autoresearch.sh",
                                                         "candidate_grid.json", "run_preflight.py")]


def _check_gate(path, required, checkpoint_hash=None):
    gate = read_json(path)
    if gate.get("passed") is not True:
        raise ValueError(f"gate not passed: {path}")
    if checkpoint_hash is not None and gate.get("source_checkpoint_sha256") != checkpoint_hash:
        raise ValueError(f"gate checkpoint hash mismatch: {path}")
    hashes = gate.get("code_sha256", {})
    if any(str(p) not in hashes for p in required):
        raise ValueError(f"gate missing required code hashes: {path}")
    for file, digest in hashes.items():
        if sha256_file(file) != digest:
            raise ValueError(f"stale gate code hash: {file}")
    return gate


def _smoke_observations(run, manifest, active):
    metrics, provenance = read_json(run / "metrics.json"), read_json(run / "provenance.json")
    checkpoint_hash = manifest["files"][manifest["source_checkpoint"]]
    if (provenance.get("status") != "complete" or provenance.get("runner_sha256") != sha256_file(ROOT / "runner.py")
            or provenance.get("source_checkpoint") != manifest["source_checkpoint"]
            or provenance.get("source_checkpoint_sha256") != checkpoint_hash
            or metrics.get("source_checkpoint_sha256") != checkpoint_hash
            or metrics.get("mode") != "all" or metrics.get("source_iteration") != 4000
            or metrics.get("end_iteration") != 4002 or metrics.get("forward_backward_evaluations") != 2):
        raise ValueError("smoke is not a completed code-bound/checkpoint-bound two-step continuation")
    source_copy = read_json(run / "source_to_copy_manifest.json")
    for relative, entry in source_copy["files"].items():
        if entry.get("copy_sha256") is not None:
            path = (ROOT / "trainer" / relative).resolve()
            if not path.is_relative_to(ROOT / "trainer") or sha256_file(path) != entry["copy_sha256"]:
                raise ValueError("smoke source-copy manifest hash changed")
    for path in smoke_code_paths()[1:]:
        relative = str(path.relative_to(ROOT / "trainer"))
        if source_copy["files"].get(relative, {}).get("copy_sha256") != sha256_file(path):
            raise ValueError("smoke manifest omits required current trainer code")
    traces = [json.loads(line) for line in (run / "step_trace.jsonl").read_text().splitlines()]
    if len(traces) != 2 or [r.get("absolute_iteration") for r in traces] != [4001, 4002]:
        raise ValueError("smoke trace must contain both actual steps")
    for row in traces:
        if row.get("gradients_finite") is not True or any(not math.isfinite(row[k]) for k in ("loss", "unclipped_grad_norm", "postclip_gradient_norm", "wall_seconds")):
            raise ValueError("smoke has invalid observed loss/gradients/runtime")
    nlr = metrics["nlr"]
    if nlr != provenance.get("nlr") or not nlr.get("all_gradients_finite") or bool(nlr.get("enabled")) != active:
        raise ValueError("smoke prior observations inconsistent")
    if active:
        if (not nlr.get("validated_active") or nlr.get("active_step_count") != 2
                or nlr.get("nonzero_weighted_step_count") != 2 or nlr.get("weighted_R_nl_min", 0) <= 0):
            raise ValueError("active smoke did not observe two nonzero prior steps")
        for scan in ("pre", "post"):
            if nlr["coefficients"][scan] <= 0 or nlr["raw_R_nl_by_channel_max"][scan] <= 0:
                raise ValueError("active smoke missing scan-wise prior")
            history = nlr["refresh_history"][scan]
            if len(history) != 1 or history[0]["iter"] != 4001:
                raise ValueError("active smoke must refresh both scans at4001")
            row = history[0]
            memory = row["kernel_stats"]["memory"]
            if (row["crop"] != [255, 232, 176] or not math.isfinite(row["target_rms_difference"]) or row["target_rms_difference"] <= 0
                    or not 0 < row["seconds"] <= 300 or row["kernel_stats"].get("valid_group_count", 0) <= 0
                    or memory["cuda_allocated_after"] - memory["cuda_allocated_before"] >= 2 * 1024**3):
                raise ValueError("active full-crop smoke failed nonidentity/runtime/memory bound")
        if not 0 < metrics["cuda_memory_bytes"]["peak_allocated"] < 75 * 1024**3:
            raise ValueError("active smoke peak memory exceeds approved capacity")
        if not 0 < metrics["cuda_memory_bytes"]["peak_reserved"] <= 46 * 1024**3:
            raise ValueError("active smoke reserved memory plus4GiB headroom exceeds50GiB requirement")
        observed = [nlr[k] for k in ("weighted_R_nl_min", "weighted_R_nl_max", "weighted_R_nl_last")]
        observed += [v for k in ("raw_R_nl_by_channel_max", "raw_R_nl_by_channel_last") for v in nlr[k].values()]
        if any(not math.isfinite(v) or v <= 0 for v in observed):
            raise ValueError("active smoke prior observations must be finite and positive")
    elif any(nlr["coefficients"][s] != 0 or nlr["refresh_history"][s] for s in ("pre", "post")):
        raise ValueError("zero smoke unexpectedly applied nonlocal prior")
    return metrics, traces


def validate_final_drift(parity):
    """Bind quantified GPU nondeterminism evidence to the actual final artifacts."""
    path = parity.get("final_drift_file")
    if not path or sha256_file(path) != parity.get("final_drift_sha256"):
        raise ValueError("missing/stale final parity drift evidence hash")
    drift = read_json(path)
    if drift.get("passed") is not True:
        raise ValueError("final parity drift evidence failed")
    for role, gate_key, drift_key in (("zero", "zero_smoke_run_dir", "zero_run_dir"),
                                      ("reference", "reference_smoke_run_dir", "reference_run_dir")):
        run = Path(parity[gate_key]).resolve()
        if Path(drift.get(drift_key, "")).resolve() != run:
            raise ValueError("final drift evidence names a different smoke run")
        metrics = read_json(run / "metrics.json")
        if drift.get("checkpoint_sha256", {}).get(role) != sha256_file(metrics["final_checkpoint"]):
            raise ValueError("final drift checkpoint artifact hash changed")
        for scan in ("pre", "post"):
            if drift.get("candidate_sha256", {}).get(role, {}).get(scan) != sha256_file(run / f"candidate_{scan}_complex.npy"):
                raise ValueError("final drift candidate artifact hash changed")
    names = {"positions", "scales", "rotations", "density_real", "density_imag", "density_real_contrast", "density_imag_contrast"}
    tolerances = {"model_relative_l2": 1e-6, "adam_relative_l2": 1e-4, "reconstruction_relative_l2": 1e-5}
    if (drift.get("tolerances") != tolerances or set(drift.get("model", {})) != names
            or set(drift.get("adam", {})) != {str(i) for i in range(7)}
            or set(drift.get("reconstruction", {})) != {"pre", "post"}):
        raise ValueError("incomplete drift evidence or altered fixed tolerances")
    def check(row, tolerance):
        if any(not isinstance(row.get(k), (int, float)) or not math.isfinite(row[k]) or row[k] < 0
               for k in ("max_absolute", "relative_l2")) or row["relative_l2"] > tolerance:
            raise ValueError("nonfinite or excessive final parity drift")
    for row in drift["model"].values():
        check(row, 1e-6)
    for group in drift["adam"].values():
        if set(group) != {"exp_avg", "exp_avg_sq"}:
            raise ValueError("both Adam moments required for every group")
        for row in group.values():
            check(row, 1e-4)
    for row in drift["reconstruction"].values():
        check(row, 1e-5)


def validate_training_gates(parity_path, active_path, review_path, manifest):
    checkpoint_hash = manifest["files"][manifest["source_checkpoint"]]
    parity = _check_gate(parity_path, smoke_code_paths(), checkpoint_hash)
    active = _check_gate(active_path, smoke_code_paths(), checkpoint_hash)
    review = _check_gate(review_path, review_code_paths())
    if review.get("round") not in (2, 3) or review.get("actionable_findings") != 0:
        raise ValueError("training requires a fresh clean round2/3 review")
    if parity.get("observedsteps") != 2 or active.get("observedsteps") != 2 or active.get("memory_runtime_passed") is not True:
        raise ValueError("both observed two-step smoke gates and memory/runtime gate required")
    _smoke_observations(Path(active["run_dir"]), manifest, True)
    parity_control(parity_path, manifest)
    return {"parity": str(parity_path), "active": str(active_path), "review": str(review_path)}


def _validate_zero_parity(parity, manifest):
    if parity.get("observedsteps") != 2:
        raise ValueError("zero parity requires observed two-step evidence")
    zero, traces = _smoke_observations(Path(parity["zero_smoke_run_dir"]), manifest, False)
    reference = Path(parity["reference_smoke_run_dir"])
    if reference.resolve() != (ROOT.parent / "block_opt/outputs/smoke_all_2").resolve():
        raise ValueError("zero smoke reference must be the audited sibling two-step control")
    prior = read_json(reference / "metrics.json")
    provenance = read_json(reference / "provenance.json")
    if (provenance.get("status") != "complete" or provenance.get("source_checkpoint_sha256") != manifest["files"][manifest["source_checkpoint"]]
            or prior.get("source_iteration") != 4000 or prior.get("end_iteration") != 4002
            or prior.get("forward_backward_evaluations") != 2):
        raise ValueError("zero smoke reference is not the completed frozen-checkpoint two-step run")
    old_traces = [json.loads(line) for line in (reference / "step_trace.jsonl").read_text().splitlines()]
    if zero["before"] != prior["before"] or len(old_traces) != 2:
        raise ValueError("zero smoke initial model/Adam differs from prior continuation")
    for row, old in zip(traces, old_traces):
        if row["absolute_iteration"] != old["absolute_iteration"] or row["wavelet_shift"] != old["wavelet_shift"]:
            raise ValueError("zero smoke iteration/wavelet sequence differs")
        for key in ("loss", "unclipped_grad_norm", "postclip_gradient_norm"):
            if not math.isfinite(old[key]) or abs(row[key] - old[key]) > 1e-7:
                raise ValueError("zero smoke loss/gradient parity failed")
    for scan in ("pre", "post"):
        for key in ("head_psnr", "head_ssim", "head_nmse", "head_hfen"):
            a, b = zero["rows"][scan][key], prior["rows"][scan][key]
            if not math.isfinite(a) or not math.isfinite(b) or abs(a - b) > .001:
                raise ValueError("zero smoke whole-head metric parity failed")


def input_freeze():
    """Hash scoring inputs plus the exact formal training checkpoint/acquisitions."""
    if sha256_file(SCORER) != SCORER_HASH or sha256_file(SCRIPTS / "arms.py") != ARMS_HASH:
        raise ValueError("frozen scorer/arms source hash changed")
    import yaml
    spec = importlib.util.spec_from_file_location("frozen_arms", SCRIPTS / "arms.py")
    arms = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arms)
    arms.R = TOPIC
    training_topic = TOPIC.with_name("CE_retro_cssense_af16_vdpois")
    source_run = training_topic / "trainer/_trainer_e778bf2_ar/results_arc/ar_i1_k30p90/ar_i1_k30p90"
    config = source_run / "config.yaml"
    checkpoint = source_run / "checkpoints/latest.pth"
    paths = {BASELINE, SCORER, SCRIPTS / "arms.py", SCRIPTS / "t1t2_common.py", ROI,
             TOPIC / "logs/summary.json", config, checkpoint}
    with config.open() as stream:
        cfg = yaml.safe_load(stream)
    for channel in ("plain", "contrast"):
        for key in ("csm_path", "data_path", "mask_path"):
            paths.add(Path(cfg["data"][channel][key]))
    for scan in ("pre", "post"):
        paths.add(TOPIC / f"recon/gt_{scan}_complex.npy")
        found = arms.arm_files(scan)
        if [label for _, label, _ in found] != list(LABELS):
            raise ValueError("frozen arms label order changed")
        paths.update(path for _, label, path in found if label != "MC-GS")
    manifest = {"version": 1, "baseline_sha256": BASELINE_SHA256,
                "source_checkpoint": str(checkpoint), "source_config": str(config),
                "files": {str(p): sha256_file(p) for p in sorted(paths)}}
    path = VERIFY / "loop_input_freeze.json"
    if path.exists():
        if read_json(path) != manifest:
            raise ValueError("frozen input hashes changed; halt rather than compare altered inputs")
    else:
        write_new(path, manifest)
    return manifest


def validate_run(run_dir, candidate, manifest, control=False):
    metrics = read_json(run_dir / "metrics.json")
    provenance = read_json(run_dir / "provenance.json")
    checkpoint_hash = manifest["files"][manifest["source_checkpoint"]]
    if (provenance.get("status") != "complete" or provenance.get("source_iteration") != 4000
            or provenance.get("end_iteration") != 4500 or provenance.get("source_checkpoint_sha256") != checkpoint_hash
            or provenance.get("source_checkpoint") != manifest["source_checkpoint"]
            or metrics.get("source_iteration") != 4000 or metrics.get("end_iteration") != 4500
            or metrics.get("forward_backward_evaluations") != 500):
        raise ValueError("run does not match the frozen independent4000+500 protocol")
    if metrics.get("mode") != "all":
        raise ValueError("formal nonlocal trial must update all parameter groups")
    head = metrics["rows"]
    if set(head) != {"pre", "post"}:
        raise ValueError("whole-head metrics for both scans required")
    for row in head.values():
        for key in ("head_psnr", "head_ssim", "head_nmse", "head_hfen"):
            if not math.isfinite(float(row[key])):
                raise ValueError("invalid whole-head metric")
    if not control:
        import yaml
        for record in (metrics, provenance):
            if any(record.get("overrides", {}).get(key) != candidate[key] for key in ("local_factor", "nlambda", "sigma_rel")):
                raise ValueError("candidate parameters disagree with observed runner overrides")
        with Path(manifest["source_config"]).open() as stream:
            original = yaml.safe_load(stream)
        with (run_dir / "actual_config.yaml").open() as stream:
            actual = yaml.safe_load(stream)
        for key in ("tv_weight", "lambda_wav", "lambda_wav_plain"):
            if actual["loss"].get(key) != float(original["loss"][key]) * candidate["local_factor"]:
                raise ValueError("actual local regularization differs from requested factor")
        expected_nl = {"lambda_nl": candidate["nlambda"], "lambda_nl_plain": candidate["nlambda"],
                       "lambda_nl_contrast": candidate["nlambda"], "nl_start": 4001, "nl_refresh": 250,
                       "nl_sigma_rel": candidate["sigma_rel"], "nl_denoiser": "patch_lowrank"}
        if any(actual["loss"].get(k) != v for k, v in expected_nl.items()):
            raise ValueError("actual nonlocal configuration differs from requested candidate")
        if actual["training"].get("resume_offset") != 4000 or actual["training"].get("max_iterations") != 4500:
            raise ValueError("actual training configuration differs from4000+500")
        nlr = metrics["nlr"]
        if nlr != provenance.get("nlr"):
            raise ValueError("metrics/provenance nonlocal observations disagree")
        if not nlr.get("all_gradients_finite"):
            raise ValueError("nonfinite observed gradients")
        if (nlr.get("sigma_rel") != candidate["sigma_rel"] or nlr.get("start") != 4001
                or nlr.get("refresh_every") != 250 or nlr.get("denoiser") != "patch_lowrank"):
            raise ValueError("observed nonlocal configuration differs from requested candidate")
        if any(nlr.get("coefficients", {}).get(scan) != candidate["nlambda"] for scan in ("pre", "post")):
            raise ValueError("requested and observed nonlocal coefficients differ")
        if candidate["nlambda"] > 0:
            if not (nlr.get("enabled") and nlr.get("validated_active") and nlr.get("all_gradients_finite")):
                raise ValueError("configured nonlocal prior was not observed active with finite gradients")
            if nlr.get("nonzero_weighted_step_count", 0) <= 0 or nlr.get("weighted_R_nl_max", 0) <= 0:
                raise ValueError("no active weighted R_nl")
            if nlr.get("active_step_count") != 500 or nlr.get("expected_refresh_iterations") != [4001, 4251]:
                raise ValueError("nonlocal prior must be active throughout all500 continuation steps")
            for scan in ("pre", "post"):
                if nlr.get("raw_R_nl_by_channel_max", {}).get(scan, 0) <= 0:
                    raise ValueError("nonlocal raw loss inactive for one scan")
                history = nlr["refresh_history"][scan]
                if [entry["iter"] for entry in history] != [4001, 4251]:
                    raise ValueError("actual nonlocal refresh schedule must be4001/4251")
                if any(entry["target_rms_difference"] <= 0 or not math.isfinite(entry["target_rms_difference"]) for entry in history):
                    raise ValueError("identity/nonfinite nonlocal target")
        elif nlr.get("enabled"):
            raise ValueError("zero-coefficient candidate unexpectedly enabled nonlocal prior")
    for scan in ("pre", "post"):
        if not (run_dir / f"candidate_{scan}_complex.npy").is_file():
            raise ValueError("missing candidate reconstruction")
    return head, provenance


def score_run(run_dir, candidate, baseline, book, manifest, control=False):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", candidate["name"]):
        raise ValueError("unsafe experiment name")
    head, provenance = validate_run(run_dir, candidate, manifest, control)
    # Re-hash before and after scoring. Read-only originals must stay fixed even
    # if another session updates a registered comparison while this run trains.
    if input_freeze() != manifest:
        raise ValueError("frozen scoring inputs changed")
    directory = VERIFY / "loop_scores" / f"{candidate['name']}_{time.time_ns()}"
    directory.mkdir(parents=True, exist_ok=False)
    raw = directory / "lesion_allmetrics.json"
    command = [str(PYTHON), str(SCORER), "--min_core", "150", "--mcgs_file",
               str(run_dir / "candidate_{scan}_complex.npy"), "--out", str(raw)]
    env = {**os.environ, "RETRO_TOPIC": str(TOPIC), "CUDA_VISIBLE_DEVICES": "",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "PYTHONPATH": ""}
    with (directory / "scorer.log").open("x") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    input_freeze()
    result = score(read_json(raw), baseline, head)
    result.update(run_dir=str(run_dir), candidate=candidate, scorer_command=command,
                  source_checkpoint_sha256=provenance["source_checkpoint_sha256"],
                  candidate_sha256={s: sha256_file(run_dir / f"candidate_{s}_complex.npy") for s in ("pre", "post")})
    output = directory / "objective.json"
    write_new(output, result)
    incumbent = score(baseline, baseline)["score"]
    for record in book.records:
        if record.get("event") == "trial_scored" and record.get("accepted"):
            incumbent = max(incumbent, record["score"])
    return {"score": result["score"], "strict_success": result["strict_success"],
            "accepted": is_improvement(result["score"], incumbent), "incumbent_before": incumbent,
            "objective": str(output), "best_plane": result["best_plane"], "whole_head_metrics": head}


def next_candidate(grid, book):
    tried = book.attempted()
    for candidate in grid["initial"]:
        if candidate["name"] not in tried:
            return dict(candidate)
    scored = [r for r in book.records if r.get("event") == "trial_scored" and r.get("candidate", {}).get("nlambda", 0) > 0]
    best = max(scored, key=lambda r: r["score"])["candidate"] if scored else grid["initial"][1]
    # Fix the measured adaptive parent once E is attempted; later measurements
    # do not silently move F/G's origin or resume into duplicate configs.
    prior = next((r["candidate"] for r in book.records if r.get("event") == "trial_started" and r.get("exp") == "E"), None)
    if prior:
        best = prior["adaptive_parent"]
    for name, field, factor in (("E", "nlambda", 1 / 3), ("F", "nlambda", 3), ("G", "sigma_rel", 2)):
        if name not in tried:
            candidate = {k: best[k] for k in ("local_factor", "nlambda", "sigma_rel")}
            candidate.update(name=name, adaptive_parent={k: best[k] for k in ("local_factor", "nlambda", "sigma_rel")},
                             rationale=f"Measured best nonlocal candidate: {field} times{factor:g}")
            candidate[field] *= factor
            return candidate
    return None


def parity_control(path, manifest):
    parity = _check_gate(path, smoke_code_paths(), manifest["files"][manifest["source_checkpoint"]])
    validate_final_drift(parity)
    _validate_zero_parity(parity, manifest)
    run = Path(parity["control_run_dir"]).resolve()
    if run != (ROOT.parent / "block_opt/outputs/all_500").resolve():
        raise ValueError("only the audited sibling all_500 control may be registered")
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--score-only", type=Path, metavar="EXISTING_RUN")
    ap.add_argument("--freeze-inputs", action="store_true", help="also create/verify input hashes in plan-only mode")
    ap.add_argument("--gpu", type=int, help="physical GPU index; mandatory only for training")
    ap.add_argument("--max-trials", type=int, default=6, help="campaign total formal attempts, bounded1..6")
    ap.add_argument("--control-parity", type=Path)
    ap.add_argument("--active-smoke", type=Path, default=VERIFY / "active_gate.json")
    ap.add_argument("--review-gate", type=Path, default=VERIFY / "review_gate.json")
    ap.add_argument("--gpu-evidence", type=Path, default=VERIFY / "loop_gpu_idle.jsonl")
    ap.add_argument("--candidate-name")
    ap.add_argument("--local-factor", type=float)
    ap.add_argument("--nlambda", type=float)
    ap.add_argument("--sigma-rel", type=float)
    args = ap.parse_args()
    if not 1 <= args.max_trials <= 6:
        ap.error("max-trials must be1..6")
    training = not args.plan_only and args.score_only is None
    if training and args.control_parity is None:
        ap.error("training requires --control-parity and valid active/review gates")
    if training and (args.gpu is None or args.gpu < 0):
        ap.error("training requires --gpu PHYSICAL preference")
    baseline = load_baseline(BASELINE)
    grid = read_json(ROOT / "candidate_grid.json")
    VERIFY.mkdir(exist_ok=True)
    with (VERIFY / "loop_driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        book = Ledger(VERIFY / "loop_results.jsonl")
        if args.plan_only:
            if args.freeze_inputs:
                input_freeze()
            print(json.dumps({"baseline": score(baseline, baseline)["score"], "grid": grid,
                              "attempted": sorted(book.attempted()), "next": next_candidate(grid, book)}, indent=2))
            return
        manifest = input_freeze()
        if training:
            validate_training_gates(args.control_parity, args.active_smoke, args.review_gate, manifest)
        if args.control_parity and "A_control" not in book.attempted():
            run = parity_control(args.control_parity, manifest)
            candidate = {"name": "A_control", "local_factor": 1., "nlambda": 0., "sigma_rel": .015}
            result = score_run(run, candidate, baseline, book, manifest, control=True)
            book.append({"event": "trial_scored", "exp": "A_control", "candidate": candidate,
                         "run_dir": str(run), "control_parity": str(args.control_parity.resolve()), **result})
        if args.score_only:
            run = args.score_only.resolve()
            if not run.is_relative_to(ROOT / "outputs"):
                if not args.control_parity or run != parity_control(args.control_parity, manifest):
                    ap.error("score-only outside campaign requires the passed sibling control parity gate")
                return  # Already registered above, or prior ledger record preserved.
            name = args.candidate_name or run.name
            if any(r.get("event") == "trial_scored" and r.get("exp") == name for r in book.records):
                print(f"{name} already scored; preserving existing result")
                return
            recorded = next((r.get("candidate") for r in book.records if r.get("exp") == name and r.get("candidate")), None)
            candidate = recorded or next((dict(c) for c in grid["initial"] if c["name"] == name), None)
            if candidate is None:
                if None in (args.local_factor, args.nlambda, args.sigma_rel):
                    ap.error("unregistered score-only candidate requires all three hyperparameters")
                candidate = {"name": name, "local_factor": args.local_factor, "nlambda": args.nlambda, "sigma_rel": args.sigma_rel}
            result = score_run(run, candidate, baseline, book, manifest)
            book.append({"event": "trial_scored", "exp": name, "candidate": candidate, "run_dir": str(run), **result})
            print(json.dumps(result, indent=2))
            return
        if args.gpu is None or args.gpu < 0:
            ap.error("training requires --gpu PHYSICAL")
        failures = 0
        while len(book.attempted() - {"A_control"}) < args.max_trials:
            if any(r.get("strict_success") for r in book.records if r.get("event") == "trial_scored"):
                print("Strict provisional witness observed; stop for fair single/repeat audit.")
                break
            candidate = next_candidate(grid, book)
            if candidate is None:
                break
            name = candidate["name"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise ValueError("unsafe experiment name")
            gates = validate_training_gates(args.control_parity, args.active_smoke, args.review_gate, manifest)
            observation = wait_for_gpu(args.gpu, args.gpu_evidence)
            selected_gpu = observation["gpu"]
            book.append({"event": "gpu_claim_ready", "exp": name, "observation": observation,
                         "monitor": str(args.gpu_evidence), "validated_gates": gates})
            command = [str(PYTHON), str(ROOT / "runner.py"), "--mode", "all", "--gpu", str(selected_gpu),
                       "--exp", name, "--steps", "500", "--local-factor", str(candidate["local_factor"]),
                       "--nlambda", str(candidate["nlambda"]), "--sigma-rel", str(candidate["sigma_rel"]),
                       "--output", str(ROOT / "outputs")]
            environment = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(selected_gpu),
                           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "PYTHONPATH": ""}
            ok = execute_trial(book, candidate, ROOT / "outputs" / name, command, environment,
                               lambda run, c: score_run(run, c, baseline, book, manifest))
            failures = 0 if ok else failures + 1
            if not ok and (book.records[-1].get("stage") == "scoring" or failures >= 2):
                print("Stopping after scoring/invariant failure or two consecutive training failures; inspect preserved ledger.")
                break


if __name__ == "__main__":
    main()
