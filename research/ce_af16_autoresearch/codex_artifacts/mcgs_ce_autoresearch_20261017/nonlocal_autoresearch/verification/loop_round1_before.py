#!/usr/bin/env python3
"""Bounded, resumable CE AF16 development loop. Default execution waits for a quiet GPU.

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


def gpu_probe(gpu):
    # Reuse the established process-environment claim check, then strengthen its
    # 1500MiB threshold to this campaign's strict100MiB threshold.
    sys.path.insert(0, str(SCRIPTS))
    try:
        import ar_free_queue
        helper_free = gpu in ar_free_queue.empty_cards({gpu})
    finally:
        sys.path.pop(0)
    env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    cards = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=True, env=env)
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
                          capture_output=True, text=True, check=True, env=env)
    busy = {line.strip() for line in apps.stdout.splitlines() if line.strip()}
    for line in cards.stdout.splitlines():
        index, uuid, memory = [part.strip() for part in line.split(",")]
        if int(index) == gpu:
            used = float(memory)
            return {"timestamp": time.time(), "gpu": gpu, "used_mib": used, "uuid": uuid,
                    "compute_process_present": uuid in busy, "helper_free_and_unclaimed": helper_free,
                    "free": helper_free and used < 100 and uuid not in busy}
    raise ValueError(f"physical GPU{gpu} missing from nvidia-smi")


def wait_for_gpu(gpu, evidence):
    monitor = Ledger(evidence)
    while True:
        row = gpu_probe(gpu)
        monitor.append(row)
        if quiet_ready(monitor.records, gpu, row["timestamp"]):
            return row
        print(f"GPU{gpu}: free={row['free']}, used={row['used_mib']}MiB; waiting for600 quiet seconds", flush=True)
        time.sleep(30)


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
    parity = read_json(path)
    if parity.get("passed") is not True or not parity.get("control_run_dir"):
        raise ValueError("control requires parent-created passed parity_gate.json with control_run_dir")
    if parity.get("source_checkpoint_sha256") != manifest["files"][manifest["source_checkpoint"]]:
        raise ValueError("control parity source checkpoint hash differs")
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
    ap.add_argument("--gpu-evidence", type=Path, default=VERIFY / "loop_gpu_idle.jsonl")
    ap.add_argument("--candidate-name")
    ap.add_argument("--local-factor", type=float)
    ap.add_argument("--nlambda", type=float)
    ap.add_argument("--sigma-rel", type=float)
    args = ap.parse_args()
    if not 1 <= args.max_trials <= 6:
        ap.error("max-trials must be1..6")
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
            observation = wait_for_gpu(args.gpu, args.gpu_evidence)
            book.append({"event": "gpu_claim_ready", "exp": name, "observation": observation,
                         "monitor": str(args.gpu_evidence), "quiet_seconds": 600})
            command = [str(PYTHON), str(ROOT / "runner.py"), "--mode", "all", "--gpu", str(args.gpu),
                       "--exp", name, "--steps", "500", "--local-factor", str(candidate["local_factor"]),
                       "--nlambda", str(candidate["nlambda"]), "--sigma-rel", str(candidate["sigma_rel"]),
                       "--output", str(ROOT / "outputs")]
            environment = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(args.gpu),
                           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "PYTHONPATH": ""}
            ok = execute_trial(book, candidate, ROOT / "outputs" / name, command, environment,
                               lambda run, c: score_run(run, c, baseline, book, manifest))
            failures = 0 if ok else failures + 1
            if not ok and (book.records[-1].get("stage") == "scoring" or failures >= 2):
                print("Stopping after scoring/invariant failure or two consecutive training failures; inspect preserved ledger.")
                break


if __name__ == "__main__":
    main()
