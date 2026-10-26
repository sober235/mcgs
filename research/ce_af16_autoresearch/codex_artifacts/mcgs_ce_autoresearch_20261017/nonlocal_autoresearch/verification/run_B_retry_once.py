#!/usr/bin/env python3
"""Retry the interrupted B configuration once, with a visible CPU-parent GPU reservation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import autoresearch as ar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reserved-gpu", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    run = ROOT / "outputs/B_retry"
    if run.exists():
        raise ValueError("B_retry was already attempted; preserve it")
    if args.reserved_gpu is None:
        # Reserve only an actually idle card for this infrastructure retry.
        cards = ar.gpu_snapshot()
        selected = ar.select_gpu([c for c in cards if c["free"]], 7, os.environ)
        gpu = selected["gpu"]
        env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(gpu)}
        # exec makes CVD part of this parent's INITIAL environment. The main
        # selector then sees a loading reservation for the entire child run.
        os.execve(sys.executable, [sys.executable, "-u", str(Path(__file__).resolve()), "--reserved-gpu", str(gpu)], env)
    gpu = args.reserved_gpu
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise ValueError("missing initial parent reservation")
    manifest = ar.input_freeze()
    gates = ar.validate_training_gates(ROOT / "verification/parity_gate_v2.json",
        ROOT / "verification/active_gate_v2.json", ROOT / "verification/review_gate_scoring_v2.json", manifest)
    selected = ar.select_gpu(ar.gpu_snapshot(), gpu, os.environ)
    if selected["gpu"] != gpu or not selected["free"]:
        raise ValueError("reserved retry card is no longer idle; no workload was changed")
    candidate = {"name": "B_retry", "local_factor": .5, "nlambda": 0., "sigma_rel": .015}
    command = [str(ar.PYTHON), "-u", str(ROOT / "runner.py"), "--mode", "all", "--gpu", str(gpu),
        "--exp", "B_retry", "--steps", "500", "--local-factor", ".5", "--nlambda", "0", "--sigma-rel", ".015"]
    ar.write_new(ROOT / "verification/B_retry_launch.json", {"timestamp": time.time(), "candidate": candidate,
        "command": command, "resource": selected, "parent_pid": os.getpid(), "gates": gates,
        "reason": "B interrupted by concurrent audit memory use; same hypothesis, independent original checkpoint restart"})
    env = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1"}
    with (ROOT / "verification/B_retry_launch.log").open("x") as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        ar.write_new(ROOT / "verification/B_retry_failure.json", {"returncode": result.returncode, "timestamp": time.time()})
        raise RuntimeError("B retry failed; preserve artifacts and inspect logs")
    head, provenance = ar.validate_run(run, candidate, manifest)
    ar.write_new(ROOT / "verification/B_retry_completed.json", {"timestamp": time.time(), "candidate": candidate,
        "whole_head_metrics": head, "status": provenance["status"],
        "instruction": "Register with score-only after main driver releases its ledger lock; no simultaneous ledger writer"})
    print(json.dumps({"completed": "B_retry", "gpu": gpu, "whole_head_metrics": head}, indent=2))


if __name__ == "__main__":
    main()
