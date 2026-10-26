#!/usr/bin/env python3
"""Append resource observations until STOP_RESOURCE_MONITOR exists; never launch work."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent


def observe():
    gpu_result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True)
    app_result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
        check=True, text=True, capture_output=True)
    active = set(app_result.stdout.splitlines())
    card_lines = [[x.strip() for x in line.split(",")] for line in gpu_result.stdout.splitlines()]
    uuid_to_index = {uuid: int(i) for i, uuid, _, _ in card_lines}
    claimed = set()
    for entry in Path("/proc").glob("[0-9]*/environ"):
        try:
            raw = entry.read_bytes()
        except OSError:
            continue
        for item in raw.split(b"\0"):
            if item.startswith(b"CUDA_VISIBLE_DEVICES="):
                for raw_token in item.split(b"=", 1)[1].split(b","):
                    token = raw_token.decode(errors="replace").strip()
                    if token.isdigit():
                        claimed.add(int(token))
                    elif token.startswith("GPU-"):
                        claimed.update(idx for uuid, idx in uuid_to_index.items() if uuid.startswith(token))
    rows = []
    for i, uuid, used, utilization in card_lines:
        idx = int(i)
        rows.append(dict(gpu=idx, uuid=uuid, used_mib=float(used), utilization=float(utilization),
                         active_process=uuid in active, claimed=idx in claimed,
                         free=float(used) < 100 and uuid not in active and idx not in claimed))
    return dict(timestamp=time.time(), pid=os.getpid(), claim_detection_version=2, gpus=rows)


def main():
    with (ROOT / "verification/resource_observations.jsonl").open("a", buffering=1) as stream:
        while not (ROOT / "STOP_RESOURCE_MONITOR_V2").exists():
            try:
                row = observe()
            except Exception as exc:
                row = dict(timestamp=time.time(), error=repr(exc))
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            time.sleep(30)


if __name__ == "__main__":
    main()
