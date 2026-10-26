#!/usr/bin/env python3
"""Record one live resource decision for an authorized zero-step single audit."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import autoresearch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scan", choices=("pre", "post"), required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
output = args.output.resolve()
if output.parent != ROOT / "verification" or not output.name.startswith("single_gpu_"):
    parser.error("resource evidence must be a fresh verification/single_gpu_* file")
cards = autoresearch.gpu_snapshot()
report = {"scan": args.scan, "reserved_for_main_loop": [5], "full_snapshot": cards,
          "environment_constraints": {key: os.environ[key] for key in
              ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS")
              if key in os.environ}}
try:
    report["selection"] = autoresearch.select_gpu([card for card in cards if card["gpu"] == 7], 7, os.environ)
except Exception as error:
    report["selection_error"] = repr(error)
with output.open("x") as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
    stream.write("\n")
print(json.dumps(report, indent=2, allow_nan=False))
if "selection_error" in report:
    raise SystemExit(2)
