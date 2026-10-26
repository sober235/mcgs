#!/usr/bin/env python3
"""Score the completed infrastructure retry without writing the live campaign ledger."""
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import autoresearch as ar


def main():
    marker = ROOT / "verification/B_retry_completed.json"
    if not marker.exists():
        raise ValueError("Retry runner has not completed; no partial result is scored")
    output = ROOT / "verification/B_retry_score_pending_registration.json"
    if output.exists():
        raise FileExistsError(output)
    candidate = {"name": "B_retry", "local_factor": .5, "nlambda": 0., "sigma_rel": .015}
    manifest = ar.input_freeze()
    baseline = ar.load_baseline(ar.BASELINE)
    snapshot = ar.Ledger(ROOT / "verification/loop_results.jsonl")  # read only; never append here
    result = ar.score_run(ROOT / "outputs/B_retry", candidate, baseline, snapshot, manifest)
    ar.write_new(output, {"candidate": candidate, **result,
        "registration_status": "pending until main driver releases its lock; recompute incumbent acceptance then"})
    print({k: result[k] for k in ("score", "strict_success", "accepted")})


if __name__ == "__main__":
    main()
