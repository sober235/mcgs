#!/usr/bin/env python3
"""Validate completed two-step outputs on CPU; this utility never launches GPU work."""
import argparse
import json
import math
from pathlib import Path

import autoresearch as ar

ROOT = Path(__file__).resolve().parent
VERIFY = ROOT / "verification"
FILES = [ROOT / "runner.py", *(ROOT / "trainer/paired_gs" / rel for rel in (
    "trainers/joint_trainer.py", "trainers/resume_protocol.py",
    "losses/nonlocal_red.py", "losses/nonlocal_lowrank.py"))]


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def difference_record(reference, candidate):
    import numpy as np
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    require(reference.shape == candidate.shape, "final arrays have different shapes")
    require(np.isfinite(reference).all() and np.isfinite(candidate).all(), "nonfinite final array")
    dtype = np.complex128 if np.iscomplexobj(reference) or np.iscomplexobj(candidate) else np.float64
    original, current = reference.astype(dtype), candidate.astype(dtype)
    difference = current - original
    return {"max_absolute": float(np.abs(difference).max()),
            "relative_l2": float(np.linalg.norm(difference.ravel()) / max(np.linalg.norm(original.ravel()), 1e-30))}


def final_drift(reference_dir, zero_dir, reference_metrics, zero_metrics):
    import numpy as np
    import torch
    torch.set_num_threads(8)
    paths = {"reference": Path(reference_metrics["final_checkpoint"]), "zero": Path(zero_metrics["final_checkpoint"])}
    left, right = [torch.load(paths[key], map_location="cpu", weights_only=True) for key in ("reference", "zero")]
    require(left["iter"] == right["iter"] == 4002, "final checkpoint step differs")
    require(set(left["model_state"]) == set(right["model_state"]) == set(ar.NAMES if hasattr(ar, "NAMES") else (
        "positions", "scales", "rotations", "density_real", "density_imag", "density_real_contrast", "density_imag_contrast")), "model names differ")
    model = {name: difference_record(value.numpy(), right["model_state"][name].numpy()) for name, value in left["model_state"].items()}
    a, b = left["optim_state"]["state"], right["optim_state"]["state"]
    require(set(a) == set(b) and len(a) == 7, "Adam groups differ")
    adam = {}
    for group in a:
        require(float(a[group]["step"]) == float(b[group]["step"]) == 4002, "Adam step differs")
        adam[str(group)] = {key: difference_record(a[group][key].numpy(), b[group][key].numpy())
                            for key in ("exp_avg", "exp_avg_sq")}
    reconstruction, candidate_hashes = {}, {"zero": {}, "reference": {}}
    for scan in ("pre", "post"):
        p, q = reference_dir / f"candidate_{scan}_complex.npy", zero_dir / f"candidate_{scan}_complex.npy"
        reconstruction[scan] = difference_record(np.load(p), np.load(q))
        candidate_hashes["reference"][scan], candidate_hashes["zero"][scan] = ar.sha256_file(p), ar.sha256_file(q)
    require(all(row["relative_l2"] <= 1e-6 for row in model.values()), "model numerical parity failed")
    require(all(row["relative_l2"] <= 1e-4 for group in adam.values() for row in group.values()), "Adam numerical parity failed")
    require(all(row["relative_l2"] <= 1e-5 for row in reconstruction.values()), "reconstruction numerical parity failed")
    return {"passed": True, "zero_run_dir": str(zero_dir), "reference_run_dir": str(reference_dir),
            "checkpoint_sha256": {key: ar.sha256_file(path) for key, path in paths.items()},
            "candidate_sha256": candidate_hashes, "model": model, "adam": adam, "reconstruction": reconstruction,
            "tolerances": {"model_relative_l2": 1e-6, "adam_relative_l2": 1e-4, "reconstruction_relative_l2": 1e-5}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-existing", action="store_true", help="validate completed v1 outputs without launching GPU work")
    parser.add_argument("--gate-suffix", default="_v2", help="fresh immutable gate/evidence suffix")
    args = parser.parse_args()
    if not args.validate_existing:
        parser.error("Only --validate-existing is supported; GPU execution belongs to the runner and current-resource selector")
    require(args.gate_suffix and all(c.isalnum() or c in "_-" for c in args.gate_suffix), "invalid gate suffix")
    hashes = {str(path): ar.sha256_file(path) for path in FILES}
    zero_dir, active_dir = [ROOT / "outputs" / f"smoke_{kind}_nlr_v1" for kind in ("zero", "active")]
    if hashes != {str(path): ar.sha256_file(path) for path in FILES}:
        raise RuntimeError("Implementation changed during preflight")
    z, a = read(zero_dir / "metrics.json"), read(active_dir / "metrics.json")
    zp, ap = read(zero_dir / "provenance.json"), read(active_dir / "provenance.json")
    for directory, provenance in ((zero_dir, zp), (active_dir, ap)):
        require(provenance["runner_sha256"] == hashes[str(ROOT / "runner.py")], "runner hash differs")
        manifest = read(directory / "source_to_copy_manifest.json")
        for path in FILES[1:]:
            relative = path.relative_to(ROOT / "trainer").as_posix()
            require(manifest["files"][relative]["copy_sha256"] == hashes[str(path)], "trainer hash differs")
    old_dir = ROOT.parent / "block_opt/outputs/smoke_all_2"
    old = read(old_dir / "metrics.json")
    traces = [[json.loads(line) for line in path.read_text().splitlines()] for path in (
        old_dir / "step_trace.jsonl", zero_dir / "step_trace.jsonl")]
    differences = []
    for left, right in zip(*traces):
        require(left["absolute_iteration"] == right["absolute_iteration"], "iteration differs")
        require(left["wavelet_shift"] == right["wavelet_shift"], "wavelet sequence differs")
        require(left["learning_rates_used"] == right["learning_rates_used"], "learning rates differ")
        for key in ("loss", "unclipped_grad_norm", "postclip_gradient_norm"):
            differences.append(abs(left[key] - right[key]))
        differences.extend(abs(left["weighted_components"][k] - right["weighted_components"][k])
                           for k in left["weighted_components"])
    require(len(traces[0]) == len(traces[1]) == 2, "not two observed steps")
    require(all(math.isfinite(d) and d <= 1e-7 for d in differences), "loss/gradient numerical parity failed")
    require(z["before"] == old["before"], "initial model or Adam differs")
    for scan in ("pre", "post"):
        require(abs(z["rows"][scan]["head_psnr"] - old["rows"][scan]["head_psnr"]) <= .001, "head PSNR parity failed")
    require(zp["status"] == ap["status"] == "complete", "incomplete smoke")
    require(z["source_checkpoint_sha256"] == a["source_checkpoint_sha256"], "source checkpoint differs")
    nlr = a["nlr"]
    require(nlr["validated_active"] and nlr["all_gradients_finite"] and nlr["active_step_count"] == 2, "inactive or invalid prior")
    refreshes = []
    for scan in ("pre", "post"):
        history = nlr["refresh_history"][scan]
        require([r["iter"] for r in history] == [4001], "wrong actual refresh schedule")
        require(nlr["raw_R_nl_by_channel_max"][scan] > 0, "inactive channel")
        for row in history:
            require(row["target_rms_difference"] > 0 and math.isfinite(row["target_rms_difference"]) and 0 <= row["seconds"] < 300, "invalid target or excessive runtime")
            mem = row["kernel_stats"]["memory"]
            require(mem["cuda_allocated_after"] - mem["cuda_allocated_before"] < 2 * 1024**3, "excessive scratch allocation")
            refreshes.append({"scan": scan, **row})
    require(a["cuda_memory_bytes"]["peak_allocated"] < 75 * 1024**3, "excessive allocated peak")
    drift = final_drift(old_dir, zero_dir, old, z)
    drift_path = VERIFY / f"parity_final_drift{args.gate_suffix}.json"
    ar.write_new(drift_path, drift)
    common = dict(passed=True, source_checkpoint_sha256=z["source_checkpoint_sha256"], code_sha256=hashes)
    ar.write_new(VERIFY / f"parity_gate{args.gate_suffix}.json", {**common,
        "control_run_dir": str(ROOT.parent / "block_opt/outputs/all_500"),
        "zero_smoke_run_dir": str(zero_dir), "reference_smoke_run_dir": str(old_dir),
        "maximum_loss_gradient_difference": max(differences), "observedsteps": 2,
        "final_drift_file": str(drift_path), "final_drift_sha256": ar.sha256_file(drift_path)})
    ar.write_new(VERIFY / f"active_gate{args.gate_suffix}.json", {**common, "run_dir": str(active_dir),
        "observedsteps": 2, "memory_runtime_passed": True, "refreshes": refreshes,
        "peak_allocated_bytes": a["cuda_memory_bytes"]["peak_allocated"]})
    print(json.dumps({"parity": "passed", "active": "passed", "refresh_seconds": [r["seconds"] for r in refreshes]}, indent=2))


if __name__ == "__main__":
    main()
