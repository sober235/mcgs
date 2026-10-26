#!/usr/bin/env python3
"""Frozen CPU three-metric scoring; candidate arrays already use the scoring gauge.

The primary objective is whole-head min-six, while the original max-plane S6 is
reported separately. This module never fits a scale or computes another metric.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "mcgs_ce_autoresearch_20261017"
NONLOCAL = OLD / "nonlocal_autoresearch"
TOPIC = Path("/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois_cmp")
SCRIPTS = Path("/data0/congcong/code/GS/mcgs_ce_mask_wt/ce_retro_cssense_af16/scripts")
ROI = Path("/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast/lesion_candidates/enhance_rois_cssense.npz")
BASELINE = OLD / "baseline_lesion_allmetrics.json"
CONTRACT = OLD / "reassessment_without_hfen_20261019/metric_contract_v1.json"
LOOP_FREEZE = NONLOCAL / "verification/loop_input_freeze.json"
REPORT_FREEZE = NONLOCAL / "verification/report_reference_freeze.json"
SCANS = ("pre", "post")
METRICS = ("psnr", "ssim", "nmse")
CONDITIONS = tuple(f"{scan}:{metric}" for scan in SCANS for metric in METRICS)
OPPONENTS = ("ZF", "CG-SENSE", "CS-SENSE", "Joint SENSE", "L1-SPIRiT", "ZS-SSDU",
             "3D-DIP", "3D-INR", "MC-INR", "GS-single")
FROZEN_SOURCE_HASHES = {
    str(LOOP_FREEZE): "fecc6fba1edc03fe06377717f45c8c51923c6b8656b123f7ed878f5288a358d7",
    str(REPORT_FREEZE): "6ebe48fa543af773274f7d0c10bc0c21593669b9305784a8382fab7e9896e634",
    str(CONTRACT): "7404413f695bce3b59886de802815011615f7473a0a59626542ffd4dbe60cd13",
    str(NONLOCAL / "report.py"): "fcce12451ba3983a00b232e4c7c34f927a7c7a6827949d9e0e5f8c657d20b800",
    str(SCRIPTS / "diag_artefact/headonly.py"): "a005c5de667a9f671eb33531cb68648c55cea064d3594c83e1c0d748483f0f9e",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError(f"nonfinite JSON value {value} in {path}")
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def verify_hashes(hashes):
    for path, digest in hashes.items():
        if sha256_file(path) != digest:
            raise ValueError(f"frozen input/source changed: {path}")


def _validate_value(value, metric):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value):
        raise ValueError(f"requires finite numeric {metric}")
    if metric == "ssim" and not -1 <= value < 1:
        raise ValueError("SSIM must be in [-1,1) for finite margins")
    if metric == "nmse" and value <= 0:
        raise ValueError("NMSE must be strictly positive for finite margins")


def _validate_heads(heads):
    if set(heads) != set(SCANS):
        raise ValueError("both and only pre/post scans required")
    for values in heads.values():
        if set(values) != set(METRICS):
            raise ValueError("exactly PSNR, SSIM, NMSE required")
        for metric, value in values.items():
            _validate_value(value, metric)


def _log_ratio(numerator, denominator):
    delta = numerator - denominator
    return (math.log1p(delta / denominator) if abs(delta) < denominator / 2
            else math.log(numerator) - math.log(denominator))


def margin(candidate, best, metric):
    if metric not in METRICS:
        raise ValueError(f"unknown decision metric: {metric}")
    _validate_value(candidate, metric)
    _validate_value(best, metric)
    if metric == "psnr":
        return float((candidate - best) * math.log(10) / 10)
    if metric == "ssim":
        return _log_ratio(1 - best, 1 - candidate)
    return _log_ratio(best, candidate)


def comparisons(candidate, references):
    _validate_heads(candidate)
    if set(references) != set(OPPONENTS):
        raise ValueError("all ten complete frozen opponents required")
    for heads in references.values():
        _validate_heads(heads)
    result = {}
    for scan in SCANS:
        for metric in METRICS:
            values = {label: float(references[label][scan][metric]) for label in OPPONENTS}
            best = (max if metric != "nmse" else min)(values.values())
            value = float(candidate[scan][metric])
            result[f"{scan}:{metric}"] = {
                "candidate": value, "best_opponent": best,
                "labels": [label for label, number in values.items() if number == best],
                "opponents": values, "strict_win": value < best if metric == "nmse" else value > best,
                "margin": margin(value, best, metric),
            }
    return result


def summarize_whole_head(candidate, references):
    compared = comparisons(candidate, references)
    return {"score": min(value["margin"] for value in compared.values()),
            "strict_success": all(value["strict_win"] for value in compared.values()),
            "metrics": candidate, "comparisons": compared,
            "region": "whole-head mask, independently for pre and post"}


def summarize_planes(observed, baseline):
    if len(observed) != 59 or len(baseline) != 59:
        raise ValueError("exactly 59 frozen lesion-containing planes required")
    planes, seen = [], set()
    for candidate, frozen in zip(observed, baseline):
        key = (frozen["axis"], frozen["index"], frozen["core"])
        if (key != (candidate["axis"], candidate["index"], candidate["core"])
                or key[:2] in seen or key[0] not in (0, 1, 2) or key[1] < 0 or key[2] < 150):
            raise ValueError("frozen plane axis/index/core/order changed")
        seen.add(key[:2])
        refs = {label: {s: {m: frozen["scans"][s]["metrics"][label][m] for m in METRICS}
                        for s in SCANS} for label in OPPONENTS}
        compared = comparisons(candidate["metrics"], refs)
        planes.append({"axis": key[0], "index": key[1], "core": key[2],
                       "score_six": min(c["margin"] for c in compared.values()),
                       "strict_success": all(c["strict_win"] for c in compared.values()),
                       "comparisons": compared})
    best = max(planes, key=lambda plane: plane["score_six"])
    return {"score_six": best["score_six"], "strict_success": any(p["strict_success"] for p in planes),
            "eligible_planes": 59, "best_plane": best,
            "passing_planes_both_scans": sum(p["strict_success"] for p in planes),
            "passing_planes_by_scan": {s: sum(all(p["comparisons"][f"{s}:{m}"]["strict_win"] for m in METRICS)
                                                for p in planes) for s in SCANS},
            "per_condition_passing_planes": {c: sum(p["comparisons"][c]["strict_win"] for p in planes) for c in CONDITIONS},
            "region": "head mask on each frozen lesion-containing plane; not lesion ROI",
            "planes": planes}


def _load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _headonly():
    # A private package keeps the original relative import independent of any
    # caller's diag_artefact module. External sources remain read-only.
    sys.dont_write_bytecode = True
    name = "_ce_campaign_frozen_headonly"
    if name + ".headonly" not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(SCRIPTS / "diag_artefact")]
        sys.modules[name] = package
        _load_source(SCRIPTS / "diag_artefact/headonly.py", name + ".headonly")
    return sys.modules[name + ".headonly"]


def whole_head_metrics(candidate, gt):
    """Exact head_* subset of the original Pair.score CPU128 path.

    GT complex64 magnitude is calculated before promotion, as Pair.__init__
    does. PSNR/SSIM convert to float32 inside score_headonly; NMSE does not.
    """
    import torch
    helper = _headonly()
    rec = torch.from_numpy(np.array(candidate, copy=True)).to(torch.complex128).abs()
    truth = torch.from_numpy(np.array(gt, dtype=np.complex64, copy=True)).abs().to(torch.float64)
    if rec.shape != truth.shape or rec.ndim != 3 or not bool(torch.isfinite(rec).all()) or not bool(torch.isfinite(truth).all()):
        raise ValueError("whole-head requires matching finite 3-D complex volumes")
    rec32, gt32 = rec.numpy().astype(np.float32), truth.numpy().astype(np.float32)
    head32 = helper.head_mask(gt32)
    head64 = torch.from_numpy(helper.head_mask(truth.numpy()))
    if not head32.any() or not bool(head64.any()):
        raise ValueError("empty whole-head mask")
    gmax = float(gt32.max())
    result = {"psnr": helper.head_psnr(rec32, gt32, head32, gmax),
              "ssim": helper.head_ssim(rec32, gt32, head32, gmax),
              "nmse": float(((rec - truth)[head64] ** 2).sum()) / float((truth[head64] ** 2).sum())}
    for metric, value in result.items():
        _validate_value(value, metric)
    return result


def load_whole_head_gt(topic, reference_freeze):
    """Reproduce Pair.__init__ using the original stored scalar, without fitting."""
    import torch
    result = {}
    with np.load(Path(topic) / "sim/gt.npz", allow_pickle=False) as archive:
        for scan in SCANS:
            scale = float(archive[f"scale_{scan}"])
            if (not math.isfinite(scale) or scale <= 0
                    or scale != reference_freeze["gt_equivalence"][scan]["scale_float64"]):
                raise ValueError(f"frozen {scan} scoring scale changed")
            source = torch.from_numpy(np.asarray(archive[scan])).to(torch.complex64)
            result[scan] = (source / scale).numpy()
    return result


def plane_metrics(rec, gt, head, gmax, axis, index):
    """Original lesion_slice_allmetrics arithmetic, without unused metrics."""
    from skimage.metrics import structural_similarity
    truth = np.take(gt, index, axis=axis).astype(np.float64)
    value = np.take(rec, index, axis=axis).astype(np.float64)
    mask = np.take(head, index, axis=axis)
    if not mask.any():
        raise ValueError("empty lesion-plane head mask")
    _, smap = structural_similarity(value, truth, data_range=gmax, full=True)
    result = {"psnr": float(20 * np.log10(gmax) - 10 * np.log10(np.mean((value[mask] - truth[mask]) ** 2))),
              "ssim": float(smap[mask].mean()),
              "nmse": float(np.sum((value[mask] - truth[mask]) ** 2) / np.sum(truth[mask] ** 2))}
    for metric, number in result.items():
        _validate_value(number, metric)
    return result


def load_candidate_pair(run_dir, shape=None):
    pair, evidence = {}, {}
    for scan in SCANS:
        path = Path(run_dir) / f"candidate_{scan}_complex.npy"
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if (value.ndim != 3 or not np.iscomplexobj(value) or not np.isfinite(value).all()
                or (shape is not None and tuple(value.shape) != tuple(shape))):
            raise ValueError(f"requires matching finite complex 3-D {scan} volume: {path}")
        shape = value.shape
        pair[scan] = value
        evidence[scan] = {"path": str(path.resolve()), "sha256": sha256_file(path),
                          "shape": list(value.shape), "dtype": str(value.dtype)}
    return pair, evidence


def _load_report():
    # The reused enhancement helper imports old CLI modules. Temporarily bind
    # those imports without replacing a supervisor's own objective module.
    sys.dont_write_bytecode = True
    saved = {name: sys.modules.get(name) for name in ("objective", "autoresearch")}
    try:
        _load_source(NONLOCAL / "objective.py", "objective")
        _load_source(NONLOCAL / "autoresearch.py", "autoresearch")
        return _load_source(NONLOCAL / "report.py", "_ce_campaign_frozen_report")
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def load_frozen_context():
    verify_hashes(FROZEN_SOURCE_HASHES)
    manifest, report_freeze, contract = map(read_json, (LOOP_FREEZE, REPORT_FREEZE, CONTRACT))
    if (contract["metrics"] != list(METRICS) or contract["opponents"] != list(OPPONENTS)
            or contract["eligible_planes"] != 59
            or report_freeze["loop_input_freeze_sha256"] != FROZEN_SOURCE_HASHES[str(LOOP_FREEZE)]):
        raise ValueError("frozen three-metric contract/manifests disagree")
    hashes = dict(FROZEN_SOURCE_HASHES)
    # Score-only inputs; acquired k-space/CSM and training checkpoints are not loaded.
    required = {str(BASELINE), str(SCRIPTS / "arms.py"), str(SCRIPTS / "t1t2_common.py"),
                str(SCRIPTS / "lesion_slice_allmetrics.py"), str(TOPIC / "logs/summary.json")}
    required.update(path for path in manifest["files"] if path.startswith(str(TOPIC / "recon")))
    for path in required:
        hashes[path] = manifest["files"][path]
    for path, digest in report_freeze["files"].items():
        if path in hashes and hashes[path] != digest:
            raise ValueError(f"conflicting frozen hash: {path}")
        hashes[path] = digest
    for path in (NONLOCAL / "objective.py", NONLOCAL / "autoresearch.py",
                 SCRIPTS / "diag_artefact/volumes.py", SCRIPTS / "enhancement_fidelity.py",
                 *(TOPIC / f"logs/zsssdu_{scan}.json" for scan in SCANS)):
        hashes[str(path)] = sha256_file(path)
    verify_hashes(hashes)
    report = _load_report()
    references = report.load_references(TOPIC, manifest)
    if set(references) != {*OPPONENTS, "MC-GS"}:
        raise ValueError("all ten complete frozen opponents required")
    heads = {label: {s: {m: references[label]["whole_head_metrics"][s][m] for m in METRICS}
                     for s in SCANS} for label in OPPONENTS}
    comparisons(heads[OPPONENTS[0]], heads)  # Strict validation, without altering values.
    canonical = {label: {"key": references[label]["key"],
                         "whole_head_metrics": heads[label],
                         "volumes": {s: {"path": references[label]["volumes"][s],
                                          "sha256": hashes[references[label]["volumes"][s]]} for s in SCANS},
                         "metric_source": {"path": str(TOPIC / "logs/summary.json"),
                                           "sha256": hashes[str(TOPIC / "logs/summary.json")]}}
                 for label in OPPONENTS}
    whole_contract = {
        "version": 1, "metrics": list(METRICS), "scans": list(SCANS),
        "opponents": canonical, "ranking": "minimum of six normalized whole-head margins",
        "strict_success": "all six raw inequalities strictly beat all ten frozen opponents; no tolerance",
        "normalization": {m: contract["normalization"][m] for m in METRICS},
        "helper_sources": {str(path): hashes[str(path)] for path in
                           (SCRIPTS / "t1t2_common.py", SCRIPTS / "diag_artefact/headonly.py")},
        "mask": "GT magnitude > 0.05 * P99.9; headonly float32 mask for PSNR/SSIM, Pair float64 mask for NMSE",
        "psnr": "headonly.head_psnr; float32 magnitudes, float64 squared difference, global GT max; no amplitude fitting",
        "ssim": "headonly.head_ssim; float32 magnitudes; skimage 7^3 uniform SSIM map, K1=.01 K2=.03, data_range=max GT; head mean",
        "nmse": "Pair.score CPU: sum((abs(candidate complex128)-GT magnitude float64)^2)[head] / sum(GT magnitude^2)[head]; GT magnitude formed in complex64 before promotion",
        "gauge": report_freeze["gt_equivalence"],
        "whole_head_gt_loading": "Pair.__init__: torch complex64(sim/gt.npz[scan]) / stored scale; no fitted normalization",
    }
    return {"manifest": manifest, "report_freeze": report_freeze, "report": report,
            "references": references, "reference_heads": heads, "baseline": read_json(BASELINE),
            "hashes": hashes, "whole_head_contract": whole_contract,
            "slice_contract": {"path": str(CONTRACT), "sha256": hashes[str(CONTRACT)],
                               "formula_source": str(SCRIPTS / "lesion_slice_allmetrics.py"),
                               "formula_source_sha256": hashes[str(SCRIPTS / "lesion_slice_allmetrics.py")]}}


def build_objective(name, run_dir, whole_head, lesion_planes, inputs):
    return {"schema_version": 1, "name": name, "run_dir": str(run_dir),
            "metrics": list(METRICS), "primary_objective": "whole_head_min_six_normalized_margin",
            "score": whole_head["score"], "strict_success": whole_head["strict_success"],
            "whole_head_metrics": whole_head["metrics"], "whole_head_comparisons": whole_head["comparisons"],
            "lesion_planes": {key: value for key, value in lesion_planes.items() if key != "planes"},
            "candidate_sha256": {s: inputs.get("candidate", {}).get(s, {}).get("sha256") for s in SCANS},
            "input_hashes": inputs}


def _write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def evaluate_run(run_dir, out_dir, name):
    """Score two saved candidates into a new directory and return objective.json.

    This is an actual clinical scoring operation; unit tests use CPU fixtures in
    the pure helpers instead. No training or GPU selection occurs here.
    """
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("name must be a nonempty single path component")
    run_dir, out_dir = Path(run_dir).resolve(strict=True), Path(out_dir).resolve()
    if not run_dir.is_dir() or out_dir == run_dir or run_dir in out_dir.parents:
        raise ValueError("score directory must be separate from the candidate run")
    out_dir.mkdir(parents=True, exist_ok=False)
    _write_json(out_dir / "STARTED.json", {"status": "scoring", "name": name,
                                          "created_utc": datetime.now(timezone.utc).isoformat()})
    context = load_frozen_context()
    pair, candidate_inputs = load_candidate_pair(run_dir)
    inputs = {"candidate": candidate_inputs, "files": dict(context["hashes"])}
    inputs["files"].update({e["path"]: e["sha256"] for e in candidate_inputs.values()})
    inputs["files"][str(Path(__file__).resolve())] = sha256_file(__file__)
    truth = {s: np.load(TOPIC / f"recon/gt_{s}_complex.npy", mmap_mode="r", allow_pickle=False) for s in SCANS}
    for scan in SCANS:
        if (truth[scan].shape != pair[scan].shape or truth[scan].dtype != np.complex64
                or not np.isfinite(truth[scan]).all()):
            raise ValueError("candidate/GT shape, complex64 gauge or finite check failed")
    whole_truth = load_whole_head_gt(TOPIC, context["report_freeze"])
    heads = {s: whole_head_metrics(pair[s], whole_truth[s]) for s in SCANS}
    del whole_truth
    whole = summarize_whole_head(heads, context["reference_heads"])
    with np.load(ROI, allow_pickle=False) as archive:
        names = [str(n) for n in archive["names"]]
        core = archive[f"roi{names.index('病灶核心')}"] > 0
    eligible = [(axis, index, int(np.take(core, index, axis=axis).sum()))
                for axis in range(3) for index in range(core.shape[axis])
                if np.take(core, index, axis=axis).sum() >= 150]
    expected = [(p["axis"], p["index"], p["core"]) for p in context["baseline"]]
    if core.shape != pair["pre"].shape or eligible != expected or len(eligible) != 59:
        raise ValueError("ROI does not reproduce all 59 frozen lesion-containing planes")
    observed = [{"axis": axis, "index": index, "core": count, "metrics": {}} for axis, index, count in eligible]
    for scan in SCANS:
        gt, rec = np.abs(truth[scan]), np.abs(pair[scan])
        gmax, head = float(gt.max()), gt > .05 * float(np.quantile(gt, .999))
        for row in observed:
            row["metrics"][scan] = plane_metrics(rec, gt, head, gmax, row["axis"], row["index"])
    planes = summarize_planes(observed, context["baseline"])
    enhancement = context["report"].enhancement_bundle(
        TOPIC, ROI, context["references"], [{"name": name, "run_dir": str(run_dir)}],
        context["manifest"], context["report_freeze"])
    verify_hashes(inputs["files"])
    result = build_objective(name, run_dir, whole, planes, inputs)
    result.update(whole_head_contract=context["whole_head_contract"], slice_contract=context["slice_contract"],
                  all_inputs_unchanged=True,
                  runtime={"python": sys.version.split()[0], "executable": sys.executable,
                           "torch": __import__("torch").__version__, "numpy": np.__version__,
                           "skimage": __import__("skimage").__version__, "device": "cpu"})
    _write_json(out_dir / "whole_head_contract.json", context["whole_head_contract"])
    _write_json(out_dir / "lesion_planes.json", planes)
    _write_json(out_dir / "enhancement.json", enhancement)
    _write_json(out_dir / "input_hashes.json", inputs)
    with (out_dir / "whole_head.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("method", "scan", *METRICS))
        writer.writeheader()
        for label, values in [*context["reference_heads"].items(), (name, heads)]:
            for scan in SCANS:
                writer.writerow({"method": label, "scan": scan, **values[scan]})
    # Publish the completed objective only after all mandatory reports exist.
    verify_hashes(inputs["files"])
    _write_json(out_dir / "objective.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    result = evaluate_run(args.run_dir, args.out_dir, args.name)
    print(json.dumps({"name": result["name"], "score": result["score"],
                      "strict_success": result["strict_success"]}, allow_nan=False))


if __name__ == "__main__":
    main()
