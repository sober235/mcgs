#!/usr/bin/env python3
"""Score aliasing and patch/fragment proxies for one dual-contrast run.

This complements ``arc/cycle.py``'s fidelity composite.  It does not replace
the final fixed-window visual gate: the gradient-distribution measurements are
proxies, and the repository's own calibration shows that no scalar fully names
the visible artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diag_artefact.controls import assert_control_is_not_vacuous  # noqa: E402
from diag_artefact.metrics import (  # noqa: E402
    gradient_ratio_curve,
    ncc,
    spatial_edge_error_autocorrelation,
)
from diag_artefact.volumes import brain_mask, wm_mask  # noqa: E402
from scripts.autoresearch_compare import (  # noqa: E402
    ARRAY_KEYS,
    EXPECTED_NPZ_KEYS,
    RESULTS_ROOT,
    _all_numeric_finite,
    _load_json_and_sha256,
    _sha256_array,
    _sha256_file,
    _validate_score_document,
)


QUANTILES = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99, 0.999)


def _finite_metric(parent: dict[str, Any], key: str, where: str) -> float:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}.{key} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{where}.{key} must be finite")
    return number


def _validate_flat_region(seac: dict[str, Any]) -> None:
    flat = seac.get("flat_region")
    if not isinstance(flat, dict):
        raise ValueError(
            "spatial_edge_error_autocorrelation.flat_region must be an object"
        )
    rms = _finite_metric(flat, "rms", "spatial_edge_error_autocorrelation.flat_region")
    score_value = _finite_metric(
        flat, "score", "spatial_edge_error_autocorrelation.flat_region"
    )
    if rms < 0.0 or score_value < 0.0:
        raise ValueError(
            "spatial_edge_error_autocorrelation.flat_region values must be non-negative"
        )


def _canonical_input(path: Path, experiment: str, suffix: str, label: str) -> Path:
    observed = path.expanduser().resolve()
    expected = (RESULTS_ROOT / f"{experiment}_{suffix}").resolve()
    if observed != expected:
        raise ValueError(f"{label} must be canonical {expected}, got {observed}")
    if not observed.is_file():
        raise FileNotFoundError(f"{label} not found: {observed}")
    return observed


def _array_source(value: np.ndarray, npz_key: str) -> dict[str, Any]:
    return {
        "npz_key": npz_key,
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "sha256": _sha256_array(value),
    }


def _contrast_metrics(
    rec: np.ndarray,
    gt: np.ndarray,
    zf: np.ndarray,
    cycle_metrics: dict,
) -> dict:
    if rec.shape != gt.shape or zf.shape != gt.shape:
        raise ValueError(f"volume shape mismatch: rec={rec.shape}, gt={gt.shape}, zf={zf.shape}")
    if rec.shape != (256, 220, 256):
        raise ValueError(f"unexpected volume shape: {rec.shape}")
    if not all(np.isfinite(v).all() for v in (rec, gt, zf)):
        raise ValueError("non-finite volume value")

    brain = brain_mask(gt)
    white_matter = wm_mask(gt, brain)
    curve = gradient_ratio_curve(rec, gt, white_matter, quantiles=QUANTILES)
    ratios = np.asarray([curve[q] for q in QUANTILES], dtype=np.float64)
    if not np.all(np.isfinite(ratios)) or np.any(ratios <= 0):
        raise ValueError(f"invalid gradient-ratio curve: {ratios}")

    rec_error = np.abs(rec.astype(np.float64) - gt.astype(np.float64))
    zf_error = np.abs(zf.astype(np.float64) - gt.astype(np.float64))
    nonvacuity = assert_control_is_not_vacuous(rec, gt)
    selectivity = float(curve[0.999] / curve[0.10])
    spatial_edge_error = spatial_edge_error_autocorrelation(
        rec, gt, wm=white_matter
    )
    if not isinstance(spatial_edge_error, dict):
        raise ValueError("spatial_edge_error_autocorrelation must return an object")
    _validate_flat_region(spatial_edge_error)

    result = {
        "cycle": {key: cycle_metrics[key] for key in (
            "psnr", "ssim", "nmse", "hfen", "sharp", "grain", "comp", "beat"
        )},
        "gradient_ratio_curve": {str(q): float(curve[q]) for q in QUANTILES},
        "gradient_log_rmse_to_one": float(np.sqrt(np.mean(np.log(ratios) ** 2))),
        "gradient_span": float(ratios.max() - ratios.min()),
        "selectivity": selectivity,
        "abs_log_selectivity": float(abs(math.log(selectivity))),
        "spatial_edge_error_autocorrelation": spatial_edge_error,
        "alias_ncc_with_zf_error": ncc(rec_error, zf_error, white_matter),
        "nonvacuity": nonvacuity,
    }
    if not _all_numeric_finite(result):
        raise ValueError("artifact metrics contain a non-finite or unsupported value")
    return result


def score(npz_path: Path, score_path: Path) -> dict:
    score_path = score_path.expanduser().resolve()
    cycle, score_sha256 = _load_json_and_sha256(score_path)
    experiment = _validate_score_document(cycle, "artifact")
    score_path = _canonical_input(score_path, experiment, "score.json", "--score")
    npz_path = _canonical_input(npz_path, experiment, "volumes.npz", "--npz")

    volumes_sha256 = _sha256_file(npz_path)
    with np.load(npz_path, allow_pickle=False) as volumes:
        if set(volumes.files) != EXPECTED_NPZ_KEYS:
            raise ValueError(f"unexpected NPZ keys: {volumes.files}")
        arrays = {key: volumes[key] for key in EXPECTED_NPZ_KEYS}
        t1 = _contrast_metrics(
            arrays["recT1"], arrays["gtT1"], arrays["zfT1"], cycle["plain"]
        )
        flair = _contrast_metrics(
            arrays["recFL"], arrays["gtFL"], arrays["zfFL"], cycle["contrast"]
        )
        for modality, channel in (("T1W", t1), ("FLAIR", flair)):
            channel["source_arrays"] = {
                role: _array_source(arrays[npz_key], npz_key)
                for role, npz_key in ARRAY_KEYS[modality].items()
            }

    if _sha256_file(npz_path) != volumes_sha256:
        raise RuntimeError(f"volumes source changed while scoring: {npz_path}")
    if _sha256_file(score_path) != score_sha256:
        raise RuntimeError(f"score source changed while scoring: {score_path}")

    channels = (t1, flair)
    aggregate = {
        "combined": float(cycle["combined"]),
        "both_beat_zf": bool(cycle["both_beat"]),
        "mean_hfen": float(np.mean([c["cycle"]["hfen"] for c in channels])),
        "worst_hfen": float(max(c["cycle"]["hfen"] for c in channels)),
        "mean_grain": float(np.mean([c["cycle"]["grain"] for c in channels])),
        "worst_grain": float(max(c["cycle"]["grain"] for c in channels)),
        "min_sharp": float(min(c["cycle"]["sharp"] for c in channels)),
        "mean_gradient_log_rmse_to_one": float(
            np.mean([c["gradient_log_rmse_to_one"] for c in channels])
        ),
        "worst_gradient_log_rmse_to_one": float(
            max(c["gradient_log_rmse_to_one"] for c in channels)
        ),
        "mean_abs_log_selectivity": float(
            np.mean([c["abs_log_selectivity"] for c in channels])
        ),
        "worst_spatial_edge_error_autocorrelation_q90": float(max(
            c["spatial_edge_error_autocorrelation"]["primary_score"] for c in channels
        )),
        "all_finite": True,
        "both_nonvacuous": True,
    }
    result = {
        "schema_version": 2,
        "experiment": experiment,
        "npz": str(npz_path),
        "cycle_score": str(score_path),
        "sources": {
            "volumes": {"path": str(npz_path), "sha256": volumes_sha256},
            "score": {"path": str(score_path), "sha256": score_sha256},
        },
        "T1W": t1,
        "FLAIR": flair,
        "aggregate": aggregate,
        "interpretation": {
            "aliasing": "lower HFEN/grain is better; sharpness gate prevents blur gaming",
            "fragmentation_proxy": (
                "lower per-modality spatial edge-error autocorrelation is better; q90 is primary; "
                "the complete q85/q90/q95 profile is retained"
            ),
            "limitation": "final fixed-window three-view human inspection is mandatory",
        },
    }
    if not _all_numeric_finite(result):
        raise ValueError("artifact report contains a non-finite or unsupported value")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.out}")
    result = score(args.npz, args.score)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, allow_nan=False)
        output_file.write("\n")
    print(json.dumps(result["aggregate"], indent=2, allow_nan=False))
    print(args.out)


if __name__ == "__main__":
    main()
