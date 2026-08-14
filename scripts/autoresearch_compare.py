#!/usr/bin/env python3
"""Screen one autoresearch candidate against its same-budget baseline.

The decision is deliberately per contrast: a gain in T1W cannot compensate
for a regression in FLAIR, or vice versa.  This command is SCREEN_ONLY: only
``autoresearch_repeat_gate.py`` can make a numerical promotion decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = (ROOT / "results_arc").resolve()
MODALITIES = {
    "T1W": "plain",
    "FLAIR": "contrast",
}
CYCLE_KEYS = ("psnr", "ssim", "nmse", "hfen", "sharp", "grain", "comp", "beat")
NUMERIC_CYCLE_KEYS = CYCLE_KEYS[:-1]
ARRAY_KEYS = {
    "T1W": {"gt": "gtT1", "zf": "zfT1"},
    "FLAIR": {"gt": "gtFL", "zf": "zfFL"},
}
EXPECTED_NPZ_KEYS = {"recT1", "gtT1", "zfT1", "recFL", "gtFL", "zfFL"}
EXPERIMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
HASH_BLOCK_SIZE = 4 * 1024 * 1024


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _load_json_and_sha256(path: Path) -> tuple[dict[str, Any], str]:
    """Parse the same bytes used for the streaming source digest."""

    digest = hashlib.sha256()
    chunks: list[bytes] = []
    try:
        with path.open("rb") as handle:
            while block := handle.read(HASH_BLOCK_SIZE):
                digest.update(block)
                chunks.append(block)
        value = json.loads(b"".join(chunks))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value, digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(HASH_BLOCK_SIZE):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    """Hash C-order array bytes without materializing one giant ``bytes`` value."""

    array = np.ascontiguousarray(value)
    raw = memoryview(array).cast("B")
    digest = hashlib.sha256()
    for offset in range(0, raw.nbytes, HASH_BLOCK_SIZE):
        digest.update(raw[offset : offset + HASH_BLOCK_SIZE])
    return digest.hexdigest()


def _sha256_claim(parent: dict[str, Any], where: str) -> str:
    value = parent.get("sha256")
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{where}.sha256 must be a 64-character hexadecimal digest")
    return value.lower()


def _validate_experiment(value: Any, where: str) -> str:
    if not isinstance(value, str) or EXPERIMENT_RE.fullmatch(value) is None:
        raise ValueError(
            f"{where} must be one safe experiment name using letters, digits, '.', '_', or '-'"
        )
    return value


def _canonical_result_path(experiment: str, suffix: str) -> Path:
    return (RESULTS_ROOT / f"{experiment}_{suffix}").resolve()


def _source_path(
    parent: dict[str, Any], experiment: str, suffix: str, where: str
) -> Path:
    raw_path = parent.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{where}.path must be a non-empty string")
    observed = Path(raw_path).expanduser().resolve()
    expected = _canonical_result_path(experiment, suffix)
    if observed != expected:
        raise ValueError(f"{where}.path must be canonical {expected}, got {observed}")
    if not observed.is_file():
        raise ValueError(f"{where}.path is not an existing file: {observed}")
    return observed


def _mapping(parent: dict[str, Any], key: str, where: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{where}.{key} must be an object")
    return value


def _number(parent: dict[str, Any], key: str, where: str) -> float:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}.{key} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{where}.{key} must be finite")
    return number


def _boolean(parent: dict[str, Any], key: str, where: str) -> bool:
    value = parent.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{where}.{key} must be boolean")
    return value


def _experiment(score: dict[str, Any], artifact: dict[str, Any], label: str) -> str:
    score_exp = _validate_experiment(score.get("exp"), f"{label} score.exp")
    artifact_exp = _validate_experiment(
        artifact.get("experiment"), f"{label} artifact.experiment"
    )
    if artifact_exp != score_exp:
        raise ValueError(
            f"{label} experiment mismatch: artifact={artifact_exp!r}, score={score_exp!r}"
        )
    return score_exp


def _canonical_modality(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = "".join(character for character in value.upper() if character.isalnum())
    if normalized in {"T1", "T1W"}:
        return "T1W"
    if normalized in {"FL", "FLAIR"}:
        return "FLAIR"
    return None


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        left_float = float(left)
        right_float = float(right)
        if not math.isfinite(left_float) or not math.isfinite(right_float):
            return False
        return math.isclose(left_float, right_float, rel_tol=1e-9, abs_tol=1e-12)
    return left == right


def _zf_metrics(channel: dict[str, Any], where: str) -> tuple[float, float, float, float]:
    value = channel.get("zf")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{where}.zf must contain [psnr, ssim, nmse, hfen]")
    numbers: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{where}.zf[{index}] must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{where}.zf[{index}] must be finite")
        numbers.append(number)
    return tuple(numbers)  # type: ignore[return-value]


def _validate_score_document(score: dict[str, Any], label: str) -> str:
    """Validate consumed score fields and independently recompute decision flags."""

    experiment = _validate_experiment(score.get("exp"), f"{label} score.exp")
    iterations = score.get("iters")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise ValueError(f"{label} score.iters must be a positive integer")
    if not _all_numeric_finite(score):
        raise ValueError(f"{label} score contains a non-finite or unsupported value")

    expected_names = {"plain_name": "T1W", "contrast_name": "FLAIR"}
    for key, expected in expected_names.items():
        observed = _canonical_modality(score.get(key))
        if observed != expected:
            raise ValueError(
                f"{label} score.{key} must identify {expected}, got {score.get(key)!r}"
            )

    recomputed_beats: list[bool] = []
    comps: list[float] = []
    for score_key in MODALITIES.values():
        where = f"{label} score.{score_key}"
        channel = _mapping(score, score_key, f"{label} score")
        metrics = {key: _number(channel, key, where) for key in NUMERIC_CYCLE_KEYS}
        for nonnegative in ("nmse", "hfen", "sharp", "grain"):
            if metrics[nonnegative] < 0.0:
                raise ValueError(f"{where}.{nonnegative} must be non-negative")
        for optional in ("e_unobs", "nmse_unobs"):
            if optional in channel:
                if _number(channel, optional, where) < 0.0:
                    raise ValueError(f"{where}.{optional} must be non-negative")
        _, zf_ssim, zf_nmse, zf_hfen = _zf_metrics(channel, where)
        if zf_nmse < 0.0 or zf_hfen < 0.0:
            raise ValueError(f"{where}.zf NMSE and HFEN must be non-negative")
        declared_beat = _boolean(channel, "beat", where)
        recomputed_beat = bool(
            metrics["ssim"] >= zf_ssim
            and metrics["nmse"] <= zf_nmse
            and metrics["hfen"] <= zf_hfen
        )
        if declared_beat != recomputed_beat:
            raise ValueError(
                f"{where}.beat={declared_beat} disagrees with recomputed {recomputed_beat}"
            )
        recomputed_beats.append(recomputed_beat)
        comps.append(metrics["comp"])

    combined = _number(score, "combined", f"{label} score")
    expected_combined = sum(comps) / len(comps)
    if not math.isclose(combined, expected_combined, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            f"{label} score.combined={combined} != mean modality comp {expected_combined}"
        )
    declared_both = _boolean(score, "both_beat", f"{label} score")
    recomputed_both = all(recomputed_beats)
    if declared_both != recomputed_both:
        raise ValueError(
            f"{label} score.both_beat={declared_both} disagrees with recomputed {recomputed_both}"
        )
    return experiment


def _optional_flat_metrics(channel: dict[str, Any], where: str) -> dict[str, float] | None:
    seac = _mapping(channel, "spatial_edge_error_autocorrelation", where)
    if "flat_region" not in seac:
        return None
    flat = _mapping(seac, "flat_region", f"{where}.spatial_edge_error_autocorrelation")
    values = {
        key: _number(flat, key, f"{where}.spatial_edge_error_autocorrelation.flat_region")
        for key in ("rms", "score")
    }
    if any(value < 0.0 for value in values.values()):
        raise ValueError(f"{where}.spatial_edge_error_autocorrelation.flat_region must be non-negative")
    return values


def _array_claim(
    channel: dict[str, Any], role: str, expected_key: str, where: str
) -> dict[str, Any]:
    arrays = _mapping(channel, "source_arrays", where)
    claim = _mapping(arrays, role, f"{where}.source_arrays")
    if claim.get("npz_key") != expected_key:
        raise ValueError(
            f"{where}.source_arrays.{role}.npz_key must be {expected_key!r}"
        )
    shape = claim.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape)
    ):
        raise ValueError(f"{where}.source_arrays.{role}.shape must be three positive integers")
    dtype = claim.get("dtype")
    if not isinstance(dtype, str) or not dtype:
        raise ValueError(f"{where}.source_arrays.{role}.dtype must be a non-empty string")
    return {
        "npz_key": expected_key,
        "shape": shape,
        "dtype": dtype,
        "sha256": _sha256_claim(claim, f"{where}.source_arrays.{role}"),
    }


def _artifact_source_claims(
    artifact: dict[str, Any], experiment: str, label: str
) -> dict[str, Any]:
    sources = _mapping(artifact, "sources", f"{label} artifact")
    volumes = _mapping(sources, "volumes", f"{label} artifact.sources")
    score_source = _mapping(sources, "score", f"{label} artifact.sources")
    volumes_path = _source_path(
        volumes, experiment, "volumes.npz", f"{label} artifact.sources.volumes"
    )
    score_path = _source_path(
        score_source, experiment, "score.json", f"{label} artifact.sources.score"
    )

    legacy_npz = artifact.get("npz")
    legacy_score = artifact.get("cycle_score")
    if legacy_npz != str(volumes_path) or legacy_score != str(score_path):
        raise ValueError(f"{label} artifact legacy source paths are not canonical")

    array_claims = {
        modality: {
            role: _array_claim(
                _mapping(artifact, modality, f"{label} artifact"),
                role,
                npz_key,
                f"{label} artifact.{modality}",
            )
            for role, npz_key in roles.items()
        }
        for modality, roles in ARRAY_KEYS.items()
    }
    return {
        "volumes_path": volumes_path,
        "volumes_sha256": _sha256_claim(
            volumes, f"{label} artifact.sources.volumes"
        ),
        "score_path": score_path,
        "score_sha256": _sha256_claim(score_source, f"{label} artifact.sources.score"),
        "arrays": array_claims,
    }


def _validate_artifact_sources(
    artifact: dict[str, Any], score: dict[str, Any], experiment: str, label: str
) -> dict[str, Any]:
    claims = _artifact_source_claims(artifact, experiment, label)
    source_score, actual_score_sha256 = _load_json_and_sha256(claims["score_path"])
    if actual_score_sha256 != claims["score_sha256"]:
        raise ValueError(
            f"{label} artifact score SHA-256 mismatch: claimed={claims['score_sha256']}, "
            f"actual={actual_score_sha256}"
        )
    if source_score != score:
        raise ValueError(f"{label} supplied score document differs from its canonical source")

    actual_volumes_sha256 = _sha256_file(claims["volumes_path"])
    if actual_volumes_sha256 != claims["volumes_sha256"]:
        raise ValueError(
            f"{label} artifact volumes SHA-256 mismatch: claimed={claims['volumes_sha256']}, "
            f"actual={actual_volumes_sha256}"
        )
    with np.load(claims["volumes_path"], allow_pickle=False) as volumes:
        if set(volumes.files) != EXPECTED_NPZ_KEYS:
            raise ValueError(f"{label} source NPZ has unexpected keys: {volumes.files}")
        for modality, roles in claims["arrays"].items():
            for role, claim in roles.items():
                array = volumes[claim["npz_key"]]
                if list(array.shape) != claim["shape"]:
                    raise ValueError(f"{label} {modality} {role} source-array shape mismatch")
                if array.dtype.str != claim["dtype"]:
                    raise ValueError(f"{label} {modality} {role} source-array dtype mismatch")
                actual = _sha256_array(array)
                if actual != claim["sha256"]:
                    raise ValueError(
                        f"{label} {modality} {role} source-array SHA-256 mismatch: "
                        f"claimed={claim['sha256']}, actual={actual}"
                    )
    return {
        "volumes_path": str(claims["volumes_path"]),
        "volumes_sha256": actual_volumes_sha256,
        "score_path": str(claims["score_path"]),
        "score_sha256": actual_score_sha256,
        "array_sha256": {
            modality: {role: claim["sha256"] for role, claim in roles.items()}
            for modality, roles in claims["arrays"].items()
        },
        "hashes_recomputed": True,
    }


def _validate_artifact_score_pair(
    artifact: dict[str, Any], score: dict[str, Any], label: str
) -> str:
    experiment = _experiment(score, artifact, label)
    if _validate_score_document(score, label) != experiment:
        raise ValueError(f"{label} score experiment changed during validation")
    schema_version = artifact.get("schema_version")
    if schema_version != 2 or isinstance(schema_version, bool):
        raise ValueError(f"{label} artifact.schema_version must be 2")

    for modality, score_key in MODALITIES.items():
        artifact_channel = _mapping(artifact, modality, f"{label} artifact")
        artifact_cycle = _mapping(artifact_channel, "cycle", f"{label} artifact.{modality}")
        score_cycle = _mapping(score, score_key, f"{label} score")
        for metric in CYCLE_KEYS:
            if metric not in artifact_cycle or metric not in score_cycle:
                raise ValueError(f"{label} missing {modality} cycle metric {metric!r}")
            if not _same_value(artifact_cycle[metric], score_cycle[metric]):
                raise ValueError(
                    f"{label} {modality}.{metric} mismatch between artifact and score: "
                    f"{artifact_cycle[metric]!r} != {score_cycle[metric]!r}"
                )
        _optional_flat_metrics(artifact_channel, f"{label} artifact.{modality}")

    aggregate = _mapping(artifact, "aggregate", f"{label} artifact")
    if not _same_value(aggregate.get("combined"), score.get("combined")):
        raise ValueError(f"{label} aggregate.combined does not match score.combined")
    if not _same_value(aggregate.get("both_beat_zf"), score.get("both_beat")):
        raise ValueError(f"{label} aggregate.both_beat_zf does not match score.both_beat")
    _validate_artifact_sources(artifact, score, experiment, label)
    return experiment


def _all_numeric_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_numeric_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_numeric_finite(item) for item in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return False


def _nonvacuous(artifact: dict[str, Any]) -> bool:
    aggregate = _mapping(artifact, "aggregate", "artifact")
    aggregate_flag = _boolean(aggregate, "both_nonvacuous", "artifact.aggregate")
    channel_flags = []
    for modality in MODALITIES:
        channel = _mapping(artifact, modality, "artifact")
        evidence = _mapping(channel, "nonvacuity", f"artifact.{modality}")
        max_abs_diff = _number(evidence, "max_abs_diff", f"artifact.{modality}.nonvacuity")
        rel_max_abs_diff = _number(
            evidence, "rel_max_abs_diff", f"artifact.{modality}.nonvacuity"
        )
        channel_flags.append(
            math.isfinite(max_abs_diff)
            and math.isfinite(rel_max_abs_diff)
            and max_abs_diff > 0.0
            and rel_max_abs_diff > 0.0
        )
    return aggregate_flag and all(channel_flags)


def _both_beat_zf(artifact: dict[str, Any], score: dict[str, Any]) -> bool:
    aggregate = _mapping(artifact, "aggregate", "artifact")
    flags = [
        _boolean(aggregate, "both_beat_zf", "artifact.aggregate"),
        _boolean(score, "both_beat", "score"),
    ]
    for modality, score_key in MODALITIES.items():
        artifact_cycle = _mapping(
            _mapping(artifact, modality, "artifact"), "cycle", f"artifact.{modality}"
        )
        score_cycle = _mapping(score, score_key, "score")
        flags.extend(
            [
                _boolean(artifact_cycle, "beat", f"artifact.{modality}.cycle"),
                _boolean(score_cycle, "beat", f"score.{score_key}"),
            ]
        )
    return all(flags)


def _gate(pass_value: bool, **details: Any) -> dict[str, Any]:
    return {"pass": bool(pass_value), **details}


def _metric_gate(
    baseline: float,
    candidate: float,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> dict[str, Any]:
    finite = math.isfinite(baseline) and math.isfinite(candidate)
    passed = finite
    tolerance = 1e-12
    if lower is not None:
        passed = passed and candidate >= lower - tolerance
    if upper is not None:
        passed = passed and candidate <= upper + tolerance
    return _gate(
        passed,
        baseline=baseline if math.isfinite(baseline) else None,
        candidate=candidate if math.isfinite(candidate) else None,
        delta=(candidate - baseline) if finite else None,
        allowed_min=lower if lower is None or math.isfinite(lower) else None,
        allowed_max=upper if upper is None or math.isfinite(upper) else None,
    )


def _run_health(artifact: dict[str, Any], score: dict[str, Any]) -> dict[str, bool]:
    aggregate = _mapping(artifact, "aggregate", "artifact")
    finite = _all_numeric_finite(artifact) and _all_numeric_finite(score)
    finite = finite and _boolean(aggregate, "all_finite", "artifact.aggregate")
    return {
        "all_finite": finite,
        "both_nonvacuous": _nonvacuous(artifact),
        "both_beat_zf": _both_beat_zf(artifact, score),
    }


def compare_documents(
    baseline_artifact: dict[str, Any],
    candidate_artifact: dict[str, Any],
    baseline_score: dict[str, Any],
    candidate_score: dict[str, Any],
) -> dict[str, Any]:
    baseline_exp = _validate_artifact_score_pair(
        baseline_artifact, baseline_score, "baseline"
    )
    candidate_exp = _validate_artifact_score_pair(
        candidate_artifact, candidate_score, "candidate"
    )

    baseline_iters = baseline_score.get("iters")
    candidate_iters = candidate_score.get("iters")
    if isinstance(baseline_iters, bool) or not isinstance(baseline_iters, int):
        raise ValueError("baseline score.iters must be an integer")
    if isinstance(candidate_iters, bool) or not isinstance(candidate_iters, int):
        raise ValueError("candidate score.iters must be an integer")

    baseline_health = _run_health(baseline_artifact, baseline_score)
    candidate_health = _run_health(candidate_artifact, candidate_score)
    global_gates = {
        "same_budget": _gate(
            baseline_iters == candidate_iters and baseline_iters > 0,
            baseline_iters=baseline_iters,
            candidate_iters=candidate_iters,
        )
    }
    for label, health in (("baseline", baseline_health), ("candidate", candidate_health)):
        for name, observed in health.items():
            global_gates[f"{label}_{name}"] = _gate(observed, observed=observed)

    modality_results: dict[str, Any] = {}
    for modality in MODALITIES:
        baseline_channel = _mapping(baseline_artifact, modality, "baseline artifact")
        candidate_channel = _mapping(candidate_artifact, modality, "candidate artifact")
        baseline_cycle = _mapping(
            baseline_channel, "cycle", f"baseline artifact.{modality}"
        )
        candidate_cycle = _mapping(
            candidate_channel, "cycle", f"candidate artifact.{modality}"
        )

        baseline_sharp = _number(baseline_cycle, "sharp", f"baseline {modality}.cycle")
        candidate_sharp = _number(candidate_cycle, "sharp", f"candidate {modality}.cycle")
        baseline_grain = _number(baseline_cycle, "grain", f"baseline {modality}.cycle")
        candidate_grain = _number(candidate_cycle, "grain", f"candidate {modality}.cycle")
        baseline_hfen = _number(baseline_cycle, "hfen", f"baseline {modality}.cycle")
        candidate_hfen = _number(candidate_cycle, "hfen", f"candidate {modality}.cycle")
        baseline_gradient = _number(
            baseline_channel,
            "gradient_log_rmse_to_one",
            f"baseline artifact.{modality}",
        )
        candidate_gradient = _number(
            candidate_channel,
            "gradient_log_rmse_to_one",
            f"candidate artifact.{modality}",
        )
        baseline_flat = _optional_flat_metrics(
            baseline_channel, f"baseline artifact.{modality}"
        )
        candidate_flat = _optional_flat_metrics(
            candidate_channel, f"candidate artifact.{modality}"
        )

        gates = {
            "sharp": _metric_gate(
                baseline_sharp,
                candidate_sharp,
                lower=baseline_sharp - 0.01,
                upper=baseline_sharp + 0.08,
            ),
            "grain": _metric_gate(
                baseline_grain,
                candidate_grain,
                upper=baseline_grain * 1.03,
            ),
            "hfen": _metric_gate(
                baseline_hfen,
                candidate_hfen,
                upper=baseline_hfen * 1.01,
            ),
            "gradient_log_rmse_to_one": _metric_gate(
                baseline_gradient,
                candidate_gradient,
                upper=baseline_gradient,
            ),
        }
        modality_results[modality] = {
            "pass": all(gate["pass"] for gate in gates.values()),
            "gates": gates,
            "flat_region_diagnostic": {
                "baseline": baseline_flat,
                "candidate": candidate_flat,
                "gated_here": False,
            },
        }

    passed = all(gate["pass"] for gate in global_gates.values()) and all(
        result["pass"] for result in modality_results.values()
    )
    return {
        "schema_version": 2,
        "decision_scope": "SCREEN_ONLY",
        "verdict": "SCREEN_PASS" if passed else "SCREEN_FAIL",
        "screen_pass": passed,
        "promotion_eligible": False,
        "baseline_experiment": baseline_exp,
        "candidate_experiment": candidate_exp,
        "identity": {
            "baseline_artifact_matches_score": True,
            "candidate_artifact_matches_score": True,
            "artifact_source_hashes_recomputed": True,
        },
        "global_gates": global_gates,
        "modalities": modality_results,
        "decision_policy": {
            "compensation": "none: every gate must pass separately for T1W and FLAIR",
            "alias_ncc_with_zf_error": "diagnostic only; excluded from the verdict",
            "promotion_gate": "scripts/autoresearch_repeat_gate.py only",
            "flat_region": "validated when present; not gated by this screen",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-artifact", type=Path, required=True)
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument("--baseline-score", type=Path, required=True)
    parser.add_argument("--candidate-score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")

    result = compare_documents(
        _load_json(args.baseline_artifact),
        _load_json(args.candidate_artifact),
        _load_json(args.baseline_score),
        _load_json(args.candidate_score),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output_file:
        output_file.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "decision_scope": result["decision_scope"],
                "verdict": result["verdict"],
                "promotion_eligible": result["promotion_eligible"],
                "modalities": result["modalities"],
            },
            indent=2,
        )
    )
    print(args.output)


if __name__ == "__main__":
    main()
