#!/usr/bin/env python3
"""Promote a candidate only beyond two same-budget baseline repeats.

The gate is deliberately modality-local.  T1W never compensates for FLAIR,
and scalar image-quality guards never compensate for the required spatial
edge-error autocorrelation (SEAC) improvement.  Aliasing NCC is diagnostic
only and is not read here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

try:  # ``python scripts/autoresearch_repeat_gate.py``
    from autoresearch_compare import (
        MODALITIES,
        RESULTS_ROOT,
        _all_numeric_finite,
        _artifact_source_claims,
        _boolean,
        _canonical_modality,
        _gate,
        _load_json,
        _mapping,
        _number,
        _run_health,
        _validate_artifact_score_pair,
    )
except ModuleNotFoundError:  # imported as ``scripts.autoresearch_repeat_gate``
    from scripts.autoresearch_compare import (
        MODALITIES,
        RESULTS_ROOT,
        _all_numeric_finite,
        _artifact_source_claims,
        _boolean,
        _canonical_modality,
        _gate,
        _load_json,
        _mapping,
        _number,
        _run_health,
        _validate_artifact_score_pair,
    )


SEAC_QUANTILES = (0.85, 0.90, 0.95)
LABELS = ("baseline_a", "baseline_b", "candidate")
ALLOWED_TUNABLE_LOSS_KEYS = {
    "lambda_geo",
    "tv_weight",
    "tv_eps",
    "edge_tv_kappa_q",
    "edge_tv_wfloor",
    "kspace_high_weight",
    "kspace_high_weight_plain",
}


def _positive_iterations(score: dict[str, Any], label: str) -> int:
    value = score.get("iters")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} score.iters must be a positive integer")
    return value


def _sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _sha256_claim(parent: dict[str, Any], where: str) -> str:
    value = parent.get("sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{where}.sha256 must be a 64-character SHA-256 digest")
    normalized = value.lower()
    if any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{where}.sha256 is not hexadecimal")
    return normalized


def _provenance_path(
    parent: dict[str, Any], experiment: str, kind: str, where: str
) -> Path:
    raw_path = parent.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{where}.path must be a non-empty string")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{where}.path is not an existing file: {path}")

    run_dir = RESULTS_ROOT / experiment / experiment
    if kind == "config":
        expected = (run_dir / "config.yaml").resolve()
    elif kind == "checkpoint":
        expected = (run_dir / "checkpoints" / "latest.pth").resolve()
    else:  # internal invariant
        raise ValueError(f"unknown provenance kind: {kind!r}")
    if path != expected:
        raise ValueError(
            f"{where}.path must be {experiment!r}'s canonical {kind} {expected}, got {path}"
        )
    return path


def _optional_number(parent: dict[str, Any], key: str, where: str) -> float | None:
    value = parent.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}.{key} must be numeric or null")
    return float(value)


def _finite_number(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def _safe_number(value: float | None) -> float | None:
    return value if _finite_number(value) else None


def _all_relevant_numeric_finite(value: Any) -> bool:
    """Finite scan that deliberately excludes the diagnostic-only alias NCC."""

    if isinstance(value, dict):
        return all(
            _all_relevant_numeric_finite(item)
            for key, item in value.items()
            if key != "alias_ncc_with_zf_error"
        )
    if isinstance(value, (list, tuple)):
        return all(_all_relevant_numeric_finite(item) for item in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return False


def _validate_kspace(
    document: dict[str, Any], experiment: str, iterations: int, label: str
) -> dict[str, Any]:
    """Validate checkpoint-replay identity and independently rehash its inputs."""

    observed_experiment = document.get("experiment")
    if observed_experiment != experiment:
        raise ValueError(
            f"{label} kspace experiment mismatch: "
            f"{observed_experiment!r} != {experiment!r}"
        )

    config = _mapping(document, "config", f"{label} kspace")
    checkpoint = _mapping(document, "checkpoint", f"{label} kspace")
    config_path = _provenance_path(
        config, experiment, "config", f"{label} kspace.config"
    )
    checkpoint_path = _provenance_path(
        checkpoint, experiment, "checkpoint", f"{label} kspace.checkpoint"
    )
    if config_path.parent != checkpoint_path.parent.parent:
        raise ValueError(f"{label} kspace config/checkpoint resolve to different runs")

    config_claim = _sha256_claim(config, f"{label} kspace.config")
    checkpoint_claim = _sha256_claim(checkpoint, f"{label} kspace.checkpoint")
    config_actual = _sha256_file(config_path)
    checkpoint_actual = _sha256_file(checkpoint_path)
    if config_claim != config_actual:
        raise ValueError(
            f"{label} kspace config SHA-256 mismatch: "
            f"claimed={config_claim}, actual={config_actual}"
        )
    if checkpoint_claim != checkpoint_actual:
        raise ValueError(
            f"{label} kspace checkpoint SHA-256 mismatch: "
            f"claimed={checkpoint_claim}, actual={checkpoint_actual}"
        )

    checkpoint_iteration = checkpoint.get("iteration")
    if isinstance(checkpoint_iteration, bool) or not isinstance(
        checkpoint_iteration, int
    ):
        raise ValueError(f"{label} kspace.checkpoint.iteration must be an integer")
    if checkpoint_iteration != iterations:
        raise ValueError(
            f"{label} kspace checkpoint iteration {checkpoint_iteration} "
            f"!= score.iters {iterations}"
        )

    source_guards = _mapping(document, "source_guards", f"{label} kspace")
    if not _boolean(
        source_guards, "checkpoint_replayed", f"{label} kspace.source_guards"
    ):
        raise ValueError(f"{label} kspace did not replay a checkpoint")
    if _boolean(
        source_guards, "magnitude_npz_used", f"{label} kspace.source_guards"
    ):
        raise ValueError(f"{label} kspace improperly used a magnitude NPZ")
    reference = source_guards.get("complex_multicoil_reference")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(
            f"{label} kspace source_guards.complex_multicoil_reference is invalid"
        )

    modalities = _mapping(document, "modalities", f"{label} kspace")
    modality_finite: dict[str, bool] = {}
    for modality, channel_key in MODALITIES.items():
        channel = _mapping(modalities, channel_key, f"{label} kspace.modalities")
        if channel.get("channel") != channel_key:
            raise ValueError(
                f"{label} kspace {channel_key}.channel must equal {channel_key!r}"
            )
        if _canonical_modality(channel.get("name")) != modality:
            raise ValueError(
                f"{label} kspace {channel_key}.name does not identify {modality}"
            )
        e_unobs = _optional_number(
            channel, "e_unobs", f"{label} kspace.modalities.{channel_key}"
        )
        nmse_unobs = _optional_number(
            channel, "nmse_unobs", f"{label} kspace.modalities.{channel_key}"
        )
        declared_finite = _boolean(
            channel, "finite", f"{label} kspace.modalities.{channel_key}"
        )
        values_finite = (
            _finite_number(e_unobs)
            and _finite_number(nmse_unobs)
            and e_unobs >= 0.0
            and nmse_unobs >= 0.0
        )
        if declared_finite and values_finite and not math.isclose(
            e_unobs * e_unobs, nmse_unobs, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError(
                f"{label} kspace {channel_key} e_unobs^2 does not match nmse_unobs"
            )
        modality_finite[modality] = bool(declared_finite and values_finite)

    declared_all_finite = _boolean(document, "finite", f"{label} kspace")
    kspace_finite = bool(
        declared_all_finite
        and all(modality_finite.values())
        and _all_numeric_finite(document)
    )
    return {
        "experiment_matches": True,
        "iteration_matches_score": True,
        "config_path": str(config_path),
        "config_sha256": config_actual,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_actual,
        "checkpoint_iteration": checkpoint_iteration,
        "hashes_recomputed": True,
        "kspace_finite": kspace_finite,
        "modality_finite": modality_finite,
    }


def _resolved_config(path: str) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"resolved config root must be a mapping: {path}")
    return value


def _without_output(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    normalized.pop("output", None)
    return normalized


def _fixed_protocol(config: dict[str, Any]) -> dict[str, Any]:
    normalized = _without_output(config)
    loss = normalized.get("loss")
    if isinstance(loss, dict):
        for key in ALLOWED_TUNABLE_LOSS_KEYS:
            loss.pop(key, None)
    return normalized


def _repeat_stats(
    baseline_a: float | None, baseline_b: float | None
) -> dict[str, float] | None:
    if not _finite_number(baseline_a) or not _finite_number(baseline_b):
        return None
    return {
        "min": min(baseline_a, baseline_b),
        "max": max(baseline_a, baseline_b),
        "mean": (baseline_a + baseline_b) / 2.0,
        "delta": abs(baseline_a - baseline_b),
    }


def _repeat_gate(
    baseline_a: float | None,
    baseline_b: float | None,
    candidate: float | None,
    *,
    comparator: str,
    threshold_fn: Callable[[dict[str, float]], float],
    tolerance_fn: Callable[[dict[str, float]], float],
    strict: bool = False,
) -> dict[str, Any]:
    stats = _repeat_stats(baseline_a, baseline_b)
    finite = stats is not None and _finite_number(candidate)
    threshold = threshold_fn(stats) if stats is not None else None
    tolerance = tolerance_fn(stats) if stats is not None else None
    if not finite:
        passed = False
    elif comparator == "<=":
        passed = candidate <= threshold + 1e-12
    elif comparator == ">=":
        passed = candidate >= threshold - 1e-12
    elif comparator == "<" and strict:
        passed = candidate < threshold
    else:  # internal invariant
        raise ValueError(f"unsupported comparator: {comparator!r}")
    return _gate(
        passed,
        baseline_a=_safe_number(baseline_a),
        baseline_b=_safe_number(baseline_b),
        candidate=_safe_number(candidate),
        baseline_min=stats["min"] if stats is not None else None,
        baseline_max=stats["max"] if stats is not None else None,
        baseline_mean=stats["mean"] if stats is not None else None,
        repeat_delta=stats["delta"] if stats is not None else None,
        tolerance=tolerance,
        comparator=comparator,
        threshold=threshold,
        strict=bool(strict),
    )


def _cycle_metric(
    artifact: dict[str, Any], modality: str, metric: str, label: str
) -> float:
    channel = _mapping(artifact, modality, f"{label} artifact")
    cycle = _mapping(channel, "cycle", f"{label} artifact.{modality}")
    return _number(cycle, metric, f"{label} artifact.{modality}.cycle")


def _gradient_metric(
    artifact: dict[str, Any], modality: str, label: str
) -> float:
    channel = _mapping(artifact, modality, f"{label} artifact")
    return _number(
        channel,
        "gradient_log_rmse_to_one",
        f"{label} artifact.{modality}",
    )


def _seac_metric(
    artifact: dict[str, Any], modality: str, quantile: float, label: str
) -> float:
    channel = _mapping(artifact, modality, f"{label} artifact")
    seac = _mapping(
        channel,
        "spatial_edge_error_autocorrelation",
        f"{label} artifact.{modality}",
    )
    quantiles = _mapping(
        seac,
        "quantiles",
        f"{label} artifact.{modality}.spatial_edge_error_autocorrelation",
    )
    key = str(quantile)
    row = _mapping(
        quantiles,
        key,
        f"{label} artifact.{modality}.spatial_edge_error_autocorrelation.quantiles",
    )
    return _number(row, "score", f"{label} artifact.{modality}.SEAC[{key}]")


def _flat_metric(
    artifact: dict[str, Any], modality: str, metric: str, label: str
) -> float:
    channel = _mapping(artifact, modality, f"{label} artifact")
    seac = _mapping(
        channel,
        "spatial_edge_error_autocorrelation",
        f"{label} artifact.{modality}",
    )
    flat = _mapping(
        seac,
        "flat_region",
        f"{label} artifact.{modality}.spatial_edge_error_autocorrelation",
    )
    value = _number(
        flat,
        metric,
        f"{label} artifact.{modality}.spatial_edge_error_autocorrelation.flat_region",
    )
    if value < 0.0:
        raise ValueError(f"{label} artifact.{modality}.flat_region.{metric} must be non-negative")
    return value


def _kspace_e(
    kspace: dict[str, Any], modality: str, label: str
) -> float | None:
    channel_key = MODALITIES[modality]
    modalities = _mapping(kspace, "modalities", f"{label} kspace")
    channel = _mapping(modalities, channel_key, f"{label} kspace.modalities")
    return _optional_number(
        channel, "e_unobs", f"{label} kspace.modalities.{channel_key}"
    )


def compare_documents(
    baseline_a_artifact: dict[str, Any],
    baseline_a_score: dict[str, Any],
    baseline_a_kspace: dict[str, Any],
    baseline_b_artifact: dict[str, Any],
    baseline_b_score: dict[str, Any],
    baseline_b_kspace: dict[str, Any],
    candidate_artifact: dict[str, Any],
    candidate_score: dict[str, Any],
    candidate_kspace: dict[str, Any],
) -> dict[str, Any]:
    artifacts = {
        "baseline_a": baseline_a_artifact,
        "baseline_b": baseline_b_artifact,
        "candidate": candidate_artifact,
    }
    scores = {
        "baseline_a": baseline_a_score,
        "baseline_b": baseline_b_score,
        "candidate": candidate_score,
    }
    kspaces = {
        "baseline_a": baseline_a_kspace,
        "baseline_b": baseline_b_kspace,
        "candidate": candidate_kspace,
    }

    experiments: dict[str, str] = {}
    iterations: dict[str, int] = {}
    provenance: dict[str, dict[str, Any]] = {}
    artifact_provenance: dict[str, dict[str, Any]] = {}
    health: dict[str, dict[str, bool]] = {}
    for label in LABELS:
        experiments[label] = _validate_artifact_score_pair(
            artifacts[label], scores[label], label
        )
        artifact_claims = _artifact_source_claims(
            artifacts[label], experiments[label], label
        )
        artifact_provenance[label] = {
            "volumes_path": str(artifact_claims["volumes_path"]),
            "volumes_sha256": artifact_claims["volumes_sha256"],
            "score_path": str(artifact_claims["score_path"]),
            "score_sha256": artifact_claims["score_sha256"],
            "array_sha256": {
                modality: {
                    role: claim["sha256"] for role, claim in roles.items()
                }
                for modality, roles in artifact_claims["arrays"].items()
            },
            "hashes_recomputed": True,
        }
        iterations[label] = _positive_iterations(scores[label], label)
        provenance[label] = _validate_kspace(
            kspaces[label], experiments[label], iterations[label], label
        )
        artifact_health = _run_health(artifacts[label], scores[label])
        aggregate = _mapping(artifacts[label], "aggregate", f"{label} artifact")
        relevant_finite = bool(
            _all_relevant_numeric_finite(artifacts[label])
            and _all_numeric_finite(scores[label])
            and _boolean(aggregate, "all_finite", f"{label} artifact.aggregate")
        )
        health[label] = {
            "all_finite": bool(
                relevant_finite
                and provenance[label]["kspace_finite"]
            ),
            "both_nonvacuous": artifact_health["both_nonvacuous"],
            "both_beat_zf": artifact_health["both_beat_zf"],
        }

    all_iterations = tuple(iterations[label] for label in LABELS)
    configs = {
        label: _resolved_config(provenance[label]["config_path"])
        for label in LABELS
    }
    baseline_configs_match = (
        _without_output(configs["baseline_a"])
        == _without_output(configs["baseline_b"])
    )
    fixed_protocols = {
        label: _fixed_protocol(configs[label])
        for label in LABELS
    }
    fixed_protocol_matches = (
        fixed_protocols["baseline_a"]
        == fixed_protocols["baseline_b"]
        == fixed_protocols["candidate"]
    )
    distinct_experiments = len(set(experiments.values())) == len(LABELS)
    checkpoint_paths = {
        label: provenance[label]["checkpoint_path"] for label in LABELS
    }
    distinct_checkpoints = len(set(checkpoint_paths.values())) == len(LABELS)
    source_array_details = {
        modality: {
            role: {
                label: artifact_provenance[label]["array_sha256"][modality][role]
                for label in LABELS
            }
            for role in ("gt", "zf")
        }
        for modality in MODALITIES
    }
    source_arrays_identical = all(
        len(set(hashes.values())) == 1
        for roles in source_array_details.values()
        for hashes in roles.values()
    )
    global_gates: dict[str, Any] = {
        "three_distinct_experiments": _gate(
            distinct_experiments,
            observed=experiments,
        ),
        "three_distinct_canonical_checkpoints": _gate(
            distinct_checkpoints,
            observed=checkpoint_paths,
        ),
        "identical_gt_zf_source_arrays": _gate(
            source_arrays_identical,
            observed=source_array_details,
        ),
        "same_iterations": _gate(
            len(set(all_iterations)) == 1,
            baseline_a=iterations["baseline_a"],
            baseline_b=iterations["baseline_b"],
            candidate=iterations["candidate"],
        ),
        "baseline_repeat_same_config_except_output": _gate(
            baseline_configs_match,
            observed=baseline_configs_match,
        ),
        "fixed_protocol_same_except_allowed_loss_knobs_and_output": _gate(
            fixed_protocol_matches,
            observed=fixed_protocol_matches,
            allowed_loss_knobs=sorted(ALLOWED_TUNABLE_LOSS_KEYS),
        ),
    }
    for label in LABELS:
        for name, observed in health[label].items():
            global_gates[f"{label}_{name}"] = _gate(observed, observed=observed)

    modality_results: dict[str, Any] = {}
    for modality in MODALITIES:
        values = {
            label: {
                "sharp": _cycle_metric(artifacts[label], modality, "sharp", label),
                "grain": _cycle_metric(artifacts[label], modality, "grain", label),
                "hfen": _cycle_metric(artifacts[label], modality, "hfen", label),
                "e_unobs": _kspace_e(kspaces[label], modality, label),
                "gradient_log_rmse_to_one": _gradient_metric(
                    artifacts[label], modality, label
                ),
                "flat_region_rms": _flat_metric(
                    artifacts[label], modality, "rms", label
                ),
                "flat_region_score": _flat_metric(
                    artifacts[label], modality, "score", label
                ),
                **{
                    f"seac_q{int(round(q * 100))}": _seac_metric(
                        artifacts[label], modality, q, label
                    )
                    for q in SEAC_QUANTILES
                },
            }
            for label in LABELS
        }

        def metric_gate(
            metric: str,
            *,
            comparator: str,
            threshold_fn: Callable[[dict[str, float]], float],
            tolerance_fn: Callable[[dict[str, float]], float],
            strict: bool = False,
        ) -> dict[str, Any]:
            return _repeat_gate(
                values["baseline_a"][metric],
                values["baseline_b"][metric],
                values["candidate"][metric],
                comparator=comparator,
                threshold_fn=threshold_fn,
                tolerance_fn=tolerance_fn,
                strict=strict,
            )

        gates: dict[str, Any] = {
            "sharp_lower": metric_gate(
                "sharp",
                comparator=">=",
                threshold_fn=lambda s: s["min"] - 0.01,
                tolerance_fn=lambda s: 0.01,
            ),
            "grain": metric_gate(
                "grain",
                comparator="<=",
                threshold_fn=lambda s: s["max"] * 1.03,
                tolerance_fn=lambda s: s["max"] * 0.03,
            ),
            "hfen": metric_gate(
                "hfen",
                comparator="<=",
                threshold_fn=lambda s: s["max"] * 1.01,
                tolerance_fn=lambda s: s["max"] * 0.01,
            ),
            "e_unobs": metric_gate(
                "e_unobs",
                comparator="<=",
                threshold_fn=lambda s: s["max"] + s["delta"],
                tolerance_fn=lambda s: s["delta"],
            ),
            "gradient_log_rmse_to_one": metric_gate(
                "gradient_log_rmse_to_one",
                comparator="<=",
                threshold_fn=lambda s: s["max"] + s["delta"],
                tolerance_fn=lambda s: s["delta"],
            ),
            "flat_region_rms_no_regression": metric_gate(
                "flat_region_rms",
                comparator="<=",
                threshold_fn=lambda s: s["max"],
                tolerance_fn=lambda s: 0.0,
            ),
            "flat_region_score_no_regression": metric_gate(
                "flat_region_score",
                comparator="<=",
                threshold_fn=lambda s: s["max"],
                tolerance_fn=lambda s: 0.0,
            ),
        }

        for q in SEAC_QUANTILES:
            metric = f"seac_q{int(round(q * 100))}"
            gates[f"{metric}_no_regression"] = metric_gate(
                metric,
                comparator="<=",
                threshold_fn=lambda s: s["max"] + s["delta"],
                tolerance_fn=lambda s: s["delta"],
            )

        q90_improvement = metric_gate(
            "seac_q90",
            comparator="<",
            threshold_fn=lambda s: s["min"] - s["delta"],
            tolerance_fn=lambda s: s["delta"],
            strict=True,
        )
        neighbor_details = {
            name: metric_gate(
                name,
                comparator="<",
                threshold_fn=lambda s: s["min"] - s["delta"],
                tolerance_fn=lambda s: s["delta"],
                strict=True,
            )
            for name in ("seac_q85", "seac_q95")
        }
        gates["seac_q90_strict_improvement"] = q90_improvement
        gates["seac_neighbor_strict_improvement"] = _gate(
            any(detail["pass"] for detail in neighbor_details.values()),
            requirement="candidate < baseline_min - repeat_delta for q85 or q95",
            quantiles=neighbor_details,
        )

        modality_results[modality] = {
            "pass": all(gate["pass"] for gate in gates.values()),
            "gates": gates,
        }

    passed = all(gate["pass"] for gate in global_gates.values()) and all(
        result["pass"] for result in modality_results.values()
    )
    return {
        "schema_version": 2,
        "decision_scope": "PROMOTION_GATE",
        "verdict": "PASS" if passed else "FAIL",
        "promotion_eligible": passed,
        "experiments": experiments,
        "identity": provenance,
        "artifact_identity": artifact_provenance,
        "global_gates": global_gates,
        "modalities": modality_results,
        "decision_policy": {
            "baseline_repeat_delta": (
                "used by e_unobs, gradient, and SEAC only; excluded from fixed guards "
                "and flat-region no-regression"
            ),
            "compensation": "none: every gate passes separately for T1W and FLAIR",
            "fixed_guards": (
                "sharp >= baseline_min - 0.01; grain <= baseline_max * 1.03; "
                "HFEN <= baseline_max * 1.01; repeat delta is not added"
            ),
            "flat_region": "rms and score each require candidate <= baseline_max per modality",
            "seac": (
                "q85/q90/q95 cannot regress; q90 and at least one neighboring "
                "quantile must improve strictly beyond repeat delta"
            ),
            "alias_ncc_with_zf_error": "ignored by metric gates and finite scan",
            "visual_gate": "fixed-window three-view inspection remains mandatory",
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("baseline-a", "baseline-b", "candidate"):
        parser.add_argument(f"--{label}-artifact", type=Path, required=True)
        parser.add_argument(f"--{label}-score", type=Path, required=True)
        parser.add_argument(f"--{label}-kspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")

    result = compare_documents(
        _load_json(args.baseline_a_artifact),
        _load_json(args.baseline_a_score),
        _load_json(args.baseline_a_kspace),
        _load_json(args.baseline_b_artifact),
        _load_json(args.baseline_b_score),
        _load_json(args.baseline_b_kspace),
        _load_json(args.candidate_artifact),
        _load_json(args.candidate_score),
        _load_json(args.candidate_kspace),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output_file:
        output_file.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "decision_scope": result["decision_scope"],
                "verdict": result["verdict"],
                "promotion_eligible": result["promotion_eligible"],
                "experiments": result["experiments"],
                "modalities": {
                    key: value["pass"] for key, value in result["modalities"].items()
                },
            },
            indent=2,
        )
    )
    print(args.output)


if __name__ == "__main__":
    main()
