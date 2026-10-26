"""Re-evaluate saved reconstructions under the user-requested three-metric contract."""
from pathlib import Path
import collections
import csv
import datetime
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
METRICS = ("psnr", "ssim", "nmse")
SCANS = ("pre", "post")
CONDITIONS = tuple(f"{s}:{m}" for s in SCANS for m in METRICS)
SUMMARY = ROOT / "phase_firstorder/scoring/combined_phase_firstorder_v1/summary.json"
BASELINE = ROOT / "baseline_lesion_allmetrics.json"
ARCHIVE = ROOT / "archival_allplanes_20261018"
EXPECTED_SUMMARY = "ce12805b84710cfe119fabfc063bf62931aa7e797f28174a627df000d2422413"
EXPECTED_BASELINE = "eb351e158b8029b846b37a45d85bf0b6a99825b326ad5c411f59095d9db0df22"
SEEN = {}


def read(path, expected=None):
    path = Path(path)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected is not None:
        assert digest == expected, str(path)
    SEEN[str(path)] = digest
    return json.loads(data)


def margin(candidate, best, metric):
    assert math.isfinite(candidate) and math.isfinite(best)
    if metric == "psnr":
        return (candidate - best) * math.log(10) / 10
    if metric == "ssim":
        assert candidate < 1 and best < 1
        return math.log((1 - best) / (1 - candidate))
    assert candidate > 0 and best > 0
    return math.log(best / candidate)


def comparisons(candidate, references):
    result = {}
    for scan in SCANS:
        for metric in METRICS:
            values = {k: v[scan][metric] for k, v in references.items()}
            best = (max if metric in ("psnr", "ssim") else min)(values.values())
            value = candidate[scan][metric]
            win = value > best if metric in ("psnr", "ssim") else value < best
            result[f"{scan}:{metric}"] = {
                "candidate": value,
                "best_opponent": best,
                "labels": [k for k, v in values.items() if v == best],
                "strict_win": win,
                "margin": margin(value, best, metric),
            }
    return result


def re_evaluate(name, cohort, path, expected, head):
    original = read(path, expected)
    assert original["baseline_sha256"] == EXPECTED_BASELINE
    assert len(original["planes"]) == 59
    assert {(p["axis"], p["index"]) for p in original["planes"]} == set(BY_PLANE)
    planes = []
    for p in original["planes"]:
        frozen = BY_PLANE[p["axis"], p["index"]]
        assert p["core"] == frozen["core"]
        values = {s: {m: p["comparisons"][f"{s}:{m}"]["candidate"] for m in METRICS} for s in SCANS}
        references = {label: {s: frozen["scans"][s]["metrics"][label] for s in SCANS} for label in LABELS}
        cs = comparisons(values, references)
        for condition, c in cs.items():
            old = p["comparisons"][condition]
            assert c["best_opponent"] == old["best_opponent"]
            assert c["strict_win"] == old["strict_win"]
            assert math.isclose(c["margin"], p["margins"][condition], abs_tol=2e-14)
        score = min(c["margin"] for c in cs.values())
        win = all(c["strict_win"] for c in cs.values())
        assert (score > 0) == win
        planes.append({"axis": p["axis"], "index": p["index"], "core": p["core"], "score_six": score,
                       "strict_success": win, "comparisons": cs})
    best = max(planes, key=lambda p: p["score_six"])
    assert best["score_six"] >= original["score"] - 2e-14
    result = {
        "name": name, "cohort": cohort, "source_objective": str(path),
        "source_objective_sha256": SEEN[str(Path(path))],
        "score_six": best["score_six"], "strict_success": any(p["strict_success"] for p in planes),
        "passing_planes_both_scans": sum(p["strict_success"] for p in planes),
        "passing_planes_by_scan": {s: sum(all(p["comparisons"][f"{s}:{m}"]["strict_win"] for m in METRICS) for p in planes) for s in SCANS},
        "per_condition_passing_planes": {c: sum(p["comparisons"][c]["strict_win"] for p in planes) for c in CONDITIONS},
        "best_plane": best, "planes": planes,
    }
    if head is not None:
        result["whole_head_metrics"] = {s: {m: head[s][m] for m in METRICS} for s in SCANS}
        result["whole_head_comparisons"] = comparisons(result["whole_head_metrics"], HEAD_REFS)
    return result


def dump(path, obj):
    assert not path.exists(), f"Refuse to overwrite {path}"
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    assert not (OUT / "summary.json").exists(), "Use a new versioned output for a rerun"
    summary = read(SUMMARY, EXPECTED_SUMMARY)
    baseline = read(BASELINE, EXPECTED_BASELINE)
    old_global = read(ROOT / "development_incumbent.json")
    BY_PLANE = {(p["axis"], p["index"]): p for p in baseline}
    assert len(BY_PLANE) == 59
    LABELS = tuple(k for k in baseline[0]["scans"]["pre"]["metrics"] if k != "MC-GS")
    assert len(LABELS) == 10
    HEAD_REFS = {k: summary["references"][k]["whole_head_metrics"] for k in LABELS}
    assert len(summary["trials"]) == 25
    recent = [re_evaluate(t["name"], "formal25", t["objective"], t["objective_sha256"], t["whole_head_metrics"]) for t in summary["trials"]]
    archive_summary = read(ARCHIVE / "summary.json")
    archival_paths = sorted(ARCHIVE.glob("objective_ar_*.json"))
    assert len(archival_paths) == archive_summary["completed"] == 65
    historical = [re_evaluate(p.stem.removeprefix("objective_"), "archival65", p, None, None) for p in archival_paths]
    recent.sort(key=lambda t: t["score_six"], reverse=True)
    historical.sort(key=lambda t: t["score_six"], reverse=True)
    contract = {
        "version": 1, "user_instruction": "不考虑HFEN指标，重新评估", "metrics": list(METRICS),
        "scans": list(SCANS), "conditions": list(CONDITIONS), "opponents": list(LABELS),
        "eligible_planes": 59, "strict_success_rule": "Same predeclared plane; both scans; all three metrics strictly beat every one of the ten frozen opponents. No equality tolerance in wins.",
        "metric_region": "Head mask on each lesion-containing plane, not lesion ROI", "ranking": "max over planes of min over the six original normalized margins",
        "normalization": {"psnr": "(candidate-best)*ln(10)/10", "ssim": "ln((1-best)/(1-candidate))", "nmse": "ln(best/candidate)"},
        "HFEN": "Excluded from ranking, acceptance and reported comparisons", "keep_threshold": 1e-4,
        "fixed": "Existing reconstructions, masks, normalization, reference images, 59 planes and full-size opponents; no retraining or amplitude fitting",
        "scope": "Retrospective metric ablation; one case/one seed adaptive development. The 65 archival records have the pre-existing provenance limits and are not 65 new training experiments. Archival whole-head and ROI scores are not recomputed.",
        "parent_summary": str(SUMMARY), "parent_summary_sha256": EXPECTED_SUMMARY,
    }
    dump(OUT / "metric_contract_v1.json", contract)
    dump(OUT / "all_results.json", {"formal25": recent, "archival65": historical})
    with (OUT / "ranking.csv").open("x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cohort", "rank", "name", "score_six", "passing_planes_both_scans", "pre_all_three", "post_all_three", *CONDITIONS, "best_axis", "best_index"])
        for cohort in (recent, historical):
            for i, t in enumerate(cohort, 1):
                writer.writerow([t["cohort"], i, t["name"], t["score_six"], t["passing_planes_both_scans"], t["passing_planes_by_scan"]["pre"], t["passing_planes_by_scan"]["post"], *[t["per_condition_passing_planes"][c] for c in CONDITIONS], t["best_plane"]["axis"], t["best_plane"]["index"]])
    result = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "contract": contract,
              "completed_formal_trials": 25, "archival_reconstructions": 65,
              "formal_trials_with_strict_win": sum(t["strict_success"] for t in recent),
              "archival_models_with_strict_win": sum(t["strict_success"] for t in historical),
              "best_formal": {k: v for k, v in recent[0].items() if k != "planes"},
              "best_archival": {k: v for k, v in historical[0].items() if k != "planes"},
              "old_four_metric_incumbent": old_global["exp"],
              "best_formal_changed": recent[0]["name"] != old_global["exp"],
              "top_five_formal": [{k: t[k] for k in ("name", "score_six", "passing_planes_both_scans", "passing_planes_by_scan", "per_condition_passing_planes")} for t in recent[:5]],
              "whole_head_opponents": {k: {s: {m: v[s][m] for m in METRICS} for s in SCANS} for k, v in HEAD_REFS.items()},
              "source_sha256": dict(SEEN), "runtime": {"python": sys.version.split()[0], "executable": sys.executable, "numerics": "Python stdlib exact saved float values; no CUDA/array reconstruction"}}
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in SEEN.items())
    result["all_inputs_unchanged"] = True
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dump(OUT / "summary.json", result)
    print(json.dumps({k: result[k] for k in ("formal_trials_with_strict_win", "archival_models_with_strict_win", "best_formal_changed", "top_five_formal")}, indent=2))
    print("BEST_FORMAL", json.dumps(result["best_formal"], ensure_ascii=False))
    print("BEST_ARCHIVAL", result["best_archival"]["name"], result["best_archival"]["score_six"])
