"""CPU acceptance tests: changing a tie into a win or omitting an arm must fail."""
import copy
import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("objective", ROOT / "objective.py")
if spec is not None and spec.loader is not None and (ROOT / "objective.py").exists():
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
else:
    obj = None


def baseline():
    assert obj is not None, "objective implementation is missing"
    return obj.load_baseline(ROOT.parent / "baseline_lesion_allmetrics.json")


def winning_candidate(rows):
    candidate = copy.deepcopy(rows)
    # Put one real plane strictly above every opponent, on both scans.
    for scan in ("pre", "post"):
        m = candidate[0]["scans"][scan]["metrics"]
        other = [v for k, v in m.items() if k != "MC-GS"]
        m["MC-GS"] = {"psnr": max(v["psnr"] for v in other) + 1,
                        "ssim": 1 - (1 - max(v["ssim"] for v in other)) / 2,
                        "nmse": min(v["nmse"] for v in other) / 2,
                        "hfen": min(v["hfen"] for v in other) / 2}
    return candidate


def test_true_witness_uses_same_plane_both_scans_and_all_metrics():
    rows = baseline()
    result = obj.score(winning_candidate(rows), rows)
    assert result["strict_success"] is True
    assert result["score"] == pytest.approx(math.log(10) / 10)
    assert result["witness"]["axis"] == 0
    assert result["witness"]["index"] == 194
    assert len(result["planes"]) == 59
    assert len(result["witness"]["margins"]) == 8


def test_false_witness_cannot_mix_pre_and_post_planes():
    rows = baseline()
    c = winning_candidate(rows)
    c[1]["scans"]["post"]["metrics"]["MC-GS"] = c[0]["scans"]["post"]["metrics"]["MC-GS"]
    c[0]["scans"]["post"]["metrics"]["MC-GS"] = copy.deepcopy(rows[0]["scans"]["post"]["metrics"]["MC-GS"])
    result = obj.score(c, rows)
    assert result["strict_success"] is False
    assert result["score"] < 0
    assert result["witness"] is None
    assert result["best_plane"]["blockers"]


def test_tie_is_not_success_even_when_scorer_flags_claim_win():
    rows = baseline()
    c = winning_candidate(rows)
    m = c[0]["scans"]["post"]["metrics"]
    m["MC-GS"]["hfen"] = min(v["hfen"] for k, v in m.items() if k != "MC-GS")
    c[0]["all_first"] = True
    result = obj.score(c, rows)
    assert result["score"] == 0
    assert result["strict_success"] is False


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0, -1])
def test_invalid_positive_metrics_raise_instead_of_clipping(value):
    rows = baseline()
    c = winning_candidate(rows)
    c[0]["scans"]["pre"]["metrics"]["MC-GS"]["nmse"] = value
    with pytest.raises(ValueError):
        obj.score(c, rows)


@pytest.mark.parametrize("mutation", ["label", "opponent", "plane", "missing", "ssim"])
def test_frozen_comparison_cannot_be_weakened(mutation):
    rows = baseline()
    c = winning_candidate(rows)
    if mutation == "label":
        del c[0]["scans"]["pre"]["metrics"]["ZS-SSDU"]
    elif mutation == "opponent":
        c[0]["scans"]["pre"]["metrics"]["ZS-SSDU"]["psnr"] -= 1e-10
    elif mutation == "plane":
        c[0]["core"] += 1
    elif mutation == "missing":
        c.pop()
    else:
        c[0]["scans"]["pre"]["metrics"]["MC-GS"]["ssim"] = 1
    with pytest.raises(ValueError):
        obj.score(c, rows)


def test_baseline_is_frozen_to_audited_bytes(tmp_path):
    rows = baseline()
    import json
    p = tmp_path / "modified.json"
    p.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="hash"):
        obj.load_baseline(p)


def shifted(value, ulps):
    for _ in range(abs(ulps)):
        value = math.nextafter(value, math.inf if ulps > 0 else -math.inf)
    return value


def test_real_unchanged_control_scorer_replay_allows_only_recorded_roundoff():
    rows = baseline()
    path = ROOT / "verification/loop_scores/A_control_1792220439827229756/lesion_allmetrics.json"
    candidate = obj.read_json(path)
    canonical = copy.deepcopy(candidate)
    for observed, frozen in zip(canonical, rows):
        for scan in ("pre", "post"):
            for label in obj.LABELS[:-1]:
                observed["scans"][scan]["metrics"][label] = copy.deepcopy(frozen["scans"][scan]["metrics"][label])
    result = obj.score(candidate, rows)
    expected = obj.score(canonical, rows)
    assert result["planes"] == expected["planes"]
    assert result["score"] == expected["score"]
    assert result["strict_success"] == expected["strict_success"]
    assert result["witness"] == expected["witness"]
    check = result["input_validation"]
    assert check["allowed_roundoff_count"] == 750
    assert check["max_absolute_roundoff"] == 4.440892098500626e-16
    assert check["max_roundoff_ulp"] == 6
    assert check["roundoff_count_by_metric"] == {"psnr": 0, "ssim": 0, "nmse": 0, "hfen": 750}


@pytest.mark.parametrize("metric", ["psnr", "ssim", "nmse", "hfen"])
@pytest.mark.parametrize("ulps,accepted", [(8, True), (9, False)])
def test_opponent_recomputation_has_fixed_eight_ulp_boundary(metric, ulps, accepted):
    rows = baseline()
    candidate = winning_candidate(rows)
    metrics = candidate[0]["scans"]["pre"]["metrics"]["ZS-SSDU"]
    metrics[metric] = shifted(metrics[metric], ulps)
    if accepted:
        result = obj.score(candidate, rows)
        assert result["input_validation"]["allowed_roundoff_count"] == 1
        assert result["input_validation"]["max_roundoff_ulp"] == 8
    else:
        with pytest.raises(ValueError, match="opponent"):
            obj.score(candidate, rows)


@pytest.mark.parametrize("candidate_ulps,opponent_ulps,strict_win", [(1, 2, False), (-1, -2, True), (0, 2, False)])
def test_frozen_threshold_controls_micro_win_loss_and_tie(candidate_ulps, opponent_ulps, strict_win):
    rows = baseline()
    candidate = winning_candidate(rows)
    frozen_hfen = rows[0]["scans"]["pre"]["metrics"]["ZS-SSDU"]["hfen"]
    metrics = candidate[0]["scans"]["pre"]["metrics"]
    metrics["MC-GS"]["hfen"] = shifted(frozen_hfen, candidate_ulps)
    metrics["ZS-SSDU"]["hfen"] = shifted(frozen_hfen, opponent_ulps)
    result = obj.score(candidate, rows)
    comparison = result["planes"][0]["comparisons"]["pre:hfen"]
    assert comparison["best_opponent"] == frozen_hfen
    assert comparison["candidate"] == metrics["MC-GS"]["hfen"]
    assert comparison["strict_win"] is strict_win
    assert result["strict_success"] is strict_win
    assert (result["score"] > 0) is strict_win
    if candidate_ulps == 0:
        assert result["score"] == 0
