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
