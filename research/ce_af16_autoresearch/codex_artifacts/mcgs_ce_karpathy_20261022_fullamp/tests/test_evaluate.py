"""CPU fixtures only: no clinical volumes, GPU, or external writes."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def scorer():
    spec = importlib.util.spec_from_file_location("campaign_evaluate", ROOT / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def heads(module, psnr=30., ssim=.9, nmse=.01):
    return {s: {"psnr": psnr, "ssim": ssim, "nmse": nmse} for s in module.SCANS}


def references(module):
    result = {label: heads(module) for label in module.OPPONENTS}
    result[module.OPPONENTS[-1]]["post"]["psnr"] = 31.
    return result


def test_normalized_margins_preserve_direction_and_strict_tiny_differences():
    m = scorer()
    assert m.margin(31., 30., "psnr") == pytest.approx(math.log(10) / 10)
    assert m.margin(.95, .9, "ssim") == pytest.approx(math.log(2))
    assert m.margin(.005, .01, "nmse") == pytest.approx(math.log(2))
    assert m.margin(math.nextafter(.9, 1), .9, "ssim") > 0
    with pytest.raises(ValueError):
        m.margin(1., 2., "hfen")


def test_primary_is_worst_whole_head_condition_and_compares_all_ten():
    m = scorer()
    candidate = heads(m, psnr=30.5, ssim=.95, nmse=.005)
    result = m.summarize_whole_head(candidate, references(m))
    assert result["score"] == pytest.approx(-.5 * math.log(10) / 10)
    assert result["strict_success"] is False
    assert result["comparisons"]["post:psnr"]["labels"] == [m.OPPONENTS[-1]]
    assert set(result["comparisons"]) == set(m.CONDITIONS)
    for comparison in result["comparisons"].values():
        assert set(comparison["opponents"]) == set(m.OPPONENTS)
    tied = m.summarize_whole_head(heads(m), {k: heads(m) for k in m.OPPONENTS})
    assert tied["score"] == 0 and tied["strict_success"] is False


@pytest.mark.parametrize("corruption", ["scan", "opponent", "nonfinite", "ssim", "nmse", "metric"])
def test_comparisons_reject_incomplete_or_invalid_values(corruption):
    m = scorer()
    candidate, refs = heads(m), references(m)
    if corruption == "scan":
        del candidate["post"]
    elif corruption == "opponent":
        del refs[m.OPPONENTS[-1]]
    elif corruption == "nonfinite":
        candidate["pre"]["psnr"] = float("nan")
    elif corruption == "ssim":
        candidate["pre"]["ssim"] = 1.
    elif corruption == "nmse":
        candidate["pre"]["nmse"] = 0.
    else:
        candidate["pre"]["hfen"] = .1
    with pytest.raises(ValueError):
        m.comparisons(candidate, refs)


def test_slice_success_stays_separate_from_whole_head_and_json_has_only_three_metrics():
    m = scorer()
    candidate = heads(m, psnr=32., ssim=.95, nmse=.005)
    baseline, observed = [], []
    for index in range(59):
        baseline.append({"axis": 0, "index": index, "core": 150,
                         "scans": {s: {"metrics": {**{k: {**heads(m)[s], "hfen": .1} for k in m.OPPONENTS},
                                                        "MC-GS": {**heads(m)[s], "hfen": .2}}} for s in m.SCANS}})
        observed.append({"axis": 0, "index": index, "core": 150, "metrics": copy.deepcopy(candidate)})
    planes = m.summarize_planes(observed, baseline)
    whole = m.summarize_whole_head(heads(m, psnr=29.), references(m))
    objective = m.build_objective("fixture", Path("/fixture"), whole, planes, {"files": {}})
    assert objective["strict_success"] is False
    assert objective["score"] == whole["score"] < 0
    assert objective["lesion_planes"]["strict_success"] is True
    assert objective["lesion_planes"]["passing_planes_both_scans"] == 59
    assert objective["metrics"] == ["psnr", "ssim", "nmse"]
    json.dumps(objective, allow_nan=False)
    assert "hfen" not in json.dumps(objective).lower()


def test_candidate_loading_requires_finite_complex_matching_scans(tmp_path):
    m = scorer()
    run = tmp_path / "run"
    run.mkdir()
    np.save(run / "candidate_pre_complex.npy", np.ones((9, 9, 9), np.complex64))
    with pytest.raises((ValueError, FileNotFoundError)):
        m.load_candidate_pair(run)
    np.save(run / "candidate_post_complex.npy", np.ones((9, 9, 9), np.float32))
    with pytest.raises(ValueError):
        m.load_candidate_pair(run)
    bad = np.ones((9, 9, 9), np.complex64)
    bad[0, 0, 0] = np.inf
    np.save(run / "candidate_post_complex.npy", bad)
    with pytest.raises(ValueError):
        m.load_candidate_pair(run)


def test_whole_head_matches_original_cpu_pair_arithmetic_without_fourth_metric():
    m = scorer()
    import torch
    from skimage.metrics import structural_similarity
    rng = np.random.default_rng(51)
    gt = (rng.uniform(.05, 1.2, (9, 9, 9)) + 1j * rng.uniform(-.3, .3, (9, 9, 9))).astype(np.complex64)
    candidate = (gt + rng.normal(0, .015, gt.shape)).astype(np.complex64)
    result = m.whole_head_metrics(candidate, gt)
    rec = torch.from_numpy(candidate).to(torch.complex128).abs()
    truth = torch.from_numpy(gt).abs().to(torch.float64)
    g32, r32 = truth.numpy().astype(np.float32), rec.numpy().astype(np.float32)
    mask32 = g32 > .05 * float(np.quantile(g32, .999))
    mse = float(np.mean((r32[mask32].astype(np.float64) - g32[mask32].astype(np.float64)) ** 2))
    _, smap = structural_similarity(r32, g32, data_range=float(g32.max()), full=True)
    head = torch.from_numpy(truth.numpy() > .05 * float(np.quantile(truth.numpy(), .999)))
    expected = {"psnr": float(20 * np.log10(float(g32.max())) - 10 * np.log10(mse)),
                "ssim": float(smap[mask32].mean()),
                "nmse": float(((rec - truth)[head] ** 2).sum()) / float((truth[head] ** 2).sum())}
    assert result == expected
    assert set(result) == {"psnr", "ssim", "nmse"}


def test_plane_metric_matches_original_numpy_slice_formula():
    m = scorer()
    from skimage.metrics import structural_similarity
    rng = np.random.default_rng(52)
    gt = rng.uniform(.01, 1.2, (9, 9, 9)).astype(np.float32)
    rec = (gt + rng.normal(0, .01, gt.shape)).astype(np.float32)
    head = gt > .05 * float(np.quantile(gt, .999))
    index = 4
    result = m.plane_metrics(rec, gt, head, float(gt.max()), 1, index)
    g, r, hd = np.take(gt, index, 1).astype(np.float64), np.take(rec, index, 1).astype(np.float64), np.take(head, index, 1)
    _, smap = structural_similarity(r, g, data_range=float(gt.max()), full=True)
    assert result == {"psnr": float(20 * np.log10(float(gt.max())) - 10 * np.log10(np.mean((r[hd] - g[hd]) ** 2))),
                      "ssim": float(smap[hd].mean()),
                      "nmse": float(np.sum((r[hd] - g[hd]) ** 2) / np.sum(g[hd] ** 2))}


def test_freeze_recheck_rejects_changed_inputs(tmp_path):
    m = scorer()
    path = tmp_path / "input.txt"
    path.write_text("frozen")
    hashes = {str(path): m.sha256_file(path)}
    m.verify_hashes(hashes)
    path.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        m.verify_hashes(hashes)


def test_whole_head_gt_uses_pair_stored_scale_arithmetic(tmp_path):
    m = scorer()
    import torch
    (tmp_path / "sim").mkdir()
    rng = np.random.default_rng(95)
    source = (rng.normal(size=(9, 9, 9)) + 1j * rng.normal(size=(9, 9, 9))).astype(np.complex64)
    scales = {"pre": 183.57667347717228, "post": 243.4532769927959}
    np.savez(tmp_path / "sim/gt.npz", pre=source, post=source, **{f"scale_{s}": v for s, v in scales.items()})
    freeze = {"gt_equivalence": {s: {"scale_float64": v} for s, v in scales.items()}}
    result = m.load_whole_head_gt(tmp_path, freeze)
    for scan in m.SCANS:
        expected = (torch.from_numpy(source).to(torch.complex64) / scales[scan]).numpy()
        assert np.array_equal(result[scan], expected)
    freeze["gt_equivalence"]["post"]["scale_float64"] = 5.
    with pytest.raises(ValueError, match="scale"):
        m.load_whole_head_gt(tmp_path, freeze)


def test_fresh_run_exports_complete_schema_and_refuses_overwrite(tmp_path, monkeypatch):
    """An artificial 41^3 head yields exactly 21+20+18 eligible planes."""
    m = scorer()
    report = m._load_report()
    topic, run, output = tmp_path / "topic", tmp_path / "run", tmp_path / "scores"
    for directory in (topic / "recon", topic / "sim", run):
        directory.mkdir(parents=True)
    shape = (41, 41, 41)
    rng = np.random.default_rng(96)
    core = np.zeros(shape, bool)
    core[10:31, 10:30, 10:28] = True
    gt = np.zeros(shape, np.complex64)
    gt[core] = rng.uniform(.4, 1.2, core.sum())
    truth = {"pre": gt, "post": gt * np.complex64(1.1)}
    for scan in m.SCANS:
        np.save(topic / f"recon/gt_{scan}_complex.npy", truth[scan])
        candidate = (truth[scan] + rng.normal(0, .02, shape)).astype(np.complex64)
        np.save(run / f"candidate_{scan}_complex.npy", candidate)
        np.save(topic / f"recon/reference_{scan}_complex.npy", candidate)
    gt_path = topic / "sim/gt.npz"
    np.savez(gt_path, **truth, scale_pre=1., scale_post=1.)
    roi_path = tmp_path / "roi.npz"
    roi_names = list(report.ROI_KEYS)
    rois = {"roi0": core}
    for index in range(1, 6):
        mask = np.zeros(shape, bool)
        mask[12 + index:14 + index, 14:17, 14:17] = True
        rois[f"roi{index}"] = mask
    np.savez(roi_path, names=np.array(roi_names), **rois)
    refs = {label: {"key": label, "volumes": {s: str(topic / f"recon/reference_{s}_complex.npy") for s in m.SCANS}}
            for label in (*m.OPPONENTS, "MC-GS")}
    paths = [roi_path, gt_path, *(topic / f"recon/gt_{s}_complex.npy" for s in m.SCANS),
             *(topic / f"recon/reference_{s}_complex.npy" for s in m.SCANS)]
    manifest = {"files": {str(p): m.sha256_file(p) for p in paths}}
    manifest_path = tmp_path / "freeze.json"
    manifest_path.write_text(json.dumps(manifest))
    reference_freeze = {"loop_input_freeze_path": str(manifest_path),
                        "loop_input_freeze_sha256": m.sha256_file(manifest_path),
                        "files": {str(p): m.sha256_file(p) for p in paths},
                        "gt_equivalence": {s: {"scale_float64": 1.} for s in m.SCANS}}
    baseline = []
    for axis in range(3):
        for index in range(shape[axis]):
            count = int(np.take(core, index, axis).sum())
            if count >= 150:
                baseline.append({"axis": axis, "index": index, "core": count,
                                 "scans": {s: {"metrics": {label: heads(m)[s] for label in m.OPPONENTS}} for s in m.SCANS}})
    context = {"hashes": {**manifest["files"], str(manifest_path): m.sha256_file(manifest_path)},
               "manifest": manifest, "report_freeze": reference_freeze, "report": report,
               "references": refs, "reference_heads": references(m), "baseline": baseline,
               "whole_head_contract": {"metrics": list(m.METRICS)}, "slice_contract": {"eligible_planes": 59}}
    monkeypatch.setattr(m, "TOPIC", topic)
    monkeypatch.setattr(m, "ROI", roi_path)
    monkeypatch.setattr(m, "load_frozen_context", lambda: context)
    result = m.evaluate_run(run, output, "fixture")
    assert len(baseline) == 59
    assert result == json.loads((output / "objective.json").read_text())
    assert result["all_inputs_unchanged"] is True
    assert result["primary_objective"] == "whole_head_min_six_normalized_margin"
    assert set(result["whole_head_metrics"]["pre"]) == set(m.METRICS)
    assert len(json.loads((output / "lesion_planes.json").read_text())["planes"]) == 59
    enhancement = json.loads((output / "enhancement.json").read_text())
    assert len(enhancement["rows"]) == 13  # truth + original 11 + candidate
    assert enhancement["regions"] == list(report.REGIONS)
    assert "background_post" in enhancement["rows"][-1]
    assert (output / "whole_head.csv").read_text().splitlines()[0] == "method,scan,psnr,ssim,nmse"
    assert len((output / "whole_head.csv").read_text().splitlines()) == 23
    original_hash = m.sha256_file(output / "objective.json")
    with pytest.raises(FileExistsError):
        m.evaluate_run(run, output, "fixture")
    assert m.sha256_file(output / "objective.json") == original_hash
