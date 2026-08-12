"""`run_manifest.json` is the CITATION GATE -- `report._controls_from_manifest` decides which runs
may be cited from its `status`/`exists` fields, including the rule that the vacuous all-ones
ctrlA_lam010 may never be cited for anything. Until the final whole-branch review those fields were
hand-typed and no generator for the file existed anywhere in the repo (C2a).

These tests cover `scripts/build_run_manifest.py` -- that statuses are DERIVED from the volumes
rather than declared, and that the committed manifest agrees, to the last digit, with the committed
non-vacuity record it is supposed to share one computation with.

`scripts/` has no `__init__.py`, so the module is loaded by path (same as test_check_nonvacuity.py).
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_run_manifest.py"

spec = importlib.util.spec_from_file_location("build_run_manifest", SCRIPT_PATH)
build_run_manifest = importlib.util.module_from_spec(spec)
sys.modules["build_run_manifest"] = build_run_manifest
spec.loader.exec_module(build_run_manifest)

MANIFEST_JSON = REPO_ROOT / "results_arc" / "diag_artefact" / "run_manifest.json"
NONVACUITY_JSON = REPO_ROOT / "results_arc" / "diag_artefact" / "nonvacuity_report.json"


def _write_volumes(path, *, vacuous: bool, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    gt = (rng.random((8, 8, 8)).astype(np.float32)) * 1.887
    rec = gt.copy() if vacuous else gt + rng.normal(scale=0.15, size=gt.shape).astype(np.float32)
    np.savez(path, recT1=rec, gtT1=gt, zfT1=gt, recFL=rec, gtFL=gt, zfFL=gt)


def test_status_is_derived_from_the_volumes_not_declared(tmp_path):
    (tmp_path / "results_arc").mkdir()
    genuine = tmp_path / "results_arc" / "genuine_volumes.npz"
    vacuous = tmp_path / "results_arc" / "vacuous_volumes.npz"
    _write_volumes(genuine, vacuous=False, seed=1)
    _write_volumes(vacuous, vacuous=True, seed=2)
    (tmp_path / "results_arc" / "genuine.log").write_text(
        "junk\nT1    psnr=38.362 ssim=0.9073 nmse=0.0208 beat=True\n"
        "FLAIR psnr=38.036 ssim=0.9131 nmse=0.0089 beat=True\n")

    registry = {
        "genuine": {"npz": "results_arc/genuine_volumes.npz", "log": "results_arc/genuine.log"},
        "vacuous": {"npz": "results_arc/vacuous_volumes.npz", "log": "results_arc/vacuous.log",
                    "reason": "all-ones mask"},
        "never_ran": {"npz": "results_arc/absent_volumes.npz", "log": "results_arc/absent.log"},
    }
    manifest = build_run_manifest.build(registry=registry, repo_root=tmp_path)
    runs = manifest["runs"]

    assert runs["genuine"]["status"] == "completed_genuine", runs["genuine"]
    assert runs["vacuous"]["status"] == "completed_vacuous", runs["vacuous"]
    assert runs["never_ran"]["status"] == "not_run", runs["never_ran"]
    assert runs["never_ran"]["exists"] is False and runs["never_ran"]["keys"] is None
    assert "never created" in runs["never_ran"]["log"]
    assert runs["genuine"]["final_metrics_from_log"]["T1"]["psnr"] == 38.362
    assert "beat" not in runs["genuine"]["final_metrics_from_log"]["T1"]
    assert runs["vacuous"]["final_metrics_from_log"] is None  # no log => no invented numbers
    # the vacuous fixture is the real ctrlA_lam010 failure mode: hard-DC hands the measured data
    # back, so rec == zf == gt and the gate's own numbers are unavailable (it recorded the error
    # message instead) -- they must be measured here rather than left null.
    assert runs["vacuous"]["measured_from_volumes"]["T1"]["rec_identical_to_zf"] is True
    assert runs["vacuous"]["measured_from_volumes"]["T1"]["max_rec_gt"] == 0.0
    assert runs["genuine"]["measured_from_volumes"]["T1"]["max_rec_gt"] > 0.0
    print("PASS test_status_is_derived_from_the_volumes_not_declared")


def test_parse_final_metrics_takes_the_last_line_per_contrast(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("T1    psnr=1.0 ssim=0.1 beat=False\n"
                   "[iter 10/20] loss=1e-2\n"
                   "T1    psnr=2.0 ssim=0.2 beat=True\n"
                   "FLAIR psnr=3.0 ssim=0.3 beat=True\n")
    got = build_run_manifest.parse_final_metrics(log)
    assert got == {"T1": {"psnr": 2.0, "ssim": 0.2}, "FLAIR": {"psnr": 3.0, "ssim": 0.3}}, got
    assert build_run_manifest.parse_final_metrics(tmp_path / "nope.log") is None
    print("PASS test_parse_final_metrics_takes_the_last_line_per_contrast")


@pytest.mark.skipif(not (MANIFEST_JSON.exists() and NONVACUITY_JSON.exists()),
                    reason="committed diagnosis records not present")
def test_committed_manifest_agrees_with_the_committed_nonvacuity_record():
    """The cross-check C2a asks for. Two committed files describe the same runs; before this fix
    they were produced by two independent uncommitted computations and disagreed in the last digits
    of max|rec-gt| (ctrlB FLAIR 0.3010830879211426 vs 0.3010830730199814). They now come from one
    code path, so agreement must be EXACT, and the manifest's citation-gate status must be exactly
    what the non-vacuity gate says.
    """
    manifest = json.loads(MANIFEST_JSON.read_text())
    nonvacuity = json.loads(NONVACUITY_JSON.read_text())

    assert manifest["generated_by"] == "scripts/build_run_manifest.py", manifest["generated_by"]
    assert nonvacuity["generated_by"] == "scripts/check_nonvacuity.py", nonvacuity["generated_by"]

    checked = 0
    for name, gate_entry in nonvacuity["runs"].items():
        run = manifest["runs"].get(name)
        if run is None:
            continue
        gate = {c: v["status"] for c, v in gate_entry["contrasts"].items()}
        expected = ("completed_genuine" if all(s == "PASS" for s in gate.values())
                    else "completed_vacuous")
        assert run["status"] == expected, (name, run["status"], gate)
        assert run["nonvacuity_gate"] == gate, (name, run["nonvacuity_gate"], gate)
        for contrast, reading in gate_entry["contrasts"].items():
            if reading["status"] != "PASS":
                continue
            assert run["measured_from_volumes"][contrast]["max_rec_gt"] == reading["max_abs_diff"], (
                f"{name}/{contrast}: the manifest and the non-vacuity report must come from ONE "
                f"computation of max|rec-gt|")
            checked += 1
    assert checked >= 4, f"expected several genuine runs to cross-check, got {checked}"
    print("PASS test_committed_manifest_agrees_with_the_committed_nonvacuity_record")


@pytest.mark.skipif(not MANIFEST_JSON.exists(), reason="committed manifest not present")
def test_the_committed_manifest_still_marks_the_all_ones_control_uncitable():
    """The one rule the whole citation gate exists to enforce (task-9-brief.md point 3)."""
    manifest = json.loads(MANIFEST_JSON.read_text())
    assert manifest["runs"]["ctrlA_lam010"]["status"] == "completed_vacuous"
    assert manifest["runs"]["ctrlA_lam120"]["status"] == "not_run"
    for name in ("ctrlB_seed1", "ctrlA_af2_lam010", "ctrlA_af2_lam060"):
        assert manifest["runs"][name]["status"] == "completed_genuine", name
    print("PASS test_the_committed_manifest_still_marks_the_all_ones_control_uncitable")
