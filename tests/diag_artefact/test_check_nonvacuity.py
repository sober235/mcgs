"""The non-vacuity GATE (diag_artefact.controls.assert_control_is_not_vacuous) is unit-tested in
test_controls.py against synthetic arrays only. Nothing wired it to a real control run's npz and
wrote a committed record -- exactly the gap this task exists to close, per fix-round-1 review.
`scripts/check_nonvacuity.py` is the runner that closes it; these tests cover ITS failure path,
since "a runner that cannot fail is the same defect one level up" (a gate that is never actually
exercised against a vacuous run is no better than no gate at all).

`scripts/` has no `__init__.py`, so the module is loaded by path rather than imported as a package
member -- the same reason build_ctrl_mask.py/build_ctrlB_mask.py insert the repo root onto
sys.path instead of being importable directly.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_nonvacuity.py"

spec = importlib.util.spec_from_file_location("check_nonvacuity", SCRIPT_PATH)
check_nonvacuity = importlib.util.module_from_spec(spec)
sys.modules["check_nonvacuity"] = check_nonvacuity
spec.loader.exec_module(check_nonvacuity)


def _write_volumes(path, *, vacuous: bool, seed: int = 0) -> None:
    """A minimal *_volumes.npz fixture with the same 6 keys load_volumes expects.
    `vacuous=True` reproduces the real ctrlA_lam010 failure mode: rec == gt to float32 rounding.
    `vacuous=False` gives rec a material, GT-scale difference from gt -- comparable in magnitude to
    the genuine ctrlA_af2 runs (~10% of peak), not just barely over the 1e-4 relative floor.
    """
    rng = np.random.default_rng(seed)
    gt = (rng.random((8, 8, 8)).astype(np.float32)) * 1.887
    if vacuous:
        rec = gt.copy()  # bit-identical: the exact hard-DC-with-(1-m)=0 failure mode
    else:
        rec = gt + rng.normal(scale=0.15, size=gt.shape).astype(np.float32)
    zf = gt.copy()
    np.savez(path, recT1=rec, gtT1=gt, zfT1=zf, recFL=rec, gtFL=gt, zfFL=zf)


def test_main_exits_nonzero_on_a_vacuous_volumes_file(tmp_path):
    npz = tmp_path / "ctrlA_lam010_volumes.npz"
    _write_volumes(npz, vacuous=True)
    json_out = tmp_path / "report.json"

    rc = check_nonvacuity.main([str(npz), "--json-out", str(json_out)])

    assert rc != 0, "the runner must exit non-zero when any contrast fails the non-vacuity gate"
    print("PASS test_main_exits_nonzero_on_a_vacuous_volumes_file")


def test_main_records_the_failure_and_its_measured_value_in_the_json(tmp_path):
    npz = tmp_path / "ctrlA_lam010_volumes.npz"
    _write_volumes(npz, vacuous=True)
    json_out = tmp_path / "report.json"

    check_nonvacuity.main([str(npz), "--json-out", str(json_out)])
    report = json.loads(json_out.read_text())

    assert report["any_fail"] is True
    entry = report["runs"]["ctrlA_lam010"]["contrasts"]["T1"]
    assert entry["status"] == "FAIL"
    assert "identical" in entry["error"]
    print("PASS test_main_records_the_failure_and_its_measured_value_in_the_json")


def test_main_exits_zero_on_a_genuine_volumes_file(tmp_path):
    npz = tmp_path / "ctrlB_seed1_volumes.npz"
    _write_volumes(npz, vacuous=False)
    json_out = tmp_path / "report.json"

    rc = check_nonvacuity.main([str(npz), "--json-out", str(json_out)])
    report = json.loads(json_out.read_text())

    assert rc == 0
    assert report["any_fail"] is False
    assert report["runs"]["ctrlB_seed1"]["contrasts"]["T1"]["status"] == "PASS"
    assert report["runs"]["ctrlB_seed1"]["contrasts"]["FLAIR"]["status"] == "PASS"
    print("PASS test_main_exits_zero_on_a_genuine_volumes_file")


def test_main_keeps_checking_every_file_after_an_earlier_one_fails(tmp_path):
    # One vacuous input in a batch must not stop the rest from being checked/recorded -- a batch
    # runner that aborts on the first failure would silently skip auditing everything after it.
    vac = tmp_path / "ctrlA_lam010_volumes.npz"
    genuine = tmp_path / "ctrlB_seed1_volumes.npz"
    _write_volumes(vac, vacuous=True)
    _write_volumes(genuine, vacuous=False, seed=1)
    json_out = tmp_path / "report.json"

    rc = check_nonvacuity.main([str(vac), str(genuine), "--json-out", str(json_out)])
    report = json.loads(json_out.read_text())

    assert rc != 0
    assert report["runs"]["ctrlA_lam010"]["contrasts"]["T1"]["status"] == "FAIL"
    assert report["runs"]["ctrlB_seed1"]["contrasts"]["T1"]["status"] == "PASS"
    print("PASS test_main_keeps_checking_every_file_after_an_earlier_one_fails")


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_main_exits_nonzero_on_a_vacuous_volumes_file(tmp)
    print("ALL PASS")
