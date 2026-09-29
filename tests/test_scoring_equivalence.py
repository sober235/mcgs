"""spec_scoring (arc/run_spec.py) and arc/cycle.py give the same numbers on the same checkpoint (hp_n36).
Reads the newest (by modification time) score jsons that scripts/regress_brain_pin.sh writes; skipped when they
are absent. The run_spec json must come from code equal to HEAD in the four files of the strict entry.

Run: env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_scoring_equivalence.py -q
"""
import glob
import json
import math
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["psnr", "ssim", "nmse", "hfen", "sharp", "grain", "head_psnr", "head_ssim", "head_hfen",
        "grain_ratio", "head_psnr_cplx", "cos_phase", "e_unobs", "e_unobs_excl_z", "nmse_unobs", "comp"]
ENTRY_FILES = ["arc/spec_scoring.py", "arc/run_spec.py", "arc/spec_io.py", "arc/strict_config.py"]


def _latest(pattern):
    files = glob.glob(str(ROOT / pattern))
    return json.loads(Path(max(files, key=os.path.getmtime)).read_text()) if files else None


CYC = _latest("results_arc/pin_hp_n36_*_score.json")
SPC = _latest("results_spec/brainpin_equiv*_score.json")
pytestmark = pytest.mark.skipif(CYC is None or SPC is None, reason="run scripts/regress_brain_pin.sh first")


@pytest.mark.parametrize("channel", ["plain", "contrast"])
def test_spec_scoring_equals_cycle(channel):
    for k in KEYS:
        a, b = CYC[channel][k], SPC[channel][k]
        assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12), f"{channel}.{k}: cycle {a!r} vs run_spec {b!r}"


def test_run_spec_run_had_no_fallbacks():
    assert SPC["fallbacks"] == []


def _blob(rev, path):
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", f"{rev}:{path}"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def test_run_spec_json_comes_from_the_code_at_head():
    rerun = "re-run `bash scripts/regress_brain_pin.sh GPU`"
    assert "git_dirty_paths" in SPC, f"the newest brainpin_equiv json predates git_dirty_paths; {rerun}"
    for f in ENTRY_FILES:
        assert _blob(SPC["git_commit"], f) == _blob("HEAD", f), f"{f} differs from {SPC['git_commit'][:10]}; {rerun}"
        assert f not in SPC["git_dirty_paths"], f"{f} had uncommitted changes when the json was written; {rerun}"
