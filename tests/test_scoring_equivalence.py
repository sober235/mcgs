"""spec_scoring (arc/run_spec.py) and arc/cycle.py give the same numbers on the same checkpoint (hp_n36).
Needs the score jsons of plan Task 5 steps 4-5; skipped when they are absent.

Run: env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_scoring_equivalence.py -q
"""
import glob
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["psnr", "ssim", "nmse", "hfen", "sharp", "grain", "head_psnr", "head_ssim", "head_hfen",
        "grain_ratio", "head_psnr_cplx", "cos_phase", "e_unobs", "e_unobs_excl_z", "nmse_unobs", "comp"]


def _latest(pattern):
    files = sorted(glob.glob(str(ROOT / pattern)))
    return json.loads(Path(files[-1]).read_text()) if files else None


CYC = _latest("results_arc/pin_hp_n36_*_score.json")
SPC = _latest("results_spec/brainpin_equiv*_score.json")
pytestmark = pytest.mark.skipif(CYC is None or SPC is None,
                                reason="run scripts/regress_brain_pin.sh and the brainpin_equiv run first")


@pytest.mark.parametrize("channel", ["plain", "contrast"])
def test_spec_scoring_equals_cycle(channel):
    for k in KEYS:
        a, b = CYC[channel][k], SPC[channel][k]
        assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12), f"{channel}.{k}: cycle {a!r} vs run_spec {b!r}"


def test_run_spec_run_had_no_fallbacks():
    assert SPC["fallbacks"] == []
