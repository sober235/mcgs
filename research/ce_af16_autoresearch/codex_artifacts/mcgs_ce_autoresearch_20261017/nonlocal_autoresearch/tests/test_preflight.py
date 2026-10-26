"""CPU acceptance checks for the artifact-only preflight validator."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_preflight as preflight


def test_complex_drift_keeps_imaginary_component():
    row = preflight.difference_record(np.array([1+0j]), np.array([1+0.25j]))
    assert row == {"max_absolute": .25, "relative_l2": .25}


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_drift_is_rejected(value):
    with pytest.raises(ValueError, match="nonfinite"):
        preflight.difference_record(np.array([1.]), np.array([value]))


def test_optimized_python_preserves_validation():
    result = subprocess.run([sys.executable, "-O", "-c",
        "import sys;sys.path.insert(0,sys.argv[1]);import run_preflight;p=run_preflight.require(False,'rejected')", str(ROOT)],
        capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert result.returncode != 0 and "ValueError: rejected" in result.stderr


def test_default_cli_is_read_only_and_requires_existing_artifacts():
    result = subprocess.run([sys.executable, "-O", str(ROOT / "run_preflight.py")],
                            capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert result.returncode == 2
    assert "Only --validate-existing is supported" in result.stderr
