"""run_spec.py CLI guard: --audit below 3 must be refused before any GPU/env setup (fix round 1).

CPU only: no GPU, no data files. Calls run_spec.main() with a --spec that does not exist; the audit
value must be rejected by argparse before the spec is even read, so CUDA_VISIBLE_DEVICES is never set.

Run: env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_run_spec_cli.py -q
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "arc"))
import run_spec  # noqa: E402


@pytest.mark.parametrize("audit", [2, 1, -5])
def test_audit_below_3_refused_before_gpu_env_is_touched(audit):
    before = os.environ.get("CUDA_VISIBLE_DEVICES")
    with pytest.raises(SystemExit):
        run_spec.main(["--spec", "/nonexistent.yaml", "--exp", "x", "--gpu", "0", "--audit", str(audit)])
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == before
