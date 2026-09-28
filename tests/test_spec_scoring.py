"""spec_scoring: the scoring copy used by run_spec.py. The copied helpers must stay textually identical to
arc/cycle.py (checked without importing cycle.py, whose argparse runs at import), the anatomy metrics must
equal diag_artefact.headonly.score_headonly at the brain settings, and the >16M-element path must work.
The full score_channel == cycle.score_contrast check is tests/test_scoring_equivalence.py (GPU runs).

Run: env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_spec_scoring.py -q
"""
import ast
import inspect
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "arc"))
sys.path.insert(0, str(ROOT))
import spec_scoring  # noqa: E402
from diag_artefact.headonly import score_headonly  # noqa: E402


def _cycle_function_source(name):
    src = (ROOT / "arc/cycle.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    raise AssertionError(f"{name} not found in cycle.py")


def test_copied_helpers_match_cycle_source():
    for name in ("hfen", "sharp_ratio"):
        assert inspect.getsource(getattr(spec_scoring, name)).strip() == _cycle_function_source(name).strip()
    cyc = _cycle_function_source("grain")
    ours = inspect.getsource(spec_scoring.grain).strip()
    assert ours == cyc.replace("thr = torch.quantile(gg[brain], 0.30)", "thr = quantile(gg[brain], 0.30)").strip()


def test_quantile_torch_path_below_the_cap_and_numpy_above():
    small = torch.rand(1_000_000, generator=torch.Generator().manual_seed(0))
    assert torch.equal(spec_scoring.quantile(small, 0.3), torch.quantile(small, 0.3))
    big = torch.rand(17_000_000, generator=torch.Generator().manual_seed(1))
    assert float(spec_scoring.quantile(big, 0.3)) == float(np.float32(np.quantile(big.numpy(), 0.3)))


def _volume(seed=0, shape=(48, 56, 40)):
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.meshgrid(*[np.linspace(-1, 1, n) for n in shape], indexing="ij")
    obj = (zz ** 2 + yy ** 2 + xx ** 2 < 0.7).astype(np.float32)
    gt = ndimage.gaussian_filter(obj * (1 + 0.3 * rng.standard_normal(shape)), 1.0).astype(np.float32)
    rec = (gt + 0.02 * rng.standard_normal(shape)).astype(np.float32)
    return rec, gt


def test_anatomy_metrics_equal_headonly_at_brain_settings():
    rec, gt = _volume()
    ours = spec_scoring.score_anatomy(rec, gt, 0.05, True)
    ref = score_headonly(rec, gt)
    assert set(ours) == set(ref)
    for k in ref:
        a, b = ours[k], ref[k]
        assert (math.isnan(a) and math.isnan(b)) or a == b, k


def test_anatomy_threshold_and_grain_switch():
    rec, gt = _volume(1)
    loose = spec_scoring.score_anatomy(rec, gt, 0.05, False)
    tight = spec_scoring.score_anatomy(rec, gt, 0.40, False)
    assert tight["head_frac"] < loose["head_frac"]
    assert loose["grain_ratio"] is None
