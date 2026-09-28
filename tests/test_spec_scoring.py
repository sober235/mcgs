"""spec_scoring: the scoring copy used by run_spec.py. The copied helpers must stay textually identical to
arc/cycle.py (checked without importing cycle.py, whose argparse runs at import), the anatomy metrics must
equal diag_artefact.headonly.score_headonly at the brain settings, and the >16M-element path must work.
The full score_channel == cycle.score_contrast check is tests/test_scoring_equivalence.py (GPU runs).

Run: env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_spec_scoring.py -q
"""
import ast
import inspect
import math
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest
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
    assert set(ours) == set(ref) | {"head_nmse"}          # head_nmse: spec §5.4, not in headonly
    for k in ref:
        a, b = ours[k], ref[k]
        assert (math.isnan(a) and math.isnan(b)) or a == b, k


def test_anatomy_threshold_and_grain_switch():
    rec, gt = _volume(1)
    loose = spec_scoring.score_anatomy(rec, gt, 0.05, False)
    tight = spec_scoring.score_anatomy(rec, gt, 0.40, False)
    assert tight["head_frac"] < loose["head_frac"]
    assert loose["grain_ratio"] is None


# --- final review fixes (2026-09-28) ---------------------------------------------------------------------------
def test_head_nmse_by_hand():
    gt = np.zeros((16, 16, 16), np.float32)
    gt[4:12, 4:12, 4:12] = 1.0                 # 512 anatomy voxels; P99.9 = 1, so the mask is exactly this cube
    rec = gt.copy()
    assert spec_scoring.score_anatomy(rec, gt, 0.05, False)["head_nmse"] == 0.0
    rec[5, 5, 5] = 1.5                         # inside the mask: error 0.5
    rec[0, 0, 0] = 3.0                         # outside the mask: not counted
    assert spec_scoring.score_anatomy(rec, gt, 0.05, False)["head_nmse"] == pytest.approx(0.25 / 512, rel=1e-12)


def test_score_channel_reports_the_ssim_mode(monkeypatch):
    """evaluate_reconstruction falls back to slice-wise SSIM on any exception and marks it only by ssim_std."""
    rec, gt = _volume(2, (16, 16, 16))
    gt_t = torch.from_numpy(gt).to(torch.complex64)
    csm = torch.ones((1, 16, 16, 16), dtype=torch.complex64)
    y_full = spec_scoring.fft3c(csm * gt_t.unsqueeze(0))
    mask = torch.zeros(16, 16, 16)
    mask[:, ::2, :] = 1.0
    ds = types.SimpleNamespace(get_data=lambda: {"ground_truth": gt_t}, kspace_multicoil=y_full)
    tr = types.SimpleNamespace(csm_p=csm, kspace_p=mask * y_full, mask_p=mask, ds_p=ds, mask_fit_p=mask,
                               kspace_dc_p=mask * y_full, physics_info={},
                               _render=lambda ch: torch.from_numpy(rec).to(torch.complex64))
    calls = []

    def fake_evaluate(x, target, compute_extended=False):
        calls.append(x)
        met = {"psnr": 30.0, "ssim": 0.9, "nmse": 0.01}
        if len(calls) == 1:                    # the reconstruction's SSIM fell back to slice-wise
            met["ssim_std"] = 0.05
        return met

    monkeypatch.setattr(spec_scoring, "evaluate_reconstruction", fake_evaluate)
    s = spec_scoring.score_channel(tr, "plain", torch.device("cpu"), 0.05, False)
    assert len(calls) == 2 and s["ssim_mode"] == "slicewise" and s["zf_ssim_mode"] == "3d"
    assert "head_nmse" in s


@pytest.mark.parametrize("start", ["arc_only", "root_ahead_of_paired_gs"])
def test_paired_gs_is_searched_before_the_repo_root(start):
    """A future root-level metrics.py or data/ must not shadow paired_gs, whatever sys.path the importer had."""
    root, pg, arc = str(ROOT), str(ROOT / "paired_gs"), str(ROOT / "arc")
    code = textwrap.dedent(f"""
        import inspect, sys
        sys.path[:] = [p for p in sys.path if p not in ({root!r}, {pg!r}, {arc!r}, "")]
        sys.path.insert(0, {arc!r})
        if {start!r} == "root_ahead_of_paired_gs":
            sys.path[1:1] = [{root!r}, {pg!r}]
        import spec_scoring
        print(sys.path.index({pg!r}) < sys.path.index({root!r}))
        print(inspect.getsourcefile(spec_scoring.evaluate_reconstruction))
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.splitlines()
    assert out[0] == "True", "the repo root is searched before paired_gs"
    assert Path(out[1]).resolve() == ROOT / "paired_gs" / "metrics" / "metrics.py"
