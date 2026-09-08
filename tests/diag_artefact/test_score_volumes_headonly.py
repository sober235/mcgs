"""scripts/ has no __init__.py, so the module is loaded by path (same as test_check_nonvacuity.py)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "score_volumes_headonly.py"
spec = importlib.util.spec_from_file_location("score_volumes_headonly", SCRIPT)
svh = importlib.util.module_from_spec(spec)
sys.modules["score_volumes_headonly"] = svh
spec.loader.exec_module(svh)


def _npz(tmp_path, phantom):
    rng = np.random.default_rng(0)
    rec = (phantom + rng.normal(0, 0.01, phantom.shape)).astype(np.float32)
    zf = (phantom * 0.8).astype(np.float32)
    p = tmp_path / "toy_volumes.npz"
    np.savez_compressed(p, recT1=rec, gtT1=phantom, zfT1=zf, recFL=rec, gtFL=phantom, zfFL=zf)
    return p


def test_score_npz_has_both_contrasts_and_zf(tmp_path, phantom):
    out = svh.score_npz(_npz(tmp_path, phantom), run="toy")
    assert out["run"] == "toy"
    for key in ("plain", "contrast"):
        for m in ("head_psnr", "head_ssim", "head_hfen", "grain_ratio"):
            assert isinstance(out[key][m], float)
            assert isinstance(out[key]["zf"][m], float)
        assert out[key]["head_psnr"] > out[key]["zf"]["head_psnr"]


def test_cli_writes_json(tmp_path, phantom):
    p = _npz(tmp_path, phantom)
    out = tmp_path / "s.json"
    svh.main([str(p), "--run", "toy", "--out", str(out)])
    d = json.loads(out.read_text())
    assert d["run"] == "toy" and "plain" in d and "contrast" in d
