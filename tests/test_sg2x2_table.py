"""scripts/sg2x2_table.py: the four cells per contrast, the two shared-geometry differences and the interaction,
from fake score jsons. Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_sg2x2_table.py -q
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sg2x2_table import collect, table  # noqa: E402


def _score(h):
    return {"head_psnr": h, "head_ssim": 0.9, "head_hfen": 0.1, "psnr": h + 3}


def _write(arc, exp, joint, t1=None, fl=None):
    if joint:
        row = {"plain": _score(t1), "contrast": _score(fl), "plain_name": "T1", "contrast_name": "FLAIR"}
    else:
        row = {"plain": _score(t1 if t1 is not None else fl), "contrast": None,
               "plain_name": "T1" if t1 is not None else "FLAIR", "contrast_name": None}
    (arc / f"{exp}_score.json").write_text(json.dumps(row))


def test_collect_and_differences(tmp_path):
    arc = tmp_path
    for s, (a_t1, a_fl, b_t1, b_fl, c_t1, c_fl, d_t1, d_fl) in {42: (35.0, 34.0, 34.9, 33.95, 33.0, 32.0, 32.8, 31.9),
                                                                   43: (35.1, 34.1, 35.0, 34.05, 33.1, 32.1, 32.9, 32.0)}.items():
        _write(arc, f"p_s{s}_A", True, a_t1, a_fl); _write(arc, f"p_s{s}_C", True, c_t1, c_fl)
        _write(arc, f"p_s{s}_B_T1", False, t1=b_t1); _write(arc, f"p_s{s}_B_FLAIR", False, fl=b_fl)
        _write(arc, f"p_s{s}_D_T1", False, t1=d_t1); _write(arc, f"p_s{s}_D_FLAIR", False, fl=d_fl)
    got = collect(arc, "p", [42, 43])
    assert got[("A", "T1", 42)]["head_psnr"] == 35.0 and got[("B", "FLAIR", 43)]["head_psnr"] == 34.05
    assert np.isnan(got[("A", "T1", 42)]["head_nmse"])          # no volumes written in this test
    md = table(got, [42, 43])
    assert "| A |" in md and "| A−B |" in md and "| (A−B)−(C−D) |" in md
    assert "+0.100 ± 0.000" in md                                 # A-B on T1: 0.1 in both seeds
    assert "+0.050 ± 0.000" in md                                 # A-B on FLAIR
    assert "−0.100 ± 0.000" in md                                 # interaction on T1: (0.1) - (0.2)


def test_missing_runs_render_as_dash(tmp_path):
    md = table(collect(tmp_path, "none", [42]), [42])
    assert "—" in md
