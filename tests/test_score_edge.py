#!/usr/bin/env python3
"""Validate scripts/score_edge.py on the existing baseline volumes.
Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python tests/test_score_edge.py"""
import sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from score_edge import score_contrast

d = np.load(ROOT / "results_arc" / "reg_aligned_4k_volumes.npz")
r = score_contrast(np.abs(d["recT1"]), np.abs(d["gtT1"]), "T1")
for k in ["nmse_whole","ssim","hfen","nmse_edge","hfen_edge","nmse_vent","quintiles"]:
    assert k in r, f"missing key {k}"
assert all(np.isfinite(v) for v in [r["nmse_whole"], r["ssim"], r["hfen"], r["nmse_edge"], r["nmse_vent"]])
assert 0.0 < r["ssim"] < 1.0, f"ssim out of range: {r['ssim']}"
assert r["nmse_edge"] > r["nmse_whole"], f"edge NMSE {r['nmse_edge']} should exceed whole {r['nmse_whole']}"
assert r["quintiles"][4] > r["quintiles"][0], "top gradient quintile error should exceed the flattest"
print("PASS test_score_edge:", {k: round(r[k],4) for k in ["nmse_whole","ssim","nmse_edge","nmse_vent"]})
print("ALL SCORE-EDGE TESTS PASSED")
