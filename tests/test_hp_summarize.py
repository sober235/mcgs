#!/usr/bin/env python3
"""Unit test for scripts/hp_summarize.py on synthetic score jsons and a tiny volumes npz (no GPU).
Run: PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_hp_summarize.py -q"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hp_summarize as hs  # noqa: E402


def _score(t1_psnr, fl_psnr, npts=3600000):
    def c(p):
        return {"head_psnr": p, "head_ssim": 0.93, "head_hfen": 0.11, "psnr": p + 3.5, "ssim": 0.92,
                "nmse": 0.004, "head_psnr_cplx": p - 5.0}
    return {"npts": npts, "elapsed_s": 25200.0, "vram_gb": 33.0, "plain": c(t1_psnr), "contrast": c(fl_psnr)}


def _volumes(path):
    rng = np.random.default_rng(0)
    gt = np.zeros((12, 12, 12), np.float32)
    gt[3:9, 3:9, 3:9] = 1.0                      # a bright cube = the "head"
    rec = gt + rng.normal(0, 0.01, gt.shape).astype(np.float32)
    np.savez(path, recT1=rec, gtT1=gt, recFL=rec, gtFL=gt)
    return rec, gt


def test_head_nmse_matches_direct_formula(tmp_path):
    rec, gt = _volumes(tmp_path / "v.npz")
    head = gt > 0.05 * np.quantile(gt, 0.999)
    direct = float(((rec - gt)[head] ** 2).sum() / (gt[head] ** 2).sum())
    assert abs(hs.head_nmse(rec, gt) - direct) < 1e-9
    assert 0.0 < direct < 1e-2


def test_rows_order_ratio_and_missing(tmp_path):
    arc = tmp_path
    (arc / "hp_n36_score.json").write_text(json.dumps(_score(35.70, 34.93)))
    (arc / "hp_n18_score.json").write_text(json.dumps(_score(35.40, 34.60, npts=1800000)))
    (arc / "hp_p05_score.json").write_text(json.dumps(_score(35.71, 34.94)))
    _volumes(arc / "hp_n36_volumes.npz")
    rows = hs.build_rows(arc)
    assert [r["exp"] for r in rows] == ["hp_n09", "hp_n18", "hp_n36", "hp_n72", "hp_p05", "hp_p10"]
    by = {r["exp"]: r for r in rows}
    assert by["hp_n09"]["status"] == "missing" and by["hp_n36"]["status"] == "done"
    assert by["hp_p05"]["ratio"] == 0.05 and by["hp_p10"]["ratio"] == 0.10 and by["hp_n36"]["ratio"] == 0.0
    assert by["hp_n36"]["T1_head_psnr"] == 35.70 and by["hp_n36"]["FLAIR_head_psnr"] == 34.93
    assert np.isfinite(by["hp_n36"]["T1_head_nmse"]) and np.isnan(by["hp_n18"]["T1_head_nmse"])
    assert abs(by["hp_n36"]["elapsed_h"] - 7.0) < 1e-9


def test_anchor_check_and_rendering(tmp_path):
    arc = tmp_path
    (arc / "hp_n36_score.json").write_text(json.dumps(_score(35.70, 34.93)))
    rows = hs.build_rows(arc)
    ok = hs.anchor_check(rows, "hp_n36", (35.706, 34.928), 0.02)
    assert ok["passed"] is True and abs(ok["delta"][0] - (-0.006)) < 1e-9
    bad = hs.anchor_check(rows, "hp_n36", (35.80, 34.928), 0.02)
    assert bad["passed"] is False
    md = hs.render_md(rows, ok)
    assert "hp_n36" in md and "anchor" in md.lower() and "PASS" in md
    hs.write_csv(rows, arc / "hp_summary.csv")
    head = (arc / "hp_summary.csv").read_text().splitlines()[0]
    assert head.startswith("exp,n_shared,n_private,ratio,status")
