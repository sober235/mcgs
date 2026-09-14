"""Registering the MC-INR arm in this topic's table (uih_mtp.register_mcinr)."""
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, "/data0/congcong/code/GS/mcgs_peds_wt")

CONTRASTS = ("T1w", "T2s")


def _run(tmp_path, gauge_ok=True, with_volumes=True):
    """A minimal MC-INR result folder: the reproduction's score json and its gauge-unit volumes."""
    root = tmp_path / "mcinr"
    (root / "logs").mkdir(parents=True)
    (root / "recon" / "af12").mkdir(parents=True)
    base = {"method": "mcinr", "psnr": 30.0, "ssim": 0.80, "nmse": 0.010, "hfen": 0.50,
            "head_psnr": 28.5, "head_ssim": 0.90, "head_hfen": 0.20, "sharp": 0.70,
            "grain": 0.01, "seconds": 900.0}
    doc = {"af": 12, "contrasts": {c: {"row": dict(base, contrast=c), "gauge_scale": 2.0,
                                       "gauge_check": {"head_psnr": {"recorded": 34.46, "rescored": 34.46,
                                                                     "ok": gauge_ok}}}
                                   for c in CONTRASTS}}
    json.dump(doc, open(root / "logs" / "score_af12.json", "w"))
    if with_volumes:
        for c in CONTRASTS:
            np.save(root / "recon" / "af12" / f"mcinr_{c}_complex.npy",
                    np.zeros((2, 2, 2), dtype=np.complex64))
    topic = tmp_path / "topic"
    (topic / "logs").mkdir(parents=True)
    (topic / "recon").mkdir(parents=True)
    return root, topic


def test_registers_the_row_in_the_format_the_table_builder_reads(tmp_path):
    from uih_mtp.register_mcinr import register
    root, topic = _run(tmp_path)
    register(root, topic, af=12, contrasts=CONTRASTS)
    doc = json.load(open(topic / "logs" / "mcinr_T1w.json"))
    assert [r["method"] for r in doc["rows"]] == ["mcinr"]
    assert doc["contrast"] == "T1w"
    assert doc["rows"][0]["head_psnr"] == 28.5
    assert (topic / "recon" / "mcinr_T2s_complex.npy").resolve() == \
           (root / "recon" / "af12" / "mcinr_T2s_complex.npy").resolve()


def test_refuses_a_run_whose_gauge_recheck_failed(tmp_path):
    from uih_mtp.register_mcinr import register
    root, topic = _run(tmp_path, gauge_ok=False)
    with pytest.raises(ValueError):
        register(root, topic, af=12, contrasts=CONTRASTS)
    assert not (topic / "logs" / "mcinr_T1w.json").exists()


def test_refuses_a_run_that_saved_no_volume(tmp_path):
    from uih_mtp.register_mcinr import register
    root, topic = _run(tmp_path, with_volumes=False)
    with pytest.raises(FileNotFoundError):
        register(root, topic, af=12, contrasts=CONTRASTS)
