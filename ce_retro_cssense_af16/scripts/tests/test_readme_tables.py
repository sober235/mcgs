"""readme_tables.py has no __main__ guard (it is meant to be run as `python readme_tables.py > table.md`), so it is
exercised here as a subprocess -- importing it in-process would execute it immediately against whatever RETRO_TOPIC
happens to be set, and would cache its `from arms import LABEL` in sys.modules (arms.py freezes its own module-level
R at first import), which would leak into any other test in this session that also imports arms (test_summary.py
does). A fresh subprocess per call has no shared sys.modules, so it cannot contaminate sibling tests."""
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]


def _row(c, m, hp, **kw):
    r = dict(contrast=c, method=m, head_psnr=hp, head_ssim=0.9, head_nmse=0.01, head_hfen=0.3, psnr_cplx=28.0, psnr=34.0)
    r.update(kw)
    return r


def _run(tmp_path):
    env = dict(os.environ, RETRO_TOPIC=str(tmp_path), PYTHONNOUSERSITE="1", PYTHONPATH="")
    return subprocess.run([sys.executable, str(SCRIPTS / "readme_tables.py")], cwd=SCRIPTS, env=env, capture_output=True, text=True, timeout=30)


def test_tables_sort_bold_the_best_and_report_the_headline_margin(tmp_path):
    (tmp_path / "logs").mkdir()
    rows = []
    for c, base in (("pre", 0.0), ("post", 1.0)):
        rows += [_row(c, "zf", 24.0 + base, lam=0.001), _row(c, "gs_single", 33.0 + base), _row(c, "gs_joint", 33.5 + base)]
    json.dump(rows, open(tmp_path / "logs/summary.json", "w"))
    p = _run(tmp_path)
    assert p.returncode == 0, p.stderr
    out = p.stdout
    assert "**pre**" in out and "**post**" in out
    assert out.count("**33.500**") == 1 and out.count("**34.500**") == 1  # MC-GS is the per-scan head-PSNR best
    assert "MC-GS minus GS-single" in out
    assert "| pre | +0.500 |" in out and "| post | +0.500 |" in out
    assert "Enhancement" not in out  # no enhancement_fidelity.json written -> that table is skipped, not blank


def test_enhancement_table_prints_err_against_truth_when_the_json_exists(tmp_path):
    (tmp_path / "logs").mkdir()
    rows = [_row(c, m, hp) for c in ("pre", "post") for m, hp in (("gs_single", 33.0), ("gs_joint", 33.5))]
    json.dump(rows, open(tmp_path / "logs/summary.json", "w"))
    ef = {"rows": [
        {"method": "truth", "label": "truth (CS-SENSE)", "vessels": 0.30, "plexus": 0.20, "cortex": 0.05,
         "white_matter": 0.0, "lesion_core": 0.50, "rim_0_3mm": 0.40, "background_post": 0.01},
        {"method": "gs_joint", "label": "MC-GS", "vessels": 0.28, "plexus": 0.22, "cortex": 0.06,
         "white_matter": 0.01, "lesion_core": 0.47, "rim_0_3mm": 0.41, "background_post": 0.02,
         "err_vessels": -0.02, "err_plexus": 0.02, "err_cortex": 0.01, "err_white_matter": 0.01,
         "err_lesion_core": -0.03, "err_rim_0_3mm": 0.01, "err_background_post": 0.01},
    ]}
    json.dump(ef, open(tmp_path / "logs/enhancement_fidelity.json", "w"))
    p = _run(tmp_path)
    assert p.returncode == 0, p.stderr
    assert "Enhancement (post" in p.stdout
    assert "+0.280 (-0.020)" in p.stdout
