"""scripts/sg2x2_core_table.py: per-subject seed means, then paired A-B / C-D / interaction across the TEST subjects
with bootstrap intervals; the development subject never enters the statistics; a missing cell removes that subject
from the rows that need it only.
Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_sg2x2_core_table.py -q
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sg2x2_core_table import differences, subject_means, summary, table  # noqa: E402

DEV = "jiangyizhou_20230705"
TESTS = ["huangyuwei_20230705", "jiangdian_20230705", "sunxinyi_20230705"]


def _score(psnr):
    return {"head_psnr": psnr, "head_ssim": 0.9, "head_hfen": 0.1}


def _write(arc, exp, joint, t1=None, fl=None):
    row = ({"plain": _score(t1), "contrast": _score(fl)} if joint else
           {"plain": _score(t1 if t1 is not None else fl), "contrast": None})
    (arc / f"{exp}_score.json").write_text(json.dumps(row))


def _subject(arc, subj, seed, a, b, c, d, skip=()):
    """a, b, c, d = (T1, FLAIR) head PSNR of the four cells"""
    pre = f"p_{subj}_s{seed}"
    _write(arc, f"{pre}_A", True, *a)
    _write(arc, f"{pre}_C", True, *c)
    for cell, (t1, fl) in (("B", b), ("D", d)):
        if f"{cell}_T1" not in skip:
            _write(arc, f"{pre}_{cell}_T1", False, t1=t1)
        if f"{cell}_FLAIR" not in skip:
            _write(arc, f"{pre}_{cell}_FLAIR", False, fl=fl)


def _fill(arc):
    # A-B on T1 = 0.1, 0.2, 0.3 for the three test subjects; C-D on T1 = 1.0 for all
    for i, s in enumerate(TESTS):
        ab = 0.1 * (i + 1)
        _subject(arc, s, 42, a=(35.0 + ab, 34.0), b=(35.0, 34.05), c=(24.0, 25.0), d=(23.0, 24.0))
    _subject(arc, DEV, 42, a=(99.0, 99.0), b=(0.0, 0.0), c=(0.0, 0.0), d=(0.0, 0.0))   # absurd: must not count


def test_subject_means_average_the_seeds(tmp_path):
    s = TESTS[0]
    _subject(tmp_path, s, 42, a=(35.0, 34.0), b=(35.0, 34.0), c=(24.0, 25.0), d=(23.0, 24.0))
    _subject(tmp_path, s, 43, a=(36.0, 34.0), b=(35.0, 34.0), c=(24.0, 25.0), d=(23.0, 24.0))
    m = subject_means(tmp_path, "p", s, [42, 43])
    assert m[("A", "T1")]["head_psnr"] == 35.5 and m[("A", "T1")]["n_seeds"] == 2
    assert math.isnan(m[("A", "T1")]["head_nmse"])                     # no volumes written


def test_differences_and_summary():
    m = {("A", "T1"): {"head_psnr": 35.2}, ("B", "T1"): {"head_psnr": 35.0},
         ("C", "T1"): {"head_psnr": 24.0}, ("D", "T1"): {"head_psnr": 23.0}}
    d = differences(m, "T1", "head_psnr")
    assert np.isclose(d["A-B"], 0.2) and np.isclose(d["C-D"], 1.0) and np.isclose(d["(A-B)-(C-D)"], -0.8)
    s = summary([0.1, 0.2, 0.3], higher_better=True)
    assert s["n"] == 3 and np.isclose(s["mean"], 0.2) and s["k_better"] == 3
    assert s["ci"][0] <= 0.2 <= s["ci"][1] and summary([0.1, 0.2, 0.3], True) == s     # deterministic
    assert summary([0.01, -0.02], higher_better=False)["k_better"] == 1
    one = summary([0.5], True)
    assert one["n"] == 1 and math.isnan(one["sd"]) and math.isnan(one["ci"][0])


def test_dev_subject_never_enters_the_statistics(tmp_path):
    _fill(tmp_path)
    md = table(tmp_path, "p", [42])
    row = next(l for l in md.splitlines() if l.startswith("| A−B | T1 | PSNR |"))
    assert "| 3 |" in row and "+0.200" in row and "3/3" in row
    assert "jiangyizhou_20230705" in md and "development subject" in md


def test_missing_cell_drops_the_subject_from_that_row_only(tmp_path):
    _fill(tmp_path)
    _subject(tmp_path, "sunxinyi_20230705", 43, a=(1, 1), b=(1, 1), c=(1, 1), d=(1, 1))   # a second seed, harmless
    s = "zhoushuo_20231030"
    _subject(tmp_path, s, 42, a=(35.5, 34.0), b=(35.0, 34.0), c=(24.0, 25.0), d=(23.0, 24.0), skip=("B_T1",))
    md = table(tmp_path, "p", [42])
    t1 = next(l for l in md.splitlines() if l.startswith("| A−B | T1 | PSNR |"))
    fl = next(l for l in md.splitlines() if l.startswith("| A−B | FLAIR | PSNR |"))
    assert "| 3 |" in t1 and "| 4 |" in fl
    cd_t1 = next(l for l in md.splitlines() if l.startswith("| C−D | T1 | PSNR |"))
    ia_t1 = next(l for l in md.splitlines() if l.startswith("| (A−B)−(C−D) | T1 | PSNR |"))
    ia_fl = next(l for l in md.splitlines() if l.startswith("| (A−B)−(C−D) | FLAIR | PSNR |"))
    assert "| 4 |" in cd_t1 and "| 3 |" in ia_t1 and "| 4 |" in ia_fl
    assert "+0.200" in t1


def test_unequal_seed_sets_are_paired_seed_by_seed(tmp_path):
    s = TESTS[0]
    _subject(tmp_path, s, 42, a=(35.0, 34.0), b=(35.0, 34.0), c=(24.0, 25.0), d=(23.0, 24.0))
    _write(tmp_path, f"p_{s}_s43_A", True, 36.0, 34.0)          # seed 43: A only
    md = table(tmp_path, "p", [42, 43])
    row = next(l for l in md.splitlines() if l.startswith("| A−B | T1 | PSNR |"))
    assert "| 1 |" in row and "+0.000" in row                   # seed 42 only; per-cell means would give +0.500
