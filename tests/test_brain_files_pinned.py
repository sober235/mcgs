"""The brain line's files must stay byte-identical to 2eef592 on every branch cut from nonbrain-infra
(spec 2026-09-28 §2.2): configs/, run_*_mcgs.sh, arc/cycle.py, the joint trainer and the scoring modules.

Run: env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_brain_files_pinned.py -q
"""
import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = [line.split("  ", 1) for line in (ROOT / "tests/data/brain_pinned_md5.txt").read_text().splitlines()
           if line.strip()]


def test_manifest_covers_the_pinned_set():
    names = {p for _, p in ENTRIES}
    for f in ("arc/cycle.py", "paired_gs/trainers/joint_trainer.py", "diag_artefact/headonly.py",
              "diag_artefact/volumes.py", "paired_gs/metrics/metrics.py", "run_wd_mcgs.sh"):
        assert f in names
    assert sum(1 for n in names if n.startswith("configs/")) == 57


@pytest.mark.parametrize("md5,path", ENTRIES, ids=[p for _, p in ENTRIES])
def test_brain_file_unchanged(md5, path):
    assert hashlib.md5((ROOT / path).read_bytes()).hexdigest() == md5, f"{path} differs from 2eef592"
