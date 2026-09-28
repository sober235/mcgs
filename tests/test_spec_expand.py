"""spec_expand: one spec per subject from a template; only strings carrying {subject} may differ.

Run: env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_spec_expand.py -q
"""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "arc"))
import spec_expand  # noqa: E402

T = {"spec": {"dataset": "knee", "results_root": "/r"},
     "trainer": {"data": {"plain": {"data_path": "/d/{subject}/ksp_E1.h5", "name": "E1"}},
                 "loss": {"tv_weight": 0.1, "list": [1, "{subject}"]}}}


def test_expand_replaces_every_placeholder_and_nothing_else():
    e = spec_expand.expand(T, "MTR_013")
    assert e["trainer"]["data"]["plain"]["data_path"] == "/d/MTR_013/ksp_E1.h5"
    assert e["trainer"]["loss"] == {"tv_weight": 0.1, "list": [1, "MTR_013"]}
    assert spec_expand.differs_only_by_subject(T, e, "MTR_013")
    assert "{subject}" in T["trainer"]["data"]["plain"]["data_path"]


def test_tampered_expansion_is_detected():
    e = spec_expand.expand(T, "MTR_013")
    e["trainer"]["loss"]["tv_weight"] = 0.2
    assert not spec_expand.differs_only_by_subject(T, e, "MTR_013")


def test_template_without_placeholder_is_refused():
    with pytest.raises(ValueError, match="placeholder"):
        spec_expand.expand({"a": "/no/subject"}, "MTR_013")


def test_cli_writes_one_file_per_subject(tmp_path):
    tpl = tmp_path / "t.yaml"
    tpl.write_text(yaml.safe_dump(T))
    spec_expand.main([str(tpl), str(tmp_path / "out"), "joint", "MTR_013", "MTR_045"])
    for s in ("MTR_013", "MTR_045"):
        e = yaml.safe_load((tmp_path / "out" / f"joint_{s}.yaml").read_text())
        assert spec_expand.differs_only_by_subject(T, e, s)
