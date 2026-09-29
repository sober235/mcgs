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


def _valid_template():
    """A template whose expansions pass spec_io.validate_spec (the CLI checks every expanded spec)."""
    ch = {"data_path": "/d/{subject}/ksp_E1.h5", "csm_path": "/d/{subject}/csm.h5", "mask_path": "/m/af16.h5"}
    return {"spec": {"dataset": "knee", "exp_prefix": "knee", "results_root": "/r", "save_volumes": True,
                     "from_ckpt": None},
            "scoring": {"anatomy_rel": 0.05, "grain_ratio": False},
            "trainer": {"data": {"plain": dict(ch, name="E1"),
                                 "contrast": dict(ch, name="E2", data_path="/d/{subject}/ksp_E2.h5")},
                        "training": {"seed": 42, "deterministic": False, "max_iterations": 10,
                                     "shared_warmup_iters": 2},
                        "loss": {"tv_weight": 0.1}}}


def _template_file(tmp_path, template):
    tpl = tmp_path / "t.yaml"
    tpl.write_text(yaml.safe_dump(template))
    return tpl


def test_cli_writes_one_file_per_subject(tmp_path):
    t = _valid_template()
    spec_expand.main([str(_template_file(tmp_path, t)), str(tmp_path / "out"), "joint", "MTR_013", "MTR_045"])
    for s in ("MTR_013", "MTR_045"):
        e = yaml.safe_load((tmp_path / "out" / f"joint_{s}.yaml").read_text())
        assert spec_expand.differs_only_by_subject(t, e, s)


@pytest.mark.parametrize("mutate", [
    lambda e: e["trainer"]["loss"]["list"].__setitem__(0, 1.0),  # int 1 -> float 1.0
    lambda e: e["trainer"]["loss"]["list"].__setitem__(0, True),  # int 1 -> bool True
    lambda e: e["trainer"]["loss"].update({"extra": 0}),  # added key
    lambda e: e["trainer"]["loss"]["list"].append(2),  # longer list
    lambda e: e["trainer"]["data"]["plain"].update({"name": "E2"}),  # changed string without placeholder
])
def test_type_and_shape_changes_are_detected(mutate):
    """Verify that differs_only_by_subject rejects type changes, added keys, and shape changes."""
    e = spec_expand.expand(T, "MTR_013")
    mutate(e)
    assert not spec_expand.differs_only_by_subject(T, e, "MTR_013")


# --- final review fixes (2026-09-28) ---------------------------------------------------------------------------
@pytest.mark.parametrize("template", [
    {"a": {"{subject}_x": 1}, "b": "/d/{subject}"},
    {"a": [{"k_{subject}": 1}], "b": "/d/{subject}"},
])
def test_placeholder_in_a_key_is_refused(template):
    with pytest.raises(ValueError, match="mapping key"):
        spec_expand.expand(template, "MTR_013")


def test_cli_refuses_an_invalid_template_and_writes_nothing(tmp_path):
    t = _valid_template()
    t["trainer"]["data"]["plain"]["csm_path"] = "d/{subject}/csm.h5"          # relative once expanded
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="MTR_013.*csm_path must be an absolute path"):
        spec_expand.main([str(_template_file(tmp_path, t)), str(out), "joint", "MTR_013", "MTR_045"])
    assert not out.exists()


def test_cli_refuses_to_overwrite_and_writes_nothing(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "joint_MTR_045.yaml").write_text("keep")
    with pytest.raises(SystemExit, match="joint_MTR_045.yaml"):
        spec_expand.main([str(_template_file(tmp_path, _valid_template())), str(out), "joint", "MTR_013", "MTR_045"])
    assert sorted(p.name for p in out.iterdir()) == ["joint_MTR_045.yaml"]
    assert (out / "joint_MTR_045.yaml").read_text() == "keep"
