"""spec_io: spec validation, run naming, provenance and default filling (no torch, no GPU).

Run: env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_spec_io.py -q
"""
import copy
import hashlib
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "arc"))
import spec_io  # noqa: E402
from spec_io import SpecError  # noqa: E402


def _spec(tmp_path):
    paths = {}
    for n in ("ksp.h5", "csm.h5", "mask.h5"):
        p = tmp_path / n
        p.write_bytes(n.encode())
        paths[n] = str(p)
    ch = {"name": "E1", "data_path": paths["ksp.h5"], "csm_path": paths["csm.h5"], "mask_path": paths["mask.h5"]}
    return {"spec": {"dataset": "toy", "exp_prefix": "toy", "results_root": str(tmp_path / "res"),
                     "save_volumes": False, "from_ckpt": None},
            "scoring": {"anatomy_rel": 0.05, "grain_ratio": False},
            "trainer": {"data": {"plain": dict(ch), "contrast": dict(ch, name="E2")},
                        "training": {"seed": 42, "deterministic": False, "max_iterations": 10,
                                     "shared_warmup_iters": 2},
                        "loss": {"tv_weight": 0.1}, "output": {"use_tensorboard": False}}}


def test_valid_spec_passes_and_loads_from_file(tmp_path):
    s = _spec(tmp_path)
    spec_io.validate_spec(s)
    f = tmp_path / "s.yaml"
    f.write_text(yaml.safe_dump(s))
    assert spec_io.load_spec(f) == s


@pytest.mark.parametrize("mutate, message", [
    (lambda s: s.pop("scoring"), "top-level"),
    (lambda s: s["spec"].update(extra=1), "spec block"),
    (lambda s: s["scoring"].update(extra=1), "scoring block"),
    (lambda s: s["spec"].update(results_root="rel/path"), "absolute"),
    (lambda s: s["spec"].update(exp_prefix="Knee"), "exp_prefix"),
    (lambda s: s["trainer"]["output"].update(output_dir="/x"), "output_dir"),
    (lambda s: s["trainer"]["loss"].update(lambda_geo=0.1), "lambda_geo"),
    (lambda s: s["trainer"]["training"].pop("seed"), "training.seed"),
    (lambda s: s["trainer"]["data"]["plain"].update(csm_path="rel.h5"), "absolute"),
])
def test_invalid_specs_are_refused(tmp_path, mutate, message):
    s = _spec(tmp_path)
    mutate(s)
    with pytest.raises(SpecError, match=message):
        spec_io.validate_spec(s)


def test_exp_name_must_carry_the_prefix(tmp_path):
    s = _spec(tmp_path)
    spec_io.check_exp("toy_joint_01", s)
    for bad in ("joint_01", "toyjoint", "toy_a/b"):
        with pytest.raises(SpecError):
            spec_io.check_exp(bad, s)


def test_run_dir_refuses_an_existing_run(tmp_path):
    s = _spec(tmp_path)
    d = spec_io.run_dir("toy_a", s)
    assert d == Path(s["spec"]["results_root"]) / "toy_a"
    d.mkdir(parents=True)
    with pytest.raises(SpecError, match="exists"):
        spec_io.run_dir("toy_a", s)


def test_trainer_config_fills_output_and_leaves_the_spec_alone(tmp_path):
    s = _spec(tmp_path)
    before = copy.deepcopy(s)
    cfg = spec_io.trainer_config(s, "toy_a")
    assert cfg["output"]["output_dir"] == str(Path(s["spec"]["results_root"]) / "toy_a")
    assert cfg["output"]["exp_name"] == "toy_a"
    assert s == before


def test_provenance_hashes(tmp_path):
    s = _spec(tmp_path)
    f = tmp_path / "s.yaml"
    f.write_text(yaml.safe_dump(s))
    p = spec_io.provenance(f, s, ROOT)
    assert p["spec_sha256"] == hashlib.sha256(f.read_bytes()).hexdigest()
    ksp = s["trainer"]["data"]["plain"]["data_path"]
    assert p["data_md5"][ksp] == hashlib.md5(Path(ksp).read_bytes()).hexdigest()
    assert len(p["git_commit"]) == 40 and isinstance(p["git_dirty"], bool)


def test_fill_defaults_inserts_missing_keys_only(tmp_path):
    s = _spec(tmp_path)
    out = spec_io.fill_defaults(s, [("loss.tv_weight", 9.9), ("loss.lambda_ptv", 0.0),
                                    ("mode", {}), ("mode.strict_cuda", False),
                                    ("gaussian.init.scale", 0.008)])
    assert out["trainer"]["loss"] == {"tv_weight": 0.1, "lambda_ptv": 0.0}
    assert out["trainer"]["mode"] == {"strict_cuda": False}
    assert out["trainer"]["gaussian"] == {"init": {"scale": 0.008}}
    assert "mode" not in s["trainer"]


@pytest.mark.parametrize("mutate, message", [
    (lambda s: s.update(spec=None), "spec block must be a mapping"),
    (lambda s: s.update(scoring="notadict"), "scoring block must be a mapping"),
    (lambda s: s.update(trainer=[]), "trainer block must be a mapping"),
    (lambda s: s["trainer"].update(loss=["tv_weight", "lambda_geo"]), "trainer.loss must be a mapping"),
    (lambda s: s["trainer"]["data"].update(plain=None), "trainer.data.plain must be a mapping"),
])
def test_non_mapping_blocks_are_refused(tmp_path, mutate, message):
    s = _spec(tmp_path)
    mutate(s)
    with pytest.raises(SpecError, match=message):
        spec_io.validate_spec(s)
