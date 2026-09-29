"""spec_io: spec validation, run naming, provenance and default filling (no torch, no GPU).

Run: env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_spec_io.py -q
"""
import copy
import hashlib
import json
import subprocess
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


# --- final review fixes (2026-09-28) ---------------------------------------------------------------------------
def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                    *args], check=True, capture_output=True)


def test_provenance_counts_untracked_files_and_lists_the_paths(tmp_path):
    s = _spec(tmp_path)                                   # data files sit in tmp_path, outside the repo below
    f = tmp_path / "s.yaml"
    f.write_text(yaml.safe_dump(s))
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.py").write_text("x = 1\n")
    _git(repo, "add", "a.py")
    _git(repo, "commit", "-q", "-m", "one")
    clean = spec_io.provenance(f, s, repo)
    assert clean["git_dirty"] is False and clean["git_dirty_paths"] == []
    (repo / "new.py").write_text("y = 2\n")               # untracked
    (repo / "sub").mkdir()
    (repo / "sub" / "b.txt").write_text("b")              # untracked, inside an untracked directory
    dirty = spec_io.provenance(f, s, repo)
    assert dirty["git_dirty"] is True and dirty["git_dirty_paths"] == ["new.py", "sub/b.txt"]
    (repo / "a.py").write_text("x = 2\n")                 # tracked and modified
    assert spec_io.provenance(f, s, repo)["git_dirty_paths"] == ["a.py", "new.py", "sub/b.txt"]


def test_results_root_must_be_inside_the_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    spec_io.check_results_root(repo / "results_spec", repo)
    spec_io.check_results_root(str(repo / "a" / "b"), str(repo))
    for bad in (tmp_path / "elsewhere", repo, tmp_path / "repo_other" / "res", repo / ".." / "x"):
        with pytest.raises(SpecError, match="not inside the repository root"):
            spec_io.check_results_root(bad, repo)


@pytest.mark.parametrize("taken", ["toy_a_score.json", "toy_a_audit.json", "toy_a_volumes.npz"])
def test_run_dir_refuses_a_name_whose_outputs_exist(tmp_path, taken):
    s = _spec(tmp_path)
    res = Path(s["spec"]["results_root"])
    res.mkdir()
    (res / taken).write_text("{}")
    with pytest.raises(SpecError, match=f"{taken} exists"):
        spec_io.run_dir("toy_a", s)
    spec_io.run_dir("toy_b", s)


FILE_KEYS = [("trainer", "data", "plain", "known_zero_support"), ("trainer", "data", "contrast", "known_zero_support"),
             ("trainer", "gaussian", "warm_start_checkpoint"), ("trainer", "gaussian", "private_warm_start_checkpoint"),
             ("trainer", "loss", "guide_path_plain"), ("trainer", "loss", "guide_path_contrast"), ("spec", "from_ckpt")]


def _set(spec, path, value):
    node = spec
    for k in path[:-1]:
        node = node.setdefault(k, {})
    node[path[-1]] = value


@pytest.mark.parametrize("path", FILE_KEYS, ids=[".".join(p) for p in FILE_KEYS])
def test_file_keys_must_be_absolute_when_set(tmp_path, path):
    s = _spec(tmp_path)
    for ok in (None, "", str(tmp_path / "not_yet_there.pth")):   # unset, empty, absolute (existence is provenance's job)
        _set(s, path, ok)
        spec_io.validate_spec(s)
    _set(s, path, "rel/warm.pth")
    with pytest.raises(SpecError, match=".".join(path) + " must be an absolute path"):
        spec_io.validate_spec(s)


def test_provenance_hashes_every_file_input_and_refuses_a_missing_one(tmp_path):
    s = _spec(tmp_path)
    ck, kz = tmp_path / "warm.pth", tmp_path / "support.npy"
    ck.write_bytes(b"ckpt")
    kz.write_bytes(b"kz")
    s["trainer"]["gaussian"] = {"warm_start_checkpoint": str(ck), "private_warm_start_checkpoint": None}
    s["trainer"]["data"]["contrast"]["known_zero_support"] = str(kz)
    s["trainer"]["loss"]["guide_path_plain"] = ""
    f = tmp_path / "s.yaml"
    f.write_text(yaml.safe_dump(s))
    md5 = spec_io.provenance(f, s, ROOT)["data_md5"]
    assert md5[str(ck)] == hashlib.md5(b"ckpt").hexdigest() and md5[str(kz)] == hashlib.md5(b"kz").hexdigest()
    assert len(md5) == 5                                   # ksp / csm / mask (shared by both channels) + the two above
    s["spec"]["from_ckpt"] = str(tmp_path / "missing.pth")
    with pytest.raises(SpecError, match="spec.from_ckpt = .*missing.pth"):
        spec_io.provenance(f, s, ROOT)


def _fill_files(tmp_path, entries):
    sp, au = tmp_path / "spec.yaml", tmp_path / "audit.json"
    sp.write_text(yaml.safe_dump(_spec(tmp_path)))
    au.write_text(json.dumps({"fallbacks": entries}))
    return sp, au


@pytest.mark.parametrize("in_place", [False, True])
def test_fill_defaults_cli_writes_a_valid_spec(tmp_path, in_place):
    sp, au = _fill_files(tmp_path, [{"key": "loss.lambda_ptv", "default": 0.0}, {"key": "mode", "default": {}},
                                    {"key": "mode.strict_cuda", "default": False}])
    out = sp if in_place else tmp_path / "out.yaml"
    spec_io._main(["fill-defaults", str(sp), str(au), "--out", str(out)])
    got = yaml.safe_load(out.read_text())
    spec_io.validate_spec(got)
    assert got["trainer"]["loss"] == {"tv_weight": 0.1, "lambda_ptv": 0.0}
    assert got["trainer"]["mode"] == {"strict_cuda": False}


@pytest.mark.parametrize("entry, message", [
    ({"key": "voxelizer.type", "default": "chunk", "conflicting_defaults": ["chunk", "tile_cuda"]},
     r"voxelizer\.type: \['chunk', 'tile_cuda'\]"),
    ({"key": "loss.fn", "default": {"__repr__": "<function f>"}}, "loss.fn"),
])
def test_fill_defaults_cli_refuses_and_writes_nothing(tmp_path, entry, message):
    sp, au = _fill_files(tmp_path, [{"key": "loss.lambda_ptv", "default": 0.0}, entry])
    before = sp.read_bytes()
    out = tmp_path / "out.yaml"
    with pytest.raises(SystemExit, match=message):
        spec_io._main(["fill-defaults", str(sp), str(au), "--out", str(out)])
    assert not out.exists() and sp.read_bytes() == before
