"""spec_io: read, check and trace the spec files that drive arc/run_spec.py (spec 2026-09-28 §2.2, §3.2).

A spec has exactly three blocks: `spec` (run bookkeeping), `scoring` (the two scoring settings that are
constants in cycle.py) and `trainer` (the complete JointTrainer config). Nothing here imports torch.

CLI:  python arc/spec_io.py fill-defaults SPEC AUDIT_JSON --out OUT
      writes SPEC with every fallback recorded by `run_spec.py --audit` inserted under `trainer`.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable, Tuple

import yaml

SPEC_KEYS = {"dataset", "exp_prefix", "results_root", "save_volumes", "from_ckpt"}
SCORING_KEYS = {"anatomy_rel", "grain_ratio"}
CHANNELS = ("plain", "contrast")
PATH_KEYS = ("data_path", "csm_path", "mask_path")


class SpecError(ValueError):
    """The spec file breaks a rule of spec 2026-09-28 §2.2."""


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise SpecError(msg)


def validate_spec(spec: dict) -> None:
    _need(isinstance(spec, dict) and set(spec) == {"spec", "scoring", "trainer"},
          f"top-level blocks must be exactly spec/scoring/trainer, got {sorted(spec) if isinstance(spec, dict) else spec!r}")
    s, sc, tr = spec["spec"], spec["scoring"], spec["trainer"]
    _need(isinstance(s, dict), f"spec block must be a mapping, got {type(s).__name__}")
    _need(isinstance(sc, dict), f"scoring block must be a mapping, got {type(sc).__name__}")
    _need(isinstance(tr, dict), f"trainer block must be a mapping, got {type(tr).__name__}")
    _need(set(s) == SPEC_KEYS, f"spec block keys must be {sorted(SPEC_KEYS)}, got {sorted(s)}")
    _need(set(sc) == SCORING_KEYS, f"scoring block keys must be {sorted(SCORING_KEYS)}, got {sorted(sc)}")
    _need(re.fullmatch(r"[a-z0-9]+", str(s["exp_prefix"])) is not None, "exp_prefix must match [a-z0-9]+")
    _need(Path(str(s["results_root"])).is_absolute(), "spec.results_root must be an absolute path")
    _need(isinstance(s["save_volumes"], bool), "spec.save_volumes must be a bool")
    _need(s["from_ckpt"] is None or isinstance(s["from_ckpt"], str), "spec.from_ckpt must be null or a path")
    _need(isinstance(sc["anatomy_rel"], (int, float)) and 0.0 < float(sc["anatomy_rel"]) < 1.0,
          "scoring.anatomy_rel must be in (0, 1)")
    _need(isinstance(sc["grain_ratio"], bool), "scoring.grain_ratio must be a bool")
    data = tr.get("data", {})
    _need(isinstance(data, dict), f"trainer.data must be a mapping, got {type(data).__name__}")
    for ch in CHANNELS:
        _need(ch in data, f"trainer.data.{ch} is missing")
        _need(isinstance(data[ch], dict), f"trainer.data.{ch} must be a mapping, got {type(data[ch]).__name__}")
        _need("name" in data[ch], f"trainer.data.{ch}.name is missing")
        for k in PATH_KEYS:
            _need(k in data[ch] and Path(str(data[ch][k])).is_absolute(),
                  f"trainer.data.{ch}.{k} must be an absolute path")
    out = tr.get("output", {})
    _need(isinstance(out, dict), f"trainer.output must be a mapping, got {type(out).__name__}")
    for k in ("output_dir", "exp_name"):
        _need(k not in out, f"trainer.output.{k} is set by run_spec.py; remove it from the spec")
    loss = tr.get("loss", {})
    _need(isinstance(loss, dict), f"trainer.loss must be a mapping, got {type(loss).__name__}")
    _need(not ("lambda_geo" in loss and "tv_weight" in loss),
          "trainer.loss has both lambda_geo and tv_weight; JointTrainer would silently use lambda_geo "
          "(joint_trainer.py:781-791); keep tv_weight only")
    training = tr.get("training", {})
    _need(isinstance(training, dict), f"trainer.training must be a mapping, got {type(training).__name__}")
    for k in ("seed", "deterministic", "max_iterations", "shared_warmup_iters"):
        _need(k in training, f"trainer.training.{k} is missing")


def load_spec(path) -> dict:
    spec = yaml.safe_load(Path(path).read_text())
    validate_spec(spec)
    return spec


def check_exp(exp: str, spec: dict) -> None:
    prefix = spec["spec"]["exp_prefix"] + "_"
    _need(exp.startswith(prefix) and len(exp) > len(prefix), f"run name must start with '{prefix}', got {exp!r}")
    _need(re.fullmatch(r"[A-Za-z0-9_.-]+", exp) is not None, f"run name has characters outside [A-Za-z0-9_.-]: {exp!r}")


def run_dir(exp: str, spec: dict) -> Path:
    d = Path(spec["spec"]["results_root"]) / exp
    _need(not d.exists(), f"run directory {d} exists; pick a new run name (runs are never overwritten)")
    return d


def trainer_config(spec: dict, exp: str) -> dict:
    cfg = copy.deepcopy(spec["trainer"])
    out = cfg.setdefault("output", {})
    out["output_dir"] = str(Path(spec["spec"]["results_root"]) / exp)
    out["exp_name"] = exp
    return cfg


def md5_file(path, chunk: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def provenance(spec_path, spec: dict, repo_root) -> dict:
    spec_path, repo_root = Path(spec_path), Path(repo_root)
    commit = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True).stdout.strip() != ""
    paths = sorted({str(spec["trainer"]["data"][ch][k]) for ch in CHANNELS for k in PATH_KEYS})
    return {"spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
            "git_commit": commit, "git_dirty": dirty,
            "data_md5": {p: md5_file(p) for p in paths}}


def fill_defaults(spec: dict, fallbacks: Iterable[Tuple[str, Any]]) -> dict:
    out = copy.deepcopy(spec)
    for path, default in fallbacks:
        node = out["trainer"]
        parts = path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            _need(isinstance(node, dict), f"cannot insert {path}: '{part}' is not a mapping")
        if parts[-1] not in node:
            node[parts[-1]] = copy.deepcopy(default)
    return out


def _main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="spec utilities")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fill-defaults", help="insert the fallbacks recorded by run_spec.py --audit")
    f.add_argument("spec")
    f.add_argument("audit_json")
    f.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    spec = yaml.safe_load(Path(a.spec).read_text())
    entries = json.loads(Path(a.audit_json).read_text())["fallbacks"]
    bad = [e["key"] for e in entries if isinstance(e["default"], dict) and "__repr__" in e["default"]]
    if bad:
        raise SystemExit(f"defaults that are not YAML-serialisable, set them by hand: {bad}")
    out = fill_defaults(spec, [(e["key"], e["default"]) for e in entries])
    validate_spec(out)
    Path(a.out).write_text(yaml.safe_dump(out, sort_keys=True))
    print(f"filled {len(entries)} fallback(s) -> {a.out}")


if __name__ == "__main__":
    _main()
