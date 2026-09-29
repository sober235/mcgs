"""spec_io: read, check and trace the spec files that drive arc/run_spec.py (spec 2026-09-28 §2.2, §3.2).

A spec has exactly three blocks: `spec` (run bookkeeping), `scoring` (the two scoring settings that are
constants in cycle.py) and `trainer` (the complete JointTrainer config). Nothing here imports torch.
File inputs: the three PATH_KEYS of each channel must be absolute paths, and so must every FILE_KEYS entry
that is set (not null, not empty); provenance() refuses any of them that does not exist and md5s them all.

CLI:  python arc/spec_io.py fill-defaults SPEC AUDIT_JSON --out OUT
      writes SPEC with every fallback recorded by `run_spec.py --audit` inserted under `trainer` (OUT may be
      SPEC itself). Refuses, writing nothing, when a key was read with conflicting defaults or a default is
      not YAML-serialisable: those are set by hand.
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
# other files a run reads when the key is set (joint_trainer.py:189, :223, :307, :923; run_spec.py for from_ckpt)
FILE_KEYS = (("trainer", "data", "plain", "known_zero_support"), ("trainer", "data", "contrast", "known_zero_support"),
             ("trainer", "gaussian", "warm_start_checkpoint"), ("trainer", "gaussian", "private_warm_start_checkpoint"),
             ("trainer", "loss", "guide_path_plain"), ("trainer", "loss", "guide_path_contrast"),
             ("spec", "from_ckpt"))
RUN_OUTPUTS = ("{exp}", "{exp}_score.json", "{exp}_audit.json", "{exp}_volumes.npz")


class SpecError(ValueError):
    """The spec file breaks a rule of spec 2026-09-28 §2.2."""


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise SpecError(msg)


def _lookup(spec: dict, path: Tuple[str, ...]):
    node = spec
    for k in path:
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node


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
    gaussian = tr.get("gaussian", {})
    _need(isinstance(gaussian, dict), f"trainer.gaussian must be a mapping, got {type(gaussian).__name__}")
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
    for path in FILE_KEYS:
        v = _lookup(spec, path)
        _need(v in (None, "") or Path(str(v)).is_absolute(), f"{'.'.join(path)} must be an absolute path (or null)")


def load_spec(path) -> dict:
    spec = yaml.safe_load(Path(path).read_text())
    validate_spec(spec)
    return spec


def check_exp(exp: str, spec: dict) -> None:
    prefix = spec["spec"]["exp_prefix"] + "_"
    _need(exp.startswith(prefix) and len(exp) > len(prefix), f"run name must start with '{prefix}', got {exp!r}")
    _need(re.fullmatch(r"[A-Za-z0-9_.-]+", exp) is not None, f"run name has characters outside [A-Za-z0-9_.-]: {exp!r}")


def check_results_root(results_root, repo_root) -> None:
    """run_spec.py writes only below the repository it runs from, so a run started in one worktree never
    writes into another."""
    r, repo = Path(results_root).resolve(), Path(repo_root).resolve()
    _need(repo in r.parents, f"spec.results_root {results_root} is not inside the repository root {repo} that "
          f"run_spec.py runs from; use a folder of this worktree, e.g. {repo / 'results_spec'}")


def run_dir(exp: str, spec: dict) -> Path:
    """The run directory of `exp`; refuses the name when the directory or any of its result files exists."""
    res = Path(spec["spec"]["results_root"])
    for pattern in RUN_OUTPUTS:
        p = res / pattern.format(exp=exp)
        _need(not p.exists(), f"{p} exists; pick a new run name (runs are never overwritten)")
    return res / exp


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


def file_inputs(spec: dict) -> dict:
    """Every file a run reads, as {dotted key: path}: the PATH_KEYS of both channels and each FILE_KEYS entry that
    is set."""
    files = {f"trainer.data.{ch}.{k}": str(spec["trainer"]["data"][ch][k]) for ch in CHANNELS for k in PATH_KEYS}
    for path in FILE_KEYS:
        v = _lookup(spec, path)
        if v not in (None, ""):
            files[".".join(path)] = str(v)
    return files


def git_dirty_paths(repo_root) -> list:
    """Sorted paths `git status` reports: modified, staged and untracked files (ignored files excluded)."""
    out = subprocess.run(["git", "-C", str(repo_root), "status", "--porcelain", "-z", "--untracked-files=all"],
                         capture_output=True, text=True, check=True).stdout
    fields, paths, i = out.split("\0"), set(), 0
    while i < len(fields):
        entry = fields[i]
        if entry:
            paths.add(entry[3:])
            if "R" in entry[:2] or "C" in entry[:2]:     # a rename / copy entry is followed by its source path
                i += 1
                paths.add(fields[i])
        i += 1
    return sorted(paths)


def provenance(spec_path, spec: dict, repo_root) -> dict:
    """Spec sha256, git commit, uncommitted paths and the md5 of every file input. A missing file input raises
    here, before training: a missing warm-start checkpoint would otherwise be a silent cold start."""
    spec_path, repo_root = Path(spec_path), Path(repo_root)
    files = file_inputs(spec)
    for key, p in files.items():
        _need(Path(p).is_file(), f"{key} = {p} is not an existing file")
    commit = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    dirty = git_dirty_paths(repo_root)
    return {"spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
            "git_commit": commit, "git_dirty": bool(dirty), "git_dirty_paths": dirty,
            "data_md5": {p: md5_file(p) for p in sorted(set(files.values()))}}


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
    conflicts = [e for e in entries if "conflicting_defaults" in e]
    if conflicts:
        lines = "\n".join(f"  {e['key']}: {e['conflicting_defaults']}" for e in conflicts)
        raise SystemExit("the trainer reads these keys with different defaults, so no single default describes the "
                         f"audited run; set them by hand in the spec (nothing written):\n{lines}")
    bad = [e["key"] for e in entries if isinstance(e["default"], dict) and "__repr__" in e["default"]]
    if bad:
        raise SystemExit(f"defaults that are not YAML-serialisable, set them by hand: {bad}")
    out = fill_defaults(spec, [(e["key"], e["default"]) for e in entries])
    validate_spec(out)
    Path(a.out).write_text(yaml.safe_dump(out, sort_keys=True))
    print(f"filled {len(entries)} fallback(s) -> {a.out}")


if __name__ == "__main__":
    _main()
