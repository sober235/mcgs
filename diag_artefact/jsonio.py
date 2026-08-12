"""The one writer for every JSON record under `results_arc/diag_artefact/`.

Final-review C2: two committed records in this directory could not be regenerated from committed
code — one had no generator anywhere in the repo, the other did not match the driver that claimed
to produce it. Both were only discoverable by reading the files against the scripts and noticing a
key mismatch. A record that names its own producer and the commit it was produced at makes that
class of drift self-evident instead of forensic, which is why every emitter here goes through
`write_json` rather than `json.dumps` + `write_text`.

The header is deliberately two fields and no more: `generated_by` (the script, as it would be
invoked from the repo root) and `commit` (HEAD when the file was written — the record is written
BEFORE it is committed, so this names its parent commit, which is the state the numbers were
computed from).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def head_commit(repo_root=REPO_ROOT) -> str:
    """HEAD's sha, or "unknown" if git is unavailable — never raises: a provenance header must not
    be able to break the run whose result it is describing."""
    try:
        proc = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def json_header(generated_by: str, repo_root=REPO_ROOT) -> dict:
    return {"generated_by": generated_by, "commit": head_commit(repo_root)}


def _default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return float(obj)


def write_json(path, payload: dict, generated_by: str, indent: int = 2) -> str:
    """Write `payload` with the provenance header first. Returns the absolute path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({**json_header(generated_by), **payload}, indent=indent,
                            default=_default))
    return str(p.resolve())
