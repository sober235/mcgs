"""Anonymized subject-ID to filesystem-path resolver (D-14 / D-15 / CB-04).

The pair configs reference subjects by token (e.g. ``S01_plain.mat``,
``S01_contrast.mat``). Real paths live in a gitignored ``paths.private.yaml``
file. This module is the single resolution seam used by ``train_joint.py``
and any autoresearch driver that loads tokenized subject data.

The resolver is intentionally CALLER-SIDE (not pushed into ``MRIDataset``).
Baselines and historical configs that already store absolute paths
continue to work unchanged: any value that does NOT match the token
pattern is returned verbatim.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

import yaml


_TOKEN_RE = re.compile(r"^S\d+_[A-Za-z][A-Za-z0-9_]*\.mat$")


def _candidate_mapping_paths() -> Iterable[Path]:
    env = os.environ.get("MCGS_PATHS_PRIVATE")
    if env:
        yield Path(env)
    # `data/` -> `paired_gs/` -> framework root
    framework_root = Path(__file__).resolve().parents[2]
    yield framework_root / "paths.private.yaml"
    # Fallback: repo root one level above the framework root
    yield framework_root.parent / "paths.private.yaml"


def _load_mapping(mapping_file: str | None) -> dict:
    if mapping_file:
        with open(mapping_file, "r") as f:
            return yaml.safe_load(f) or {}
    for p in _candidate_mapping_paths():
        if p.is_file():
            with open(p, "r") as f:
                return yaml.safe_load(f) or {}
    raise FileNotFoundError(
        "paths.private.yaml not found. Looked in $MCGS_PATHS_PRIVATE, "
        "<framework_root>/paths.private.yaml, and <framework_root>/../paths.private.yaml."
    )


def resolve_subject_paths(value: str, mapping_file: str | None = None) -> str:
    """Resolve an anonymized token to a real path; pass through real paths.

    Parameters
    ----------
    value
        YAML value as read from config. Either an anonymized token like
        ``S01_plain.mat`` / ``S01_plain_CSM.mat`` or an absolute filesystem path.
    mapping_file
        Optional override for the ``paths.private.yaml`` location.

    Returns
    -------
    str
        Absolute path on local disk.

    Raises
    ------
    FileNotFoundError
        If ``value`` is a token but ``paths.private.yaml`` is missing.
    KeyError
        If ``value`` is a token but no matching subject entry exists.
    """
    if not _TOKEN_RE.match(value):
        return value   # already an absolute / relative real path

    mapping = _load_mapping(mapping_file)
    # token format: S<digits>_<tag>.mat  -> e.g. S01_plain.mat -> ("S01", "plain")
    token_no_ext = value[: -len(".mat")]
    subj, _, tag = token_no_ext.partition("_")
    subjects = mapping.get("subjects", {})
    entry = subjects.get(subj)
    if entry is None or tag not in entry:
        raise KeyError(
            f"paths.private.yaml has no mapping for {subj}/{tag} (token={value!r})."
        )
    return str(entry[tag])
