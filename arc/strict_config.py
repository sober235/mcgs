"""StrictConfig: a nested dict that records every read that falls back to a code default.

JointTrainer and the loader read their settings with ``cfg.get(key, default)``; a key missing from the
spec silently takes the code default. run_spec.py hands the trainer a StrictConfig instead of a dict:
in strict mode the first fallback raises StrictConfigError (the run stops before it can use a value
nobody wrote down); in record mode (``run_spec.py --audit N``) fallbacks are collected so the defaults
can be written into the spec. Spec 2026-09-28 §2.2 / §3.2.
"""
from __future__ import annotations

import copy
from typing import Any, List, Tuple

import yaml


class StrictConfigError(KeyError):
    """A config key was read that is not in the spec."""


class StrictConfig(dict):
    """dict subclass; every nested dict value is itself a StrictConfig sharing one fallback log."""

    def __init__(self, data=None, *, strict: bool = True, _path: str = "", _log: list | None = None):
        super().__init__()
        self._strict = strict
        self._path = _path
        self._log: List[Tuple[str, Any]] = [] if _log is None else _log
        for key, value in (data or {}).items():
            super().__setitem__(key, self._wrap(key, value))

    def _child_path(self, key) -> str:
        return f"{self._path}.{key}" if self._path else str(key)

    def _wrap(self, key, value):
        if isinstance(value, dict) and not isinstance(value, StrictConfig):
            return StrictConfig(value, strict=self._strict, _path=self._child_path(key), _log=self._log)
        return value

    def _fallback(self, key, default) -> None:
        path = self._child_path(key)
        try:
            recorded = copy.deepcopy(default)
        except Exception:  # noqa: BLE001 - an uncopyable default is still worth recording by reference
            recorded = default
        self._log.append((path, recorded))
        if self._strict:
            raise StrictConfigError(
                f"config key '{path}' is not in the spec; the code default {default!r} would be used")

    def get(self, key, default=None):
        if key in self:
            return super().__getitem__(key)
        self._fallback(key, default)
        if isinstance(default, dict):          # record mode: keep tracking reads below a missing section
            return StrictConfig(default, strict=False, _path=self._child_path(key), _log=self._log)
        return default

    def setdefault(self, key, default=None):
        if key in self:
            return super().__getitem__(key)
        self._fallback(key, default)
        value = self._wrap(key, default)
        super().__setitem__(key, value)
        return value

    def __setitem__(self, key, value):
        super().__setitem__(key, self._wrap(key, value))

    @property
    def fallbacks(self) -> List[Tuple[str, Any]]:
        return list(self._log)

    def to_plain(self) -> dict:
        return {k: (v.to_plain() if isinstance(v, StrictConfig) else v) for k, v in self.items()}


def _represent(dumper, data):
    return dumper.represent_dict(data)


yaml.add_representer(StrictConfig, _represent, Dumper=yaml.Dumper)
yaml.add_representer(StrictConfig, _represent, Dumper=yaml.SafeDumper)
