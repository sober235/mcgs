"""受试者注册表与冻结划分。

划分规则(spec §3.1,先于任何实验固定,不得事后调整):
  1. forced_dev 先行进入 development(huangyuwei 已承受约 62 次调参运行,已污染);
  2. 其余 development 名额,在剩下的受试者中按 SHA-256(subject_key) 十六进制升序取。
token 规则:按 subject_key 字典序编号 S001..,与划分无关,保证跨会话稳定。
"""
from __future__ import annotations

import hashlib


def _digest(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def split_subjects(subject_keys: list[str], forced_dev: str, n_dev: int) -> dict:
    keys = sorted(set(subject_keys))
    if forced_dev not in keys:
        raise ValueError(f"forced_dev {forced_dev!r} not among the {len(keys)} subject keys")
    if not 1 <= n_dev <= len(keys):
        raise ValueError(f"n_dev {n_dev} out of range for {len(keys)} subjects")
    rest = sorted((k for k in keys if k != forced_dev), key=_digest)
    development = sorted([forced_dev] + rest[: n_dev - 1])
    test = sorted(k for k in keys if k not in set(development))
    return {"development": development, "test": test}


def token_map(subject_keys: list[str]) -> dict:
    return {k: f"S{i + 1:03d}" for i, k in enumerate(sorted(set(subject_keys)))}
