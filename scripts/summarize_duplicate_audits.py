#!/usr/bin/env python3
"""Validate and combine exact-duplicate audit report directories.

This program only reads audit artifacts and writes aggregate reports.  It never
deletes, links, truncates, or otherwise changes inventoried source files.

Supported producers:

* ``exact-dedupe-v2`` (legacy, without an artifact manifest)
* ``exact-dedupe-streaming-v2`` (generation manifest required)

Manifested artifacts are accepted only after their declared size and SHA-256
match.  Legacy v2 artifacts have no signed/declared digest, so their current
size and SHA-256 are recorded and the combined result is always ``PARTIAL``.
Every JSONL record is parsed and all duplicate totals are independently
recomputed before any output is written.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
import datetime as dt
import hashlib
import json
import os
import posixpath
import re
import sqlite3
import stat
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, TextIO


SUPPORTED_ALGORITHMS = {"exact-dedupe-v2", "exact-dedupe-streaming-v2"}
STREAMING_ALGORITHM = "exact-dedupe-streaming-v2"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
SQLITE_INT_MAX = 2**63 - 1
IO_CHUNK = 8 * 1024 * 1024


class AuditError(RuntimeError):
    """An input violates the report contract."""


@dataclass(frozen=True)
class Artifact:
    path: Path
    sha256: str
    size_bytes: int
    records: int
    manifest_verified: bool


@dataclass(frozen=True)
class Interval:
    minimum: int
    maximum_exclusive: int | None


@dataclass
class ReportResult:
    report_dir: Path
    algorithm: str
    status: str
    interval: Interval
    inventory_key: dict[str, object]
    summary_sha256: str
    manifest_path: Path | None
    manifest_sha256: str | None
    manifest_verified: bool
    legacy_unmanifested: bool
    duplicates: Artifact
    errors: Artifact
    summary: dict[str, object]


def fail(message: str) -> None:
    raise AuditError(message)


def require_dict(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        fail(f"{where} must be an object")
    return value


def require_list(value: object, where: str) -> list[object]:
    if not isinstance(value, list):
        fail(f"{where} must be an array")
    return value


def require_str(value: object, where: str) -> str:
    if not isinstance(value, str):
        fail(f"{where} must be a string")
    return value


def require_bool(value: object, where: str) -> bool:
    if not isinstance(value, bool):
        fail(f"{where} must be a boolean")
    return value


def require_int(
    value: object,
    where: str,
    *,
    minimum: int = 0,
    maximum: int = SQLITE_INT_MAX,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{where} must be an integer")
    if value < minimum or value > maximum:
        fail(f"{where} must be in [{minimum}, {maximum}]")
    return value


def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def reject_json_constant(value: str) -> object:
    fail(f"non-finite JSON number is forbidden: {value}")
    raise AssertionError("unreachable")


def parse_json_bytes(payload: bytes, where: str) -> object:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AuditError(f"{where} is not UTF-8: {exc}") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=strict_object,
            parse_constant=reject_json_constant,
        )
    except AuditError:
        raise
    except json.JSONDecodeError as exc:
        raise AuditError(f"{where} is invalid JSON: {exc}") from exc


def stable_file_bytes(path: Path) -> tuple[bytes, str, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AuditError(f"cannot open artifact {path}: {exc}") from exc
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            fail(f"artifact is not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            while True:
                chunk = handle.read(IO_CHUNK)
                if not chunk:
                    break
                chunks.append(chunk)
                digest.update(chunk)
            after = os.fstat(handle.fileno())
        before_key = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_key = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_key != after_key:
            fail(f"artifact changed while reading: {path}")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            fail(f"artifact short read: {path}")
        return payload, digest.hexdigest(), len(payload)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def load_json_file(path: Path) -> tuple[dict[str, object], str, int]:
    payload, digest, size = stable_file_bytes(path)
    return require_dict(parse_json_bytes(payload, str(path)), str(path)), digest, size


def contained_path(root: Path, reference: str, where: str) -> Path:
    candidate = Path(reference)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise AuditError(f"{where} does not resolve: {candidate}: {exc}") from exc
    try:
        resolved.relative_to(root)
    except ValueError:
        fail(f"{where} escapes report directory {root}: {resolved}")
    return resolved


def locate_summary(report_dir: Path) -> Path:
    for candidate in (report_dir / "summary.json", report_dir / "manifest.json"):
        if not candidate.is_file():
            continue
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(report_dir)
        except ValueError:
            fail(f"summary/manifest symlink escapes report directory: {candidate}")
        return resolved
    fail(f"report directory has neither summary.json nor manifest.json: {report_dir}")
    raise AssertionError("unreachable")


def artifact_claim(
    report_dir: Path,
    value: object,
    where: str,
) -> tuple[Path, str, int]:
    claim = require_dict(value, where)
    path = contained_path(
        report_dir,
        require_str(claim.get("path"), f"{where}.path"),
        f"{where}.path",
    )
    digest = require_str(claim.get("sha256"), f"{where}.sha256")
    if SHA256_RE.fullmatch(digest) is None:
        fail(f"{where}.sha256 must be lowercase 64-character hexadecimal")
    size = require_int(claim.get("size_bytes"), f"{where}.size_bytes")
    return path, digest, size


def legacy_artifact_path(report_dir: Path, value: object, where: str) -> Path:
    return contained_path(report_dir, require_str(value, where), where)


def canonical_base64(value: object, where: str) -> bytes:
    encoded = require_str(value, where)
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise AuditError(f"{where} is invalid base64: {exc}") from exc
    if base64.b64encode(raw).decode("ascii") != encoded:
        fail(f"{where} is not canonical base64")
    if not raw.startswith(b"/") or b"\x00" in raw:
        fail(f"{where} must encode an absolute NUL-free path")
    return raw


def validate_error_path(
    value: object, where: str, *, display_only_ok: bool = False
) -> None:
    record = require_dict(value, where)
    raw_value = record.get("raw_base64")
    if raw_value is None:
        if not display_only_ok:
            fail(f"{where}.raw_base64 must not be null")
        require_str(record.get("display"), f"{where}.display")
        if record.get("source_encoding") != "inventory_meta_display_only":
            fail(f"{where} has an unsupported display-only path encoding")
        return
    canonical_base64(raw_value, f"{where}.raw_base64")
    if "display" in record:
        require_str(record["display"], f"{where}.display")


def validate_error_paths(value: object, where: str) -> None:
    paths = require_list(value, where)
    if not paths:
        fail(f"{where} must not be empty")
    for index, path in enumerate(paths):
        validate_error_path(path, f"{where}[{index}]")


def validate_error_identity(value: object, where: str) -> None:
    identity = require_dict(value, where)
    require_int(identity.get("dev"), f"{where}.dev")
    require_int(identity.get("ino"), f"{where}.ino")
    validate_error_paths(identity.get("paths"), f"{where}.paths")


def validate_error_record(value: object, where: str, algorithm: str) -> None:
    record = require_dict(value, where)
    stage = require_str(record.get("stage"), f"{where}.stage")
    if not stage:
        fail(f"{where}.stage must not be empty")
    require_str(record.get("error"), f"{where}.error")

    if algorithm == "exact-dedupe-v2":
        if stage == "compare":
            require_int(record.get("size"), f"{where}.size", minimum=1)
            digest = require_str(record.get("sha256"), f"{where}.sha256")
            if SHA256_RE.fullmatch(digest) is None:
                fail(f"{where}.sha256 must be lowercase SHA-256")
            require_int(record.get("dev"), f"{where}.dev")
            require_int(record.get("ino"), f"{where}.ino")
        else:
            require_int(record.get("dev"), f"{where}.dev")
            require_int(record.get("ino"), f"{where}.ino")
            validate_error_paths(record.get("paths"), f"{where}.paths")
        return

    if stage == "compare":
        require_int(record.get("size_bytes"), f"{where}.size_bytes", minimum=1)
        digest = require_str(record.get("sha256"), f"{where}.sha256")
        if SHA256_RE.fullmatch(digest) is None:
            fail(f"{where}.sha256 must be lowercase SHA-256")
        validate_error_identity(record.get("representative"), f"{where}.representative")
        validate_error_identity(record.get("member"), f"{where}.member")
    elif stage.startswith("inventory:"):
        path = record.get("path")
        if path is not None:
            validate_error_path(path, f"{where}.path")
    elif stage == "inventory_excluded":
        validate_error_path(record.get("path"), f"{where}.path", display_only_ok=True)
    elif stage == "inventory_skipped_mount":
        validate_error_path(record.get("path"), f"{where}.path")
        require_int(record.get("dev"), f"{where}.dev")
    elif stage == "inventory":
        require_int(record.get("expected"), f"{where}.expected")
        require_int(record.get("observed"), f"{where}.observed")
    elif stage == "inventory_fingerprint":
        pass
    elif stage == "report_validation":
        require_int(record.get("size_bytes"), f"{where}.size_bytes", minimum=1)
        digest = require_str(record.get("sha256"), f"{where}.sha256")
        if SHA256_RE.fullmatch(digest) is None:
            fail(f"{where}.sha256 must be lowercase SHA-256")
        if "dev" in record or "ino" in record or "paths" in record:
            require_int(record.get("dev"), f"{where}.dev")
            require_int(record.get("ino"), f"{where}.ino")
            validate_error_paths(record.get("paths"), f"{where}.paths")
    else:
        require_int(record.get("dev"), f"{where}.dev")
        require_int(record.get("ino"), f"{where}.ino")
        validate_error_paths(record.get("paths"), f"{where}.paths")


def validate_path_record(
    value: object,
    where: str,
    *,
    dev: int,
    ino: int,
    size: int,
    allocated: int,
    nlink: int,
) -> bytes:
    record = require_dict(value, where)
    raw = canonical_base64(record.get("raw_base64"), f"{where}.raw_base64")
    if "display" in record:
        require_str(record["display"], f"{where}.display")
    if (
        "verified" in record
        and require_bool(record["verified"], f"{where}.verified") is not True
    ):
        fail(f"{where}.verified must be true")
    for key, expected in (
        ("dev", dev),
        ("ino", ino),
        ("size_bytes", size),
        ("allocated_bytes", allocated),
        ("nlink", nlink),
    ):
        if key in record and require_int(record[key], f"{where}.{key}") != expected:
            fail(f"{where}.{key} disagrees with member identity")
    return raw


def member_record(
    value: object,
    where: str,
    size: int,
) -> tuple[int, int, int, bool, bytes]:
    member = require_dict(value, where)
    dev = require_int(member.get("dev"), f"{where}.dev")
    ino = require_int(member.get("ino"), f"{where}.ino")
    allocated = require_int(member.get("allocated_bytes"), f"{where}.allocated_bytes")
    link_complete = require_bool(member.get("link_complete"), f"{where}.link_complete")
    if (
        require_bool(member.get("stable_at_report"), f"{where}.stable_at_report")
        is not True
    ):
        fail(f"{where}.stable_at_report must be true")
    nlink = require_int(member.get("nlink"), f"{where}.nlink", minimum=1)
    aliases_seen = require_int(
        member.get("aliases_seen"), f"{where}.aliases_seen", minimum=1
    )
    if "inventory_nlink" in member:
        require_int(member["inventory_nlink"], f"{where}.inventory_nlink", minimum=1)
    if "inventory_allocated_bytes" in member:
        require_int(
            member["inventory_allocated_bytes"],
            f"{where}.inventory_allocated_bytes",
        )
    paths = require_list(member.get("paths"), f"{where}.paths")
    if len(paths) != aliases_seen:
        fail(f"{where}.paths length disagrees with aliases_seen")
    if "verified_aliases" in member and require_int(
        member["verified_aliases"], f"{where}.verified_aliases", minimum=1
    ) != len(paths):
        fail(f"{where}.verified_aliases disagrees with paths length")
    if link_complete != (nlink == aliases_seen == len(paths)):
        fail(f"{where}.link_complete disagrees with nlink/aliases_seen/paths")
    raw_paths = [
        validate_path_record(
            item,
            f"{where}.paths[{index}]",
            dev=dev,
            ino=ino,
            size=size,
            allocated=allocated,
            nlink=nlink,
        )
        for index, item in enumerate(paths)
    ]
    if len(set(raw_paths)) != len(raw_paths):
        fail(f"{where}.paths contains duplicates")
    return dev, ino, allocated, link_complete, min(raw_paths)


def exact_metric(summary: dict[str, object], key: str, actual: int, where: str) -> None:
    claimed = require_int(summary.get(key), f"{where}.{key}")
    if claimed != actual:
        fail(f"{where}.{key} mismatch: claimed={claimed}, recomputed={actual}")


class GroupStore:
    def __init__(self, database: sqlite3.Connection) -> None:
        self.database = database

    def add_group(
        self,
        value: object,
        where: str,
        report_index: int,
        interval: Interval,
        expected_generation: str | None,
    ) -> tuple[int, int]:
        group = require_dict(value, where)
        group_id = require_int(group.get("group_id"), f"{where}.group_id", minimum=1)
        size = require_int(group.get("size_bytes"), f"{where}.size_bytes", minimum=1)
        if size < interval.minimum or (
            interval.maximum_exclusive is not None
            and size >= interval.maximum_exclusive
        ):
            fail(f"{where}.size_bytes lies outside report interval")
        digest = require_str(group.get("sha256"), f"{where}.sha256")
        if SHA256_RE.fullmatch(digest) is None:
            fail(f"{where}.sha256 must be lowercase SHA-256")
        if (
            require_bool(
                group.get("byte_verified_exact"), f"{where}.byte_verified_exact"
            )
            is not True
        ):
            fail(f"{where}.byte_verified_exact must be true")
        if expected_generation is not None:
            generation = require_str(
                group.get("generation_id"), f"{where}.generation_id"
            )
            if generation != expected_generation:
                fail(f"{where}.generation_id disagrees with manifest")

        members_json = require_list(group.get("members"), f"{where}.members")
        if len(members_json) < 2:
            fail(f"{where}.members must contain at least two inodes")
        members = [
            member_record(item, f"{where}.members[{index}]", size)
            for index, item in enumerate(members_json)
        ]
        identities = [(member[0], member[1]) for member in members]
        if len(set(identities)) != len(identities):
            fail(f"{where}.members repeats a dev/inode identity")
        count = require_int(
            group.get("distinct_inode_count"),
            f"{where}.distinct_inode_count",
            minimum=2,
        )
        if count != len(members):
            fail(f"{where}.distinct_inode_count disagrees with members")
        logical = size * (count - 1)
        allocations = [
            allocated if complete else 0 for _, _, allocated, complete, _ in members
        ]
        actionable = sum(allocations) - min(allocations)
        if (
            require_int(
                group.get("logical_reclaim_bytes"), f"{where}.logical_reclaim_bytes"
            )
            != logical
        ):
            fail(f"{where}.logical_reclaim_bytes is not size*(inode_count-1)")
        if (
            require_int(
                group.get("actionable_allocated_reclaim_bytes"),
                f"{where}.actionable_allocated_reclaim_bytes",
            )
            != actionable
        ):
            fail(f"{where}.actionable_allocated_reclaim_bytes is inconsistent")

        existing = self.database.execute(
            "SELECT member_count,logical_bytes,actionable_bytes FROM groups "
            "WHERE size=? AND sha256=?",
            (size, digest),
        ).fetchone()
        expected = (count, logical, actionable)
        if existing is None:
            self.database.execute(
                "INSERT INTO groups(size,sha256,member_count,logical_bytes,"
                "actionable_bytes) VALUES(?,?,?,?,?)",
                (size, digest, count, logical, actionable),
            )
        elif tuple(existing) != expected:
            fail(f"{where} conflicts with another report's group totals")
        collision = self.database.execute(
            "SELECT size FROM hash_sizes WHERE sha256=?", (digest,)
        ).fetchone()
        if collision is None:
            self.database.execute(
                "INSERT INTO hash_sizes(sha256,size) VALUES(?,?)", (digest, size)
            )
        elif int(collision[0]) != size:
            fail(f"{where} uses one SHA-256 for two different sizes")

        try:
            self.database.execute(
                "INSERT INTO group_sources(report_index,local_group_id,size,sha256) "
                "VALUES(?,?,?,?)",
                (report_index, group_id, size, digest),
            )
        except sqlite3.IntegrityError as exc:
            raise AuditError(f"{where} duplicates group_id or group key") from exc

        for dev, ino, allocated, link_complete, canonical_path in members:
            identity = self.database.execute(
                "SELECT size,sha256 FROM identities WHERE dev=? AND ino=?",
                (dev, ino),
            ).fetchone()
            if identity is None:
                self.database.execute(
                    "INSERT INTO identities(dev,ino,size,sha256) VALUES(?,?,?,?)",
                    (dev, ino, size, digest),
                )
            elif (int(identity[0]), str(identity[1])) != (size, digest):
                fail(f"{where} assigns one inode identity to different content")
            old_member = self.database.execute(
                "SELECT allocated,link_complete,canonical_path FROM members "
                "WHERE size=? AND sha256=? AND dev=? AND ino=?",
                (size, digest, dev, ino),
            ).fetchone()
            candidate = (allocated, int(link_complete), canonical_path)
            if old_member is None:
                self.database.execute(
                    "INSERT INTO members(size,sha256,dev,ino,allocated,link_complete,"
                    "canonical_path) VALUES(?,?,?,?,?,?,?)",
                    (size, digest, dev, ino, *candidate),
                )
            elif (
                int(old_member[0]),
                int(old_member[1]),
                bytes(old_member[2]),
            ) != candidate:
                fail(f"{where} conflicts with another report's member snapshot")
        return logical, actionable


def create_store(path: Path) -> sqlite3.Connection:
    database = sqlite3.connect(path)
    database.execute("PRAGMA journal_mode=OFF")
    database.execute("PRAGMA synchronous=OFF")
    database.execute("PRAGMA temp_store=FILE")
    database.executescript(
        """
        CREATE TABLE groups(
          size INTEGER NOT NULL,
          sha256 TEXT NOT NULL,
          member_count INTEGER NOT NULL,
          logical_bytes INTEGER NOT NULL,
          actionable_bytes INTEGER NOT NULL,
          PRIMARY KEY(size,sha256)
        ) WITHOUT ROWID;
        CREATE TABLE members(
          size INTEGER NOT NULL,
          sha256 TEXT NOT NULL,
          dev INTEGER NOT NULL,
          ino INTEGER NOT NULL,
          allocated INTEGER NOT NULL,
          link_complete INTEGER NOT NULL,
          canonical_path BLOB NOT NULL,
          PRIMARY KEY(size,sha256,dev,ino)
        ) WITHOUT ROWID;
        CREATE TABLE identities(
          dev INTEGER NOT NULL,
          ino INTEGER NOT NULL,
          size INTEGER NOT NULL,
          sha256 TEXT NOT NULL,
          PRIMARY KEY(dev,ino)
        ) WITHOUT ROWID;
        CREATE TABLE hash_sizes(
          sha256 TEXT PRIMARY KEY,
          size INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE group_sources(
          report_index INTEGER NOT NULL,
          local_group_id INTEGER NOT NULL,
          size INTEGER NOT NULL,
          sha256 TEXT NOT NULL,
          PRIMARY KEY(report_index,local_group_id),
          UNIQUE(report_index,size,sha256)
        ) WITHOUT ROWID;
        CREATE TABLE pair_contributions(
          dir_a BLOB NOT NULL,
          dir_b BLOB NOT NULL,
          size INTEGER NOT NULL,
          member_pairs INTEGER NOT NULL
        );
        CREATE INDEX pair_contributions_dirs
          ON pair_contributions(dir_a,dir_b);
        CREATE TABLE directory_pairs(
          dir_a BLOB NOT NULL,
          dir_b BLOB NOT NULL,
          exact_group_count INTEGER NOT NULL,
          inode_pair_count TEXT NOT NULL,
          inode_pair_count_digits INTEGER NOT NULL,
          pair_logical_bytes TEXT NOT NULL,
          pair_logical_bytes_digits INTEGER NOT NULL,
          PRIMARY KEY(dir_a,dir_b)
        ) WITHOUT ROWID;
        """
    )
    return database


def validate_summary_contract(
    summary: dict[str, object], where: str
) -> tuple[str, str, Interval, dict[str, object]]:
    algorithm = require_str(summary.get("algorithm"), f"{where}.algorithm")
    if algorithm not in SUPPORTED_ALGORITHMS:
        fail(f"{where}.algorithm is unsupported: {algorithm!r}")
    status_value = require_str(summary.get("status"), f"{where}.status")
    if status_value not in {"COMPLETE", "PARTIAL"}:
        fail(f"{where}.status must be COMPLETE or PARTIAL")
    if (
        require_bool(
            summary.get("deletion_or_deduplication_performed"),
            f"{where}.deletion_or_deduplication_performed",
        )
        is not False
    ):
        fail(f"{where} claims destructive action was performed")
    scope = require_dict(summary.get("scope"), f"{where}.scope")
    if require_str(scope.get("algorithm"), f"{where}.scope.algorithm") != algorithm:
        fail(f"{where}.scope.algorithm disagrees with summary")
    minimum = require_int(scope.get("min_size"), f"{where}.scope.min_size", minimum=1)
    maximum_key = (
        "max_size_exclusive" if algorithm == STREAMING_ALGORITHM else "max_size"
    )
    maximum_value = scope.get(maximum_key)
    maximum = None
    if maximum_value is not None:
        maximum = require_int(
            maximum_value, f"{where}.scope.{maximum_key}", minimum=minimum + 1
        )
    inventory = require_dict(scope.get("inventory"), f"{where}.scope.inventory")
    metrics: dict[str, int] = {}
    for key in (
        "candidate_objects",
        "prefix_hashed",
        "sample_hashed",
        "full_hashed",
        "exact_duplicate_groups",
        "logical_reclaim_bytes",
        "actionable_allocated_reclaim_bytes",
        "inventory_error_count",
        "object_error_count",
        "comparison_error_count",
    ):
        metrics[key] = require_int(summary.get(key), f"{where}.{key}")
    counter_chain = (
        metrics["candidate_objects"],
        metrics["prefix_hashed"],
        metrics["sample_hashed"],
        metrics["full_hashed"],
    )
    if any(left < right for left, right in zip(counter_chain, counter_chain[1:])):
        fail(
            f"{where} hash counters must satisfy "
            "candidate_objects >= prefix_hashed >= sample_hashed >= full_hashed"
        )
    pipeline_incomplete = require_bool(
        summary.get("pipeline_incomplete"), f"{where}.pipeline_incomplete"
    )
    inventory_unchanged = require_bool(
        summary.get("inventory_unchanged_during_run"),
        f"{where}.inventory_unchanged_during_run",
    )
    complete_failures = [
        metrics["inventory_error_count"] != 0,
        metrics["object_error_count"] != 0,
        metrics["comparison_error_count"] != 0,
        pipeline_incomplete,
        not inventory_unchanged,
    ]
    if algorithm == STREAMING_ALGORITHM:
        streaming_checks = {
            "inventory_error_records_written": require_int(
                summary.get("inventory_error_records_written"),
                f"{where}.inventory_error_records_written",
            ),
            "inventory_skipped_mount_count": require_int(
                summary.get("inventory_skipped_mount_count"),
                f"{where}.inventory_skipped_mount_count",
            ),
            "inventory_excluded_count": require_int(
                summary.get("inventory_excluded_count"),
                f"{where}.inventory_excluded_count",
            ),
            "report_validation_error_count": require_int(
                summary.get("report_validation_error_count"),
                f"{where}.report_validation_error_count",
            ),
            "pending_prefix_count": require_int(
                summary.get("pending_prefix_count"),
                f"{where}.pending_prefix_count",
            ),
        }
        count_consistent = require_bool(
            summary.get("inventory_error_count_consistent"),
            f"{where}.inventory_error_count_consistent",
        )
        coverage_complete = require_bool(
            summary.get("source_coverage_complete"),
            f"{where}.source_coverage_complete",
        )
        observed_consistency = (
            streaming_checks["inventory_error_records_written"]
            == metrics["inventory_error_count"]
        )
        if count_consistent != observed_consistency:
            fail(f"{where}.inventory_error_count_consistent is inconsistent")
        observed_coverage = not (
            metrics["inventory_error_count"]
            or streaming_checks["inventory_skipped_mount_count"]
            or streaming_checks["inventory_excluded_count"]
        )
        if coverage_complete != observed_coverage:
            fail(f"{where}.source_coverage_complete is inconsistent")
        complete_failures.extend(
            (
                any(streaming_checks.values()),
                not count_consistent,
                not coverage_complete,
            )
        )
    else:
        complete_failures.append(
            require_int(
                summary.get("stale_at_report_count"),
                f"{where}.stale_at_report_count",
            )
            != 0
        )
    if status_value == "COMPLETE" and any(complete_failures):
        fail(f"{where}.status=COMPLETE contradicts error/coverage fields")
    return algorithm, status_value, Interval(minimum, maximum), inventory


def fingerprint_part(value: object, where: str) -> dict[str, object]:
    part = require_dict(value, where)
    if part.get("empty") is True:
        return {"empty": True}
    result: dict[str, object] = {
        "dev": require_int(part.get("dev"), f"{where}.dev"),
        "ino": require_int(part.get("ino"), f"{where}.ino"),
        "size": require_int(part.get("size"), f"{where}.size"),
        "mtime_ns": require_int(part.get("mtime_ns"), f"{where}.mtime_ns"),
    }
    if "ctime_ns" in part:
        result["ctime_ns"] = require_int(part["ctime_ns"], f"{where}.ctime_ns")
    return result


def normalize_inventory_fingerprint(
    value: dict[str, object], where: str
) -> dict[str, object]:
    database = require_dict(value.get("database"), f"{where}.database")
    normalized = {
        "database": fingerprint_part(database, f"{where}.database"),
        "wal": fingerprint_part(value.get("wal"), f"{where}.wal"),
    }
    path_value = value.get("path", database.get("path"))
    normalized["path"] = require_str(path_value, f"{where}.path")
    return normalized


def fingerprints_compatible(left: dict[str, object], right: dict[str, object]) -> bool:
    if left["path"] != right["path"]:
        return False
    for name in ("database", "wal"):
        left_part = require_dict(left[name], f"left fingerprint {name}")
        right_part = require_dict(right[name], f"right fingerprint {name}")
        if (left_part.get("empty") is True) != (right_part.get("empty") is True):
            return False
        if left_part.get("empty") is True:
            continue
        for key in ("dev", "ino", "size", "mtime_ns"):
            if left_part[key] != right_part[key]:
                return False
        if (
            "ctime_ns" in left_part
            and "ctime_ns" in right_part
            and left_part["ctime_ns"] != right_part["ctime_ns"]
        ):
            return False
    return True


def richest_inventory_fingerprint(reports: list[ReportResult]) -> dict[str, object]:
    result = json.loads(json.dumps(reports[0].inventory_key))
    for report in reports[1:]:
        for name in ("database", "wal"):
            target = require_dict(result[name], f"combined inventory {name}")
            source = require_dict(
                report.inventory_key[name], f"report inventory {name}"
            )
            if "ctime_ns" not in target and "ctime_ns" in source:
                target["ctime_ns"] = source["ctime_ns"]
    return result


def consume_jsonl(
    path: Path,
    callback: Callable[[object, int], None] | None,
) -> tuple[str, int, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AuditError(f"cannot open artifact {path}: {exc}") from exc
    digest = hashlib.sha256()
    size = records = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            fail(f"artifact is not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            for line_number, raw_line in enumerate(handle, start=1):
                digest.update(raw_line)
                size += len(raw_line)
                if not raw_line.strip():
                    fail(f"{path}:{line_number}: blank JSONL line")
                value = parse_json_bytes(raw_line, f"{path}:{line_number}")
                require_dict(value, f"{path}:{line_number}")
                if callback is not None:
                    callback(value, line_number)
                records += 1
            after = os.fstat(handle.fileno())
        before_key = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_key = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_key != after_key:
            fail(f"artifact changed while reading: {path}")
        if size != before.st_size:
            fail(f"artifact short read: {path}")
        return digest.hexdigest(), size, records
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def validate_manifest_artifact(
    path: Path,
    claimed_sha: str,
    claimed_size: int,
    callback: Callable[[object, int], None] | None,
) -> Artifact:
    digest, size, records = consume_jsonl(path, callback)
    if size != claimed_size:
        fail(
            f"artifact size mismatch for {path}: claimed={claimed_size}, actual={size}"
        )
    if digest != claimed_sha:
        fail(
            f"artifact SHA-256 mismatch for {path}: claimed={claimed_sha}, actual={digest}"
        )
    return Artifact(path, digest, size, records, True)


def expected_error_records(summary: dict[str, object], algorithm: str) -> int:
    if algorithm == "exact-dedupe-v2":
        return require_int(
            summary["object_error_count"], "object_error_count"
        ) + require_int(summary["comparison_error_count"], "comparison_error_count")
    total = (
        require_int(
            summary.get("inventory_error_records_written"),
            "inventory_error_records_written",
        )
        + require_int(
            summary.get("inventory_excluded_count"), "inventory_excluded_count"
        )
        + require_int(
            summary.get("inventory_skipped_mount_count"),
            "inventory_skipped_mount_count",
        )
        + require_int(summary["object_error_count"], "object_error_count")
        + require_int(summary["comparison_error_count"], "comparison_error_count")
        + require_int(
            summary.get("report_validation_error_count"),
            "report_validation_error_count",
        )
    )
    consistent = require_bool(
        summary.get("inventory_error_count_consistent"),
        "inventory_error_count_consistent",
    )
    unchanged = require_bool(
        summary.get("inventory_unchanged_during_run"),
        "inventory_unchanged_during_run",
    )
    return total + int(not consistent) + int(not unchanged)


def load_report(
    report_dir: Path,
    report_index: int,
    store: GroupStore,
) -> ReportResult:
    summary_path = locate_summary(report_dir)
    summary, summary_sha, _ = load_json_file(summary_path)
    where = str(summary_path)
    algorithm, status_value, interval, inventory_raw = validate_summary_contract(
        summary, where
    )
    reports = require_dict(summary.get("reports"), f"{where}.reports")
    manifest_reference = reports.get("generation_manifest")
    manifest_path: Path | None = None
    manifest_sha: str | None = None
    manifest_verified = False
    legacy = False
    generation_id: str | None = None

    if manifest_reference is not None:
        manifest_path = contained_path(
            report_dir,
            require_str(manifest_reference, f"{where}.reports.generation_manifest"),
            f"{where}.reports.generation_manifest",
        )
        manifest, manifest_sha, _ = load_json_file(manifest_path)
        if manifest != summary:
            fail(
                f"summary and generation manifest differ: {summary_path} vs {manifest_path}"
            )
        if (
            require_int(summary.get("manifest_version"), f"{where}.manifest_version")
            != 1
        ):
            fail(f"{where}.manifest_version is unsupported")
        generation_id = require_str(
            summary.get("generation_id"), f"{where}.generation_id"
        )
        if not generation_id or manifest_path.parent.name != generation_id:
            fail(f"{where}.generation_id disagrees with manifest directory")
        manifest_verified = True
    elif algorithm == STREAMING_ALGORITHM:
        fail(f"streaming report lacks generation manifest: {report_dir}")
    else:
        legacy = True

    if manifest_verified:
        duplicates_path, duplicates_sha, duplicates_size = artifact_claim(
            report_dir, reports.get("duplicates"), f"{where}.reports.duplicates"
        )
        errors_path, errors_sha, errors_size = artifact_claim(
            report_dir, reports.get("errors"), f"{where}.reports.errors"
        )
        if (
            duplicates_path.parent != manifest_path.parent
            or errors_path.parent != manifest_path.parent
        ):
            fail("manifested artifacts must be in the generation manifest directory")
    else:
        duplicates_path = legacy_artifact_path(
            report_dir, reports.get("duplicates"), f"{where}.reports.duplicates"
        )
        errors_path = legacy_artifact_path(
            report_dir, reports.get("errors"), f"{where}.reports.errors"
        )
        duplicates_sha = errors_sha = ""
        duplicates_size = errors_size = -1

    group_count = logical_total = actionable_total = 0

    def add_group(value: object, line_number: int) -> None:
        nonlocal group_count, logical_total, actionable_total
        logical, actionable = store.add_group(
            value,
            f"{duplicates_path}:{line_number}",
            report_index,
            interval,
            generation_id,
        )
        group_count += 1
        logical_total += logical
        actionable_total += actionable

    def check_error(value: object, line_number: int) -> None:
        validate_error_record(value, f"{errors_path}:{line_number}", algorithm)

    if manifest_verified:
        duplicates = validate_manifest_artifact(
            duplicates_path, duplicates_sha, duplicates_size, add_group
        )
        errors = validate_manifest_artifact(
            errors_path, errors_sha, errors_size, check_error
        )
    else:
        digest, size, records = consume_jsonl(duplicates_path, add_group)
        duplicates = Artifact(duplicates_path, digest, size, records, False)
        digest, size, records = consume_jsonl(errors_path, check_error)
        errors = Artifact(errors_path, digest, size, records, False)

    exact_metric(summary, "exact_duplicate_groups", group_count, where)
    exact_metric(summary, "logical_reclaim_bytes", logical_total, where)
    exact_metric(
        summary,
        "actionable_allocated_reclaim_bytes",
        actionable_total,
        where,
    )
    expected_errors = expected_error_records(summary, algorithm)
    if errors.records != expected_errors:
        fail(
            f"{where} error record count mismatch: "
            f"expected={expected_errors}, parsed={errors.records}"
        )
    inventory = normalize_inventory_fingerprint(
        inventory_raw, f"{where}.scope.inventory"
    )
    if algorithm == STREAMING_ALGORITHM:
        for part_name in ("database", "wal"):
            part = require_dict(
                inventory[part_name], f"{where}.scope.inventory.{part_name}"
            )
            if part.get("empty") is not True and "ctime_ns" not in part:
                fail(
                    f"{where}.scope.inventory.{part_name}.ctime_ns is required "
                    "for streaming reports"
                )
    return ReportResult(
        report_dir=report_dir,
        algorithm=algorithm,
        status=status_value,
        interval=interval,
        inventory_key=inventory,
        summary_sha256=summary_sha,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        manifest_verified=manifest_verified,
        legacy_unmanifested=legacy,
        duplicates=duplicates,
        errors=errors,
        summary=summary,
    )


def verify_report_set(reports: list[ReportResult]) -> None:
    expected_inventory_errors = require_int(
        reports[0].summary["inventory_error_count"], "inventory_error_count"
    )
    for index, report in enumerate(reports[1:], start=1):
        for previous in reports[:index]:
            if not fingerprints_compatible(
                previous.inventory_key, report.inventory_key
            ):
                fail(
                    "report inventory fingerprints differ: "
                    f"{previous.report_dir} vs {report.report_dir}"
                )
        observed_errors = require_int(
            report.summary["inventory_error_count"], "inventory_error_count"
        )
        if observed_errors != expected_inventory_errors:
            fail(
                "global inventory_error_count differs across shards: "
                f"{reports[0].report_dir}={expected_inventory_errors}, "
                f"{report.report_dir}={observed_errors}"
            )
    ordered = sorted(reports, key=lambda item: item.interval.minimum)
    for left, right in zip(ordered, ordered[1:]):
        left_max = left.interval.maximum_exclusive
        if left_max is None or right.interval.minimum < left_max:
            fail(
                "report size intervals overlap: "
                f"{left.report_dir} and {right.report_dir}"
            )


def verify_merged_groups(database: sqlite3.Connection) -> None:
    mismatch = database.execute(
        "SELECT g.size,g.sha256,g.member_count,COUNT(m.ino) actual "
        "FROM groups g LEFT JOIN members m "
        "ON m.size=g.size AND m.sha256=g.sha256 "
        "GROUP BY g.size,g.sha256 HAVING actual<>g.member_count LIMIT 1"
    ).fetchone()
    if mismatch is not None:
        fail(
            "reports disagree on merged group membership: "
            f"size={mismatch[0]}, sha256={mismatch[1]}, "
            f"claimed={mismatch[2]}, merged={mismatch[3]}"
        )


def build_directory_pairs(database: sqlite3.Connection) -> None:
    groups = database.execute("SELECT size,sha256 FROM groups ORDER BY size,sha256")
    pending: list[tuple[bytes, bytes, int, int]] = []
    for size, digest in groups:
        directories = Counter(
            posixpath.dirname(bytes(row[0]))
            for row in database.execute(
                "SELECT canonical_path FROM members WHERE size=? AND sha256=?",
                (size, digest),
            )
        )
        names = sorted(directories)
        for index, left in enumerate(names):
            same_pairs = directories[left] * (directories[left] - 1) // 2
            if same_pairs:
                pending.append((left, left, int(size), same_pairs))
            for right in names[index + 1 :]:
                pending.append(
                    (left, right, int(size), directories[left] * directories[right])
                )
            if len(pending) >= 10_000:
                database.executemany(
                    "INSERT INTO pair_contributions(dir_a,dir_b,size,member_pairs) "
                    "VALUES(?,?,?,?)",
                    pending,
                )
                pending.clear()
    if pending:
        database.executemany(
            "INSERT INTO pair_contributions(dir_a,dir_b,size,member_pairs) VALUES(?,?,?,?)",
            pending,
        )
    database.commit()

    cursor = database.execute(
        "SELECT dir_a,dir_b,size,member_pairs FROM pair_contributions "
        "ORDER BY dir_a,dir_b"
    )
    current: tuple[bytes, bytes] | None = None
    group_count = pair_count = pair_bytes = 0
    aggregate_rows: list[tuple[bytes, bytes, int, str, int, str, int]] = []

    def flush() -> None:
        nonlocal current, group_count, pair_count, pair_bytes
        if current is None:
            return
        pair_text = str(pair_count)
        bytes_text = str(pair_bytes)
        aggregate_rows.append(
            (
                current[0],
                current[1],
                group_count,
                pair_text,
                len(pair_text),
                bytes_text,
                len(bytes_text),
            )
        )
        if len(aggregate_rows) >= 10_000:
            database.executemany(
                "INSERT INTO directory_pairs VALUES(?,?,?,?,?,?,?)", aggregate_rows
            )
            aggregate_rows.clear()
        current = None
        group_count = pair_count = pair_bytes = 0

    for dir_a, dir_b, size, member_pairs in cursor:
        key = (bytes(dir_a), bytes(dir_b))
        if current != key:
            flush()
            current = key
        group_count += 1
        pair_count += int(member_pairs)
        pair_bytes += int(size) * int(member_pairs)
    flush()
    if aggregate_rows:
        database.executemany(
            "INSERT INTO directory_pairs VALUES(?,?,?,?,?,?,?)", aggregate_rows
        )
    database.commit()


def b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    number = float(value)
    for unit in units:
        if number < 1024 or unit == units[-1]:
            return f"{number:.2f} {unit}"
        number /= 1024
    raise AssertionError("unreachable")


def interval_record(interval: Interval) -> dict[str, object]:
    return {
        "min_size_bytes_inclusive": interval.minimum,
        "max_size_bytes_exclusive": interval.maximum_exclusive,
    }


def report_record(index: int, report: ReportResult) -> dict[str, object]:
    summary = report.summary
    metrics = {
        key: require_int(summary[key], f"summary.{key}")
        for key in (
            "candidate_objects",
            "prefix_hashed",
            "sample_hashed",
            "full_hashed",
            "exact_duplicate_groups",
            "logical_reclaim_bytes",
            "actionable_allocated_reclaim_bytes",
            "inventory_error_count",
            "object_error_count",
            "comparison_error_count",
        )
    }
    metrics["stale_at_report_count"] = require_int(
        summary.get("stale_at_report_count", 0), "summary.stale_at_report_count"
    )
    metrics["report_validation_error_count"] = require_int(
        summary.get("report_validation_error_count", 0),
        "summary.report_validation_error_count",
    )
    return {
        "index": index,
        "report_dir": str(report.report_dir),
        "algorithm": report.algorithm,
        "status": report.status,
        "size_interval": interval_record(report.interval),
        "summary_sha256": report.summary_sha256,
        "generation_manifest": None
        if report.manifest_path is None
        else {
            "path": str(report.manifest_path),
            "sha256": report.manifest_sha256,
            "verified_equal_to_summary": True,
        },
        "source_manifest_verified": report.manifest_verified,
        "legacy_unmanifested": report.legacy_unmanifested,
        "duplicates": {
            "path": str(report.duplicates.path),
            "sha256": report.duplicates.sha256,
            "size_bytes": report.duplicates.size_bytes,
            "records": report.duplicates.records,
            "manifest_verified": report.duplicates.manifest_verified,
        },
        "errors": {
            "path": str(report.errors.path),
            "sha256": report.errors.sha256,
            "size_bytes": report.errors.size_bytes,
            "records": report.errors.records,
            "manifest_verified": report.errors.manifest_verified,
        },
        "metrics": metrics,
    }


def combined_summary(
    database: sqlite3.Connection,
    reports: list[ReportResult],
    top_limit: int,
) -> dict[str, object]:
    group_count, logical, actionable = database.execute(
        "SELECT COUNT(*),COALESCE(SUM(logical_bytes),0),"
        "COALESCE(SUM(actionable_bytes),0) FROM groups"
    ).fetchone()
    inode_count = int(database.execute("SELECT COUNT(*) FROM identities").fetchone()[0])
    pair_count = int(
        database.execute("SELECT COUNT(*) FROM directory_pairs").fetchone()[0]
    )
    legacy_count = sum(report.legacy_unmanifested for report in reports)
    partial_inputs = [
        index
        for index, report in enumerate(reports, start=1)
        if report.status != "COMPLETE"
    ]
    partial_reasons: list[str] = []
    if legacy_count:
        partial_reasons.append(
            f"{legacy_count} legacy exact-dedupe-v2 report(s) lack producer manifests"
        )
    if partial_inputs:
        partial_reasons.append(f"input report(s) not COMPLETE: {partial_inputs}")
    status_value = "PARTIAL" if partial_reasons else "COMPLETE"
    metric_names = (
        "candidate_objects",
        "prefix_hashed",
        "sample_hashed",
        "full_hashed",
        "object_error_count",
        "comparison_error_count",
    )
    summed_metrics = {
        key: sum(
            require_int(report.summary[key], f"summary.{key}") for report in reports
        )
        for key in metric_names
    }
    summed_metrics["inventory_error_count"] = require_int(
        reports[0].summary["inventory_error_count"], "inventory_error_count"
    )
    summed_metrics["stale_at_report_count"] = sum(
        require_int(
            report.summary.get("stale_at_report_count", 0),
            "stale_at_report_count",
        )
        for report in reports
    )
    summed_metrics["report_validation_error_count"] = sum(
        require_int(
            report.summary.get("report_validation_error_count", 0),
            "report_validation_error_count",
        )
        for report in reports
    )
    ordered_intervals = [
        interval_record(report.interval)
        for report in sorted(reports, key=lambda item: item.interval.minimum)
    ]
    return {
        "schema_version": 1,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": status_value,
        "partial_reasons": partial_reasons,
        "deletion_or_deduplication_performed": False,
        "contract": (
            "report-only aggregation; exact groups were byte-verified by input producers; "
            "no deletion, hardlinking, reflinking, or source mutation was performed"
        ),
        "inventory_fingerprint": richest_inventory_fingerprint(reports),
        "coverage_intervals": ordered_intervals,
        "input_report_count": len(reports),
        "source_manifest_verified_count": sum(
            report.manifest_verified for report in reports
        ),
        "legacy_unmanifested_count": legacy_count,
        "input_reports": [
            report_record(index, report)
            for index, report in enumerate(reports, start=1)
        ],
        "aggregate": {
            **summed_metrics,
            "exact_duplicate_groups": int(group_count),
            "distinct_duplicate_inodes": inode_count,
            "logical_reclaim_bytes": int(logical),
            "logical_reclaim_human": human_bytes(int(logical)),
            "actionable_allocated_reclaim_bytes": int(actionable),
            "actionable_allocated_reclaim_human": human_bytes(int(actionable)),
            "directory_pair_rows": pair_count,
            "top_groups_rows": min(int(group_count), top_limit),
        },
        "path_encoding": {
            "tsv_paths": "RFC 4648 canonical base64 of raw POSIX path bytes",
            "separator_for_path_lists": ";",
        },
    }


def open_temp_outputs(out: Path) -> dict[str, tuple[Path, TextIO]]:
    result: dict[str, tuple[Path, TextIO]] = {}
    for name in (
        "combined_summary.json",
        "top_groups.tsv",
        "directory_pairs.tsv",
        "README.md",
    ):
        descriptor, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=out)
        os.chmod(temporary, 0o600)
        result[name] = (
            Path(temporary),
            os.fdopen(descriptor, "w", encoding="utf-8", newline=""),
        )
    return result


def write_top_groups(
    handle: TextIO,
    database: sqlite3.Connection,
    limit: int,
) -> None:
    writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
    writer.writerow(
        (
            "rank",
            "size_bytes",
            "distinct_inode_count",
            "logical_reclaim_bytes",
            "actionable_allocated_reclaim_bytes",
            "sha256",
            "source_report_indexes",
            "member_dev_ino",
            "member_paths_raw_base64",
        )
    )
    groups = database.execute(
        "SELECT size,sha256,member_count,logical_bytes,actionable_bytes "
        "FROM groups ORDER BY logical_bytes DESC,actionable_bytes DESC,size DESC,sha256 "
        "LIMIT ?",
        (limit,),
    )
    for rank, row in enumerate(groups, start=1):
        size, digest, count, logical, actionable = row
        members = list(
            database.execute(
                "SELECT dev,ino,canonical_path FROM members "
                "WHERE size=? AND sha256=? ORDER BY dev,ino",
                (size, digest),
            )
        )
        report_indexes = [
            str(item[0])
            for item in database.execute(
                "SELECT report_index FROM group_sources WHERE size=? AND sha256=? "
                "ORDER BY report_index",
                (size, digest),
            )
        ]
        writer.writerow(
            (
                rank,
                size,
                count,
                logical,
                actionable,
                digest,
                ";".join(report_indexes),
                ";".join(f"{dev}:{ino}" for dev, ino, _ in members),
                ";".join(b64(bytes(path)) for _, _, path in members),
            )
        )


def write_directory_pairs(handle: TextIO, database: sqlite3.Connection) -> None:
    writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
    writer.writerow(
        (
            "rank",
            "directory_a_raw_base64",
            "directory_b_raw_base64",
            "same_directory",
            "exact_group_count",
            "inode_pair_count",
            "pair_logical_bytes",
        )
    )
    rows = database.execute(
        "SELECT dir_a,dir_b,exact_group_count,inode_pair_count,pair_logical_bytes "
        "FROM directory_pairs ORDER BY pair_logical_bytes_digits DESC,"
        "pair_logical_bytes DESC,inode_pair_count_digits DESC,inode_pair_count DESC,"
        "dir_a,dir_b"
    )
    for rank, (left, right, groups, pairs, byte_count) in enumerate(rows, start=1):
        writer.writerow(
            (
                rank,
                b64(bytes(left)),
                b64(bytes(right)),
                int(bytes(left) == bytes(right)),
                groups,
                pairs,
                byte_count,
            )
        )


def make_readme(summary: dict[str, object]) -> str:
    aggregate = require_dict(summary["aggregate"], "aggregate")
    intervals = require_list(summary["coverage_intervals"], "coverage_intervals")
    lines = [
        "# Combined exact-duplicate audit",
        "",
        f"Status: **{summary['status']}**.",
        "",
        "Report-only output. No file was deleted, linked, reflinked, truncated, or modified.",
        "",
        "## Totals",
        "",
        f"- Input reports: {summary['input_report_count']}",
        f"- Exact groups: {aggregate['exact_duplicate_groups']}",
        f"- Distinct duplicate inodes: {aggregate['distinct_duplicate_inodes']}",
        f"- Logical reclaim: {aggregate['logical_reclaim_bytes']} bytes ({aggregate['logical_reclaim_human']})",
        "- Actionable allocated reclaim: "
        f"{aggregate['actionable_allocated_reclaim_bytes']} bytes "
        f"({aggregate['actionable_allocated_reclaim_human']})",
        f"- Global inventory errors: {aggregate['inventory_error_count']} (counted once across shards)",
        f"- Late stale exclusions: {aggregate['stale_at_report_count']}",
        "",
        "Logical reclaim is `size_bytes * (distinct_inode_count - 1)`. "
        "Actionable allocated reclaim counts only members whose hardlink aliases were complete, "
        "then retains one independent allocation.",
        "",
        "## Validated size intervals",
        "",
    ]
    for value in intervals:
        interval = require_dict(value, "coverage interval")
        maximum = interval["max_size_bytes_exclusive"]
        lines.append(
            "- `"
            f"{interval['min_size_bytes_inclusive']} <= size < "
            f"{'unbounded' if maximum is None else maximum}`"
        )
    reasons = require_list(summary["partial_reasons"], "partial_reasons")
    if reasons:
        lines.extend(("", "## Why status is PARTIAL", ""))
        lines.extend(f"- {require_str(reason, 'partial reason')}" for reason in reasons)
    lines.extend(
        (
            "",
            "## Files",
            "",
            "- `combined_summary.json`: machine-readable provenance, validation, intervals, and totals.",
            "- `top_groups.tsv`: largest exact groups. Member paths are canonical RFC 4648 base64 of raw path bytes.",
            "- `directory_pairs.tsv`: unordered pairs of canonical member-parent directories.",
            "",
            "In `directory_pairs.tsv`, `inode_pair_count` is the number of unordered inode pairs. "
            "`pair_logical_bytes` is `sum(group_size * inode_pairs)` and is a relationship weight, "
            "not additive reclaimable capacity; groups with more than two members appear in multiple pairs.",
            "",
            "Decode one raw path with `printf '%s' BASE64 | base64 -d`. "
            "Semicolon separates base64 paths in `top_groups.tsv`; semicolon is not in the base64 alphabet.",
            "",
        )
    )
    return "\n".join(lines)


def write_outputs(
    out: Path,
    database: sqlite3.Connection,
    summary: dict[str, object],
    top_limit: int,
) -> None:
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = open_temp_outputs(out)
    try:
        json.dump(
            summary,
            temporary["combined_summary.json"][1],
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        temporary["combined_summary.json"][1].write("\n")
        write_top_groups(temporary["top_groups.tsv"][1], database, top_limit)
        write_directory_pairs(temporary["directory_pairs.tsv"][1], database)
        temporary["README.md"][1].write(make_readme(summary))
        for _, handle in temporary.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        for name, (path, _) in temporary.items():
            os.replace(path, out / name)
        directory_fd = os.open(
            out,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        for path, handle in temporary.values():
            if not handle.closed:
                handle.close()
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and combine strict/streaming exact-duplicate reports"
    )
    parser.add_argument("report_dirs", nargs="+", help="input report directories")
    parser.add_argument("--out", required=True, help="aggregate output directory")
    parser.add_argument(
        "--top-groups",
        type=int,
        default=100,
        help="rows in top_groups.tsv (default: 100)",
    )
    return parser.parse_args(argv)


def resolved_inputs(args: argparse.Namespace) -> tuple[list[Path], Path]:
    if args.top_groups <= 0:
        fail("--top-groups must be > 0")
    reports: list[Path] = []
    for value in args.report_dirs:
        path = Path(value).resolve(strict=True)
        if not path.is_dir():
            fail(f"report input is not a directory: {path}")
        if path in reports:
            fail(f"duplicate report directory: {path}")
        reports.append(path)
    out = Path(args.out).resolve(strict=False)
    for report in reports:
        if out == report or report in out.parents:
            fail(f"output directory must not be an input or lie inside one: {out}")
    return reports, out


def run(args: argparse.Namespace) -> dict[str, object]:
    report_dirs, out = resolved_inputs(args)
    with tempfile.TemporaryDirectory(prefix="duplicate-audit-summary-") as temporary:
        database = create_store(Path(temporary) / "aggregate.sqlite3")
        try:
            store = GroupStore(database)
            reports = [
                load_report(report, index, store)
                for index, report in enumerate(report_dirs, start=1)
            ]
            database.commit()
            verify_report_set(reports)
            verify_merged_groups(database)
            build_directory_pairs(database)
            summary = combined_summary(database, reports, args.top_groups)
            write_outputs(out, database, summary, args.top_groups)
            return summary
        finally:
            database.close()


def main(argv: Iterable[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        summary = run(args)
    except (AuditError, OSError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    aggregate = require_dict(summary["aggregate"], "aggregate")
    print(
        f"{summary['status']}: reports={summary['input_report_count']} "
        f"groups={aggregate['exact_duplicate_groups']} "
        f"logical_reclaim_bytes={aggregate['logical_reclaim_bytes']}"
    )
    return 0 if summary["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
