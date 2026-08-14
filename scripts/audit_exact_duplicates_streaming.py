#!/usr/bin/env python3
"""Streaming, read-only exact-duplicate verification from an inventory DB.

Required inventory tables are produced by ``audit_data0_duplicates.py``.  This
program never changes the inventory or source files.  It writes an independent,
resumable state database and atomically replaces JSON reports.

Pipeline: distinct inode -> size -> first 4 KiB -> middle/tail 64 KiB -> full
SHA-256 -> byte comparison -> final metadata sweep of every readable candidate.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import fcntl
import hashlib
import json
import os
import sqlite3
import stat
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Iterable, Iterator
from urllib.parse import quote


ALGORITHM = "exact-dedupe-streaming-v2"
SCHEMA_VERSION = 3
PREFIX_BYTES = 4 * 1024
SAMPLE_BYTES = 64 * 1024
IO_CHUNK = 8 * 1024 * 1024
INIT_BATCH = 100_000
WORK_PAGE = 4_096
GROUP_PAGE = 512
MAX_SAMPLE_WORKERS = 128
MAX_FULL_WORKERS = 8


class FileChanged(RuntimeError):
    """Source inode identity or metadata changed."""


@dataclass(frozen=True)
class Signature:
    dev: int
    ino: int
    size: int
    mtime_ns: int
    ctime_ns: int
    nlink: int


@dataclass(frozen=True)
class WorkItem:
    source_rowid: int
    dev: int
    ino: int
    size: int
    inventory_mtime_ns: int
    paths: tuple[bytes, ...]
    expected: Signature | None


@dataclass(frozen=True)
class HashResult:
    dev: int
    ino: int
    digest: bytes
    signature: Signature
    bytes_read: int
    path: bytes


@dataclass(frozen=True)
class AliasSnapshot:
    path: bytes
    signature: Signature
    allocated: int
    validated_at_ns: int


@dataclass(frozen=True)
class MemberSnapshot:
    aliases: tuple[AliasSnapshot, ...]
    signature: Signature
    allocated: int
    link_complete: bool


@dataclass(frozen=True)
class ObjectRecord:
    source_rowid: int
    dev: int
    ino: int
    size: int
    allocated: int
    nlink: int
    aliases_seen: int
    inventory_mtime_ns: int
    sig_mtime_ns: int | None
    sig_ctime_ns: int | None
    sig_nlink: int | None
    prefix_hash: bytes | None
    sample_hash: bytes | None
    full_hash: bytes | None
    error_stage: str | None
    error: str | None
    paths: tuple[bytes, ...]

    def work_item(self) -> WorkItem:
        expected = None
        if (
            self.sig_mtime_ns is not None
            and self.sig_ctime_ns is not None
            and self.sig_nlink is not None
        ):
            expected = Signature(
                self.dev, self.ino, self.size,
                self.sig_mtime_ns, self.sig_ctime_ns, self.sig_nlink,
            )
        return WorkItem(
            self.source_rowid, self.dev, self.ino, self.size,
            self.inventory_mtime_ns, self.paths, expected,
        )


def human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    number = float(value)
    for unit in units:
        if number < 1024 or unit == units[-1]:
            return f"{number:.2f} {unit}"
        number /= 1024
    raise AssertionError("unreachable")


def raw_path(value: bytes | str) -> bytes:
    return value if isinstance(value, bytes) else os.fsencode(value)


def display_path(path: bytes) -> str:
    return os.fsdecode(path).encode("utf-8", "backslashreplace").decode("utf-8")


def path_record(path: bytes) -> dict[str, str]:
    return {
        "display": display_path(path),
        "raw_base64": base64.b64encode(path).decode("ascii"),
    }


def signature(st: os.stat_result) -> Signature:
    return Signature(
        int(st.st_dev), int(st.st_ino), int(st.st_size),
        int(st.st_mtime_ns), int(st.st_ctime_ns), int(st.st_nlink),
    )


def open_verified(item: WorkItem) -> tuple[BinaryIO, Signature, bytes]:
    failures: list[str] = []
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    for path in item.paths:
        handle: BinaryIO | None = None
        try:
            descriptor = os.open(path, flags)
            handle = open(descriptor, "rb", buffering=0, closefd=True)
            st = os.fstat(handle.fileno())
            current = signature(st)
            if not stat.S_ISREG(st.st_mode):
                raise RuntimeError("not a regular file")
            if (current.dev, current.ino, current.size) != (
                item.dev, item.ino, item.size,
            ):
                raise FileChanged(
                    "identity changed: "
                    f"expected={(item.dev, item.ino, item.size)}, got={current}"
                )
            if current.mtime_ns != item.inventory_mtime_ns:
                raise FileChanged(
                    "mtime differs from inventory: "
                    f"expected={item.inventory_mtime_ns}, got={current.mtime_ns}"
                )
            if item.expected is not None and current != item.expected:
                raise FileChanged(
                    f"metadata changed: expected={item.expected}, got={current}"
                )
            if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_SEQUENTIAL"):
                try:
                    os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
                except OSError:
                    pass
            return handle, current, path
        except Exception as exc:
            if handle is not None:
                handle.close()
            failures.append(f"{display_path(path)}: {exc!r}")
    raise RuntimeError("all aliases failed: " + " | ".join(failures))


def allocated_bytes(st: os.stat_result) -> int:
    return int(st.st_blocks) * 512


def finish_verified(
    handle: BinaryIO,
    before: Signature,
    before_allocated: int | None = None,
) -> int:
    after_stat = os.fstat(handle.fileno())
    after = signature(after_stat)
    if before != after:
        raise FileChanged(f"file changed during read: before={before}, after={after}")
    after_allocated = allocated_bytes(after_stat)
    if before_allocated is not None and before_allocated != after_allocated:
        raise FileChanged(
            "allocation changed during read: "
            f"before={before_allocated}, after={after_allocated}"
        )
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        try:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass
    return after_allocated


def verify_all_aliases(record: ObjectRecord) -> MemberSnapshot:
    """Validate every inventoried alias; never substitute one alias for another."""
    item = record.work_item()
    if item.expected is None:
        raise RuntimeError("object has no complete hash-time signature")
    if len(item.paths) != record.aliases_seen:
        raise RuntimeError(
            "state alias count inconsistent: "
            f"expected={record.aliases_seen}, stored={len(item.paths)}"
        )
    snapshots: list[AliasSnapshot] = []
    common_allocated: int | None = None
    for path in item.paths:
        alias_item = WorkItem(
            item.source_rowid, item.dev, item.ino, item.size,
            item.inventory_mtime_ns, (path,), item.expected,
        )
        handle, before, _ = open_verified(alias_item)
        with handle:
            before_stat = os.fstat(handle.fileno())
            if signature(before_stat) != before:
                raise FileChanged("alias changed immediately after open")
            before_allocated = allocated_bytes(before_stat)
            current_allocated = finish_verified(
                handle, before, before_allocated
            )
        if common_allocated is None:
            common_allocated = current_allocated
        elif common_allocated != current_allocated:
            raise FileChanged(
                "aliases report different allocation: "
                f"first={common_allocated}, current={current_allocated}"
            )
        snapshots.append(
            AliasSnapshot(path, before, current_allocated, time.time_ns())
        )
    if not snapshots or common_allocated is None:
        raise RuntimeError("object has no inventoried aliases")
    first_signature = snapshots[0].signature
    if any(alias.signature != first_signature for alias in snapshots[1:]):
        raise FileChanged("aliases do not share one current inode signature")
    link_complete = (
        len(snapshots) == record.aliases_seen == first_signature.nlink
    )
    return MemberSnapshot(
        tuple(snapshots), first_signature, common_allocated, link_complete
    )


def pread_exact(handle: BinaryIO, offset: int, length: int) -> bytes:
    parts: list[bytes] = []
    cursor = offset
    remaining = length
    while remaining:
        part = os.pread(handle.fileno(), remaining, cursor)
        if not part:
            raise EOFError(f"short read offset={cursor}, remaining={remaining}")
        parts.append(part)
        cursor += len(part)
        remaining -= len(part)
    return b"".join(parts)


def hash_prefix(item: WorkItem) -> HashResult:
    handle, before, used_path = open_verified(item)
    with handle:
        length = min(item.size, PREFIX_BYTES)
        payload = pread_exact(handle, 0, length)
        digest = hashlib.sha256(
            b"exact-prefix-v1\0" + item.size.to_bytes(16, "little") + payload
        ).digest()
        finish_verified(handle, before)
    return HashResult(item.dev, item.ino, digest, before, length, used_path)


def hash_sample(item: WorkItem) -> HashResult:
    handle, before, used_path = open_verified(item)
    with handle:
        length = min(item.size, SAMPLE_BYTES)
        middle = max(0, item.size // 2 - length // 2)
        tail = max(0, item.size - length)
        digest = hashlib.sha256()
        digest.update(b"exact-mid-tail-v1\0")
        digest.update(item.size.to_bytes(16, "little"))
        total = 0
        for offset in dict.fromkeys((middle, tail)):
            payload = pread_exact(handle, offset, length)
            digest.update(offset.to_bytes(16, "little"))
            digest.update(payload)
            total += len(payload)
        finish_verified(handle, before)
    return HashResult(item.dev, item.ino, digest.digest(), before, total, used_path)


def hash_full(item: WorkItem) -> HashResult:
    handle, before, used_path = open_verified(item)
    digest = hashlib.sha256()
    total = 0
    with handle:
        while True:
            chunk = handle.read(IO_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        if total != item.size:
            raise EOFError(f"full read expected={item.size}, got={total}")
        finish_verified(handle, before)
    return HashResult(item.dev, item.ino, digest.digest(), before, total, used_path)


def configure_state(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA wal_autocheckpoint=10000")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta(
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS candidate_sizes(
          size INTEGER PRIMARY KEY
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS objects(
          source_rowid INTEGER PRIMARY KEY,
          dev INTEGER NOT NULL,
          ino INTEGER NOT NULL,
          size INTEGER NOT NULL,
          allocated INTEGER NOT NULL,
          nlink INTEGER NOT NULL,
          aliases_seen INTEGER NOT NULL,
          inventory_mtime_ns INTEGER NOT NULL,
          primary_path BLOB NOT NULL,
          sig_mtime_ns INTEGER,
          sig_ctime_ns INTEGER,
          sig_nlink INTEGER,
          prefix_hash BLOB CHECK(prefix_hash IS NULL OR length(prefix_hash)=32),
          sample_hash BLOB CHECK(sample_hash IS NULL OR length(sample_hash)=32),
          full_hash BLOB CHECK(full_hash IS NULL OR length(full_hash)=32),
          used_path BLOB,
          error_stage TEXT,
          error TEXT,
          attempts INTEGER NOT NULL DEFAULT 0,
          final_run INTEGER,
          final_ok INTEGER,
          final_error TEXT,
          UNIQUE(dev,ino)
        );
        CREATE TABLE IF NOT EXISTS paths(
          dev INTEGER NOT NULL,
          ino INTEGER NOT NULL,
          path BLOB NOT NULL,
          PRIMARY KEY(dev,ino,path)
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS idx_objects_prefix
          ON objects(size,prefix_hash)
          WHERE prefix_hash IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_objects_sample
          ON objects(size,prefix_hash,sample_hash)
          WHERE sample_hash IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_objects_full
          ON objects(size,full_hash)
          WHERE full_hash IS NOT NULL;
        CREATE TABLE IF NOT EXISTS comparison_groups(
          size INTEGER NOT NULL,
          full_hash BLOB NOT NULL CHECK(length(full_hash)=32),
          rep_dev INTEGER NOT NULL,
          rep_ino INTEGER NOT NULL,
          member_count INTEGER NOT NULL,
          complete INTEGER NOT NULL,
          PRIMARY KEY(size,full_hash)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS comparisons(
          size INTEGER NOT NULL,
          full_hash BLOB NOT NULL CHECK(length(full_hash)=32),
          rep_dev INTEGER NOT NULL,
          rep_ino INTEGER NOT NULL,
          member_dev INTEGER NOT NULL,
          member_ino INTEGER NOT NULL,
          exact INTEGER NOT NULL,
          error TEXT,
          PRIMARY KEY(size,full_hash,member_dev,member_ino),
          FOREIGN KEY(size,full_hash)
            REFERENCES comparison_groups(size,full_hash) ON DELETE CASCADE
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS idx_comparison_errors
          ON comparisons(size,full_hash)
          WHERE error IS NOT NULL;
        """
    )


def set_meta(conn: sqlite3.Connection, key: str, value: object) -> None:
    conn.execute(
        "INSERT INTO meta(key,value) VALUES(?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, json.dumps(value, ensure_ascii=False, sort_keys=True)),
    )


def get_meta(conn: sqlite3.Connection, key: str, default: object = None) -> object:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return default if row is None else json.loads(row[0])


def existing_config(conn: sqlite3.Connection) -> object:
    table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'"
    ).fetchone()
    if table is None:
        return None
    return get_meta(conn, "config")


def state_has_payload(conn: sqlite3.Connection) -> bool:
    tables = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    for table in (
        "meta", "candidate_sizes", "objects", "paths",
        "comparison_groups", "comparisons",
    ):
        if table in tables and conn.execute(
            f"SELECT 1 FROM {table} LIMIT 1"
        ).fetchone() is not None:
            return True
    return False


def sidecar_fingerprint(path: Path) -> dict[str, object]:
    try:
        st = path.stat()
    except FileNotFoundError:
        return {"empty": True}
    if st.st_size == 0:
        return {"empty": True}
    return {
        "dev": int(st.st_dev), "ino": int(st.st_ino), "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns), "ctime_ns": int(st.st_ctime_ns),
    }


def inventory_fingerprint(path: Path) -> dict[str, object]:
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "database": {
            "dev": int(st.st_dev), "ino": int(st.st_ino), "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns), "ctime_ns": int(st.st_ctime_ns),
        },
        "wal": sidecar_fingerprint(Path(str(path) + "-wal")),
    }


def inventory_uri(path: Path) -> str:
    return f"file:{quote(str(path.resolve()), safe='/')}?mode=ro"


def source_meta(conn: sqlite3.Connection, key: str) -> object:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    if row is None:
        return None
    try:
        return json.loads(row[0])
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"inventory meta {key!r} is invalid JSON") from exc


def validate_source(path: Path) -> tuple[int, dict[str, object]]:
    uri = inventory_uri(path)
    source = sqlite3.connect(uri, uri=True)
    try:
        source.execute("PRAGMA query_only=ON")
        source.execute("PRAGMA busy_timeout=30000")
        tables = {
            row[0] for row in source.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        required_tables = {
            "meta", "files", "inodes", "scan_errors", "skipped_mounts",
        }
        if not required_tables.issubset(tables):
            raise RuntimeError(
                f"inventory missing tables: {sorted(required_tables - tables)}"
            )
        if source_meta(source, "inventory_complete") is not True:
            raise RuntimeError("inventory meta inventory_complete is not true")
        if source_meta(source, "inode_table_complete") is not True:
            raise RuntimeError("inventory meta inode_table_complete is not true")
        file_columns = {row[1] for row in source.execute("PRAGMA table_info(files)")}
        inode_columns = {row[1] for row in source.execute("PRAGMA table_info(inodes)")}
        missing_files = {
            "path", "size", "allocated", "dev", "ino", "nlink", "mtime_ns",
        } - file_columns
        missing_inodes = {
            "path", "size", "allocated", "dev", "ino", "path_count",
        } - inode_columns
        if missing_files or missing_inodes:
            raise RuntimeError(
                "inventory schema incomplete: "
                f"files={sorted(missing_files)}, inodes={sorted(missing_inodes)}"
            )
        file_indexes = {
            row[1] for row in source.execute("PRAGMA index_list(files)")
        }
        inode_indexes = {
            row[1] for row in source.execute("PRAGMA index_list(inodes)")
        }
        if "idx_files_identity" not in file_indexes:
            raise RuntimeError("inventory lacks required idx_files_identity index")
        if "idx_inodes_size" not in inode_indexes:
            raise RuntimeError("inventory lacks required idx_inodes_size index")
        root = source_meta(source, "root")
        excluded = source_meta(source, "excluded")
        if not isinstance(root, str):
            raise RuntimeError("inventory meta root must be a string")
        if not isinstance(excluded, list) or not all(
            isinstance(item, str) for item in excluded
        ):
            raise RuntimeError("inventory meta excluded must be a string list")
        inventory_errors = int(
            source.execute("SELECT COUNT(*) FROM scan_errors").fetchone()[0]
        )
        skipped_mounts = [
            {"path": path_record(raw_path(item[0])), "dev": int(item[1])}
            for item in source.execute(
                "SELECT path,dev FROM skipped_mounts ORDER BY path"
            )
        ]
        details = {
            "inventory_complete": True,
            "inode_table_complete": True,
            "root": root,
            "excluded": [
                {
                    "display": item,
                    "raw_base64": None,
                    "source_encoding": "inventory_meta_display_only",
                }
                for item in excluded
            ],
            "excluded_count": len(excluded),
            "scan_error_count": inventory_errors,
            "skipped_mount_count": len(skipped_mounts),
            "skipped_mounts": skipped_mounts,
            "coverage_complete": not (
                inventory_errors or excluded or skipped_mounts
            ),
        }
        return inventory_errors, details
    finally:
        source.close()


def attach_inventory(state: sqlite3.Connection, inventory: Path) -> None:
    state.execute("ATTACH DATABASE ? AS inv", (inventory_uri(inventory),))


def initialize_candidates(
    state: sqlite3.Connection,
    inventory: Path,
    minimum: int,
    maximum: int | None,
) -> None:
    if get_meta(state, "candidate_init_complete", False) is True:
        return
    attach_inventory(state, inventory)
    try:
        if get_meta(state, "candidate_sizes_complete", False) is not True:
            state.execute("BEGIN IMMEDIATE")
            state.execute("DELETE FROM candidate_sizes")
            state.execute(
                "INSERT INTO candidate_sizes(size) "
                "SELECT size FROM inv.inodes INDEXED BY idx_inodes_size "
                "WHERE size>=? AND (? IS NULL OR size<?) "
                "GROUP BY size HAVING COUNT(*)>1",
                (minimum, maximum, maximum),
            )
            set_meta(state, "candidate_sizes_complete", True)
            state.commit()

        if get_meta(state, "candidate_objects_complete", False) is not True:
            last = int(get_meta(state, "objects_last_source_rowid", 0))
            while True:
                row = state.execute(
                    "SELECT MAX(source_rowid),COUNT(*) FROM ("
                    " SELECT i.rowid AS source_rowid "
                    " FROM inv.inodes AS i NOT INDEXED "
                    " WHERE i.rowid>? AND EXISTS("
                    "   SELECT 1 FROM candidate_sizes c WHERE c.size=i.size"
                    " ) ORDER BY i.rowid LIMIT ?"
                    ")",
                    (last, INIT_BATCH),
                ).fetchone()
                high = None if row is None else row[0]
                if high is None:
                    state.execute("BEGIN IMMEDIATE")
                    set_meta(state, "candidate_objects_complete", True)
                    state.commit()
                    break
                expected = int(row[1])
                state.execute("BEGIN IMMEDIATE")
                state.execute(
                    "INSERT INTO objects("
                    " source_rowid,dev,ino,size,allocated,nlink,aliases_seen,"
                    " inventory_mtime_ns,primary_path"
                    ") "
                    "SELECT i.rowid,i.dev,i.ino,i.size,i.allocated,f.nlink,"
                    " i.path_count,f.mtime_ns,CAST(i.path AS BLOB) "
                    "FROM inv.inodes AS i NOT INDEXED "
                    "JOIN inv.files AS f ON f.path=i.path "
                    " AND f.dev=i.dev AND f.ino=i.ino "
                    "WHERE i.rowid>? AND i.rowid<=? AND EXISTS("
                    " SELECT 1 FROM candidate_sizes c WHERE c.size=i.size) "
                    "ORDER BY i.rowid",
                    (last, high),
                )
                inserted = int(state.execute("SELECT changes()").fetchone()[0])
                if inserted != expected:
                    state.rollback()
                    raise RuntimeError(
                        "inventory primary-path join inconsistent: "
                        f"expected={expected}, inserted={inserted}"
                    )
                set_meta(state, "objects_last_source_rowid", int(high))
                state.commit()
                last = int(high)
                print(f"[init objects] source_rowid={last:,}", flush=True)

        if get_meta(state, "candidate_aliases_complete", False) is not True:
            last = int(get_meta(state, "aliases_last_source_rowid", 0))
            while True:
                row = state.execute(
                    "SELECT MAX(source_rowid),COALESCE(SUM(aliases_seen-1),0) FROM ("
                    " SELECT source_rowid,aliases_seen FROM objects WHERE source_rowid>? "
                    " ORDER BY source_rowid LIMIT ?"
                    ")",
                    (last, INIT_BATCH),
                ).fetchone()
                high = None if row is None else row[0]
                if high is None:
                    state.execute("BEGIN IMMEDIATE")
                    set_meta(state, "candidate_aliases_complete", True)
                    state.commit()
                    break
                high = int(high)
                expected = int(row[1])
                state.execute("BEGIN IMMEDIATE")
                state.execute(
                    "INSERT INTO paths(dev,ino,path) "
                    "SELECT o.dev,o.ino,CAST(f.path AS BLOB) "
                    "FROM objects AS o "
                    "JOIN inv.files AS f INDEXED BY idx_files_identity "
                    " ON f.dev=o.dev AND f.ino=o.ino "
                    "WHERE o.source_rowid>? AND o.source_rowid<=? "
                    " AND o.aliases_seen>1 AND CAST(f.path AS BLOB)<>o.primary_path",
                    (last, high),
                )
                inserted = int(state.execute("SELECT changes()").fetchone()[0])
                if inserted != expected:
                    state.rollback()
                    raise RuntimeError(
                        "inventory alias count inconsistent: "
                        f"expected={expected}, inserted={inserted}"
                    )
                set_meta(state, "aliases_last_source_rowid", high)
                state.commit()
                last = high
                print(f"[init aliases] source_rowid={last:,}", flush=True)

        state.execute("BEGIN IMMEDIATE")
        set_meta(
            state,
            "candidate_objects",
            int(state.execute("SELECT COUNT(*) FROM objects").fetchone()[0]),
        )
        set_meta(state, "candidate_init_complete", True)
        state.commit()
    except Exception:
        state.rollback()
        raise
    finally:
        state.execute("DETACH DATABASE inv")
    print(
        f"[init complete] objects={int(get_meta(state, 'candidate_objects', 0)):,}",
        flush=True,
    )


def iter_objects(
    state: sqlite3.Connection,
    where_sql: str,
    params: tuple[object, ...] = (),
    page_size: int = WORK_PAGE,
) -> Iterator[ObjectRecord]:
    """Keyset-page objects and collect aliases with one query per page."""
    last = 0
    while True:
        rows = state.execute(
            "WITH batch AS MATERIALIZED ("
            " SELECT o.source_rowid,o.dev,o.ino,o.size,o.allocated,o.nlink,"
            " o.aliases_seen,o.inventory_mtime_ns,o.primary_path,"
            " o.sig_mtime_ns,o.sig_ctime_ns,o.sig_nlink,o.prefix_hash,o.sample_hash,"
            " o.full_hash,o.error_stage,o.error "
            " FROM objects AS o WHERE o.source_rowid>? AND (" + where_sql + ") "
            " ORDER BY o.source_rowid LIMIT ?"
            ") "
            "SELECT b.source_rowid,b.dev,b.ino,b.size,b.allocated,b.nlink,"
            " b.aliases_seen,b.inventory_mtime_ns,b.primary_path,"
            " b.sig_mtime_ns,b.sig_ctime_ns,b.sig_nlink,b.prefix_hash,b.sample_hash,"
            " b.full_hash,b.error_stage,b.error,p.path "
            "FROM batch AS b LEFT JOIN paths AS p "
            " ON p.dev=b.dev AND p.ino=b.ino "
            "ORDER BY b.source_rowid,p.path",
            (last, *params, page_size),
        ).fetchall()
        if not rows:
            return
        page: list[ObjectRecord] = []
        index = 0
        while index < len(rows):
            first = rows[index]
            source_rowid = int(first[0])
            paths = [raw_path(first[8])]
            while index < len(rows) and int(rows[index][0]) == source_rowid:
                alternate = rows[index][17]
                if alternate is not None:
                    alternate_raw = raw_path(alternate)
                    if alternate_raw not in paths:
                        paths.append(alternate_raw)
                index += 1
            page.append(
                ObjectRecord(
                    source_rowid=source_rowid,
                    dev=int(first[1]), ino=int(first[2]), size=int(first[3]),
                    allocated=int(first[4]), nlink=int(first[5]),
                    aliases_seen=int(first[6]), inventory_mtime_ns=int(first[7]),
                    sig_mtime_ns=first[9], sig_ctime_ns=first[10], sig_nlink=first[11],
                    prefix_hash=first[12], sample_hash=first[13], full_hash=first[14],
                    error_stage=first[15], error=first[16], paths=tuple(paths),
                )
            )
        last = page[-1].source_rowid
        yield from page


def iter_work_items(state: sqlite3.Connection, stage: str) -> Iterator[WorkItem]:
    if stage == "prefix":
        where = "o.prefix_hash IS NULL"
    elif stage == "sample":
        where = (
            "o.sample_hash IS NULL AND o.prefix_hash IS NOT NULL AND EXISTS("
            " SELECT 1 FROM objects AS peer INDEXED BY idx_objects_prefix"
            " WHERE peer.size=o.size AND peer.prefix_hash=o.prefix_hash"
            " AND peer.source_rowid<>o.source_rowid"
            ")"
        )
    elif stage == "full":
        where = (
            "o.full_hash IS NULL AND o.sample_hash IS NOT NULL AND EXISTS("
            " SELECT 1 FROM objects AS peer INDEXED BY idx_objects_sample"
            " WHERE peer.size=o.size AND peer.prefix_hash=o.prefix_hash"
            " AND peer.sample_hash=o.sample_hash"
            " AND peer.source_rowid<>o.source_rowid"
            ")"
        )
    elif stage == "final":
        where = (
            "o.prefix_hash IS NOT NULL AND o.sig_mtime_ns IS NOT NULL"
            " AND o.sig_ctime_ns IS NOT NULL AND o.sig_nlink IS NOT NULL"
        )
    else:
        raise ValueError(stage)
    for record in iter_objects(state, where):
        yield record.work_item()


def discard_object_results(
    state: sqlite3.Connection,
    dev: int,
    ino: int,
    stage: str,
    error: str,
) -> None:
    set_meta(state, "comparison_pass_complete", False)
    set_meta(state, "final_sweep_complete", False)
    state.execute(
        "DELETE FROM comparison_groups WHERE "
        "(rep_dev=? AND rep_ino=?) OR EXISTS("
        " SELECT 1 FROM comparisons c WHERE "
        " c.size=comparison_groups.size AND c.full_hash=comparison_groups.full_hash"
        " AND c.member_dev=? AND c.member_ino=?"
        ")",
        (dev, ino, dev, ino),
    )
    state.execute(
        "UPDATE objects SET prefix_hash=NULL,sample_hash=NULL,full_hash=NULL,"
        "sig_mtime_ns=NULL,sig_ctime_ns=NULL,sig_nlink=NULL,used_path=NULL,"
        "error_stage=?,error=?,attempts=attempts+1,"
        "final_ok=0,final_error=? WHERE dev=? AND ino=?",
        (stage, error, error, dev, ino),
    )


def run_hash_stage(
    state: sqlite3.Connection,
    stage: str,
    items: Iterable[WorkItem],
    function: Callable[[WorkItem], HashResult],
    workers: int,
) -> dict[str, int]:
    started = time.time()
    done = read_bytes = errors = 0
    iterator = iter(items)
    inflight: dict[Future[HashResult], WorkItem] = {}
    limit = max(8, workers * 4)
    exhausted = False

    def fill(pool: ThreadPoolExecutor) -> None:
        nonlocal exhausted
        while not exhausted and len(inflight) < limit:
            try:
                item = next(iterator)
            except StopIteration:
                exhausted = True
                return
            inflight[pool.submit(function, item)] = item

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=stage) as pool:
        fill(pool)
        last_report = time.time()
        while inflight:
            completed, _ = wait(inflight, timeout=30, return_when=FIRST_COMPLETED)
            for future in completed:
                item = inflight.pop(future)
                try:
                    result = future.result()
                    column = {
                        "prefix": "prefix_hash",
                        "sample": "sample_hash",
                        "full": "full_hash",
                    }[stage]
                    if stage == "prefix":
                        state.execute(
                            f"UPDATE objects SET {column}=?,sig_mtime_ns=?,sig_ctime_ns=?,"
                            "sig_nlink=?,"
                            "used_path=?,error_stage=NULL,error=NULL,attempts=attempts+1 "
                            "WHERE dev=? AND ino=?",
                            (
                                result.digest, result.signature.mtime_ns,
                                result.signature.ctime_ns, result.signature.nlink, result.path,
                                result.dev, result.ino,
                            ),
                        )
                    else:
                        state.execute(
                            f"UPDATE objects SET {column}=?,used_path=?,"
                            "error_stage=NULL,error=NULL,attempts=attempts+1 "
                            "WHERE dev=? AND ino=?",
                            (result.digest, result.path, result.dev, result.ino),
                        )
                    read_bytes += result.bytes_read
                except Exception as exc:
                    discard_object_results(
                        state, item.dev, item.ino, stage, repr(exc)
                    )
                    errors += 1
                done += 1
            if completed:
                state.commit()
                fill(pool)
            now = time.time()
            if now - last_report >= 30:
                elapsed = max(now - started, 1e-6)
                print(
                    f"[{stage}] done={done:,} read={human_bytes(read_bytes)} "
                    f"rate={human_bytes(int(read_bytes/elapsed))}/s "
                    f"inflight={len(inflight):,} errors={errors:,}",
                    flush=True,
                )
                last_report = now
    state.commit()
    print(
        f"[{stage} complete] done={done:,} read={human_bytes(read_bytes)} "
        f"errors={errors:,}",
        flush=True,
    )
    return {"processed": done, "bytes_read": read_bytes, "errors": errors}


def compare_pair(rep: WorkItem, member: WorkItem) -> tuple[bool, str | None, int]:
    total = 0
    try:
        with contextlib.ExitStack() as stack:
            rep_handle, rep_before, _ = open_verified(rep)
            stack.enter_context(rep_handle)
            mem_handle, mem_before, _ = open_verified(member)
            stack.enter_context(mem_handle)
            mismatch = False
            while True:
                left = rep_handle.read(IO_CHUNK)
                right = mem_handle.read(IO_CHUNK)
                total += len(left) + len(right)
                if left != right:
                    mismatch = True
                    break
                if not left:
                    break
            finish_verified(rep_handle, rep_before)
            finish_verified(mem_handle, mem_before)
            if mismatch:
                return False, "stable byte mismatch after equal SHA-256", total
        return True, None, total
    except Exception as exc:
        return False, repr(exc), total


def iter_full_groups(
    state: sqlite3.Connection,
) -> Iterator[tuple[int, bytes, int]]:
    last_size = -1
    last_hash = b""
    while True:
        rows = state.execute(
            "SELECT size,full_hash,COUNT(*) FROM objects "
            "WHERE full_hash IS NOT NULL AND "
            "(size>? OR (size=? AND full_hash>?)) "
            "GROUP BY size,full_hash HAVING COUNT(*)>1 "
            "ORDER BY size,full_hash LIMIT ?",
            (last_size, last_size, last_hash, GROUP_PAGE),
        ).fetchall()
        if not rows:
            return
        for size, full_hash, count in rows:
            yield int(size), bytes(full_hash), int(count)
        last_size = int(rows[-1][0])
        last_hash = bytes(rows[-1][1])


def compare_groups(state: sqlite3.Connection) -> dict[str, int]:
    set_meta(state, "comparison_pass_complete", False)
    state.commit()
    processed = cached = errors = 0
    for size, full_hash, count in iter_full_groups(state):
        rep_row = state.execute(
            "SELECT dev,ino FROM objects WHERE size=? AND full_hash=? "
            "ORDER BY source_rowid LIMIT 1",
            (size, full_hash),
        ).fetchone()
        if rep_row is None:
            continue
        rep_dev, rep_ino = int(rep_row[0]), int(rep_row[1])
        prior = state.execute(
            "SELECT rep_dev,rep_ino,member_count,complete FROM comparison_groups "
            "WHERE size=? AND full_hash=?",
            (size, full_hash),
        ).fetchone()
        prior_errors = 0
        prior_members = prior_current_members = 0
        if prior is not None:
            prior_counts = state.execute(
                "SELECT COUNT(*),COALESCE(SUM(error IS NOT NULL),0),"
                "COALESCE(SUM(EXISTS("
                " SELECT 1 FROM objects o WHERE o.dev=comparisons.member_dev"
                " AND o.ino=comparisons.member_ino AND o.size=comparisons.size"
                " AND o.full_hash=comparisons.full_hash"
                ")),0) FROM comparisons WHERE size=? AND full_hash=?",
                (size, full_hash),
            ).fetchone()
            prior_members = int(prior_counts[0])
            prior_errors = int(prior_counts[1])
            prior_current_members = int(prior_counts[2])
        if (
            prior is not None
            and (int(prior[0]), int(prior[1])) == (rep_dev, rep_ino)
            and int(prior[2]) == count
            and int(prior[3]) == 1
            and prior_errors == 0
            and prior_members == count - 1
            and prior_current_members == count - 1
        ):
            cached += 1
            continue

        state.execute(
            "DELETE FROM comparison_groups WHERE size=? AND full_hash=?",
            (size, full_hash),
        )
        state.execute(
            "INSERT INTO comparison_groups("
            "size,full_hash,rep_dev,rep_ino,member_count,complete"
            ") VALUES(?,?,?,?,?,0)",
            (size, full_hash, rep_dev, rep_ino, count),
        )
        state.commit()
        members = iter_objects(
            state,
            "o.size=? AND o.full_hash=?",
            (size, full_hash),
        )
        try:
            rep = next(members).work_item()
        except StopIteration as exc:
            raise RuntimeError("full-hash group disappeared") from exc
        compared = 1
        for record in members:
            exact, error, _ = compare_pair(rep, record.work_item())
            state.execute(
                "INSERT INTO comparisons("
                "size,full_hash,rep_dev,rep_ino,member_dev,member_ino,exact,error"
                ") VALUES(?,?,?,?,?,?,?,?)",
                (
                    size, full_hash, rep.dev, rep.ino,
                    record.dev, record.ino, int(exact), error,
                ),
            )
            compared += 1
            if error is not None:
                errors += 1
            if compared % 100 == 0:
                state.commit()
        if compared != count:
            state.commit()
            raise RuntimeError(
                f"full-hash group changed during comparison: expected={count}, got={compared}"
            )
        state.execute(
            "UPDATE comparison_groups SET complete=1 WHERE size=? AND full_hash=?",
            (size, full_hash),
        )
        state.commit()
        processed += 1
        print(
            f"[compare] groups={processed:,} cached={cached:,} "
            f"size={human_bytes(size)} members={count:,}",
            flush=True,
        )
    set_meta(state, "comparison_pass_complete", True)
    state.commit()
    print(
        f"[compare complete] processed={processed:,} cached={cached:,} errors={errors:,}",
        flush=True,
    )
    return {"processed_groups": processed, "cached_groups": cached, "errors": errors}


def run_final_sweep(state: sqlite3.Connection) -> dict[str, int]:
    run_id = int(get_meta(state, "last_final_run", 0)) + 1
    set_meta(state, "last_final_run", run_id)
    set_meta(state, "final_sweep_complete", False)
    state.commit()
    checked = changed = 0
    final_where = (
        "o.prefix_hash IS NOT NULL AND o.sig_mtime_ns IS NOT NULL"
        " AND o.sig_ctime_ns IS NOT NULL AND o.sig_nlink IS NOT NULL"
    )
    for record in iter_objects(state, final_where):
        item = record.work_item()
        try:
            snapshot = verify_all_aliases(record)
            state.execute(
                "UPDATE objects SET final_run=?,final_ok=1,final_error=NULL,used_path=? "
                "WHERE dev=? AND ino=?",
                (run_id, snapshot.aliases[0].path, item.dev, item.ino),
            )
        except Exception as exc:
            error = repr(exc)
            discard_object_results(state, item.dev, item.ino, "final_stat", error)
            state.execute(
                "UPDATE objects SET final_run=?,final_ok=0,final_error=? "
                "WHERE dev=? AND ino=?",
                (run_id, error, item.dev, item.ino),
            )
            changed += 1
        checked += 1
        if checked % 1000 == 0:
            state.commit()
            print(
                f"[final stat] checked={checked:,} changed={changed:,}",
                flush=True,
            )
    state.commit()
    missing = int(
        state.execute(
            "SELECT COUNT(*) FROM objects WHERE prefix_hash IS NOT NULL AND "
            "(final_run IS NULL OR final_run<>? OR final_ok<>1)",
            (run_id,),
        ).fetchone()[0]
    )
    stats = {
        "run_id": run_id,
        "checked": checked,
        "changed": changed,
        "missing": missing,
        "all_inventoried_aliases_checked": changed == 0 and missing == 0,
        "all_inventoried_aliases_verified": changed == 0 and missing == 0,
    }
    set_meta(state, "final_sweep_stats", stats)
    set_meta(state, "final_sweep_complete", missing == 0)
    state.commit()
    print(
        f"[final stat complete] checked={checked:,} changed={changed:,} missing={missing:,}",
        flush=True,
    )
    return stats


def iter_report_groups(
    state: sqlite3.Connection,
    final_run: int,
) -> Iterator[tuple[int, bytes, int, int, int]]:
    last_size = -1
    last_hash = b""
    while True:
        rows = state.execute(
            "SELECT g.size,g.full_hash,g.rep_dev,g.rep_ino,"
            "1+COUNT(c.member_dev) AS members "
            "FROM comparison_groups AS g "
            "JOIN objects AS r ON r.dev=g.rep_dev AND r.ino=g.rep_ino "
            " AND r.size=g.size AND r.full_hash=g.full_hash "
            " AND r.final_run=? AND r.final_ok=1 "
            "JOIN comparisons AS c ON c.size=g.size AND c.full_hash=g.full_hash "
            " AND c.exact=1 AND c.error IS NULL "
            "JOIN objects AS m ON m.dev=c.member_dev AND m.ino=c.member_ino "
            " AND m.size=g.size AND m.full_hash=g.full_hash "
            " AND m.final_run=? AND m.final_ok=1 "
            "WHERE g.complete=1 AND (g.size>? OR (g.size=? AND g.full_hash>?)) "
            "GROUP BY g.size,g.full_hash,g.rep_dev,g.rep_ino "
            "ORDER BY g.size,g.full_hash LIMIT ?",
            (final_run, final_run, last_size, last_size, last_hash, GROUP_PAGE),
        ).fetchall()
        if not rows:
            return
        for row in rows:
            yield (
                int(row[0]), bytes(row[1]), int(row[2]), int(row[3]), int(row[4]),
            )
        last_size = int(rows[-1][0])
        last_hash = bytes(rows[-1][1])


def report_member_where() -> str:
    return (
        "o.size=? AND o.full_hash=? AND o.final_run=? AND o.final_ok=1 AND ("
        " (o.dev=? AND o.ino=?) OR EXISTS("
        "  SELECT 1 FROM comparisons c WHERE c.size=? AND c.full_hash=?"
        "  AND c.member_dev=o.dev AND c.member_ino=o.ino"
        "  AND c.exact=1 AND c.error IS NULL"
        " ))"
    )


def fsync_text(handle: object) -> None:
    handle.flush()  # type: ignore[attr-defined]
    os.fsync(handle.fileno())  # type: ignore[attr-defined]


def fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(IO_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_json_line(handle: object, value: object) -> None:
    handle.write(  # type: ignore[attr-defined]
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
    )


def generation_directory(out: Path) -> tuple[str, Path, Path]:
    root = out / "generations"
    root.mkdir(mode=0o700, exist_ok=True)
    fsync_directory(out)
    for suffix in range(1000):
        generation_id = f"{time.time_ns()}-{os.getpid()}-{suffix}"
        temporary = root / f".{generation_id}.tmp"
        final = root / generation_id
        try:
            temporary.mkdir(mode=0o700)
            return generation_id, temporary, final
        except FileExistsError:
            continue
    raise RuntimeError("unable to allocate a unique report generation")


def alias_snapshot_record(alias: AliasSnapshot) -> dict[str, object]:
    result: dict[str, object] = path_record(alias.path)
    result.update({
        "verified": True,
        "validated_at_unix_ns": alias.validated_at_ns,
        "dev": alias.signature.dev,
        "ino": alias.signature.ino,
        "size_bytes": alias.signature.size,
        "mtime_ns": alias.signature.mtime_ns,
        "ctime_ns": alias.signature.ctime_ns,
        "nlink": alias.signature.nlink,
        "allocated_bytes": alias.allocated,
    })
    return result


def member_snapshot_record(
    record: ObjectRecord,
    snapshot: MemberSnapshot,
) -> dict[str, object]:
    return {
        "dev": record.dev,
        "ino": record.ino,
        "allocated_bytes": snapshot.allocated,
        "inventory_allocated_bytes": record.allocated,
        "nlink": snapshot.signature.nlink,
        "inventory_nlink": record.nlink,
        "aliases_seen": record.aliases_seen,
        "verified_aliases": len(snapshot.aliases),
        "link_complete": snapshot.link_complete,
        "stable_at_report": True,
        "paths": [alias_snapshot_record(alias) for alias in snapshot.aliases],
    }


def identity_paths(state: sqlite3.Connection, dev: int, ino: int) -> list[dict[str, str]]:
    row = state.execute(
        "SELECT primary_path FROM objects WHERE dev=? AND ino=?", (dev, ino)
    ).fetchone()
    if row is None:
        return []
    paths = [raw_path(row[0])]
    paths.extend(
        raw_path(item[0]) for item in state.execute(
            "SELECT path FROM paths WHERE dev=? AND ino=? ORDER BY path", (dev, ino)
        )
    )
    return [path_record(path) for path in paths]


def write_inventory_error_records(handle: object, inventory: Path) -> int:
    source = sqlite3.connect(inventory_uri(inventory), uri=True)
    count = 0
    try:
        source.execute("PRAGMA query_only=ON")
        source.execute("PRAGMA busy_timeout=30000")
        cursor = source.execute(
            "SELECT stage,path,error FROM scan_errors ORDER BY stage,path"
        )
        while True:
            rows = cursor.fetchmany(WORK_PAGE)
            if not rows:
                break
            for stage, path, error in rows:
                write_json_line(handle, {
                    "stage": f"inventory:{stage}",
                    "path": None if path is None else path_record(raw_path(path)),
                    "error": error,
                })
                count += 1
    finally:
        source.close()
    return count


def emit_reports(
    state: sqlite3.Connection,
    out: Path,
    config: dict[str, object],
    inventory: Path,
    expected_inventory_fingerprint: dict[str, object],
    inventory_errors: int,
    source_details: dict[str, object],
) -> dict[str, object]:
    generated_started_ns = time.time_ns()
    final_stats = get_meta(state, "final_sweep_stats", {})
    final_run = int(final_stats.get("run_id", 0))
    generation_id, generation_tmp, generation_final = generation_directory(out)
    duplicates_tmp = generation_tmp / "exact_duplicates.jsonl"
    errors_tmp = generation_tmp / "errors.jsonl"
    duplicates_path = generation_final / duplicates_tmp.name
    errors_path = generation_final / errors_tmp.name
    group_count = logical_reclaim = actionable_reclaim = 0
    comparison_errors = report_validation_errors = 0
    inventory_unchanged_start = (
        inventory_fingerprint(inventory) == expected_inventory_fingerprint
    )
    with (
        duplicates_tmp.open("w", encoding="utf-8") as duplicates_handle,
        errors_tmp.open("w", encoding="utf-8") as errors_handle,
    ):
        written_inventory_errors = write_inventory_error_records(
            errors_handle, inventory
        )
        inventory_error_count_consistent = (
            written_inventory_errors == inventory_errors
        )
        if not inventory_error_count_consistent:
            write_json_line(errors_handle, {
                "stage": "inventory",
                "error": "inventory error count changed during generation",
                "expected": inventory_errors,
                "observed": written_inventory_errors,
            })
        if not inventory_unchanged_start:
            write_json_line(errors_handle, {
                "stage": "inventory_fingerprint",
                "error": "inventory database or WAL changed before report generation",
            })
        for excluded in source_details.get("excluded", []):
            write_json_line(errors_handle, {
                "stage": "inventory_excluded",
                "path": excluded,
                "error": "subtree explicitly excluded from inventory",
            })
        for skipped in source_details.get("skipped_mounts", []):
            write_json_line(errors_handle, {
                "stage": "inventory_skipped_mount",
                **skipped,
                "error": "different filesystem skipped by inventory",
            })
        for record in iter_objects(state, "o.error IS NOT NULL"):
            write_json_line(errors_handle, {
                "stage": record.error_stage,
                "error": record.error,
                "dev": record.dev,
                "ino": record.ino,
                "paths": [path_record(path) for path in record.paths],
            })
        cursor = state.execute(
            "SELECT size,full_hash,rep_dev,rep_ino,member_dev,member_ino,error "
            "FROM comparisons WHERE error IS NOT NULL "
            "ORDER BY size,full_hash,member_dev,member_ino"
        )
        while True:
            rows = cursor.fetchmany(WORK_PAGE)
            if not rows:
                break
            for row in rows:
                comparison_errors += 1
                write_json_line(errors_handle, {
                    "stage": "compare",
                    "size_bytes": int(row[0]),
                    "sha256": bytes(row[1]).hex(),
                    "representative": {
                        "dev": int(row[2]), "ino": int(row[3]),
                        "paths": identity_paths(state, int(row[2]), int(row[3])),
                    },
                    "member": {
                        "dev": int(row[4]), "ino": int(row[5]),
                        "paths": identity_paths(state, int(row[4]), int(row[5])),
                    },
                    "error": row[6],
                })

        members_scratch_path = generation_tmp / ".members.json.tmp"
        with members_scratch_path.open("w+", encoding="utf-8") as members_scratch:
            for size, full_hash, rep_dev, rep_ino, count in iter_report_groups(
                state, final_run
            ):
                members_scratch.seek(0)
                members_scratch.truncate()
                params: tuple[object, ...] = (
                    size, full_hash, final_run, rep_dev, rep_ino, size, full_hash,
                )
                records = iter_objects(state, report_member_where(), params)
                emitted = reclaimable_sum = 0
                reclaimable_min: int | None = None
                validation_error: tuple[ObjectRecord, Exception] | None = None
                for record in records:
                    try:
                        snapshot = verify_all_aliases(record)
                    except Exception as exc:
                        validation_error = (record, exc)
                        records.close()
                        break
                    reclaimable = snapshot.allocated if snapshot.link_complete else 0
                    reclaimable_sum += reclaimable
                    reclaimable_min = (
                        reclaimable
                        if reclaimable_min is None
                        else min(reclaimable_min, reclaimable)
                    )
                    if emitted:
                        members_scratch.write(",")
                    members_scratch.write(json.dumps(
                        member_snapshot_record(record, snapshot),
                        ensure_ascii=False,
                        sort_keys=True,
                    ))
                    emitted += 1
                if validation_error is not None:
                    report_validation_errors += 1
                    record, error = validation_error
                    write_json_line(errors_handle, {
                        "stage": "report_validation",
                        "size_bytes": size,
                        "sha256": full_hash.hex(),
                        "dev": record.dev,
                        "ino": record.ino,
                        "error": repr(error),
                        "paths": [path_record(path) for path in record.paths],
                    })
                    continue
                if emitted != count or reclaimable_min is None:
                    report_validation_errors += 1
                    write_json_line(errors_handle, {
                        "stage": "report_validation",
                        "size_bytes": size,
                        "sha256": full_hash.hex(),
                        "error": (
                            "group membership changed during generation: "
                            f"expected={count}, got={emitted}"
                        ),
                    })
                    continue
                group_actionable = reclaimable_sum - reclaimable_min
                group_logical = size * (count - 1)
                group_count += 1
                logical_reclaim += group_logical
                actionable_reclaim += group_actionable
                header = {
                    "group_id": group_count,
                    "generation_id": generation_id,
                    "size_bytes": size,
                    "sha256": full_hash.hex(),
                    "byte_verified_exact": True,
                    "distinct_inode_count": count,
                    "logical_reclaim_bytes": group_logical,
                    "actionable_allocated_reclaim_bytes": group_actionable,
                }
                encoded_header = json.dumps(
                    header, ensure_ascii=False, sort_keys=True
                )
                duplicates_handle.write(encoded_header[:-1] + ', "members": [')
                members_scratch.flush()
                members_scratch.seek(0)
                while True:
                    chunk = members_scratch.read(1024 * 1024)
                    if not chunk:
                        break
                    duplicates_handle.write(chunk)
                duplicates_handle.write("]}\n")
        members_scratch_path.unlink()

        inventory_unchanged_end = (
            inventory_fingerprint(inventory) == expected_inventory_fingerprint
        )
        if not inventory_unchanged_end and inventory_unchanged_start:
            write_json_line(errors_handle, {
                "stage": "inventory_fingerprint",
                "error": "inventory database or WAL changed during report generation",
            })
        fsync_text(duplicates_handle)
        fsync_text(errors_handle)

    counts = state.execute(
        "SELECT COUNT(*),"
        "COALESCE(SUM(prefix_hash IS NOT NULL),0),"
        "COALESCE(SUM(sample_hash IS NOT NULL),0),"
        "COALESCE(SUM(full_hash IS NOT NULL),0),"
        "COALESCE(SUM(error IS NOT NULL),0),"
        "COALESCE(SUM(prefix_hash IS NULL AND error IS NULL),0) FROM objects"
    ).fetchone()
    candidate_count = int(counts[0])
    prefix_count, sample_count, full_count = map(int, counts[1:4])
    object_errors, pending_prefix = map(int, counts[4:6])
    final_complete = get_meta(state, "final_sweep_complete", False) is True
    comparison_complete = get_meta(state, "comparison_pass_complete", False) is True
    init_complete = get_meta(state, "candidate_init_complete", False) is True
    pipeline_incomplete = not (
        init_complete and comparison_complete and final_complete and pending_prefix == 0
        and report_validation_errors == 0
    )
    source_coverage_complete = source_details.get("coverage_complete") is True
    inventory_unchanged = inventory_unchanged_start and inventory_unchanged_end
    status = "COMPLETE" if not (
        inventory_errors or not inventory_unchanged or object_errors
        or comparison_errors or report_validation_errors or pipeline_incomplete
        or not source_coverage_complete or not inventory_error_count_consistent
    ) else "PARTIAL"
    generated_finished_ns = time.time_ns()
    duplicate_artifact = {
        "path": str(duplicates_path),
        "sha256": file_sha256(duplicates_tmp),
        "size_bytes": duplicates_tmp.stat().st_size,
    }
    error_artifact = {
        "path": str(errors_path),
        "sha256": file_sha256(errors_tmp),
        "size_bytes": errors_tmp.stat().st_size,
    }
    summary: dict[str, object] = {
        "manifest_version": 1,
        "generation_id": generation_id,
        "status": status,
        "algorithm": ALGORITHM,
        "scope": config,
        "size_interval": "min_size <= size < max_size; max_size null means unbounded",
        "candidate_objects": candidate_count,
        "prefix_hashed": prefix_count,
        "sample_hashed": sample_count,
        "full_hashed": full_count,
        "exact_duplicate_groups": group_count,
        "logical_reclaim_bytes": logical_reclaim,
        "logical_reclaim_human": human_bytes(logical_reclaim),
        "actionable_allocated_reclaim_bytes": actionable_reclaim,
        "actionable_allocated_reclaim_human": human_bytes(actionable_reclaim),
        "inventory_error_count": inventory_errors,
        "inventory_error_records_written": written_inventory_errors,
        "inventory_error_count_consistent": inventory_error_count_consistent,
        "inventory_skipped_mount_count": int(
            source_details.get("skipped_mount_count", 0)
        ),
        "inventory_excluded_count": int(source_details.get("excluded_count", 0)),
        "source_coverage_complete": source_coverage_complete,
        "inventory_unchanged_during_run": inventory_unchanged,
        "object_error_count": object_errors,
        "comparison_error_count": comparison_errors,
        "report_validation_error_count": report_validation_errors,
        "pending_prefix_count": pending_prefix,
        "final_sweep": final_stats,
        "pipeline_incomplete": pipeline_incomplete,
        "deletion_or_deduplication_performed": False,
        "generation_consistency": {
            "started_at_unix_ns": generated_started_ns,
            "finished_at_unix_ns": generated_finished_ns,
            "source_files_frozen": False,
            "contract": (
                "every reported member alias was opened with O_NOFOLLOW and its "
                "dev/ino/size/mtime/ctime/nlink/allocation was stable across "
                "report-time validation; later external mutation remains possible"
            ),
        },
        "reports": {
            "duplicates": duplicate_artifact,
            "errors": error_artifact,
            "generation_manifest": str(generation_final / "manifest.json"),
        },
    }
    generation_manifest_tmp = generation_tmp / "manifest.json"
    with generation_manifest_tmp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        fsync_text(handle)
    fsync_directory(generation_tmp)
    os.replace(generation_tmp, generation_final)
    fsync_directory(generation_final.parent)

    summary_path = out / "summary.json"
    summary_tmp = out / f".summary.{generation_id}.tmp"
    with summary_tmp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        fsync_text(handle)
    os.replace(summary_tmp, summary_path)
    fsync_directory(out)
    return summary


def parse_size(value: str) -> int:
    units = {"k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}
    lowered = value.strip().lower()
    if lowered[-1:] in units:
        return int(float(lowered[:-1]) * units[lowered[-1]])
    return int(lowered)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Streaming exact-duplicate verification from an inventory DB"
    )
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-size", type=parse_size, default=1024**3)
    parser.add_argument(
        "--max-size", type=parse_size,
        help="exclusive upper bound; omit for no upper bound",
    )
    parser.add_argument("--sample-workers", type=int, default=16)
    parser.add_argument("--full-workers", type=int, default=2)
    return parser.parse_args()


def validate_args(args: argparse.Namespace, inventory: Path, out: Path) -> None:
    if args.min_size <= 0:
        raise RuntimeError("--min-size must be >0")
    if args.max_size is not None and args.max_size <= args.min_size:
        raise RuntimeError("--max-size must be greater than --min-size")
    if not 1 <= args.sample_workers <= MAX_SAMPLE_WORKERS:
        raise RuntimeError(f"--sample-workers must be 1..{MAX_SAMPLE_WORKERS}")
    if not 1 <= args.full_workers <= MAX_FULL_WORKERS:
        raise RuntimeError(f"--full-workers must be 1..{MAX_FULL_WORKERS}")
    if inventory == out / "state.sqlite3":
        raise RuntimeError("state database would overwrite inventory")
    if inventory.parent == out:
        raise RuntimeError("--out must not be inventory database directory")


def prepare_output(out: Path) -> tuple[Path, bool]:
    state_path = out / "state.sqlite3"
    existed = state_path.exists()
    if out.exists() and not out.is_dir():
        raise RuntimeError("--out exists and is not a directory")
    if out.exists() and not existed:
        unexpected = [entry for entry in out.iterdir() if entry.name != ".audit.lock"]
        if unexpected:
            raise RuntimeError("new --out directory is not empty")
    out.mkdir(parents=True, exist_ok=True)
    return state_path, existed


@contextlib.contextmanager
def output_lock(out: Path) -> Iterator[None]:
    descriptor = os.open(
        out / ".audit.lock", os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0), 0o600
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another audit process holds the output lock") from exc
        yield
    finally:
        os.close(descriptor)


def main() -> int:
    args = parse_args()
    inventory = Path(args.inventory).resolve()
    out = Path(args.out).resolve()
    validate_args(args, inventory, out)
    initial_fingerprint = inventory_fingerprint(inventory)
    inventory_errors, source_details = validate_source(inventory)
    if inventory_fingerprint(inventory) != initial_fingerprint:
        raise RuntimeError("inventory changed while validating source")

    config: dict[str, object] = {
        "algorithm": ALGORITHM,
        "schema_version": SCHEMA_VERSION,
        "inventory": initial_fingerprint,
        "source": source_details,
        "min_size": args.min_size,
        "max_size_exclusive": args.max_size,
        "prefix_bytes": PREFIX_BYTES,
        "sample_bytes": SAMPLE_BYTES,
        "full_hash": "SHA-256",
        "comparison": "byte-for-byte",
    }
    state_path, state_existed = prepare_output(out)
    with output_lock(out):
        state = sqlite3.connect(state_path, uri=True)
        try:
            initialize_state = not state_existed
            if state_existed:
                stored = existing_config(state)
                if stored is None:
                    if state_has_payload(state):
                        raise RuntimeError("existing state DB has no resume config")
                    initialize_state = True
                elif stored != config:
                    raise RuntimeError(
                        "resume config/inventory fingerprint differs; choose a new --out"
                    )
            configure_state(state)
            if initialize_state:
                set_meta(state, "config", config)
                state.commit()

            initialize_candidates(
                state, inventory, args.min_size, args.max_size
            )
            if inventory_fingerprint(inventory) != initial_fingerprint:
                raise RuntimeError("inventory changed during candidate initialization")

            run_hash_stage(
                state, "prefix", iter_work_items(state, "prefix"),
                hash_prefix, args.sample_workers,
            )
            run_hash_stage(
                state, "sample", iter_work_items(state, "sample"),
                hash_sample, args.sample_workers,
            )
            run_hash_stage(
                state, "full", iter_work_items(state, "full"),
                hash_full, args.full_workers,
            )
            compare_groups(state)
            final_stats = run_final_sweep(state)
            if final_stats["changed"]:
                rerun = compare_groups(state)
                final_stats["comparison_rerun_after_changes"] = rerun
                set_meta(state, "final_sweep_stats", final_stats)
                state.commit()
            summary = emit_reports(
                state,
                out,
                config,
                inventory,
                initial_fingerprint,
                inventory_errors,
                source_details,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            return 0 if summary["status"] == "COMPLETE" else 2
        finally:
            state.close()


if __name__ == "__main__":
    sys.exit(main())
