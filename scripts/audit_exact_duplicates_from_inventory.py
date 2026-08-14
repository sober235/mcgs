#!/usr/bin/env python3
"""Strict, read-only duplicate verification from an existing inventory SQLite DB.

The source inventory must contain a ``files`` table with at least
``path,size,allocated,dev,ino,nlink,mtime_ns``.  Source files are never changed.

Verification pipeline:
  distinct inode -> equal size -> prefix -> middle/tail sample -> full SHA-256
  -> byte-for-byte comparison -> final metadata revalidation.

The output state DB is resumable.  Errors and files that change during the run
are excluded from verified groups and make the report PARTIAL.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
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
from typing import BinaryIO, Callable
from urllib.parse import quote


ALGORITHM = "exact-dedupe-v2"
PREFIX_BYTES = 4 * 1024
SAMPLE_BYTES = 64 * 1024
IO_CHUNK = 8 * 1024 * 1024


class FileChanged(RuntimeError):
    pass


@dataclass(frozen=True)
class Signature:
    dev: int
    ino: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class WorkItem:
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
    digest: str
    signature: Signature
    bytes_read: int
    path: bytes


def human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    number = float(value)
    for unit in units:
        if number < 1024 or unit == units[-1]:
            return f"{number:.2f} {unit}"
        number /= 1024
    raise AssertionError


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
        int(st.st_mtime_ns), int(st.st_ctime_ns),
    )


def same_signature(left: Signature, right: Signature) -> bool:
    return left == right


def open_verified(
    item: WorkItem,
) -> tuple[BinaryIO, Signature, bytes]:
    failures: list[str] = []
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    for path in item.paths:
        try:
            descriptor = os.open(path, flags)
            handle = open(descriptor, "rb", buffering=0, closefd=True)
            st = os.fstat(handle.fileno())
            sig = signature(st)
            if not stat.S_ISREG(st.st_mode):
                handle.close()
                raise RuntimeError("not a regular file")
            if (sig.dev, sig.ino, sig.size) != (item.dev, item.ino, item.size):
                handle.close()
                raise FileChanged(
                    f"identity changed: expected={(item.dev,item.ino,item.size)}, got={sig}"
                )
            if sig.mtime_ns != item.inventory_mtime_ns:
                handle.close()
                raise FileChanged(
                    "mtime differs from inventory: "
                    f"expected={item.inventory_mtime_ns}, got={sig.mtime_ns}"
                )
            if item.expected is not None and not same_signature(sig, item.expected):
                handle.close()
                raise FileChanged(f"metadata changed: expected={item.expected}, got={sig}")
            if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_SEQUENTIAL"):
                try:
                    os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
                except OSError:
                    pass
            return handle, sig, path
        except Exception as exc:
            failures.append(f"{display_path(path)}: {exc!r}")
    raise RuntimeError("all aliases failed: " + " | ".join(failures))


def finish_verified(
    handle: BinaryIO,
    before: Signature,
    *,
    drop_cache: bool = True,
) -> None:
    after = signature(os.fstat(handle.fileno()))
    if not same_signature(before, after):
        raise FileChanged(f"file changed during read: before={before}, after={after}")
    if (
        drop_cache
        and hasattr(os, "posix_fadvise")
        and hasattr(os, "POSIX_FADV_DONTNEED")
    ):
        try:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass


def pread_exact(handle: BinaryIO, offset: int, length: int) -> bytes:
    chunks: list[bytes] = []
    cursor = offset
    remaining = length
    while remaining:
        part = os.pread(handle.fileno(), remaining, cursor)
        if not part:
            raise EOFError(f"short read offset={cursor}, remaining={remaining}")
        chunks.append(part)
        cursor += len(part)
        remaining -= len(part)
    return b"".join(chunks)


def hash_prefix(item: WorkItem) -> HashResult:
    handle, sig, used_path = open_verified(item)
    with handle:
        length = min(item.size, PREFIX_BYTES)
        payload = pread_exact(handle, 0, length)
        digest = hashlib.sha256(
            b"prefix-v1\0" + item.size.to_bytes(16, "little") + payload
        ).hexdigest()
        finish_verified(handle, sig)
    return HashResult(item.dev, item.ino, digest, sig, length, used_path)


def hash_sample(item: WorkItem) -> HashResult:
    handle, sig, used_path = open_verified(item)
    with handle:
        length = min(item.size, SAMPLE_BYTES)
        middle = max(0, item.size // 2 - length // 2)
        tail = max(0, item.size - length)
        digest = hashlib.sha256()
        digest.update(b"sample-v1\0")
        digest.update(item.size.to_bytes(16, "little"))
        total = 0
        for offset in dict.fromkeys((middle, tail)):
            payload = pread_exact(handle, offset, length)
            digest.update(offset.to_bytes(16, "little"))
            digest.update(payload)
            total += len(payload)
        finish_verified(handle, sig)
    return HashResult(item.dev, item.ino, digest.hexdigest(), sig, total, used_path)


def hash_full(item: WorkItem) -> HashResult:
    handle, sig, used_path = open_verified(item)
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
        # Keep freshly hashed pages resident for immediate byte comparison.
        finish_verified(handle, sig, drop_cache=False)
    return HashResult(item.dev, item.ino, digest.hexdigest(), sig, total, used_path)


def configure_state(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA temp_store=FILE")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS objects(
          dev INTEGER NOT NULL, ino INTEGER NOT NULL, size INTEGER NOT NULL,
          allocated INTEGER NOT NULL, nlink INTEGER NOT NULL, aliases_seen INTEGER NOT NULL,
          inventory_mtime_ns INTEGER NOT NULL,
          primary_path BLOB NOT NULL,
          sig_mtime_ns INTEGER, sig_ctime_ns INTEGER,
          prefix_hash TEXT, sample_hash TEXT, full_hash TEXT,
          used_path BLOB, error_stage TEXT, error TEXT, attempts INTEGER NOT NULL DEFAULT 0,
          PRIMARY KEY(dev,ino)
        );
        CREATE TABLE IF NOT EXISTS paths(
          dev INTEGER NOT NULL, ino INTEGER NOT NULL, path BLOB NOT NULL,
          PRIMARY KEY(dev,ino,path)
        );
        CREATE INDEX IF NOT EXISTS idx_objects_size ON objects(size);
        CREATE INDEX IF NOT EXISTS idx_objects_prefix ON objects(size,prefix_hash);
        CREATE INDEX IF NOT EXISTS idx_objects_sample ON objects(size,prefix_hash,sample_hash);
        CREATE INDEX IF NOT EXISTS idx_objects_full ON objects(size,full_hash);
        CREATE TABLE IF NOT EXISTS comparisons(
          size INTEGER NOT NULL, full_hash TEXT NOT NULL,
          rep_dev INTEGER NOT NULL, rep_ino INTEGER NOT NULL,
          member_dev INTEGER NOT NULL, member_ino INTEGER NOT NULL,
          exact INTEGER NOT NULL, error TEXT,
          PRIMARY KEY(size,full_hash,rep_dev,rep_ino,member_dev,member_ino)
        );
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


def file_fingerprint(path: Path) -> dict[str, object] | None:
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return {
        "path": str(path.resolve()), "dev": st.st_dev, "ino": st.st_ino,
        "size": st.st_size, "mtime_ns": st.st_mtime_ns,
    }


def inventory_fingerprint(path: Path) -> dict[str, object]:
    wal = file_fingerprint(Path(str(path) + "-wal"))
    # SQLite may create an empty WAL merely by opening a WAL-mode database.
    # Missing and zero-length WAL therefore represent the same inventory state.
    if wal is None or wal["size"] == 0:
        wal = {"empty": True}
    return {
        "database": file_fingerprint(path),
        "wal": wal,
    }


def source_connection(path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(path.resolve()))}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    columns = {row[1] for row in conn.execute("PRAGMA table_info(files)")}
    required = {"path", "size", "allocated", "dev", "ino", "nlink", "mtime_ns"}
    missing = required - columns
    if missing:
        raise RuntimeError(f"inventory files table missing columns: {sorted(missing)}")
    inode_columns = {row[1] for row in conn.execute("PRAGMA table_info(inodes)")}
    inode_required = {"path", "size", "allocated", "dev", "ino", "path_count"}
    inode_missing = inode_required - inode_columns
    if inode_missing:
        raise RuntimeError(
            f"inventory inodes table missing columns: {sorted(inode_missing)}"
        )
    meta = {
        row[0]: json.loads(row[1])
        for row in conn.execute(
            "SELECT key,value FROM meta WHERE key IN "
            "('inventory_complete','inode_table_complete')"
        )
    }
    if meta.get("inventory_complete") is not True:
        raise RuntimeError("source inventory is not marked inventory_complete=true")
    if meta.get("inode_table_complete") is not True:
        raise RuntimeError("source inventory is not marked inode_table_complete=true")
    return conn


def initialize_candidates(
    source: sqlite3.Connection,
    state: sqlite3.Connection,
    minimum: int,
    maximum: int | None,
) -> None:
    if get_meta(state, "candidate_init_complete", False) is True:
        return
    # Any interrupted initialization is discarded.  A non-empty table alone is
    # not evidence that all candidates were materialized.
    state.executescript(
        "DELETE FROM comparisons; DELETE FROM paths; DELETE FROM objects;"
    )
    set_meta(state, "candidate_init_complete", False)
    state.commit()

    upper = "" if maximum is None else "AND size<?"
    params: tuple[int, ...] = (minimum,) if maximum is None else (minimum, maximum)
    rows = source.execute(
        f"WITH duplicate_sizes AS MATERIALIZED ("
        f"  SELECT size FROM inodes WHERE size>=? {upper} "
        "  GROUP BY size HAVING COUNT(*)>1"
        ") "
        "SELECT i.dev,i.ino,i.size,i.allocated,i.path_count,"
        "       f.nlink,f.mtime_ns,i.path "
        "FROM inodes AS i JOIN duplicate_sizes AS d ON d.size=i.size "
        "JOIN files AS f ON f.path=i.path "
        "ORDER BY i.size DESC,i.dev,i.ino",
        params,
    )
    inserted = 0
    for dev, ino, size, allocated, aliases_seen, nlink, mtime_ns, primary in rows:
        primary_raw = raw_path(primary)
        state.execute(
            "INSERT INTO objects(dev,ino,size,allocated,nlink,aliases_seen,"
            "inventory_mtime_ns,primary_path) VALUES(?,?,?,?,?,?,?,?)",
            (dev, ino, size, allocated, nlink, aliases_seen, mtime_ns, primary_raw),
        )
        if aliases_seen > 1:
            aliases = source.execute(
                "SELECT path FROM files WHERE dev=? AND ino=? ORDER BY path", (dev, ino)
            )
            state.executemany(
                "INSERT INTO paths(dev,ino,path) VALUES(?,?,?)",
                ((dev, ino, raw_path(row[0])) for row in aliases),
            )
        else:
            state.execute(
                "INSERT INTO paths(dev,ino,path) VALUES(?,?,?)",
                (dev, ino, primary_raw),
            )
        inserted += 1
        if inserted % 1000 == 0:
            state.commit()
            print(f"[init] objects={inserted:,}", flush=True)
    set_meta(state, "candidate_objects", inserted)
    set_meta(state, "candidate_init_complete", True)
    state.commit()
    print(f"[init complete] objects={inserted:,}", flush=True)


def paths_for(state: sqlite3.Connection, dev: int, ino: int) -> tuple[bytes, ...]:
    return tuple(
        raw_path(row[0])
        for row in state.execute(
            "SELECT path FROM paths WHERE dev=? AND ino=? ORDER BY path", (dev, ino)
        )
    )


def work_items(state: sqlite3.Connection, query: str) -> list[WorkItem]:
    result: list[WorkItem] = []
    for (
        dev, ino, size, inventory_mtime, sig_mtime, sig_ctime,
        primary_path, aliases_seen,
    ) in state.execute(query):
        expected = None
        if sig_mtime is not None and sig_ctime is not None:
            expected = Signature(dev, ino, size, sig_mtime, sig_ctime)
        paths = (
            paths_for(state, dev, ino)
            if aliases_seen > 1
            else (raw_path(primary_path),)
        )
        result.append(
            WorkItem(dev, ino, size, inventory_mtime, paths, expected)
        )
    return result


def object_work_item(
    state: sqlite3.Connection,
    dev: int,
    ino: int,
) -> WorkItem:
    row = state.execute(
        "SELECT size,inventory_mtime_ns,sig_mtime_ns,sig_ctime_ns "
        "FROM objects WHERE dev=? AND ino=?",
        (dev, ino),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"object disappeared from state: dev={dev}, ino={ino}")
    size, inventory_mtime, sig_mtime, sig_ctime = row
    if sig_mtime is None or sig_ctime is None:
        raise RuntimeError(f"object lacks hash-time signature: dev={dev}, ino={ino}")
    return WorkItem(
        dev,
        ino,
        size,
        inventory_mtime,
        paths_for(state, dev, ino),
        Signature(dev, ino, size, sig_mtime, sig_ctime),
    )


def clear_changed(state: sqlite3.Connection, dev: int, ino: int, stage: str, error: str) -> None:
    set_meta(state, "comparison_complete", False)
    state.execute(
        "UPDATE objects SET prefix_hash=NULL,sample_hash=NULL,full_hash=NULL,"
        "sig_mtime_ns=NULL,sig_ctime_ns=NULL,used_path=NULL,"
        "error_stage=?,error=?,attempts=attempts+1 "
        "WHERE dev=? AND ino=?",
        (stage, error, dev, ino),
    )
    state.execute(
        "DELETE FROM comparisons WHERE "
        "(member_dev=? AND member_ino=?) OR (rep_dev=? AND rep_ino=?)",
        (dev, ino, dev, ino),
    )


def run_hash_stage(
    state: sqlite3.Connection,
    stage: str,
    items: list[WorkItem],
    function: Callable[[WorkItem], HashResult],
    workers: int,
) -> None:
    if not items:
        print(f"[{stage}] nothing pending", flush=True)
        return
    set_meta(state, "comparison_complete", False)
    state.commit()
    started = time.time()
    done = read_bytes = errors = 0
    compared = comparison_cached = comparison_errors = comparison_bytes = 0
    inflight: dict[Future[HashResult], tuple[int, WorkItem]] = {}
    iterator = iter(items)
    # Do not queue the next full read behind a worker.  Once a full hash
    # completes, that worker stays idle until its immediate comparison ends.
    limit = workers if stage == "full" else max(16, workers * 4)
    submitted = 0

    def fill(pool: ThreadPoolExecutor) -> None:
        nonlocal submitted
        while len(inflight) < limit:
            try:
                item = next(iterator)
            except StopIteration:
                return
            inflight[pool.submit(function, item)] = (submitted, item)
            submitted += 1

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=stage) as pool:
        fill(pool)
        last = time.time()
        while inflight:
            completed, _ = wait(
                inflight, timeout=30, return_when=FIRST_COMPLETED
            )
            for future in sorted(completed, key=lambda value: inflight[value][0]):
                _, item = inflight.pop(future)
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
                            "used_path=?,error_stage=NULL,error=NULL,attempts=attempts+1 "
                            "WHERE dev=? AND ino=?",
                            (result.digest, result.signature.mtime_ns, result.signature.ctime_ns,
                             result.path, result.dev, result.ino),
                        )
                    else:
                        state.execute(
                            f"UPDATE objects SET {column}=?,used_path=?,error_stage=NULL,error=NULL,"
                            "attempts=attempts+1 WHERE dev=? AND ino=?",
                            (result.digest, result.path, result.dev, result.ino),
                        )
                except Exception as exc:
                    clear_changed(state, item.dev, item.ino, stage, repr(exc))
                    errors += 1
                    if stage == "full":
                        state.commit()
                else:
                    read_bytes += result.bytes_read
                    if stage == "full":
                        # Digest is durable before comparison.  A crash now
                        # leaves one missing comparison for compare_groups().
                        state.commit()
                        stats = ensure_group_comparisons(
                            state, item.size, result.digest
                        )
                        compared += stats["compared"]
                        comparison_cached += stats["cached"]
                        comparison_errors += stats["errors"]
                        comparison_bytes += stats["bytes_read"]
                done += 1
            if completed:
                # Full-file results can represent hours of I/O. Persist every
                # completed batch so Ctrl-C/power loss never discards 100 files.
                state.commit()
                fill(pool)
            now = time.time()
            if now - last >= 30:
                elapsed = max(now - started, 1e-6)
                print(
                    f"[{stage}] {done:,}/{len(items):,} read={human_bytes(read_bytes)} "
                    f"rate={human_bytes(int(read_bytes/elapsed))}/s "
                    f"inflight={len(inflight):,} errors={errors:,} "
                    f"immediate_compared={compared:,}",
                    flush=True,
                )
                last = now
    state.commit()
    print(
        f"[{stage} complete] {done:,}/{len(items):,} read={human_bytes(read_bytes)} "
        f"errors={errors:,} immediate_compared={compared:,} "
        f"comparison_cached={comparison_cached:,} "
        f"comparison_errors={comparison_errors:,} "
        f"comparison_read={human_bytes(comparison_bytes)}",
        flush=True,
    )


def compare_pair(rep: WorkItem, member: WorkItem) -> tuple[bool, str | None, int]:
    total = 0
    try:
        with contextlib.ExitStack() as stack:
            rep_handle, rep_sig, _ = open_verified(rep)
            stack.enter_context(rep_handle)
            mem_handle, mem_sig, _ = open_verified(member)
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
            # Reuse representative pages across the rest of the group.  The
            # member was freshly full-hashed and is no longer needed afterward.
            finish_verified(rep_handle, rep_sig, drop_cache=False)
            finish_verified(mem_handle, mem_sig, drop_cache=True)
            if mismatch:
                return False, "stable byte mismatch after equal SHA-256", total
        return True, None, total
    except Exception as exc:
        return False, repr(exc), total


def ensure_group_comparisons(
    state: sqlite3.Connection,
    size: int,
    full_hash: str,
) -> dict[str, int]:
    """Persist only missing comparisons against deterministic group representative."""
    identities = list(
        state.execute(
            "SELECT dev,ino FROM objects WHERE size=? AND full_hash=? "
            "ORDER BY dev,ino",
            (size, full_hash),
        )
    )
    if len(identities) < 2:
        return {"compared": 0, "cached": 0, "errors": 0, "bytes_read": 0}

    rep_dev, rep_ino = identities[0]
    # Preserve all rows already rooted at canonical (dev,ino).  Remove only
    # rows made unusable by legacy representative ordering or stale membership.
    state.execute(
        "DELETE FROM comparisons WHERE size=? AND full_hash=? AND ("
        "rep_dev<>? OR rep_ino<>? OR "
        "(member_dev=? AND member_ino=?) OR NOT EXISTS("
        " SELECT 1 FROM objects o WHERE o.dev=comparisons.member_dev "
        " AND o.ino=comparisons.member_ino AND o.size=comparisons.size "
        " AND o.full_hash=comparisons.full_hash))",
        (size, full_hash, rep_dev, rep_ino, rep_dev, rep_ino),
    )
    state.commit()

    existing = {
        (int(row[0]), int(row[1]))
        for row in state.execute(
            "SELECT member_dev,member_ino FROM comparisons "
            "WHERE size=? AND full_hash=? AND rep_dev=? AND rep_ino=?",
            (size, full_hash, rep_dev, rep_ino),
        )
    }
    rep = object_work_item(state, rep_dev, rep_ino)
    compared = cached = errors = bytes_read = 0
    for member_dev, member_ino in identities[1:]:
        identity = (int(member_dev), int(member_ino))
        if identity in existing:
            cached += 1
            continue
        member = object_work_item(state, *identity)
        exact, error, read = compare_pair(rep, member)
        state.execute(
            "INSERT INTO comparisons("
            "size,full_hash,rep_dev,rep_ino,member_dev,member_ino,exact,error"
            ") VALUES(?,?,?,?,?,?,?,?)",
            (
                size,
                full_hash,
                rep.dev,
                rep.ino,
                member.dev,
                member.ino,
                int(exact),
                error,
            ),
        )
        # Each verified pair survives interruption independently.
        state.commit()
        compared += 1
        bytes_read += read
        errors += int(error is not None)

    covered = int(
        state.execute(
            "SELECT COUNT(*) FROM comparisons WHERE size=? AND full_hash=? "
            "AND rep_dev=? AND rep_ino=?",
            (size, full_hash, rep_dev, rep_ino),
        ).fetchone()[0]
    )
    expected = len(identities) - 1
    if covered != expected:
        raise RuntimeError(
            "comparison coverage mismatch: "
            f"size={size}, hash={full_hash}, expected={expected}, got={covered}"
        )
    return {
        "compared": compared,
        "cached": cached,
        "errors": errors,
        "bytes_read": bytes_read,
    }


def compare_groups(state: sqlite3.Connection) -> None:
    # This flag covers the whole current object set.  Existing verified pairs
    # stay durable; interrupted/resumed runs only fill absent rows.
    set_meta(state, "comparison_complete", False)
    state.commit()
    groups = list(
        state.execute(
            "SELECT size,full_hash,COUNT(*) FROM objects WHERE full_hash IS NOT NULL "
            "GROUP BY size,full_hash HAVING COUNT(*)>1 "
            "ORDER BY size*(COUNT(*)-1) DESC,size,full_hash"
        )
    )
    compared = cached = errors = bytes_read = 0
    for group_no, (size, full_hash, count) in enumerate(groups, start=1):
        stats = ensure_group_comparisons(state, size, full_hash)
        compared += stats["compared"]
        cached += stats["cached"]
        errors += stats["errors"]
        bytes_read += stats["bytes_read"]
        print(
            f"[compare] group={group_no:,}/{len(groups):,} "
            f"size={human_bytes(size)} members={count} "
            f"new={stats['compared']:,} cached={stats['cached']:,}",
            flush=True,
        )
    set_meta(state, "comparison_complete", True)
    state.commit()
    print(
        f"[compare complete] groups={len(groups):,} new={compared:,} "
        f"cached={cached:,} errors={errors:,} read={human_bytes(bytes_read)}",
        flush=True,
    )


def final_sweep(state: sqlite3.Connection) -> int:
    """Re-stat every candidate, including objects pruned by sampling.

    This cannot create a global filesystem snapshot, but it detects any object
    that changed between its content read and report emission.
    """
    changed = checked = 0
    rows = state.execute(
        "SELECT dev,ino,size,inventory_mtime_ns,sig_mtime_ns,sig_ctime_ns,"
        "primary_path,aliases_seen FROM objects ORDER BY dev,ino"
    )
    for row in rows:
        dev, ino, size, inventory_mt, mt, ct, primary, aliases_seen = row
        paths = (
            paths_for(state, dev, ino)
            if aliases_seen > 1
            else (raw_path(primary),)
        )
        expected = (
            Signature(dev, ino, size, mt, ct)
            if mt is not None and ct is not None else None
        )
        item = WorkItem(dev, ino, size, inventory_mt, paths, expected)
        try:
            handle, sig, _ = open_verified(item)
            with handle:
                finish_verified(handle, sig)
        except Exception as exc:
            clear_changed(state, dev, ino, "final-sweep", repr(exc))
            changed += 1
        checked += 1
        if checked % 1000 == 0:
            state.commit()
    if changed:
        set_meta(state, "comparison_complete", False)
    state.commit()
    print(
        f"[final sweep] checked={checked:,} changed_or_unreadable={changed:,}",
        flush=True,
    )
    return changed


def final_signature_ok(obj: sqlite3.Row, paths: tuple[bytes, ...]) -> tuple[bool, str | None]:
    dev, ino, size, inventory_mt, mt, ct = obj
    item = WorkItem(
        dev, ino, size, inventory_mt, paths,
        Signature(dev, ino, size, mt, ct),
    )
    try:
        handle, sig, _ = open_verified(item)
        with handle:
            finish_verified(handle, sig)
        return True, None
    except Exception as exc:
        return False, repr(exc)


def emit_reports(
    state: sqlite3.Connection,
    out: Path,
    inventory_errors: int,
    inventory_unchanged: bool,
) -> dict[str, object]:
    state.row_factory = sqlite3.Row
    exact_groups: list[dict[str, object]] = []
    stale = 0
    logical_reclaim = allocated_reclaim = 0
    group_rows = state.execute(
        "SELECT c.size,c.full_hash,c.rep_dev,c.rep_ino,COUNT(*) verified_members "
        "FROM comparisons c "
        "JOIN objects r ON r.dev=c.rep_dev AND r.ino=c.rep_ino "
        " AND r.size=c.size AND r.full_hash=c.full_hash "
        "JOIN objects m ON m.dev=c.member_dev AND m.ino=c.member_ino "
        " AND m.size=c.size AND m.full_hash=c.full_hash "
        "WHERE c.exact=1 AND c.error IS NULL "
        "GROUP BY c.size,c.full_hash,c.rep_dev,c.rep_ino"
    )
    for group_id, row in enumerate(group_rows, start=1):
        size, full_hash, rep_dev, rep_ino, _ = row
        identities = [(rep_dev, rep_ino)] + [
            (r[0], r[1]) for r in state.execute(
                "SELECT c.member_dev,c.member_ino FROM comparisons c "
                "JOIN objects m ON m.dev=c.member_dev AND m.ino=c.member_ino "
                " AND m.size=c.size AND m.full_hash=c.full_hash "
                "WHERE c.size=? AND c.full_hash=? AND c.rep_dev=? AND c.rep_ino=? "
                "AND c.exact=1 AND c.error IS NULL "
                "ORDER BY c.member_dev,c.member_ino",
                (size, full_hash, rep_dev, rep_ino),
            )
        ]
        members = []
        valid = True
        for dev, ino in identities:
            obj = state.execute(
                "SELECT dev,ino,size,allocated,nlink,aliases_seen,inventory_mtime_ns,"
                "sig_mtime_ns,sig_ctime_ns "
                "FROM objects WHERE dev=? AND ino=?", (dev, ino)
            ).fetchone()
            paths = paths_for(state, dev, ino)
            ok, error = final_signature_ok(
                (obj[0], obj[1], obj[2], obj[6], obj[7], obj[8]), paths
            )
            if not ok:
                valid = False
                stale += 1
            members.append(
                {
                    "dev": dev, "ino": ino, "allocated_bytes": obj[3],
                    "nlink": obj[4], "aliases_seen": obj[5],
                    "link_complete": obj[4] == obj[5],
                    "stable_at_report": ok, "stability_error": error,
                    "paths": [path_record(path) for path in paths],
                }
            )
        if not valid:
            continue
        group_logical = size * (len(members) - 1)
        reclaimable_allocations = [
            int(member["allocated_bytes"]) if member["link_complete"] else 0
            for member in members
        ]
        # One independent inode must remain. Keep the least reclaimable member.
        group_allocated = (
            sum(reclaimable_allocations) - min(reclaimable_allocations)
        )
        logical_reclaim += group_logical
        allocated_reclaim += group_allocated
        exact_groups.append(
            {
                "group_id": group_id, "size_bytes": size, "sha256": full_hash,
                "byte_verified_exact": True, "distinct_inode_count": len(members),
                "logical_reclaim_bytes": group_logical,
                "actionable_allocated_reclaim_bytes": group_allocated,
                "members": members,
            }
        )

    error_objects = state.execute(
        "SELECT COUNT(*) FROM objects WHERE error IS NOT NULL"
    ).fetchone()[0]
    comparison_errors = state.execute(
        "SELECT COUNT(*) FROM comparisons WHERE error IS NOT NULL"
    ).fetchone()[0]
    pipeline_incomplete = not (
        get_meta(state, "candidate_init_complete", False)
        and get_meta(state, "comparison_complete", False)
    )
    status = "COMPLETE" if not (
        inventory_errors or error_objects or comparison_errors or stale
        or pipeline_incomplete or not inventory_unchanged
    ) else "PARTIAL"
    duplicates_path = out / "exact_duplicates.jsonl"
    temp_path = out / "exact_duplicates.jsonl.tmp"
    with temp_path.open("w", encoding="utf-8") as handle:
        for group in exact_groups:
            handle.write(json.dumps(group, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temp_path, duplicates_path)

    errors_path = out / "errors.jsonl"
    with errors_path.open("w", encoding="utf-8") as handle:
        for row in state.execute(
            "SELECT dev,ino,error_stage,error FROM objects WHERE error IS NOT NULL ORDER BY dev,ino"
        ):
            handle.write(json.dumps({
                "dev": row[0], "ino": row[1], "stage": row[2], "error": row[3],
                "paths": [path_record(p) for p in paths_for(state, row[0], row[1])],
            }, ensure_ascii=False, sort_keys=True) + "\n")
        for row in state.execute(
            "SELECT size,full_hash,member_dev,member_ino,error FROM comparisons WHERE error IS NOT NULL"
        ):
            handle.write(json.dumps({
                "stage": "compare", "size": row[0], "sha256": row[1],
                "dev": row[2], "ino": row[3], "error": row[4],
            }, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "status": status, "algorithm": ALGORITHM,
        "scope": get_meta(state, "config"),
        "candidate_objects": state.execute("SELECT COUNT(*) FROM objects").fetchone()[0],
        "prefix_hashed": state.execute("SELECT COUNT(*) FROM objects WHERE prefix_hash IS NOT NULL").fetchone()[0],
        "sample_hashed": state.execute("SELECT COUNT(*) FROM objects WHERE sample_hash IS NOT NULL").fetchone()[0],
        "full_hashed": state.execute("SELECT COUNT(*) FROM objects WHERE full_hash IS NOT NULL").fetchone()[0],
        "exact_duplicate_groups": len(exact_groups),
        "logical_reclaim_bytes": logical_reclaim,
        "logical_reclaim_human": human_bytes(logical_reclaim),
        "actionable_allocated_reclaim_bytes": allocated_reclaim,
        "actionable_allocated_reclaim_human": human_bytes(allocated_reclaim),
        "inventory_error_count": inventory_errors,
        "inventory_unchanged_during_run": inventory_unchanged,
        "object_error_count": error_objects,
        "comparison_error_count": comparison_errors,
        "stale_at_report_count": stale,
        "pipeline_incomplete": pipeline_incomplete,
        "deletion_or_deduplication_performed": False,
        "reports": {"duplicates": str(duplicates_path), "errors": str(errors_path)},
    }
    summary_tmp = out / "summary.json.tmp"
    summary_tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(summary_tmp, out / "summary.json")
    return summary


def parse_size(value: str) -> int:
    units = {"k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}
    lowered = value.strip().lower()
    if lowered[-1:] in units:
        return int(float(lowered[:-1]) * units[lowered[-1]])
    return int(lowered)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-size", type=parse_size, default=1024**3)
    parser.add_argument("--max-size", type=parse_size)
    parser.add_argument("--sample-workers", type=int, default=64)
    parser.add_argument("--full-workers", type=int, default=2)
    return parser.parse_args()


def validate_output_paths(inventory: Path, out: Path) -> None:
    targets = (
        out / "state.sqlite3",
        out / "summary.json",
        out / "errors.jsonl",
        out / "exact_duplicates.jsonl",
    )
    if inventory.parent == out:
        raise RuntimeError("--out must not be the inventory database directory")
    for target in targets:
        if target.resolve() == inventory:
            raise RuntimeError(f"output would overwrite inventory: {target}")


def main() -> int:
    args = parse_args()
    inventory = Path(args.inventory).resolve()
    out = Path(args.out).resolve()
    if args.min_size <= 0:
        raise RuntimeError("--min-size must be >0; zero-byte files reclaim no data")
    if args.max_size is not None and args.max_size <= args.min_size:
        raise RuntimeError("--max-size must be greater than --min-size")
    validate_output_paths(inventory, out)
    initial_inventory_fingerprint = inventory_fingerprint(inventory)
    out.mkdir(parents=True, exist_ok=True)
    source = source_connection(inventory)
    state = sqlite3.connect(out / "state.sqlite3")
    configure_state(state)
    config = {
        "algorithm": ALGORITHM, "inventory": initial_inventory_fingerprint,
        "min_size": args.min_size, "max_size": args.max_size,
        "prefix_bytes": PREFIX_BYTES, "sample_bytes": SAMPLE_BYTES,
    }
    stored = get_meta(state, "config")
    if stored is not None and stored != config:
        raise RuntimeError("resume config/inventory fingerprint differs; choose a new --out")
    set_meta(state, "config", config)
    state.commit()
    initialize_candidates(source, state, args.min_size, args.max_size)

    prefix_items = work_items(
        state,
        "SELECT dev,ino,size,inventory_mtime_ns,sig_mtime_ns,sig_ctime_ns,"
        "primary_path,aliases_seen FROM objects "
        "WHERE prefix_hash IS NULL ORDER BY size DESC,dev,ino",
    )
    run_hash_stage(state, "prefix", prefix_items, hash_prefix, max(1, args.sample_workers))

    sample_items = work_items(
        state,
        "SELECT o.dev,o.ino,o.size,o.inventory_mtime_ns,o.sig_mtime_ns,o.sig_ctime_ns,"
        "o.primary_path,o.aliases_seen FROM objects o JOIN "
        "(SELECT size,prefix_hash FROM objects WHERE prefix_hash IS NOT NULL "
        "GROUP BY size,prefix_hash HAVING COUNT(*)>1) g "
        "ON o.size=g.size AND o.prefix_hash=g.prefix_hash "
        "WHERE o.sample_hash IS NULL ORDER BY o.size DESC,o.dev,o.ino",
    )
    run_hash_stage(state, "sample", sample_items, hash_sample, max(1, args.sample_workers))

    full_items = work_items(
        state,
        "SELECT o.dev,o.ino,o.size,o.inventory_mtime_ns,o.sig_mtime_ns,o.sig_ctime_ns,"
        "o.primary_path,o.aliases_seen FROM objects o JOIN "
        "(SELECT size,prefix_hash,sample_hash FROM objects WHERE sample_hash IS NOT NULL "
        "GROUP BY size,prefix_hash,sample_hash HAVING COUNT(*)>1) g "
        "ON o.size=g.size AND o.prefix_hash=g.prefix_hash AND o.sample_hash=g.sample_hash "
        "WHERE o.full_hash IS NULL "
        "ORDER BY o.size DESC,o.prefix_hash,o.sample_hash,o.dev,o.ino",
    )
    run_hash_stage(state, "full", full_items, hash_full, max(1, args.full_workers))
    compare_groups(state)
    if final_sweep(state):
        # clear_changed() removes both member- and representative-side rows.
        # Rebuild only newly missing survivor comparisons in this same run.
        compare_groups(state)

    inventory_errors = 0
    try:
        inventory_errors = source.execute("SELECT COUNT(*) FROM scan_errors").fetchone()[0]
    except sqlite3.OperationalError:
        pass
    inventory_unchanged = (
        inventory_fingerprint(inventory) == initial_inventory_fingerprint
    )
    summary = emit_reports(
        state, out, inventory_errors, inventory_unchanged
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    state.close()
    source.close()
    return 0 if summary["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    sys.exit(main())
