#!/usr/bin/env python3
"""Read-only exact-duplicate audit for a large filesystem tree.

Pipeline:
  1. Inventory regular files without crossing filesystem boundaries.
  2. Collapse hard-linked paths to distinct inodes.
  3. Group by logical size, then hash first/middle/last samples.
  4. Compute full SHA-256 only for surviving candidates.
  5. Emit machine-readable reports. No source file is modified.

The SQLite database makes the scan auditable and keeps memory use bounded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import stat
import sys
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import BinaryIO, Iterable


PREFIX_BLOCK = 4 * 1024
SAMPLE_BLOCK = 64 * 1024
SMALL_FULL_HASH_LIMIT = 1024 * 1024
HASH_CHUNK = 8 * 1024 * 1024
REPORT_DEPTH = 6
HASH_PIPELINE_VERSION = 2


def human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    number = float(value)
    for unit in units:
        if abs(number) < 1024.0 or unit == units[-1]:
            return f"{number:.2f} {unit}"
        number /= 1024.0
    raise AssertionError("unreachable")


def display_path(raw: bytes) -> str:
    return os.fsdecode(raw).encode("utf-8", "backslashreplace").decode("utf-8")


def json_dump_line(handle: BinaryIO, obj: object) -> None:
    handle.write((json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))


def configure_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("PRAGMA cache_size=-262144")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS files (
            path BLOB PRIMARY KEY,
            size INTEGER NOT NULL,
            allocated INTEGER NOT NULL,
            dev INTEGER NOT NULL,
            ino INTEGER NOT NULL,
            nlink INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS symlinks (
            path BLOB PRIMARY KEY,
            target BLOB,
            broken INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS scan_errors (
            stage TEXT NOT NULL,
            path BLOB,
            error TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS skipped_mounts (
            path BLOB PRIMARY KEY,
            dev INTEGER NOT NULL
        );
        """
    )


def set_meta(conn: sqlite3.Connection, key: str, value: object) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, json.dumps(value, ensure_ascii=False)),
    )


def get_meta(conn: sqlite3.Connection, key: str, default: object = None) -> object:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return default if row is None else json.loads(row[0])


def path_is_within(path: bytes, parent: bytes) -> bool:
    try:
        return os.path.commonpath((path, parent)) == parent
    except ValueError:
        return False


def inventory(root: bytes, excluded: list[bytes], conn: sqlite3.Connection) -> dict[str, int]:
    # A prior interrupted inventory is not a snapshot. Rebuild it cleanly so deleted
    # or type-changed paths cannot survive from the earlier partial pass.
    conn.executescript(
        """
        DELETE FROM files;
        DELETE FROM symlinks;
        DELETE FROM scan_errors;
        DELETE FROM skipped_mounts;
        """
    )
    conn.commit()
    root_stat = os.stat(root, follow_symlinks=False)
    root_dev = root_stat.st_dev
    started = time.time()
    last_report = started
    counters = defaultdict(int)
    rows: list[tuple[bytes, int, int, int, int, int, int]] = []
    symlink_rows: list[tuple[bytes, bytes | None, int]] = []

    def record_walk_error(exc: OSError) -> None:
        raw = os.fsencode(exc.filename) if exc.filename is not None else None
        conn.execute(
            "INSERT INTO scan_errors(stage, path, error) VALUES('walk', ?, ?)",
            (raw, repr(exc)),
        )
        counters["walk_errors"] += 1

    for dirpath, dirnames, filenames in os.walk(
        root, topdown=True, followlinks=False, onerror=record_walk_error
    ):
        counters["directories"] += 1
        kept_dirs: list[bytes] = []
        for dirname in dirnames:
            child = os.path.join(dirpath, dirname)
            if any(path_is_within(child, item) for item in excluded):
                counters["excluded_directories"] += 1
                continue
            try:
                child_stat = os.stat(child, follow_symlinks=False)
            except OSError as exc:
                conn.execute(
                    "INSERT INTO scan_errors(stage, path, error) VALUES('directory_stat', ?, ?)",
                    (child, repr(exc)),
                )
                counters["directory_stat_errors"] += 1
                continue
            if stat.S_ISLNK(child_stat.st_mode):
                try:
                    target = os.readlink(child)
                    broken = int(not os.path.exists(child))
                except OSError as exc:
                    target = None
                    broken = 1
                    conn.execute(
                        "INSERT INTO scan_errors(stage, path, error) VALUES('readlink', ?, ?)",
                        (child, repr(exc)),
                    )
                symlink_rows.append((child, target, broken))
                counters["symlinks"] += 1
                continue
            if child_stat.st_dev != root_dev:
                conn.execute(
                    "INSERT OR REPLACE INTO skipped_mounts(path, dev) VALUES(?, ?)",
                    (child, child_stat.st_dev),
                )
                counters["skipped_mounts"] += 1
                continue
            kept_dirs.append(dirname)
        dirnames[:] = kept_dirs

        for filename in filenames:
            path = os.path.join(dirpath, filename)
            if any(path_is_within(path, item) for item in excluded):
                counters["excluded_files"] += 1
                continue
            try:
                item_stat = os.stat(path, follow_symlinks=False)
            except OSError as exc:
                conn.execute(
                    "INSERT INTO scan_errors(stage, path, error) VALUES('file_stat', ?, ?)",
                    (path, repr(exc)),
                )
                counters["file_stat_errors"] += 1
                continue

            if stat.S_ISREG(item_stat.st_mode):
                allocated = int(item_stat.st_blocks) * 512
                rows.append(
                    (
                        path,
                        int(item_stat.st_size),
                        allocated,
                        int(item_stat.st_dev),
                        int(item_stat.st_ino),
                        int(item_stat.st_nlink),
                        int(item_stat.st_mtime_ns),
                    )
                )
                counters["regular_paths"] += 1
                counters["logical_bytes"] += int(item_stat.st_size)
                counters["allocated_bytes"] += allocated
                if item_stat.st_size == 0:
                    counters["zero_files"] += 1
            elif stat.S_ISLNK(item_stat.st_mode):
                try:
                    target = os.readlink(path)
                    broken = int(not os.path.exists(path))
                except OSError as exc:
                    target = None
                    broken = 1
                    conn.execute(
                        "INSERT INTO scan_errors(stage, path, error) VALUES('readlink', ?, ?)",
                        (path, repr(exc)),
                    )
                symlink_rows.append((path, target, broken))
                counters["symlinks"] += 1
            else:
                counters["special_files"] += 1

            if len(rows) >= 10_000:
                conn.executemany(
                    "INSERT OR REPLACE INTO files "
                    "(path,size,allocated,dev,ino,nlink,mtime_ns) VALUES(?,?,?,?,?,?,?)",
                    rows,
                )
                rows.clear()
            if len(symlink_rows) >= 2_000:
                conn.executemany(
                    "INSERT OR REPLACE INTO symlinks(path,target,broken) VALUES(?,?,?)",
                    symlink_rows,
                )
                symlink_rows.clear()

        now = time.time()
        if now - last_report >= 30:
            conn.commit()
            elapsed = max(now - started, 1e-6)
            print(
                "[inventory] "
                f"dirs={counters['directories']:,} files={counters['regular_paths']:,} "
                f"logical={human_bytes(counters['logical_bytes'])} "
                f"rate={counters['regular_paths']/elapsed:,.0f} files/s",
                flush=True,
            )
            last_report = now

    if rows:
        conn.executemany(
            "INSERT OR REPLACE INTO files "
            "(path,size,allocated,dev,ino,nlink,mtime_ns) VALUES(?,?,?,?,?,?,?)",
            rows,
        )
    if symlink_rows:
        conn.executemany(
            "INSERT OR REPLACE INTO symlinks(path,target,broken) VALUES(?,?,?)",
            symlink_rows,
        )
    conn.commit()
    counters["elapsed_seconds"] = int(time.time() - started)
    set_meta(conn, "inventory", dict(counters))
    set_meta(conn, "inventory_complete", True)
    conn.commit()
    print(
        "[inventory complete] "
        f"files={counters['regular_paths']:,} logical={human_bytes(counters['logical_bytes'])} "
        f"elapsed={counters['elapsed_seconds']}s",
        flush=True,
    )
    return dict(counters)


def build_inode_table(conn: sqlite3.Connection) -> dict[str, int]:
    conn.executescript(
        """
        DROP TABLE IF EXISTS inodes;
        CREATE TABLE inodes AS
        SELECT dev, ino, MIN(path) AS path, MAX(size) AS size,
               MAX(allocated) AS allocated, MAX(mtime_ns) AS mtime_ns,
               COUNT(*) AS path_count,
               CAST(NULL AS TEXT) AS prefix_hash,
               CAST(NULL AS TEXT) AS sample_hash,
               CAST(NULL AS TEXT) AS full_hash,
               'pending' AS hash_state
        FROM files
        GROUP BY dev, ino;
        CREATE UNIQUE INDEX IF NOT EXISTS idx_inodes_identity ON inodes(dev, ino);
        CREATE INDEX IF NOT EXISTS idx_inodes_size ON inodes(size);
        CREATE INDEX IF NOT EXISTS idx_files_identity ON files(dev, ino);
        """
    )
    stats = {
        "distinct_inodes": conn.execute("SELECT COUNT(*) FROM inodes").fetchone()[0],
        "hardlink_inode_groups": conn.execute(
            "SELECT COUNT(*) FROM inodes WHERE path_count > 1"
        ).fetchone()[0],
        "hardlink_extra_paths": conn.execute(
            "SELECT COALESCE(SUM(path_count - 1),0) FROM inodes WHERE path_count > 1"
        ).fetchone()[0],
        "candidate_size_groups": conn.execute(
            "SELECT COUNT(*) FROM ("
            "SELECT size FROM inodes WHERE size > 0 GROUP BY size HAVING COUNT(*) > 1)"
        ).fetchone()[0],
        "candidate_inodes": conn.execute(
            "SELECT COUNT(*) FROM inodes i JOIN ("
            "SELECT size FROM inodes WHERE size > 0 GROUP BY size HAVING COUNT(*) > 1"
            ") d ON i.size=d.size"
        ).fetchone()[0],
    }
    set_meta(conn, "inode_stats", stats)
    set_meta(conn, "inode_table_complete", True)
    conn.commit()
    print(
        "[candidates] "
        f"distinct_inodes={stats['distinct_inodes']:,} "
        f"candidate_inodes={stats['candidate_inodes']:,} "
        f"hardlink_groups={stats['hardlink_inode_groups']:,}",
        flush=True,
    )
    return stats


def prepare_progressive_hashing(conn: sqlite3.Connection) -> None:
    """Migrate an interrupted v1 database to prefix -> sample -> full hashing."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(inodes)")}
    if "prefix_hash" not in columns:
        conn.execute("ALTER TABLE inodes ADD COLUMN prefix_hash TEXT")
    version = get_meta(conn, "hash_pipeline_version", 0)
    if version != HASH_PIPELINE_VERSION:
        conn.execute(
            "UPDATE inodes SET prefix_hash=NULL,sample_hash=NULL,full_hash=NULL,hash_state='pending'"
        )
        conn.execute(
            "DELETE FROM meta WHERE key IN "
            "('prefix_hash_complete','prefix_hash_stats','sample_hash_complete',"
            "'sample_hash_stats','full_hash_complete','full_hash_stats')"
        )
        set_meta(conn, "hash_pipeline_version", HASH_PIPELINE_VERSION)
        conn.commit()
        print(f"[hash migration] pipeline=v{HASH_PIPELINE_VERSION}", flush=True)
    # Repair a crash between hashing and index creation before honoring resume flags.
    if get_meta(conn, "prefix_hash_complete", False):
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_inodes_prefix ON inodes(size,prefix_hash)"
        )
    if get_meta(conn, "sample_hash_complete", False):
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_inodes_sample "
            "ON inodes(size,prefix_hash,sample_hash)"
        )
    conn.commit()


def validate_open_file(
    path: bytes, dev: int, ino: int, size: int, mtime_ns: int
) -> tuple[BinaryIO, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    handle = open(descriptor, "rb", buffering=0, closefd=True)
    current = os.fstat(handle.fileno())
    identity = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
    expected = (dev, ino, size, mtime_ns)
    if identity != expected:
        handle.close()
        raise RuntimeError(f"file changed since inventory: expected={expected}, current={identity}")
    return handle, current


def read_exact_at(handle: BinaryIO, offset: int, length: int) -> bytes:
    chunks = []
    remaining = length
    cursor = offset
    while remaining:
        chunk = os.pread(handle.fileno(), remaining, cursor)
        if not chunk:
            raise EOFError(f"short read at offset={cursor}, remaining={remaining}")
        chunks.append(chunk)
        cursor += len(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def prefix_digest(handle: BinaryIO, size: int) -> tuple[str, str | None, int]:
    length = min(size, PREFIX_BLOCK)
    payload = read_exact_at(handle, 0, length)
    if size <= PREFIX_BLOCK:
        result = hashlib.sha256(payload).hexdigest()
        return result, result, length
    digest = hashlib.sha256()
    digest.update(b"data0-dedupe-prefix-v2\0")
    digest.update(size.to_bytes(16, "little", signed=False))
    digest.update(payload)
    return digest.hexdigest(), None, length


def sample_digest(handle: BinaryIO, size: int) -> tuple[str, str | None, int]:
    if size <= SMALL_FULL_HASH_LIMIT:
        result, bytes_read = full_digest(handle, size)
        return result, result, bytes_read
    digest = hashlib.sha256()
    digest.update(b"data0-dedupe-midtail-v2\0")
    digest.update(size.to_bytes(16, "little", signed=False))
    offsets = (max(0, size // 2 - SAMPLE_BLOCK // 2), size - SAMPLE_BLOCK)
    bytes_read = 0
    for offset in offsets:
        payload = read_exact_at(handle, offset, SAMPLE_BLOCK)
        digest.update(offset.to_bytes(16, "little", signed=False))
        digest.update(payload)
        bytes_read += len(payload)
    return digest.hexdigest(), None, bytes_read


def full_digest(handle: BinaryIO, expected_size: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    handle.seek(0)
    bytes_read = 0
    while True:
        chunk = handle.read(HASH_CHUNK)
        if not chunk:
            break
        digest.update(chunk)
        bytes_read += len(chunk)
    if bytes_read != expected_size:
        raise EOFError(f"full hash short read: expected={expected_size}, got={bytes_read}")
    return digest.hexdigest(), bytes_read


def advise_drop_cache(handle: BinaryIO) -> None:
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        try:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass


def hash_one(
    row: tuple[int, int, bytes, int, int], stage: str
) -> tuple[str, str | None, int]:
    dev, ino, path, size, mtime_ns = row
    handle, before = validate_open_file(path, dev, ino, size, mtime_ns)
    with handle:
        if stage == "prefix":
            stage_hash, complete_hash, bytes_read = prefix_digest(handle, size)
        elif stage == "sample":
            stage_hash, complete_hash, bytes_read = sample_digest(handle, size)
        elif stage == "full":
            stage_hash, bytes_read = full_digest(handle, size)
            complete_hash = stage_hash
        else:
            raise ValueError(stage)
        after = os.fstat(handle.fileno())
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
            or before.st_ino != after.st_ino
            or before.st_dev != after.st_dev
        ):
            raise RuntimeError("file changed while hashing")
        advise_drop_cache(handle)
    return stage_hash, complete_hash, bytes_read


def stage_query(stage: str) -> str:
    locality_scan = "FROM inodes i INDEXED BY idx_inodes_path"
    different_inode = "(j.dev<>i.dev OR j.ino<>i.ino)"
    if stage == "prefix":
        return f"""
            SELECT i.dev,i.ino,i.path,i.size,i.mtime_ns
            {locality_scan}
            WHERE i.size>0 AND i.prefix_hash IS NULL AND i.hash_state<>'error'
            ORDER BY i.path
        """
    if stage == "sample":
        return f"""
            SELECT i.dev,i.ino,i.path,i.size,i.mtime_ns
            {locality_scan}
            WHERE i.prefix_hash IS NOT NULL AND i.sample_hash IS NULL
              AND i.full_hash IS NULL AND i.hash_state<>'error'
              AND EXISTS (
                SELECT 1 FROM inodes j INDEXED BY idx_inodes_prefix
                WHERE j.size=i.size AND j.prefix_hash=i.prefix_hash
                  AND {different_inode} LIMIT 1)
            ORDER BY i.path
        """
    if stage == "full":
        return f"""
            SELECT i.dev,i.ino,i.path,i.size,i.mtime_ns
            {locality_scan}
            WHERE i.sample_hash IS NOT NULL AND i.full_hash IS NULL
              AND i.hash_state<>'error'
              AND EXISTS (
                SELECT 1 FROM inodes j INDEXED BY idx_inodes_sample
                WHERE j.size=i.size AND j.prefix_hash=i.prefix_hash
                  AND j.sample_hash=i.sample_hash AND {different_inode} LIMIT 1)
            ORDER BY i.path
        """
    raise ValueError(stage)


def run_hash_stage(conn: sqlite3.Connection, stage: str, workers: int) -> dict[str, int]:
    if stage == "prefix":
        update_sql = (
            "UPDATE inodes SET prefix_hash=?,full_hash=?,hash_state=? WHERE dev=? AND ino=?"
        )
    elif stage == "sample":
        update_sql = (
            "UPDATE inodes SET sample_hash=?,full_hash=?,hash_state=? WHERE dev=? AND ino=?"
        )
    elif stage == "full":
        update_sql = (
            "UPDATE inodes SET full_hash=?,hash_state='complete' WHERE dev=? AND ino=?"
        )
    else:
        raise ValueError(stage)

    query = stage_query(stage)
    total = conn.execute(f"SELECT COUNT(*) FROM ({query})").fetchone()[0]

    started = time.time()
    last_report = started
    counters = defaultdict(int)
    updates: list[tuple[object, ...]] = []
    errors: list[tuple[str, bytes, str]] = []
    max_inflight = max(workers * 8, 16)
    cursor = iter(conn.execute(query))
    exhausted = False
    pending: dict[Future[tuple[str, str | None, int]], tuple[int, int, bytes, int, int]] = {}

    def fill_queue(pool: ThreadPoolExecutor) -> None:
        nonlocal exhausted
        while not exhausted and len(pending) < max_inflight:
            try:
                row = next(cursor)
            except StopIteration:
                exhausted = True
                break
            pending[pool.submit(hash_one, row, stage)] = row

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"dedupe-{stage}") as pool:
        fill_queue(pool)
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                dev, ino, path, _size, _mtime_ns = pending.pop(future)
                try:
                    stage_hash, complete_hash, bytes_read = future.result()
                    if stage == "full":
                        updates.append((complete_hash, dev, ino))
                    else:
                        state = "complete" if complete_hash is not None else f"{stage}d"
                        updates.append((stage_hash, complete_hash, state, dev, ino))
                    counters["hashed_files"] += 1
                    counters["hashed_bytes"] += int(bytes_read)
                except Exception as exc:  # continue audit; preserve exact error in report
                    errors.append((stage, path, repr(exc)))
                    conn.execute(
                        "UPDATE inodes SET hash_state='error' WHERE dev=? AND ino=?", (dev, ino)
                    )
                    counters["errors"] += 1
            fill_queue(pool)

            if len(updates) >= 1_000:
                conn.executemany(update_sql, updates)
                updates.clear()
            if len(errors) >= 100:
                conn.executemany(
                    "INSERT INTO scan_errors(stage,path,error) VALUES(?,?,?)", errors
                )
                errors.clear()

            now = time.time()
            if now - last_report >= 30:
                if updates:
                    conn.executemany(update_sql, updates)
                    updates.clear()
                if errors:
                    conn.executemany(
                        "INSERT INTO scan_errors(stage,path,error) VALUES(?,?,?)", errors
                    )
                    errors.clear()
                conn.commit()
                elapsed = max(now - started, 1e-6)
                done = counters["hashed_files"] + counters["errors"]
                print(
                    f"[{stage}] {done:,}/{total:,} files "
                    f"read={human_bytes(counters['hashed_bytes'])} "
                    f"rate={human_bytes(int(counters['hashed_bytes']/elapsed))}/s "
                    f"files/s={done/elapsed:,.0f} errors={counters['errors']:,}",
                    flush=True,
                )
                last_report = now

    if updates:
        conn.executemany(update_sql, updates)
    if errors:
        conn.executemany("INSERT INTO scan_errors(stage,path,error) VALUES(?,?,?)", errors)
    counters["elapsed_seconds"] = int(time.time() - started)
    previous = get_meta(conn, f"{stage}_hash_stats", {})
    for key in ("hashed_files", "hashed_bytes", "errors", "elapsed_seconds"):
        counters[key] += int(previous.get(key, 0))
    set_meta(conn, f"{stage}_hash_stats", dict(counters))
    conn.commit()

    if stage == "prefix":
        print("[index] building (size,prefix_hash)", flush=True)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_inodes_prefix ON inodes(size,prefix_hash)"
        )
    elif stage == "sample":
        print("[index] building (size,prefix_hash,sample_hash)", flush=True)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_inodes_sample "
            "ON inodes(size,prefix_hash,sample_hash)"
        )
    set_meta(conn, f"{stage}_hash_complete", True)
    conn.commit()
    print(
        f"[{stage} complete] files={counters['hashed_files']:,}/{total:,} "
        f"read={human_bytes(counters['hashed_bytes'])} errors={counters['errors']:,} "
        f"elapsed={counters['elapsed_seconds']}s",
        flush=True,
    )
    return dict(counters)


def all_paths_for_inode(conn: sqlite3.Connection, dev: int, ino: int) -> list[str]:
    return [
        display_path(row[0])
        for row in conn.execute(
            "SELECT path FROM files WHERE dev=? AND ino=? ORDER BY path", (dev, ino)
        )
    ]


def prefixes(path: str, root: str, maximum_depth: int = REPORT_DEPTH) -> Iterable[str]:
    relative = os.path.relpath(path, root)
    parts = relative.split(os.sep)
    for depth in range(1, min(maximum_depth, len(parts) - 1) + 1):
        yield os.path.join(root, *parts[:depth])


def emit_reports(root: bytes, out: Path, conn: sqlite3.Connection) -> dict[str, object]:
    root_text = display_path(root)
    duplicate_json = out / "duplicates.jsonl"
    duplicate_tsv = out / "duplicates.tsv"
    hardlinks_json = out / "hardlinks.jsonl"
    zero_tsv = out / "zero_byte_files.tsv"
    errors_json = out / "errors.jsonl"
    directory_tsv = out / "directory_duplicate_summary.tsv"

    groups_query = """
        SELECT size,full_hash,COUNT(*) AS inode_count,
               SUM(allocated) AS total_allocated,
               MIN(allocated) AS min_allocated,
               MAX(allocated) AS max_allocated
        FROM inodes
        WHERE full_hash IS NOT NULL
        GROUP BY size,full_hash
        HAVING inode_count>1
        ORDER BY size*(inode_count-1) DESC, full_hash
    """
    summary: dict[str, object] = {
        "root": root_text,
        "generated_at_unix": time.time(),
        "definition": "distinct regular-file inodes with equal size and full SHA-256",
        "empty_files_excluded_from_duplicate_savings": True,
        "inventory": get_meta(conn, "inventory", {}),
        "inode_stats": get_meta(conn, "inode_stats", {}),
        "sample_hash_stats": get_meta(conn, "sample_hash_stats", {}),
        "full_hash_stats": get_meta(conn, "full_hash_stats", {}),
    }
    directory_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    duplicate_group_count = 0
    duplicate_inode_count = 0
    logical_reclaim = 0
    allocated_reclaim_reference = 0

    with duplicate_json.open("wb") as json_handle, duplicate_tsv.open(
        "w", encoding="utf-8", errors="backslashreplace"
    ) as tsv_handle:
        tsv_handle.write(
            "group_id\tsize\tinode_count\tlogical_reclaim\tsha256\tdev\tino\t"
            "allocated\thardlink_path_count\tpath\n"
        )
        for group_id, (
            size,
            full_hash,
            inode_count,
            total_allocated,
            min_allocated,
            max_allocated,
        ) in enumerate(conn.execute(groups_query), start=1):
            members = []
            member_rows = list(
                conn.execute(
                    "SELECT dev,ino,path,allocated,path_count FROM inodes "
                    "WHERE size=? AND full_hash=? ORDER BY path",
                    (size, full_hash),
                )
            )
            keep_allocated = member_rows[0][3]
            group_logical_reclaim = int(size) * (int(inode_count) - 1)
            group_allocated_reclaim = int(total_allocated) - int(keep_allocated)
            for member_index, (dev, ino, path, allocated, path_count) in enumerate(member_rows):
                aliases = all_paths_for_inode(conn, dev, ino)
                primary = display_path(path)
                members.append(
                    {
                        "dev": dev,
                        "ino": ino,
                        "allocated_bytes": allocated,
                        "paths": aliases,
                    }
                )
                for alias in aliases:
                    tsv_handle.write(
                        f"{group_id}\t{size}\t{inode_count}\t{group_logical_reclaim}\t"
                        f"{full_hash}\t{dev}\t{ino}\t{allocated}\t{path_count}\t{alias}\n"
                    )
                for prefix in prefixes(primary, root_text):
                    directory_stats[prefix][0] += int(size)
                    directory_stats[prefix][1] += 1
                    if member_index > 0:
                        directory_stats[prefix][2] += int(size)
            json_dump_line(
                json_handle,
                {
                    "group_id": group_id,
                    "sha256": full_hash,
                    "size_bytes": size,
                    "distinct_inode_count": inode_count,
                    "logical_reclaim_bytes": group_logical_reclaim,
                    "allocated_reclaim_bytes_if_keep_lexicographic_first": group_allocated_reclaim,
                    "allocated_reclaim_min_bytes": int(total_allocated) - int(max_allocated),
                    "allocated_reclaim_max_bytes": int(total_allocated) - int(min_allocated),
                    "members": members,
                },
            )
            duplicate_group_count += 1
            duplicate_inode_count += int(inode_count)
            logical_reclaim += group_logical_reclaim
            allocated_reclaim_reference += group_allocated_reclaim

    with hardlinks_json.open("wb") as handle:
        for dev, ino, size, allocated, path_count in conn.execute(
            "SELECT dev,ino,size,allocated,path_count FROM inodes "
            "WHERE path_count>1 ORDER BY size*(path_count-1) DESC"
        ):
            json_dump_line(
                handle,
                {
                    "dev": dev,
                    "ino": ino,
                    "size_bytes": size,
                    "allocated_bytes": allocated,
                    "path_count": path_count,
                    "paths": all_paths_for_inode(conn, dev, ino),
                    "additional_reclaimable_bytes": 0,
                },
            )

    with zero_tsv.open("w", encoding="utf-8", errors="backslashreplace") as handle:
        handle.write("path\n")
        for (path,) in conn.execute("SELECT path FROM files WHERE size=0 ORDER BY path"):
            handle.write(display_path(path) + "\n")

    with errors_json.open("wb") as handle:
        for stage, path, error in conn.execute(
            "SELECT stage,path,error FROM scan_errors ORDER BY stage,path"
        ):
            json_dump_line(
                handle,
                {
                    "stage": stage,
                    "path": None if path is None else display_path(path),
                    "error": error,
                },
            )
        for path, dev in conn.execute("SELECT path,dev FROM skipped_mounts ORDER BY path"):
            json_dump_line(
                handle,
                {
                    "stage": "skipped_mount",
                    "path": display_path(path),
                    "error": f"different st_dev={dev}",
                },
            )

    with directory_tsv.open("w", encoding="utf-8", errors="backslashreplace") as handle:
        handle.write(
            "directory\tduplicate_participating_bytes\tduplicate_inode_members\t"
            "redundant_bytes_by_lexicographic_reference\n"
        )
        for directory, values in sorted(
            directory_stats.items(), key=lambda item: (-item[1][2], -item[1][0], item[0])
        ):
            handle.write(f"{directory}\t{values[0]}\t{values[1]}\t{values[2]}\n")

    summary.update(
        {
            "duplicate_groups": duplicate_group_count,
            "duplicate_distinct_inodes": duplicate_inode_count,
            "logical_reclaim_bytes": logical_reclaim,
            "logical_reclaim_human": human_bytes(logical_reclaim),
            "allocated_reclaim_bytes_if_keep_lexicographic_first": allocated_reclaim_reference,
            "allocated_reclaim_human_if_keep_lexicographic_first": human_bytes(
                allocated_reclaim_reference
            ),
            "scan_error_count": conn.execute("SELECT COUNT(*) FROM scan_errors").fetchone()[0],
            "skipped_mount_count": conn.execute("SELECT COUNT(*) FROM skipped_mounts").fetchone()[0],
            "symlink_count": conn.execute("SELECT COUNT(*) FROM symlinks").fetchone()[0],
            "broken_symlink_count": conn.execute(
                "SELECT COUNT(*) FROM symlinks WHERE broken=1"
            ).fetchone()[0],
            "reports": {
                "duplicates_jsonl": str(duplicate_json),
                "duplicates_tsv": str(duplicate_tsv),
                "hardlinks_jsonl": str(hardlinks_json),
                "zero_byte_files_tsv": str(zero_tsv),
                "errors_jsonl": str(errors_json),
                "directory_summary_tsv": str(directory_tsv),
            },
        }
    )
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "[reports complete] "
        f"duplicate_groups={duplicate_group_count:,} "
        f"logical_reclaim={human_bytes(logical_reclaim)} "
        f"errors={summary['scan_error_count']:,}",
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="tree to scan")
    parser.add_argument("--out", required=True, help="new or resumable report directory")
    parser.add_argument(
        "--exclude", action="append", default=[], help="subtree to exclude; repeatable"
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="bounded concurrent readers (default: 8)"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = os.path.abspath(os.fsencode(args.root))
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    excluded = [os.path.abspath(os.fsencode(item)) for item in args.exclude]
    out_raw = os.fsencode(str(out))
    if path_is_within(out_raw, root):
        excluded.append(out_raw)

    db_path = out / "scan.sqlite3"
    conn = sqlite3.connect(db_path)
    configure_db(conn)
    requested_root = display_path(root)
    requested_excluded = [display_path(item) for item in excluded]
    stored_root = get_meta(conn, "root")
    stored_excluded = get_meta(conn, "excluded")
    if stored_root is not None and stored_root != requested_root:
        raise RuntimeError(
            f"report database belongs to root={stored_root!r}, requested={requested_root!r}"
        )
    if stored_excluded is not None and stored_excluded != requested_excluded:
        raise RuntimeError(
            "report database exclusion set differs; choose a new --out directory"
        )
    set_meta(conn, "root", requested_root)
    set_meta(conn, "excluded", requested_excluded)
    conn.commit()

    if not get_meta(conn, "inventory_complete", False):
        inventory(root, excluded, conn)
    else:
        print("[resume] inventory already complete", flush=True)
    if not get_meta(conn, "inode_table_complete", False):
        build_inode_table(conn)
    else:
        print("[resume] inode table already complete", flush=True)
    prepare_progressive_hashing(conn)
    if not get_meta(conn, "prefix_hash_complete", False):
        run_hash_stage(conn, "prefix", max(1, args.workers))
    else:
        print("[resume] prefix hashing already complete", flush=True)
    if not get_meta(conn, "sample_hash_complete", False):
        run_hash_stage(conn, "sample", max(1, args.workers))
    else:
        print("[resume] sample hashing already complete", flush=True)
    if not get_meta(conn, "full_hash_complete", False):
        run_hash_stage(conn, "full", max(1, args.workers))
    else:
        print("[resume] full hashing already complete", flush=True)
    emit_reports(root, out, conn)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
