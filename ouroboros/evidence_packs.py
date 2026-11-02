"""Whole-task xz evidence, verified readers and restore; protocol: docs/PERSISTENCE.md, evidence pack row."""
from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import lzma
import os
import pathlib
import sqlite3
import threading
import time
import uuid
import zlib
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from ouroboros import store
from ouroboros.utils import _atomic_overwrite, atomic_write_json, replace_atomic, utc_now_iso, write_bytes_atomic

PACKS_REL = pathlib.Path("observability/packs")
INDEX_NAME = "index.sqlite"
INDEX_LOCK_NAME = "packs_index.lock"  # the name lock under state/ around every index access
PUBLICATION_LOCK_NAME = "packs_publication.lock"  # manifest publication vs reference census + unlink
PUBLICATIONS_NAME = ".publications"  # pass-local refs appended before manifest publication, under the same lock
INDEX_WAIT_SEC = 5.0
UNLINK_SECTION_SEC = 0.5  # one deletion section's budget under the store write lock, publication lock included
UNLINK_SECTION_GAP_SEC = 0.15  # between sections: longer than a waiting writer's SQLite busy poll (100 ms)
PUBLICATION_WAIT_SEC = 90.0  # a manifest's contention budget for the publication lock (not a stale-owner bound)
FORMAT_VERSION = 1
INDEX_SCHEMA_VERSION = 2
CODEC, PRESET = "xz", 1
MIN_DICT_SIZE, MAX_DICT_SIZE = 4096, 64 << 20
CHECKPOINTS_REL = pathlib.PurePosixPath("source_handles/context_checkpoints")
CHECKPOINT = "checkpoint"
BLOB_KINDS = frozenset({"json", "txt", "bin"})
USAGE_ARCHIVE_REL = pathlib.Path("archive/usage_ledger")
_APPLICATION_ID = {store.TIER_ENFORCED: 0x4F555045, store.TIER_NAME: 0x4F55504E}  # 'OUPE' / 'OUPN'
_INDEX_SCHEMA = """
CREATE TABLE packs (task_id TEXT NOT NULL, generation INTEGER NOT NULL, path TEXT NOT NULL,
  raw_total INTEGER NOT NULL, packed_total INTEGER NOT NULL, verified_at TEXT, members INTEGER NOT NULL,
  PRIMARY KEY (task_id, generation));
CREATE TABLE members (sha256 TEXT NOT NULL, task_id TEXT NOT NULL, generation INTEGER NOT NULL,
  offset INTEGER NOT NULL, size INTEGER NOT NULL, kind TEXT NOT NULL, PRIMARY KEY (sha256, kind, task_id, generation));
CREATE TABLE handles (task_id TEXT NOT NULL, rel_path TEXT NOT NULL, sha256 TEXT NOT NULL,
  generation INTEGER NOT NULL, PRIMARY KEY (task_id, rel_path));
CREATE TABLE swept (task_id TEXT PRIMARY KEY, stamps TEXT NOT NULL);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""
_READ_CHUNK = 1 << 20
_READERS: Dict[str, "_PackReader"] = {}
_READERS_LOCK = threading.Lock()
_READERS_MAX = 4
_VALIDATED: Dict[str, tuple] = {}


class EvidenceCorrupt(ValueError):
    """``evidence_corrupt``: a located pack member failed verification; never empty bytes."""


class SidecarCorrupt(OSError):
    """Unreadable pack authority; restore and its preflight must refuse."""



def packs_dir(root: Any) -> pathlib.Path:
    return pathlib.Path(root) / PACKS_REL


def pack_name(task_id: str, generation: int) -> str:
    return f"{task_id}.xz" if int(generation) == 1 else f"{task_id}.{int(generation)}.xz"


def sidecar_name(task_id: str, generation: int) -> str:
    return pack_name(task_id, generation)[:-3] + ".members.json"


def dict_size_for(raw_total: int) -> int:
    """min(64 MiB, next power of two >= the task's raw total), at least 4096."""
    size = MIN_DICT_SIZE
    while size < raw_total and size < MAX_DICT_SIZE:
        size <<= 1
    return min(size, MAX_DICT_SIZE)


def read_sidecar(path: pathlib.Path, *, raw: Optional[bytes] = None) -> Dict[str, Any]:
    """Read the publication authority; a torn or invalid sidecar is never absence."""
    try:
        data = json.loads(path.read_text(encoding="utf-8") if raw is None else raw)
        if (not isinstance(data, dict) or data.get("format_version") != FORMAT_VERSION or data.get("codec") != CODEC
                or not isinstance(data.get("members"), list) or not data.get("task_id")
                or any(type(data[key]) is not int for key in ("generation", "raw_total", "packed_total", "dict_size", "preset"))
                or data["generation"] < 1 or sidecar_name(data["task_id"], data["generation"]) != path.name
                or len(data["pack_sha256"]) != 64 or data["pack_sha256"].strip("0123456789abcdef")
                or data["packed_total"] <= 0 or data["preset"] != PRESET
                or data["dict_size"] != dict_size_for(data["raw_total"])):
            raise ValueError("invalid pack header")
        offset, blobs = 0, False
        for member in sorted(data["members"], key=lambda m: m["offset"]):
            if (member["kind"] not in BLOB_KINDS | {CHECKPOINT}
                    or any(type(member[key]) is not int for key in ("size", "offset"))
                    or member["size"] < 0 or member["offset"] != offset
                    or len(member["sha256"]) != 64 or member["sha256"].strip("0123456789abcdef")):
                raise ValueError("invalid pack member")
            if member["kind"] == CHECKPOINT:
                rel = pathlib.PurePosixPath(member["path"])
                if blobs or len(rel.parts) != 3 or rel.parts[:2] != CHECKPOINTS_REL.parts or rel.stem.rsplit("-", 1)[-1] != member["sha256"]:
                    raise ValueError("invalid checkpoint prefix")
            else:
                blobs = True
            offset += member["size"]
        if offset != data["raw_total"]:
            raise ValueError("invalid pack extent")
        return data
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise SidecarCorrupt(f"evidence_corrupt: unreadable sidecar {path.name}") from exc


def _sidecar_names(root: Any) -> set[str]:
    try:
        return {name for name in os.listdir(packs_dir(root))
                if not name.startswith(".") and name.endswith(".members.json")}
    except FileNotFoundError:
        return set()


def _own_sidecar(name: str, task_id: str) -> bool:
    """A name one of this task's generations may carry: the exact spelling, including task IDs
    containing dots/digits (``a.2`` is ``a`` generation 2 or ``a.2`` generation 1; the sidecar's
    own ``task_id`` decides). A sidecar is valid only under its own name, so no other name holds
    this task's members."""
    prefix = name.removesuffix(".members.json")
    return prefix == task_id or (prefix.startswith(task_id + ".") and prefix[len(task_id) + 1:].isdigit())


def _task_sidecar_names(root: Any, task_id: str) -> set[str]:
    """This task's sidecar names, found by their spelling instead of listing the directory. The
    packer gives a task the first free generation after its newest and no published name is ever
    removed, so the task's occupied generations run from 1 (a torn pack, or a homonym's sidecar
    such as ``a.2`` generation 1 for ``a``, occupies one too) and the first free one ends them."""
    directory, names, generation = packs_dir(root), set(), 1
    while True:
        name = sidecar_name(task_id, generation)
        if (directory / name).exists():
            names.add(name)
        elif not (directory / pack_name(task_id, generation)).exists():
            return names
        generation += 1


def list_sidecars(root: Any, *, errors: Optional[list] = None,
                  names: Optional[set[str]] = None) -> List[Tuple[pathlib.Path, Dict[str, Any]]]:
    """Read pack authority, or a rebuild's filename delta; cache builders may collect faults."""
    out = []
    directory = packs_dir(root)
    for name in sorted(_sidecar_names(root) if names is None else names):
        try:
            out.append((directory / name, read_sidecar(directory / name)))
        except SidecarCorrupt as exc:
            if errors is None:
                raise
            errors.append(exc)
    return out



class _PackReader:
    """Locked cursor; LZMAFile seeks forward cheaply and rewinds for earlier offsets."""

    def __init__(self, path: pathlib.Path) -> None:
        self.path = pathlib.Path(path)
        self._lock = threading.Lock()
        self._identity = self._stream = None

    def read_at(self, offset: int, size: int, sink: Callable[[bytes], None]) -> None:
        with self._lock:
            try:
                info = self.path.stat()
                identity = (info.st_ino, info.st_dev, info.st_size, info.st_mtime_ns)
                if self._stream is None or identity != self._identity:
                    if self._stream is not None:
                        self._stream.close()
                    self._stream = lzma.LZMAFile(self.path, "rb")
                    self._identity = identity
                self._stream.seek(offset)
                while size:
                    chunk = self._stream.read(min(size, _READ_CHUNK))
                    if not chunk:
                        raise EvidenceCorrupt(f"evidence_corrupt: pack stream ends early: {self.path.name}")
                    size -= len(chunk)
                    sink(chunk)
            except Exception as exc:
                if self._stream is not None:
                    self._stream.close()
                    self._stream = None
                if isinstance(exc, (lzma.LZMAError, EOFError)):
                    raise EvidenceCorrupt(f"evidence_corrupt: pack stream unreadable: {self.path.name}") from exc
                raise


def _reader(path: pathlib.Path) -> _PackReader:
    key = str(path)
    with _READERS_LOCK:
        reader = _READERS.get(key)
        if reader is None:
            if len(_READERS) >= _READERS_MAX:
                _READERS.pop(next(iter(_READERS)))
            reader = _READERS[key] = _PackReader(path)
        return reader


def member_bytes(pack_path: pathlib.Path, member: Dict[str, Any], *, anchor: Any = None) -> bytes:
    """One member's exact bytes, verified against the sidecar's size and sha256."""
    parts: List[bytes] = []
    if anchor is None:
        _reader(pack_path).read_at(int(member["offset"]), int(member["size"]), parts.append)
    else:
        from ouroboros.confined_files import open_confined_file
        handle, _ = open_confined_file(anchor, pack_path.relative_to(anchor).parts)
        try:
            with handle, lzma.LZMAFile(handle, "rb") as stream:
                stream.seek(int(member["offset"]))
                parts.append(stream.read(int(member["size"])))
        except (lzma.LZMAError, EOFError) as exc:
            raise EvidenceCorrupt(f"evidence_corrupt: pack stream unreadable: {pack_path.name}") from exc
    raw = b"".join(parts)
    if len(raw) != int(member["size"]) or hashlib.sha256(raw).hexdigest() != member["sha256"]:
        raise EvidenceCorrupt(f"evidence_corrupt: member {member['sha256'][:12]} of {pack_path.name} failed verification")
    return raw


def verify_pack(pack_path: pathlib.Path, sidecar: Dict[str, Any]) -> None:
    """Re-read the pack from disk: the file's sha256 and EVERY member's size and sha256."""
    digest = hashlib.sha256()
    with open(pack_path, "rb") as handle:
        while chunk := handle.read(_READ_CHUNK):
            digest.update(chunk)
    if digest.hexdigest() != sidecar.get("pack_sha256"):
        raise EvidenceCorrupt(f"evidence_corrupt: pack file sha256 mismatch: {pack_path.name}")
    reader = _PackReader(pack_path)  # a private cursor: verification never shares a resolver's position
    for member in sorted(sidecar["members"], key=lambda item: int(item["offset"])):
        member_digest, seen = hashlib.sha256(), [0]

        def sink(chunk: bytes) -> None:
            member_digest.update(chunk)
            seen[0] += len(chunk)

        reader.read_at(int(member["offset"]), int(member["size"]), sink)
        if seen[0] != int(member["size"]) or member_digest.hexdigest() != member["sha256"]:
            raise EvidenceCorrupt(f"evidence_corrupt: member {member['sha256'][:12]} of {pack_path.name} failed verification")



def index_tier(root: pathlib.Path) -> str:
    from ouroboros.platform_layer import kernel_file_locks_enforced

    return store.TIER_ENFORCED if kernel_file_locks_enforced(root / "state" / store.LOCK_NAME) else store.TIER_NAME


def _index_stamp(path: pathlib.Path) -> Optional[tuple]:
    identity = store.file_identity(path)
    if identity is None:
        return None
    try:
        info = os.stat(path)
    except OSError:
        return None
    return (*identity, info.st_size, info.st_mtime_ns)


def _index_validated(path: pathlib.Path) -> bool:
    """This process already validated these exact index bytes; nothing is checked here."""
    stamp = _index_stamp(path)
    return stamp is not None and _VALIDATED.get(str(path)) == stamp


def _index_valid(path: pathlib.Path, tier: str, *, wait_sec: float = INDEX_WAIT_SEC) -> bool:
    stamp = _index_stamp(path)
    if stamp is None:
        return False
    if _VALIDATED.get(str(path)) == stamp:
        return True
    if store.header_tier(path, _APPLICATION_ID) != tier:
        return False
    try:
        with contextlib.closing(store.connect(path, tier, wait_sec=wait_sec)) as conn:
            ok = conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
            version = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    except sqlite3.OperationalError as exc:
        if store.busy(exc):
            raise store.StoreBusy("evidence index validation busy", reason="contention") from exc
        return False
    except sqlite3.Error:
        return False
    valid = ok and version is not None and version[0] == str(INDEX_SCHEMA_VERSION)
    if valid:
        _VALIDATED[str(path)] = stamp
    return valid


def _index_rows(conn: sqlite3.Connection, pack_path: pathlib.Path, sidecar: Dict[str, Any]) -> None:
    task_id, generation = str(sidecar["task_id"]), int(sidecar["generation"])
    conn.execute("DELETE FROM members WHERE task_id=? AND generation=?", (task_id, generation))
    conn.execute("DELETE FROM handles WHERE task_id=? AND generation=?", (task_id, generation))
    conn.execute("INSERT OR REPLACE INTO packs VALUES (?,?,?,?,?,?,?)", (
        task_id, generation, pack_path.name, int(sidecar.get("raw_total") or 0), int(sidecar.get("packed_total") or 0),
        sidecar.get("verified_at"), len(sidecar["members"])))
    for member in sidecar["members"]:
        conn.execute("INSERT OR REPLACE INTO members VALUES (?,?,?,?,?,?)", (
            member["sha256"], task_id, generation, int(member["offset"]), int(member["size"]), member["kind"]))
        if member["kind"] == CHECKPOINT and member.get("path"):
            conn.execute("INSERT OR REPLACE INTO handles VALUES (?,?,?,?)",
                         (task_id, str(member["path"]), member["sha256"], generation))


def _index_wait(wait_sec: float = INDEX_WAIT_SEC) -> float:
    return store.bounded_wait(min(wait_sec, INDEX_WAIT_SEC))


def _rebuild_index(root: pathlib.Path, path: pathlib.Path, tier: str, *, wait_sec: float = INDEX_WAIT_SEC) -> None:
    """Build privately, refresh changed sidecars unlocked, then publish a still-current snapshot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:8]}")
    indexed: set[str] = set()
    names, faults = _sidecar_names(root), {}
    try:
        with contextlib.closing(store.connect(tmp, tier, create=True)) as conn:
            conn.execute(f"PRAGMA application_id = {_APPLICATION_ID[tier]}")
            conn.executescript(_INDEX_SCHEMA)
            with store.transaction(conn, write=True):
                conn.execute("INSERT INTO meta VALUES ('schema_version', ?)", (str(INDEX_SCHEMA_VERSION),))
                conn.execute("INSERT INTO meta VALUES ('built_at', ?)", (utc_now_iso(),))
        while True:
            added, removed = names - indexed, indexed - names
            errors: list = []
            sidecars = list_sidecars(root, errors=errors, names=added)
            for name in removed:
                faults.pop(name, None)
            failed = sorted(added - {sidecar_path.name for sidecar_path, _ in sidecars})
            faults.update(zip(failed, map(str, errors)))
            with contextlib.closing(store.connect(tmp, tier)) as conn:
                with store.transaction(conn, write=True):
                    for name in removed:
                        row = conn.execute("SELECT task_id, generation FROM packs WHERE path=?",
                                           (name.removesuffix(".members.json") + ".xz",)).fetchone()
                        if row is not None:
                            for table in ("packs", "members", "handles"):
                                conn.execute(f"DELETE FROM {table} WHERE task_id=? AND generation=?", tuple(row))
                    for sidecar_path, sidecar in sidecars:
                        _index_rows(conn, sidecar_path.with_name(pack_name(sidecar["task_id"], sidecar["generation"])), sidecar)
                    conn.execute("INSERT OR REPLACE INTO meta VALUES ('sidecar_errors', ?)",
                                 (json.dumps(list(faults.values())),))
            indexed = names
            with store.name_lock(root, INDEX_LOCK_NAME, timeout_sec=_index_wait(wait_sec)):
                if _index_valid(path, tier, wait_sec=_index_wait(wait_sec)):
                    return  # another builder published; keep its subsequent writes too
                names = _sidecar_names(root)
                if names == indexed:
                    with contextlib.suppress(FileNotFoundError):
                        tmp.chmod(path.stat().st_mode)
                    replace_atomic(tmp, path)
                    store.fsync_directory(path.parent)
                    return
    finally:
        for leftover in (tmp, *(tmp.with_name(tmp.name + suffix) for suffix in ("-journal", "-wal", "-shm"))):
            with contextlib.suppress(FileNotFoundError):
                leftover.unlink()


@contextlib.contextmanager
def open_index(root: Any, *, write: bool = False, wait_sec: float = INDEX_WAIT_SEC,
               rebuild: bool = True) -> Iterator[sqlite3.Connection]:
    """One transaction under the name lock, narrowed by the caller's acquisition budget.

    Missing, damaged or other-version caches rebuild outside the lock. ``rebuild=False``
    is advisory discovery: unavailable cache raises instead of walking all sidecars.
    ``store.StoreBusy`` on contention.
    """
    root = pathlib.Path(root)
    path, tier = packs_dir(root) / INDEX_NAME, index_tier(root)
    wait_sec = _index_wait(wait_sec)
    while True:
        with store.name_lock(root, INDEX_LOCK_NAME, timeout_sec=wait_sec):
            if _index_valid(path, tier, wait_sec=wait_sec):
                with contextlib.closing(store.connect(path, tier, wait_sec=wait_sec)) as conn:
                    with store.transaction(conn, write=write):
                        yield conn
                return
            if not rebuild:
                raise sqlite3.DatabaseError("evidence index requires reconstruction")
        _rebuild_index(root, path, tier, wait_sec=wait_sec)


def index_upsert(root: Any, pack_path: pathlib.Path, sidecar: Dict[str, Any]) -> None:
    """Record one verified pack, or defer its caller without destroying a contended cache."""
    with open_index(root, write=True) as conn:
        _index_rows(conn, pack_path, sidecar)


def _placements(root: pathlib.Path, *, sha256: str = "", kind: str = "", task_id: str = "", rel_path: str = "",
                ) -> List[Tuple[pathlib.Path, Dict[str, Any]]]:
    """All indexed placements, or healthy sidecar rows when the cache is unavailable; a task's
    checkpoint handle is addressed by its task (``_task_handles``)."""
    if not packs_dir(root).is_dir():
        return []  # a root without packs is left untouched (no cache, no lock file)
    if rel_path:
        return _task_handles(root, task_id, rel_path)
    try:
        with open_index(root) as conn:
            rows = conn.execute(
                "SELECT p.path AS pack, m.sha256, m.offset, m.size, m.kind FROM members m "
                "JOIN packs p ON p.task_id=m.task_id AND p.generation=m.generation "
                "WHERE m.sha256=? AND m.kind=? ORDER BY p.task_id, p.generation", (sha256, kind)).fetchall()
            if not rows:
                error_row = conn.execute("SELECT value FROM meta WHERE key='sidecar_errors'").fetchone()
                if error_row and json.loads(error_row[0]):
                    raise SidecarCorrupt(json.loads(error_row[0])[0])
        return [(packs_dir(root) / row["pack"], dict(row)) for row in rows]
    except (store.StoreError, sqlite3.Error, OSError):
        pass  # the cache is unavailable: the sidecars are the authority
    errors, found = [], []
    for sidecar_path, sidecar in list_sidecars(root, errors=errors):
        for member in sidecar["members"]:
            if member["sha256"] == sha256 and member["kind"] == kind:
                found.append((sidecar_path.with_name(pack_name(sidecar["task_id"], sidecar["generation"])), member))
    if errors and not found:
        raise errors[0]
    return found


def _task_handles(root: pathlib.Path, task_id: str, rel_path: str = "") -> List[Tuple[pathlib.Path, Dict[str, Any]]]:
    """This task's packed checkpoint handles (``rel_path``: that one), read as an addressed fact.

    A handle is named by its task and the task's own sidecars are its authority, so no other
    task's pack is read: the index answers while this process has validated its current bytes and
    its lock is free now; otherwise, and for an absence it cannot prove (it recorded unreadable
    sidecars), the task's own sidecars answer, found by name (``_task_sidecar_names``). Only when
    one of them is unreadable does a valid index that still holds the placement keep it (checked,
    never rebuilt). The ordinary read never lists the directory or waits for, re-validates or
    rebuilds the shared cache: those whole-pack passes stay with the readers that need every pack
    (``blob_bytes``) and with the packer."""
    rows = _indexed_handles(root, task_id, rel_path, validated=True)
    if rows is not None:
        return rows
    errors, found = [], []
    for sidecar_path, sidecar in list_sidecars(root, errors=errors, names=_task_sidecar_names(root, task_id)):
        for member in sidecar["members"]:
            if sidecar["task_id"] == task_id and member["kind"] == CHECKPOINT and member.get("path") and (
                    not rel_path or member["path"] == rel_path):
                found.append((sidecar_path.with_name(pack_name(task_id, sidecar["generation"])), member))
    if errors and not found:
        kept = _indexed_handles(root, task_id, rel_path, validated=False)
        if kept:
            return kept
        raise errors[0]
    return found


_HANDLES_SQL = ("SELECT p.path AS pack, m.sha256, m.offset, m.size, m.kind, h.rel_path AS path FROM handles h "
                "JOIN members m ON m.sha256=h.sha256 AND m.task_id=h.task_id AND m.generation=h.generation "
                "JOIN packs p ON p.task_id=h.task_id AND p.generation=h.generation WHERE h.task_id=? AND m.kind=?")


def _indexed_handles(root: pathlib.Path, task_id: str, rel_path: str, *,
                     validated: bool) -> Optional[List[Tuple[pathlib.Path, Dict[str, Any]]]]:
    """The index's rows for this task's handles, or ``None`` when it cannot answer. ``validated``
    reads only bytes this process already validated and never waits for the lock; otherwise
    ``open_index`` checks the existing index under the caller's budget, never rebuilding it."""
    path = packs_dir(root) / INDEX_NAME
    try:
        with contextlib.ExitStack() as held:
            if not validated:
                conn = held.enter_context(open_index(root, rebuild=False))
            else:
                held.enter_context(store.name_lock(root, INDEX_LOCK_NAME, timeout_sec=0))
                if not _index_validated(path):
                    return None
                conn = held.enter_context(contextlib.closing(store.connect(path, index_tier(root), wait_sec=0)))
                held.enter_context(store.transaction(conn, write=False))
            rows = conn.execute(_HANDLES_SQL + (" AND h.rel_path=?" if rel_path else ""),
                                (task_id, CHECKPOINT, *((rel_path,) if rel_path else ()))).fetchall()
            recorded = conn.execute("SELECT value FROM meta WHERE key='sidecar_errors'").fetchone()
            if not rows and recorded and json.loads(recorded[0]):
                return None  # an unreadable sidecar may be this task's: the index proves no absence
    except (store.StoreError, sqlite3.Error, OSError, ValueError):
        return None
    return [(packs_dir(root) / row["pack"], dict(row)) for row in rows]


def _locate(root: pathlib.Path, **identity: str) -> Optional[Tuple[pathlib.Path, Dict[str, Any]]]:
    return next(iter(_placements(root, **identity)), None)



def blob_ref_fields(ref: Any) -> Tuple[str, str, int]:
    """``(sha256, kind, size)`` of a gzip CAS blob ref (size -1 when the ref carries none), or ``ValueError``."""
    if not isinstance(ref, dict):
        raise ValueError("observability blob ref must be an object")
    kind, sha256 = str(ref.get("kind") or ""), str(ref.get("sha256") or "")
    if kind not in BLOB_KINDS or ref.get("encoding") != "gzip":
        raise ValueError("observability blob ref has an unexpected kind or encoding")
    if len(sha256) != 64 or sha256.strip("0123456789abcdef"):
        raise ValueError("observability blob ref has no sha256")
    try:
        return sha256, kind, -1 if ref.get("size") is None else int(ref["size"])
    except (TypeError, ValueError) as exc:
        raise ValueError("observability blob ref has no valid size") from exc


def _blob_roots(root: pathlib.Path, ref: Dict[str, Any], sha256: str, kind: str) -> List[pathlib.Path]:
    """This root, then the retained child root an absolute ref path names (the same function)."""
    from ouroboros.source_retention import retained_child_root

    roots = [root]
    original = pathlib.Path(str(ref.get("path") or ""))
    if original.is_absolute() and original.parts[-3:] == ("observability", "blobs", f"{sha256}.{kind}.gz"):
        child = pathlib.Path(*original.parts[:-3])
        if child != root and retained_child_root(root, child):
            roots.append(child)
    return roots


def read_loose_gzip(path: pathlib.Path, max_bytes: Optional[int] = None) -> bytes:
    with gzip.open(path, "rb") as handle:
        raw = handle.read() if max_bytes is None else handle.read(max_bytes + 1)
    if max_bytes is not None and len(raw) > max_bytes:
        raise ValueError("blob exceeds byte limit")
    return raw


def blob_bytes(root: Any, ref: Any, *, expected_kind: Optional[str] = None, max_bytes: Optional[int] = None,
               admit: Optional[Callable[[pathlib.Path], None]] = None) -> bytes:
    """Verified loose bytes, then every pack placement, then admissible retained roots.

    ``admit`` (optional) sees each loose or pack file before it is opened; what it
    raises propagates, so a refusal never becomes a miss, corruption or fallback."""
    root = pathlib.Path(root)
    sha256, kind, size = blob_ref_fields(ref)
    if expected_kind is not None and kind != expected_kind:
        raise ValueError("observability blob ref has an unexpected kind or encoding")
    if max_bytes is not None and size > max_bytes:
        raise ValueError("blob exceeds byte limit")
    failure: Optional[Exception] = None
    try:
        for candidate in _blob_roots(root, ref, sha256, kind):
            loose = candidate / "observability" / "blobs" / f"{sha256}.{kind}.gz"
            if admit is not None:
                admit(loose)
            try:
                raw = read_loose_gzip(loose, max_bytes)
                if size in (-1, len(raw)) and hashlib.sha256(raw).hexdigest() == sha256:
                    return raw
                failure = EvidenceCorrupt("evidence_corrupt: observability blob ref failed size or sha256 verification")
            except FileNotFoundError as exc:
                failure = failure or exc
            except ValueError as exc:
                failure = exc
            except (OSError, EOFError, zlib.error) as exc:
                failure = exc  # retention repair distinguishes transient IO from damaged gzip
            try:
                placements = _placements(candidate, sha256=sha256, kind=kind)
            except (OSError, EvidenceCorrupt) as exc:
                failure = exc
                continue
            for located in placements:
                if admit is not None:
                    admit(located[0])
                try:
                    if max_bytes is not None and int(located[1]["size"]) > max_bytes:
                        raise ValueError("blob exceeds byte limit")
                    raw = member_bytes(*located)
                    if size not in (-1, len(raw)):
                        raise EvidenceCorrupt(f"evidence_corrupt: packed blob {sha256[:12]} size differs from its ref")
                    return raw
                except FileNotFoundError as exc:
                    failure = failure or exc
                except ValueError as exc:
                    failure = exc
                except OSError:
                    failure = EvidenceCorrupt(f"evidence_corrupt: unreadable pack {located[0].name}")
        raise failure or FileNotFoundError(str(root / "observability" / "blobs" / f"{sha256}.{kind}.gz"))
    finally:
        # The kept error's traceback holds this frame: kept past an exit, it is a cycle retaining the callers'
        # frames and payloads until a GC pass. Every exit drops it; a raised error keeps its object and traceback.
        failure = None


def _handle_path(root: Any, task_id: str, rel_path: str) -> pathlib.Path:
    from ouroboros.artifacts import task_artifact_dir_path

    return task_artifact_dir_path(root, task_id, create=False).joinpath(*pathlib.PurePosixPath(rel_path).parts)


def _confined_handle(root: Any, task_id: str, rel_path: str, anchor: Any, *, info_only: bool = False) -> Any:
    """Preserve admitted roots' placement authority; confine their payload opens.

    Physical scratch custody alone never admits a pathname-opened SQLite index.
    Such retained sources use confined sidecars without granting store authority.
    """
    from ouroboros.confined_files import (
        confined_directory,
        open_confined_file,
        physical_path,
        relative_parts,
        root_spellings,
    )
    from ouroboros.task_results import validate_task_id

    task_id = validate_task_id(task_id)
    if not relative_parts(rel_path):
        raise ValueError("invalid checkpoint path")
    spellings = root_spellings(anchor)  # ANCHOR as configured; its rows record that spelling
    anchor, root = spellings[0], physical_path(spellings, pathlib.Path(root).absolute())
    loose = _handle_path(root, task_id, rel_path)
    digest, failure = pathlib.PurePosixPath(rel_path).stem.rsplit("-", 1)[-1], None
    try:
        handle, _ = open_confined_file(anchor, loose.relative_to(anchor).parts)
        with handle:
            raw = handle.read()
        if hashlib.sha256(raw).hexdigest() == digest:
            return {"size": len(raw), "sha256": digest, "loose": True} if info_only else raw
        failure = EvidenceCorrupt("evidence_corrupt: loose handle failed sha256 verification")
    except FileNotFoundError:
        pass
    except OSError as exc:
        if not info_only:
            raise
        failure = exc
    if info_only and pathlib.PurePosixPath(rel_path).parts[:2] != CHECKPOINTS_REL.parts:
        if failure is not None:
            raise failure
        return None
    from ouroboros.task_custody import recorded_execution_drive
    from ouroboros.task_store import read_document

    if root == anchor or recorded_execution_drive(read_document(anchor, task_id) or {}, root, spellings):
        located = _locate(root, task_id=task_id, rel_path=rel_path)
        if located is not None:
            return ({"size": int(located[1]["size"]), "sha256": located[1]["sha256"], "loose": False}
                    if info_only else member_bytes(*located, anchor=anchor))
        if failure is not None:
            raise failure
        if info_only:
            return None
        raise FileNotFoundError(str(loose))
    try:
        held = confined_directory(anchor, packs_dir(root).relative_to(anchor).parts)
        directory = held.__enter__()
    except FileNotFoundError:
        directory = None
    if directory is not None:
        try:
            names = sorted(name for name, _ in directory.entries(include_links=True)
                           if not name.startswith(".") and name.endswith(".members.json"))
            errors = []
            for name in [*(name for name in names if _own_sidecar(name, task_id)),
                         *(name for name in names if not _own_sidecar(name, task_id))]:
                try:
                    try:
                        handle, _ = directory.open(name)
                        with handle:
                            sidecar = read_sidecar(packs_dir(root) / name, raw=handle.read())
                    except OSError as exc:
                        raise SidecarCorrupt(f"evidence_corrupt: unreadable sidecar {name}") from exc
                    if sidecar["task_id"] != task_id:
                        continue
                    for member in sidecar["members"]:
                        if member["kind"] != CHECKPOINT or member.get("path") != rel_path:
                            continue
                        if info_only:
                            return {"size": int(member["size"]), "sha256": member["sha256"], "loose": False}
                        return member_bytes(packs_dir(root) / pack_name(task_id, sidecar["generation"]),
                                            member, anchor=anchor)
                except SidecarCorrupt as exc:
                    errors.append((name, exc))
            if errors:
                raise min(errors, key=lambda item: item[0])[1]
        finally:
            held.__exit__(None, None, None)
    if failure is not None:
        raise failure
    if info_only:
        return None
    raise FileNotFoundError(str(loose))


def handle_bytes(root: Any, task_id: str, rel_path: str, *, anchor: Any = None) -> bytes:
    """Verified loose handle bytes, falling back to the pack if the loose digest is wrong."""
    if anchor is not None:
        return _confined_handle(root, task_id, rel_path, anchor)
    rel = pathlib.PurePosixPath(str(rel_path))
    loose, failure = _handle_path(root, task_id, rel.as_posix()), None
    try:
        if not loose.is_symlink():
            raw = loose.read_bytes()
            digest = rel.stem.rsplit("-", 1)[-1]
            if hashlib.sha256(raw).hexdigest() == digest:
                return raw
            failure = EvidenceCorrupt(f"evidence_corrupt: loose handle {rel.name} failed sha256 verification")
    except FileNotFoundError:
        pass
    located = _locate(pathlib.Path(root), task_id=str(task_id), rel_path=rel.as_posix())
    if located is None:
        raise failure or FileNotFoundError(str(loose))
    return member_bytes(*located)


def handle_info(root: Any, task_id: str, rel_path: str, *, anchor: Any = None) -> Optional[Dict[str, Any]]:
    """Verified placement metadata; ``None`` only for absence, corruption stays explicit."""
    if anchor is not None:
        return _confined_handle(root, task_id, rel_path, anchor, info_only=True)
    rel = pathlib.PurePosixPath(str(rel_path))
    loose = _handle_path(root, task_id, rel.as_posix())
    failure = None
    try:
        if loose.is_file() and not loose.is_symlink():
            raw, digest = loose.read_bytes(), rel.stem.rsplit("-", 1)[-1]
            if hashlib.sha256(raw).hexdigest() == digest:
                return {"size": len(raw), "sha256": digest, "loose": True}
            failure = EvidenceCorrupt(f"evidence_corrupt: loose handle {rel.name} failed sha256 verification")
    except OSError as exc:
        failure = exc
    if rel.parts[:2] == CHECKPOINTS_REL.parts:
        located = _locate(pathlib.Path(root), task_id=str(task_id), rel_path=rel.as_posix())
        if located is not None:
            return {"size": int(located[1]["size"]), "sha256": located[1]["sha256"], "loose": False}
    if failure is not None:
        raise failure
    return None


def packed_handle_names(root: Any, task_id: str) -> List[str]:
    """Names of this task's packed checkpoint handles (for directory listings)."""
    if not packs_dir(root).is_dir():
        return []
    return sorted({pathlib.PurePosixPath(member["path"]).name
                   for _, member in _task_handles(pathlib.Path(root), str(task_id))})


def _checkpoint_path(path: pathlib.Path) -> Optional[tuple]:
    """Parse the artifact root, checkpoint directory or one checkpoint file once."""
    from ouroboros.task_results import validate_task_id

    parts = pathlib.Path(path).parts
    k = next((i for i in range(len(parts) - 3, -1, -1) if parts[i:i + 2] == ("task_results", "artifacts")), None)
    rest = parts[k + 3:] if k is not None else None
    if rest is None or len(rest) > 3 or rest[:2] != CHECKPOINTS_REL.parts[:len(rest)]:
        return None
    try:
        return pathlib.Path(*parts[:k]), validate_task_id(parts[k + 2]), rest
    except ValueError:
        return None


def packed_handle_names_at(directory: pathlib.Path) -> List[str]:
    parsed = _checkpoint_path(directory)
    return packed_handle_names(*parsed[:2]) if parsed and len(parsed[2]) == 2 else []


def _restore_handle(target: pathlib.Path, raw: bytes, sha256: str) -> bool:
    """Write one handle beside its name (temp + rename), skipping a present file."""
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise EvidenceCorrupt(f"evidence_corrupt: handle {target.name} failed verification")
    if target.exists():
        return False
    write_bytes_atomic(target, raw)
    return True


def materialize_handle(root: Any, task_id: str, rel_path: str) -> Optional[pathlib.Path]:
    """The loose path of a checkpoint handle, written back from its pack when absent."""
    rel = pathlib.PurePosixPath(str(rel_path))
    target = _handle_path(root, task_id, rel.as_posix())
    if target.exists() or rel.parts[:2] != CHECKPOINTS_REL.parts:
        return target if target.exists() else None
    located = _locate(pathlib.Path(root), task_id=str(task_id), rel_path=rel.as_posix())
    if located is None:
        return None
    _restore_handle(target, member_bytes(*located), located[1]["sha256"])
    return target


def materialize_path(path: pathlib.Path) -> Optional[pathlib.Path]:
    parsed = _checkpoint_path(path)
    return materialize_handle(*parsed[:2], pathlib.PurePosixPath(*parsed[2]).as_posix()) if parsed and len(parsed[2]) == 3 else None


def materialize_tree(path: pathlib.Path) -> None:
    """Restore handles below the task artifact root, checkpoint directory or exact path."""
    parsed = _checkpoint_path(path)
    if parsed:
        root, task_id, rest = parsed
        for name in ([rest[2]] if len(rest) == 3 else packed_handle_names(root, task_id)):
            materialize_handle(root, task_id, (CHECKPOINTS_REL / name).as_posix())


def _pack_roots(root: pathlib.Path) -> List[pathlib.Path]:
    from ouroboros.store_import import descendant_drives

    return [root, *(child for child in descendant_drives(root) if packs_dir(child).is_dir())]


def _absent_checkpoints(root: Any, sidecar: Dict[str, Any]) -> Iterator[tuple]:
    for member in sorted((m for m in sidecar["members"] if m["kind"] == CHECKPOINT and m.get("path")), key=lambda m: int(m["offset"])):
        target = _handle_path(root, sidecar["task_id"], member["path"])
        if not target.exists():
            yield member, target


def restore_checkpoint_handles(root: Any) -> Dict[str, Any]:
    """Write every packed checkpoint handle back to its loose name (prefix decode only; blobs stay
    packed) on this root and its retained child roots; ``OSError`` on ANY failure so the switch
    seam refuses before anything moves."""
    report = {"status": "no_packs", "handles": 0, "bytes": 0, "packs": 0}
    try:
        for candidate in _pack_roots(pathlib.Path(root)):
            for sidecar_path, sidecar in list_sidecars(candidate):
                report["packs"] += 1
                reader = _PackReader(sidecar_path.with_name(pack_name(sidecar["task_id"], sidecar["generation"])))
                for member, target in _absent_checkpoints(candidate, sidecar):
                    parts: List[bytes] = []
                    reader.read_at(int(member["offset"]), int(member["size"]), parts.append)
                    if _restore_handle(target, b"".join(parts), member["sha256"]):
                        report["handles"] += 1
                        report["bytes"] += int(member["size"])
    except (ValueError, KeyError, TypeError) as exc:
        raise OSError(f"evidence pack restore failed: {exc}") from exc
    if report["packs"]:
        report["status"] = "restored"
    return report


def pending_restore_bytes(root: Any) -> int:
    """Raw bytes of packed checkpoint handles whose loose file is absent (the preflight's sum)."""
    total = 0
    for candidate in _pack_roots(pathlib.Path(root)):
        for _, sidecar in list_sidecars(candidate):
            _handle_path(candidate, sidecar["task_id"], "")  # validate even a pack with no checkpoint members
            total += sum(int(member["size"]) for member, _ in _absent_checkpoints(candidate, sidecar))
    return total



def row_packable(status: Any, review_pending: Any) -> bool:
    """The packing rule over a row's two indexed facts: the status is terminal, and no review of the
    task may still read its loose files (``review_collection_pending`` 1 names an open review
    operation, whose agent-session reader holds a retained handle's absolute path, or a row an older
    body flagged for its acceptance collection until the maintenance pass writes it back
    (``task_store.reclassify_review_collection``); NULL is a row no body with the selector ever wrote, so no such
    reader exists, and the selector stays a derived fact, never a retention duty). No duty, money or
    delivery fact is consulted (packing moves verified bytes, never content: a later write becomes
    the next generation)."""
    from ouroboros.task_status import SETTLED_STATUSES

    return status is not None and str(status) in SETTLED_STATUSES and review_pending != 1


def task_packable(txn: store.Txn, task_id: str) -> bool:
    """``row_packable`` over the task's row (``False`` without one)."""
    row = txn.row(str(task_id), "status", "review_collection_pending")
    return row is not None and row_packable(*row)



def _walk_blob_refs(value: Any, out: List[Tuple[str, str, int]], seen: set) -> None:
    if isinstance(value, dict):
        try:
            sha256, kind, size = blob_ref_fields(value)
        except ValueError:
            for item in value.values():
                _walk_blob_refs(item, out, seen)
            return
        if size >= 0 and (kind, sha256) not in seen:
            seen.add((kind, sha256))
            out.append((kind, sha256, size))
    elif isinstance(value, list):
        for item in value:
            _walk_blob_refs(item, out, seen)


def note_manifest_publication(root: Any, manifest: dict) -> None:
    """Called under the publication lock BEFORE the manifest write. Track new refs only
    while a sweep owns the scratch file; a stale file is reset at the next sweep."""
    try:
        fd = os.open(packs_dir(root) / PUBLICATIONS_NAME, os.O_WRONLY | os.O_APPEND | getattr(os, "O_BINARY", 0))
    except FileNotFoundError:
        return
    with os.fdopen(fd, "ab") as journal:
        refs: list = []
        _walk_blob_refs(manifest, refs, set())
        journal.write("".join(sha + "\n" for _, sha, _ in refs).encode("ascii"))


def manifest_blob_refs(root: Any, task_id: str) -> List[Tuple[str, str, int]]:
    """``(kind, sha256, size)`` of every blob the task's call manifests reference, first appearance first."""
    from ouroboros.deadline_utils import parse_deadline_ts

    out: List[Tuple[str, str, int]] = []
    seen: set = set()
    manifests = []
    directory = pathlib.Path(root) / "observability" / "calls" / str(task_id)
    try:
        entries = list(os.scandir(directory))
    except FileNotFoundError:
        return out
    for entry in entries:
        if entry.name.startswith(".") or not entry.name.endswith(".json"):
            continue
        with contextlib.suppress(OSError, ValueError):
            value = json.loads(pathlib.Path(entry.path).read_text(encoding="utf-8"))
            created = parse_deadline_ts(value.get("created_at")) if isinstance(value, dict) else None
            first = created.timestamp() if created else entry.stat(follow_symlinks=False).st_mtime_ns / 1e9
            manifests.append((first, entry.name, value))
    for _, _, value in sorted(manifests):
        _walk_blob_refs(value, out, seen)
    return out


def loose_members(root: Any, task_id: str, *, missing: Optional[list] = None) -> List[Dict[str, Any]]:
    """The task's members whose bytes are at hand (loose checkpoint handles first, then the manifest
    blobs that are loose or already held by some pack on this root), each with its loose source path;
    a referenced blob that is neither is appended to ``missing`` as ``(kind, sha256, size)``."""
    from ouroboros.artifacts import task_artifact_dir_path

    root = pathlib.Path(root)
    members: List[Dict[str, Any]] = []
    store_dir = task_artifact_dir_path(root, task_id, create=False).joinpath(*CHECKPOINTS_REL.parts)
    with contextlib.suppress(OSError):
        for entry in sorted(os.scandir(store_dir), key=lambda e: e.name):
            stem = entry.name.rsplit("-", 1)[-1].split(".", 1)[0]
            if entry.is_file(follow_symlinks=False) and len(stem) == 64 and not stem.strip("0123456789abcdef"):
                members.append({"sha256": stem, "kind": CHECKPOINT, "size": entry.stat(follow_symlinks=False).st_size,
                                "path": (CHECKPOINTS_REL / entry.name).as_posix(), "source": pathlib.Path(entry.path)})
    for kind, sha256, size in manifest_blob_refs(root, task_id):
        loose = root / "observability" / "blobs" / f"{sha256}.{kind}.gz"
        if loose.is_file() or _locate(root, sha256=sha256, kind=kind) is not None:  # a shared sha is duplicated per pack
            members.append({"sha256": sha256, "kind": kind, "size": size, "source": loose})
        elif missing is not None:
            missing.append((kind, sha256, size))
    return members


def published_generations(root: Any, task_id: str, listing: Optional[list] = None) -> List[Dict[str, Any]]:
    """This task's published sidecars, oldest generation first: those its names spell (``_task_sidecar_names``),
    or of one shared ``listing``; a homonym's (``a.2`` generation 1 beside ``a``) is told by its own ``task_id``."""
    found = [sidecar for _, sidecar in (list_sidecars(root, names=_task_sidecar_names(root, task_id))
                                        if listing is None else listing) if sidecar["task_id"] == str(task_id)]
    return sorted(found, key=lambda sidecar: int(sidecar["generation"]))


def _member_source_bytes(root: pathlib.Path, task_id: str, member: Dict[str, Any]) -> bytes:
    if member["kind"] == CHECKPOINT:
        raw = member["source"].read_bytes()
    else:
        raw = blob_bytes(root, {"sha256": member["sha256"], "kind": member["kind"], "encoding": "gzip", "size": member["size"]})
    if len(raw) != int(member["size"]) or hashlib.sha256(raw).hexdigest() != member["sha256"]:
        raise EvidenceCorrupt(f"evidence_corrupt: loose member {member['sha256'][:12]} of {task_id} failed verification")
    return raw


def write_pack(root: Any, task_id: str, generation: int, members: List[Dict[str, Any]]) -> Tuple[pathlib.Path, Dict[str, Any]]:
    """Pack + sidecar to temp names, fsync both, publish the pack, then the sidecar LAST. Checkpoints lead
    (the sidecar's prefix rule), then blobs smallest first by ``(size, sha256)``: in one stream a small
    response or error decodes from a short prefix; the sidecar's offsets stay the readers' only map."""
    members = sorted(members, key=lambda m: (0, 0, "", "") if m["kind"] == CHECKPOINT
                     else (1, int(m["size"]), m["sha256"], m["kind"]))
    root, directory = pathlib.Path(root), packs_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    pack_path, sidecar_path = directory / pack_name(task_id, generation), directory / sidecar_name(task_id, generation)
    if pack_path.exists() or sidecar_path.exists():
        raise FileExistsError(f"evidence pack generation already occupied: {pack_path.name}")
    raw_total = sum(int(m["size"]) for m in members)
    dict_size = dict_size_for(raw_total)
    suffix = f".tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:8]}"
    tmp_pack, tmp_sidecar = directory / f".{pack_path.name}{suffix}", directory / f".{sidecar_path.name}{suffix}"
    compressor = lzma.LZMACompressor(format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64,
                                     filters=[{"id": lzma.FILTER_LZMA2, "preset": PRESET, "dict_size": dict_size}])
    digest, packed_total, offset, recorded = hashlib.sha256(), 0, 0, []
    try:
        with open(tmp_pack, "wb") as handle:
            def emit(chunk: bytes) -> None:
                nonlocal packed_total
                if chunk:
                    handle.write(chunk)
                    digest.update(chunk)
                    packed_total += len(chunk)

            for member in members:
                raw = _member_source_bytes(root, task_id, member)
                emit(compressor.compress(raw))
                recorded.append({**{key: member[key] for key in ("sha256", "kind", "size", "path") if key in member}, "offset": offset})
                offset += len(raw)
            emit(compressor.flush())
            handle.flush()
            os.fsync(handle.fileno())
        sidecar = {"task_id": str(task_id), "generation": int(generation), "format_version": FORMAT_VERSION, "codec": CODEC,
                   "preset": PRESET, "dict_size": dict_size, "raw_total": raw_total, "packed_total": packed_total,
                   "pack_sha256": digest.hexdigest(), "created_at": utc_now_iso(), "members": recorded}
        with open(tmp_sidecar, "w", encoding="utf-8") as handle:
            json.dump(sidecar, handle, ensure_ascii=False, indent=1)
            handle.flush()
            os.fsync(handle.fileno())
        replace_atomic(tmp_pack, pack_path)
        store.fsync_directory(directory)
        with store.name_lock(root, INDEX_LOCK_NAME, timeout_sec=_index_wait()):
            replace_atomic(tmp_sidecar, sidecar_path)  # serialize publication with index revalidation
            store.fsync_directory(directory)
    finally:
        for tmp in (tmp_pack, tmp_sidecar):
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()
    return pack_path, sidecar


def _member_key(member: Dict[str, Any]) -> tuple:
    return member["kind"], member["sha256"], str(member.get("path") or "")


@contextlib.contextmanager
def _unlink_phase(root: Any) -> Iterator[tuple]:
    """Serialize loose deletion with the switch's phase change, without creating a store: the
    write gate's typed refusal (an export began, the boot import runs) defers the deletion."""
    present, _ = store.probe(root)
    if present != "present":
        yield not (store.generation_stamped(root) or store.retired_sibling(root)), None
        return
    entered = False
    try:
        with store.open_store(root, write=True, barrier=True) as txn:
            entered = True
            yield True, txn
            return
    except (store.StoreRetired, store.StoreMigrating):
        if entered:
            raise
    yield False, None


def unlink_loose(root: Any, task_id: str, sidecars: List[Dict[str, Any]], *, keep_shas: set, revision: str,
                 sweep, members: list) -> Dict[str, int]:
    """Remove loose files whose bytes a VERIFIED pack (``sidecars``) holds; a blob a live task's
    manifests still reference (``keep_shas`` and the sweep's keep-set) stays loose. A busy file is
    retried next pass."""
    from ouroboros.evidence_packer import task_stamps
    from ouroboros.task_custody import task_custody_lock

    root = pathlib.Path(root)
    packed = {_member_key(m) for s in sidecars for m in s["members"]}
    report = {"unlinked": 0, "kept": 0, "busy": 0, "deferred": 1}

    def remove(txn, todo: list, until: float) -> Optional[list]:
        """One section under both locks: this task's own recheck (its row is still terminal and its
        evidence directories are as the census saw them: the stamp carries both) and the journal
        suffix, then unlinks until ``until``; the members left, or ``None`` when the candidate went stale."""
        nonlocal revision
        if task_stamps(root, task_id, txn=txn) != revision:
            return None
        keep = sweep.refresh_keep()
        for index, member in enumerate(todo):
            if index and time.monotonic() >= until:
                # Still under the locks: this section's own unlinks are the only change to the stamps.
                revision = task_stamps(root, task_id, txn=txn)
                return todo[index:]
            if not member["source"].is_file():
                continue
            if member["kind"] != CHECKPOINT and (member["sha256"] in keep or member["sha256"] in keep_shas):
                report["kept"] += 1
                continue
            try:
                os.unlink(member["source"])
                report["unlinked"] += 1
            except FileNotFoundError:
                pass
            except PermissionError:
                report["busy"] += 1
        return []

    # Compression precedes custody -> store write -> manifest publication locks. A section holds
    # the store write lock (a rollback barrier) and the publication lock for one UNLINK_SECTION_SEC
    # budget; model calls and every store writer wait on them. Inside, only this task's recheck;
    # between sections a waiting writer gets a poll-length gap. A stale check defers the candidate.
    todo: Optional[list] = [m for m in members if _member_key(m) in packed]
    with task_custody_lock(root, task_id) as acquired:
        while acquired and todo:
            left: Optional[list] = None
            with _unlink_phase(root) as (allowed, txn):
                until = time.monotonic() + UNLINK_SECTION_SEC
                if allowed:
                    with contextlib.ExitStack() as held:
                        try:
                            held.enter_context(store.name_lock(root, PUBLICATION_LOCK_NAME,
                                                               timeout_sec=max(0.0, until - time.monotonic())))
                        except store.StoreBusy:
                            pass  # busy for the whole section budget: the next pass retries the candidate
                        else:
                            left = remove(txn, todo, until)
            todo = left
            if todo:
                time.sleep(UNLINK_SECTION_GAP_SEC)
    report["deferred"] = int(todo != [])
    return report


def pack_task(root: Any, task_id: str, *, keep_shas: set, listing: Optional[list] = None, revision: Optional[str] = None,
              sweep=None) -> Dict[str, Any]:
    """The per-task steps, each idempotent: new loose members -> next generation (write, publish,
    re-read + verify every member, index), then unlink loose files held by verified packs. Only the
    generations an unlink relies on are re-read: the new one, and an older one while it still holds a
    member whose loose file exists now (a busy retry); the rest stay closed. Referenced records that
    are neither loose nor packed are counted (``missing``), never invented."""
    root = pathlib.Path(root)
    from ouroboros.evidence_packer import keep_rows, pack_sweep, task_stamps

    if sweep is None:
        with pack_sweep(root) as sweep:
            sweep.collect_keep(keep_rows(root))
            return pack_task(root, task_id, keep_shas=keep_shas, listing=listing, revision=revision, sweep=sweep)
    if revision is None:
        revision = task_stamps(root, task_id)
    generations = published_generations(root, task_id, listing)
    already = {_member_key(m) for s in generations for m in s["members"]}
    missing: list = []
    members = loose_members(root, task_id, missing=missing)
    fresh = [m for m in members if _member_key(m) not in already]
    report: Dict[str, Any] = {"task_id": str(task_id), "generation": None, "members": 0, "raw": 0, "packed": 0,
                              "verified_bytes": 0, "missing": len(missing),
                              "missing_bytes": sum(size for _, _, size in missing)}
    loose = {_member_key(m) for m in members if m["source"].is_file()}
    if not fresh and not loose:
        if missing:  # nothing to pack, but the gap is a fact of this task: recorded, never silent
            record_storage_pack(root, phase="packed", **report)
        return report  # nothing can be unlinked: do not decompress historical generations
    relied = [s for s in generations if any(_member_key(m) in loose for m in s["members"])]
    if fresh:
        generation = 1 + max((int(s["generation"]) for s in generations), default=0)
        while any((packs_dir(root) / name).exists() for name in
                  (pack_name(task_id, generation), sidecar_name(task_id, generation))):
            generation += 1
        pack_path, sidecar = write_pack(root, task_id, generation, fresh)
        report.update(generation=generation, members=len(fresh), raw=sidecar["raw_total"], packed=sidecar["packed_total"])
        relied.append(sidecar)
        if listing is not None:
            listing.append((pack_path.with_name(sidecar_name(task_id, generation)), sidecar))
    for sidecar in relied:
        pack_path = packs_dir(root) / pack_name(task_id, sidecar["generation"])
        verify_pack(pack_path, sidecar)  # from disk, EVERY member of a relied-on generation, before any unlink
        index_upsert(root, pack_path, {**sidecar, "verified_at": utc_now_iso()})
        report["verified_bytes"] += int(sidecar["raw_total"])
    write_unpacker(root)
    report.update(unlink_loose(root, task_id, relied, keep_shas=keep_shas, revision=revision,
                               sweep=sweep, members=members))
    if report["members"] or report["unlinked"] or missing:  # a row records production or a gap, never a re-visit
        record_storage_pack(root, phase="packed", **report)
    return report



def archive_usage_segments(root: Any) -> Dict[str, int]:
    """``archive/usage_ledger/segment_ep*.jsonl`` -> ``.jsonl.xz`` (preset 1) with a sidecar
    recording the original's size and sha256; verify the compressed copy, then unlink."""
    directory = pathlib.Path(root) / USAGE_ARCHIVE_REL
    report = {"archived": 0, "bytes": 0, "compressed": 0}
    for source in sorted(directory.glob("segment_ep*.jsonl")) if directory.is_dir() else []:
        write_unpacker(root)
        target, meta = source.with_name(source.name + ".xz"), source.with_name(source.name + ".xz.json")
        digest, size = hashlib.sha256(), 0
        def compress(tmp):
            nonlocal size
            with open(source, "rb") as reader, lzma.open(tmp, "wb", format=lzma.FORMAT_XZ, preset=PRESET) as writer:
                while chunk := reader.read(_READ_CHUNK):
                    digest.update(chunk)
                    size += len(chunk)
                    writer.write(chunk)
            # Windows FlushFileBuffers requires a handle opened for writing.
            with open(tmp, "rb+") as handle:
                os.fsync(handle.fileno())
        _atomic_overwrite(target, compress)
        atomic_write_json(meta, {"name": source.name, "size": size, "sha256": digest.hexdigest()}, fsync=True)
        check = hashlib.sha256()
        with lzma.open(target, "rb") as handle:
            while chunk := handle.read(_READ_CHUNK):
                check.update(chunk)
        if check.hexdigest() != digest.hexdigest():
            raise EvidenceCorrupt(f"evidence_corrupt: archived segment {target.name} failed verification")
        with _unlink_phase(root) as (allowed, _txn):
            if not allowed:
                continue
            source.unlink()
        report["archived"] += 1
        report["bytes"] += size
        report["compressed"] += target.stat().st_size
    return report


UNPACKER_NAME, README_NAME = "unpack_evidence.py", "README.md"
_UNPACKER = r'''#!/usr/bin/env python3
"""Restore loose evidence from the xz packs beside this script (stdlib only, idempotent).

  python3 unpack_evidence.py <task_id>     one task's blobs (observability/blobs/<sha>.<kind>.gz, gzip)
                                           and its context-checkpoint handles, sha256-verified
  python3 unpack_evidence.py --all         every task
  python3 unpack_evidence.py --usage-archive   archive/usage_ledger/*.jsonl.xz -> .jsonl
"""
import gzip, hashlib, json, lzma, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def _write(path, data):
    if os.path.exists(path):
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as handle:
        handle.write(data)
    os.replace(tmp, path)
    return True


def unpack(sidecar_path):
    with open(sidecar_path, encoding="utf-8") as handle:
        sidecar = json.load(handle)
    task, members = sidecar["task_id"], sorted(sidecar["members"], key=lambda m: m["offset"])
    pack = sidecar_path[:-len(".members.json")] + ".xz"
    with lzma.open(pack, "rb") as stream:
        position, written = 0, 0
        for member in members:
            stream.read(member["offset"] - position)
            raw = stream.read(member["size"])
            position = member["offset"] + member["size"]
            if hashlib.sha256(raw).hexdigest() != member["sha256"]:
                raise SystemExit("evidence_corrupt: %s in %s" % (member["sha256"], os.path.basename(pack)))
            if member["kind"] == "checkpoint":
                target = os.path.join(ROOT, "task_results", "artifacts", task, *member["path"].split("/"))
                written += _write(target, raw)
            else:
                target = os.path.join(ROOT, "observability", "blobs", "%s.%s.gz" % (member["sha256"], member["kind"]))
                written += _write(target, gzip.compress(raw, compresslevel=1))
    return written


def main(argv):
    if argv == ["--usage-archive"]:
        archive, count = os.path.join(ROOT, "archive", "usage_ledger"), 0
        for name in sorted(os.listdir(archive)) if os.path.isdir(archive) else []:
            if not name.endswith(".jsonl.xz"):
                continue
            with open(os.path.join(archive, name + ".json"), encoding="utf-8") as handle:
                meta = json.load(handle)
            with lzma.open(os.path.join(archive, name), "rb") as stream:
                raw = stream.read()
            if len(raw) != meta["size"] or hashlib.sha256(raw).hexdigest() != meta["sha256"]:
                raise SystemExit("evidence_corrupt: %s" % name)
            count += _write(os.path.join(archive, name[:-3]), raw)
        print("restored %d usage segment(s)" % count)
        return 0
    if len(argv) != 1:
        print(__doc__)
        return 2
    sidecars = [os.path.join(HERE, n) for n in sorted(os.listdir(HERE)) if n.endswith(".members.json")]
    if argv != ["--all"]:
        sidecars = [p for p in sidecars if json.load(open(p, encoding="utf-8"))["task_id"] == argv[0]]
    print("restored %d file(s) from %d pack(s)" % (sum(unpack(p) for p in sidecars), len(sidecars)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
'''
_README = """# Evidence packs

Each `<task>[.<gen>].xz` is one xz stream (LZMA2 preset 1) of a settled task's evidence: its
`source_handles/context_checkpoints/*` files first, then every blob its call manifests
reference, raw bytes back to back. `<task>[.<gen>].members.json` is the authority: header
(task, generation, format_version, codec, preset, dict_size, raw_total, packed_total,
pack_sha256, created_at, verified_at) and members (sha256, kind, size, offset, path for
handles). `index.sqlite` is a cache rebuilt from the sidecars whenever it is missing or damaged.

Ouroboros reads packed members in place. To get loose files back without the app:

    python3 unpack_evidence.py <task_id> | --all | --usage-archive

Blobs return as `observability/blobs/<sha256>.<kind>.gz` (gzip), handles under
`task_results/artifacts/<task>/source_handles/context_checkpoints/`, both sha256-verified.
"""


def write_unpacker(root: Any) -> None:
    """Refresh the data-root unpacker and its README when their content changed."""
    directory = packs_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in ((UNPACKER_NAME, _UNPACKER), (README_NAME, _README)):
        path = directory / name
        try:
            if path.read_bytes() == text.encode("utf-8"):
                continue
        except OSError:
            pass
        write_bytes_atomic(path, text.encode("utf-8"), fsync=True)
    store.fsync_directory(directory)


def record_storage_pack(root: Any, **facts: Any) -> None:
    store.supervisor_row(pathlib.Path(root), "storage_pack", facts, best_effort=False)
