"""Private Git copies for helpers: one primitive, one record per copy.

Every private Git copy — a delegated writer's or investigation's execution
snapshot, a native acting child's task copy, an own-body candidate — is a
linked ``git worktree`` under a root OUTSIDE ``repo/`` and ``data/``. A copy of
the source's CURRENT files starts from a snapshot commit that git itself makes:
a private index seeded from the source's real index stages the tracked files and
every untracked file Git does not ignore, ``commit-tree`` records it and a
ref under ``refs/ouroboros/delegated/`` pins it. ``git worktree add --detach
--no-checkout`` plus ``reset --hard`` (no hooks) checks it out. A linked worktree
shares the source's object database, so shallow and partial clones work, and a
source edit after the snapshot cannot reach the copy. The copy shares the
source repository's refs, stash, hooks and config.

git has no automatic worktree garbage collection, so each copy has one durable
record file ``data/state/private_copies/<name>.json``, written BEFORE the effect
it names and removed after it. There is no shared list and no machine-wide lock:
a record is locked only while that one copy changes, and git serializes its own
metadata (``worktree add`` marks a new worktree "initializing" against ``prune``).
The legacy whole-file registry ``state/subagent_worktrees.json`` is imported once.

Snapshot blobs live in the source's object database; once a pin goes they are
ordinary unreachable objects that the repository's own ``git gc`` removes.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
from ouroboros.force_remove import force_rmtree as _force_rmtree  # the shared permission-granting removal
from ouroboros.utils import atomic_write_json, utc_now_iso
from ouroboros.config import DATA_DIR, get_subagent_worktree_root
from ouroboros.retention import age_cutoff, get_gc_retention_days
from ouroboros.unindexed import copy_path  # new checkouts are born where desktop search skips them
from ouroboros import owner_keep, worktree_removal

log = logging.getLogger(__name__)


def __getattr__(name: str) -> Any:
    """The two genesis names this module used to own; ``genesis_project`` owns them and imports this module."""
    if name in ("provision_genesis_project", "remove_genesis_project"):
        from ouroboros import genesis_project

        return getattr(genesis_project, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_RECORDS_NAME = "private_copies"
_LEGACY_REGISTRY_NAME = "subagent_worktrees.json"
_LOCK_TIMEOUT_SEC = 120.0
_LOCK_STALE_SEC = 600.0
_BRANCH_PREFIX = "subagent/"
# Delegated-run execution snapshots (C1): record `kind` and the protected ref
# namespaces. The baseline ref pins the snapshot commit B for as long as the
# copy or its unapplied result lives; the result ref pins the captured commit C.
_KIND_DELEGATED_EXEC = "delegated_exec"
# Own-body authoring candidates (``body_candidate.py``): retention is decided by
# unique work and last use, never by age alone, and never by a child task's id.
_KIND_BODY_CANDIDATE = "body_candidate"
_BASELINE_REF_PREFIX = "refs/ouroboros/delegated/"
_RESULT_REF_PREFIX = "refs/ouroboros/delegated-result/"


# --------------------------------------------------------------------------- #
# Paths and records
# --------------------------------------------------------------------------- #
def _data_dir(data_dir: Optional[Any] = None) -> Path:
    return Path(data_dir or os.environ.get("OUROBOROS_DATA_DIR") or DATA_DIR)


def _records_dir(data_dir: Optional[Any] = None) -> Path:
    return _data_dir(data_dir) / "state" / _RECORDS_NAME


def _legacy_registry_path(data_dir: Optional[Any] = None) -> Path:
    return _data_dir(data_dir) / "state" / _LEGACY_REGISTRY_NAME


def _resolve_root(worktree_root: Optional[Any] = None) -> Path:
    root = Path(worktree_root) if worktree_root else Path(get_subagent_worktree_root())
    return root.expanduser().resolve()


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def _assert_root_isolated(root: Path, repo_dir: Path, data_dir: Path) -> None:
    """Refuse a worktree root that overlaps the live repo or runtime data."""
    if _is_within(root, repo_dir) or _is_within(repo_dir, root):
        raise ValueError(f"subagent worktree root {root} overlaps the Ouroboros repo {repo_dir}")
    if _is_within(root, data_dir) or _is_within(data_dir, root):
        raise ValueError(f"subagent worktree root {root} overlaps runtime data {data_dir}")


def _safe_name(task_id: Any) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(task_id or "").strip())
    safe = safe or f"wt_{int(time.time())}"
    # Bound the component (a project display name is not length-validated upstream: ENAMETOOLONG);
    # a short hash of the full slug keeps two long names with one prefix apart.
    if len(safe) > 64:
        digest = hashlib.sha256(safe.encode("utf-8")).hexdigest()[:8]
        safe = f"{safe[:55]}_{digest}"
    return safe


class SubagentWorktreeRegistryCorrupt(RuntimeError):
    """A copy record file exists but cannot be read as a record.

    The record is the only durable name of that copy and of the refs it pins,
    so its bytes are kept and nothing that would rewrite or delete it runs. One
    file per copy: every other copy is unaffected."""


_reported_unreadable: set = set()


def _report_unreadable(file: Path, op: str, exc: Exception, data_dir: Optional[Any]) -> None:
    """Disclose an unreadable record (or legacy registry) once per process and file state."""
    try:
        seen = file.stat()
        key = (str(file), seen.st_mtime_ns, seen.st_size)
    except OSError:
        key = (str(file), 0, 0)
    if key in _reported_unreadable:
        return
    _reported_unreadable.add(key)
    log.error("private copy record %s is unreadable; %s skips it, bytes kept (%s)", file, op or "read", exc)
    try:
        from ouroboros.utils import append_jsonl

        append_jsonl(_data_dir(data_dir) / "logs" / "events.jsonl",
                     {"ts": utc_now_iso(), "type": "private_copy_record_unreadable", "op": str(op or ""),
                      "record": str(file), "error": str(exc)[:200]})
    except Exception:
        log.debug("unreadable-record event append failed", exc_info=True)


def _record_name(path: Any) -> str:
    """A copy's record file stem: its directory name plus a digest of the full path,
    so copies under two worktree roots never share a record."""
    text = str(Path(str(path)).resolve())
    digest = hashlib.sha256(text.encode("utf-8", errors="surrogateescape")).hexdigest()[:12]
    return f"{_safe_name(Path(text).name)}-{digest}"


def _record_file(path: Any, data_dir: Optional[Any] = None) -> Path:
    return _records_dir(data_dir) / f"{_record_name(path)}.json"


def _names_copy(row: Any) -> bool:
    return isinstance(row, dict) and bool(str(row.get("path") or "").strip())


def _parse_record(file: Path) -> Optional[Dict[str, Any]]:
    """One record, ``None`` when absent; an unreadable file raises ``ValueError`` or ``OSError``. Reports nothing."""
    try:
        row = json.loads(file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not _names_copy(row):
        raise ValueError("a copy record must be an object naming its path")
    return row


def _legacy_entries(data: bytes) -> List[Any]:
    """The rows of the legacy whole-file registry's bytes; malformed bytes raise ``ValueError``."""
    raw = json.loads(data.decode("utf-8"))
    entries = raw.get("worktrees") if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        raise ValueError("legacy registry 'worktrees' is not a list")
    return entries


def _read_record(file: Path, *, op: str = "", data_dir: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    """One record, ``None`` when absent; an unreadable file raises (and is reported)."""
    try:
        return _parse_record(file)
    except (OSError, ValueError) as exc:
        _report_unreadable(file, op, exc, data_dir)
        raise SubagentWorktreeRegistryCorrupt(f"{file.name}: {exc}") from exc


def _records(data_dir: Optional[Any] = None, *, op: str = "", strict: bool = False) -> List[Dict[str, Any]]:
    """Every readable copy record, never rewritten or removed: the copy it names stays. An unreadable record is reported
    and skipped; ``strict`` raises instead, listing with ``os.listdir`` (``Path.glob`` reads an unlistable directory as empty)."""
    _import_legacy_registry(data_dir)
    rows: List[Dict[str, Any]] = []
    directory = _records_dir(data_dir)
    try:
        names = os.listdir(directory) if strict else [f.name for f in directory.glob("*.json")]
    except OSError as exc:
        if strict and not isinstance(exc, FileNotFoundError):  # a missing directory holds no record
            raise
        return rows
    for file in sorted(directory / name for name in names if name.endswith(".json") and not name.startswith(".")):
        with contextlib.suppress(*(() if strict else (SubagentWorktreeRegistryCorrupt,))):
            row = _read_record(file, op=op, data_dir=data_dir)
            if row is not None:
                rows.append(row)
    return rows


def _record_at(path: Any, data_dir: Optional[Any] = None, *, op: str = "") -> Optional[Dict[str, Any]]:
    """The record of the copy at ``path`` (raises when that record is unreadable)."""
    _import_legacy_registry(data_dir)
    return _read_record(_record_file(path, data_dir), op=op, data_dir=data_dir)


def _put_record(row: Dict[str, Any], data_dir: Optional[Any] = None) -> None:
    file = _record_file(row["path"], data_dir)
    file.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(file, row, trailing_newline=True)


def _drop_record(path: Any, data_dir: Optional[Any] = None) -> None:
    with contextlib.suppress(FileNotFoundError):
        _record_file(path, data_dir).unlink()


_imported_roots: set = set()


def _import_legacy_registry(data_dir: Optional[Any] = None) -> None:
    """Move the legacy whole-file registry's rows into per-copy records, once.

    Each row becomes its record unless one already exists; a receipt names how
    many rows moved and the legacy bytes' digest, then the legacy file goes. An
    unreadable legacy file is reported and kept: its copies have no record and
    show as unregistered in the owner's Free action."""
    key = str(_data_dir(data_dir))
    if key in _imported_roots:
        return
    legacy = _legacy_registry_path(data_dir)
    if not legacy.is_file():
        _imported_roots.add(key)
        return
    lock = _records_dir(data_dir) / ".legacy-import.lock"
    fd = acquire_exclusive_file_lock(lock, timeout_sec=_LOCK_TIMEOUT_SEC, stale_sec=_LOCK_STALE_SEC,
                                     metadata=f"pid={os.getpid()} op=legacy_import", owner_aware_stale=True)
    if fd is None:
        return  # another process is importing; the next read retries
    try:
        if not legacy.is_file():
            _imported_roots.add(key)
            return
        data = legacy.read_bytes()
        try:
            entries = _legacy_entries(data)
        except ValueError as exc:
            _report_unreadable(legacy, "legacy_import", exc, data_dir)
            _imported_roots.add(key)
            return
        moved = 0
        for row in entries:
            if _names_copy(row) and not _record_file(row["path"], data_dir).exists():
                _put_record(row, data_dir)
                moved += 1
        atomic_write_json(_records_dir(data_dir) / ".legacy-import.json", {
            "at": utc_now_iso(), "rows": len(entries), "moved": moved,
            "legacy_sha256": hashlib.sha256(data).hexdigest()}, trailing_newline=True)
        legacy.unlink()
        _imported_roots.add(key)
    finally:
        release_exclusive_file_lock(lock, fd)


# --------------------------------------------------------------------------- #
# Locking
# --------------------------------------------------------------------------- #
class WorktreeOpsLockBusy(TimeoutError):
    """One copy's record lock stayed held for the whole wait.

    ``holder`` is what the holder wrote into the lock file (``pid``, ``task``,
    ``op``, ``since``, ``target``; ``{}`` when unreadable), so a refusal names who
    is doing what instead of a bare timeout. Only work on the SAME copy waits."""

    def __init__(self, lock_path: Path, waited_sec: float, holder: Dict[str, str]):
        self.lock_path, self.waited_sec, self.holder = str(lock_path), waited_sec, holder
        who = (f" (held by pid {holder.get('pid')} task {holder.get('task') or '?'} op "
               f"{holder.get('op') or '?'} since {holder.get('since') or '?'})") if holder else ""
        super().__init__(f"private copy record lock timeout after {waited_sec:.0f}s: {lock_path}{who}")


def _lock_holder(lock_path: Path) -> Dict[str, str]:
    """Parse ``pid=… task=… op=… since=… target=…``; ``target`` is the tail of the
    line so a path with spaces survives. Readable while held: flock never blocks reads."""
    try:
        text = lock_path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return {}
    head, sep, target = text.partition(" target=")
    holder = {key: value for key, eq, value in (field.partition("=") for field in head.split())
              if eq and key in ("pid", "task", "op", "since")}
    if sep:
        holder["target"] = target
    return holder


@contextlib.contextmanager
def _record_lock(path: Any, data_dir: Optional[Any] = None, *, op: str, task_id: str = "",
                 timeout_sec: Optional[float] = None):
    """Serialize changes to ONE copy's record across threads and processes (the
    portable file-lock SSOT, platform_layer). Never held for a tree walk, and
    never needed to create a different copy. The holder names itself, ``pid=``
    first, so a dead holder is evicted at once and a live one never is; a
    timeout is typed with the holder's facts. Not reentrant."""
    lock_path = _record_file(path, data_dir).with_suffix(".lock")
    metadata = f"pid={os.getpid()} task={task_id or '-'} op={op} since={utc_now_iso()} target={path}"
    started = time.monotonic()
    fd = acquire_exclusive_file_lock(
        lock_path, timeout_sec=_LOCK_TIMEOUT_SEC if timeout_sec is None else timeout_sec,
        stale_sec=_LOCK_STALE_SEC, metadata=metadata, owner_aware_stale=True)
    if fd is None:
        raise WorktreeOpsLockBusy(lock_path, time.monotonic() - started, _lock_holder(lock_path))
    try:
        yield
    finally:
        release_exclusive_file_lock(lock_path, fd)


def _update_record(path: Any, change: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]],
                   data_dir: Optional[Any] = None, *, op: str, task_id: str = "") -> Optional[Dict[str, Any]]:
    """Read-modify-write one record under its lock: ``change`` returns the new
    record, or ``None`` to drop it. An absent record is left absent."""
    with _record_lock(path, data_dir, op=op, task_id=task_id):
        row = _record_at(path, data_dir, op=op)
        if row is None:
            return None
        new = change(dict(row))
        if new is None:
            _drop_record(path, data_dir)
        elif new != row:
            _put_record(new, data_dir)
        return new


# --------------------------------------------------------------------------- #
# git helpers
# --------------------------------------------------------------------------- #
class GitFailed(subprocess.CalledProcessError):
    """A git command failed; its text carries the command and git's own stderr,
    so a refusal says what git said instead of "exit status 128"."""

    def __str__(self) -> str:
        err = self.stderr.decode("utf-8", "replace") if isinstance(self.stderr, bytes) else str(self.stderr or "")
        tail = " ".join(err.split())[-600:]
        verb = " ".join(str(part) for part in list(self.cmd or [])[1:3])
        return f"git {verb} exited {self.returncode}" + (f": {tail}" if tail else "")


# Host repository-location variables would redirect a command at another repository or index (a leaked
# GIT_DIR made `git init` "reinitialize" a foreign directory), so no git call here inherits them.
_LOCATION_VARS = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
                  "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR")


def _host_env(**extra: str) -> Dict[str, str]:
    """The process environment minus repository-location variables, plus ``extra``."""
    return {**{key: value for key, value in os.environ.items() if key not in _LOCATION_VARS}, **extra}


def _git(repo_dir: Path, *args: str, check: bool = True,
         env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(["git", *args], cwd=str(repo_dir), capture_output=True, text=True,
                          encoding="utf-8", errors="surrogateescape", check=False,
                          env=_host_env() if env is None else env)
    if check and proc.returncode != 0:
        raise GitFailed(proc.returncode, proc.args, proc.stdout, proc.stderr)
    return proc


def _deletable(path: Path, root: Path) -> bool:
    """A checkout path this module may delete: non-empty, not a root spelling, and
    STRICTLY inside the worktree root (the root itself holds every live snapshot) —
    the registry is durable state and a malformed row must never name an arbitrary
    path. One guard for every delete this module performs."""
    text = str(path).strip()
    if not text or text in (".", "/", "//"):
        return False
    try:
        return path.resolve() != Path(root).resolve() and _is_within(path, root)
    except OSError:
        return False


def _git_quiet(repo_dir: Path, *args: str) -> None:
    """Best-effort git: a failing command or a vanished repo is not an error here."""
    try:
        _git(repo_dir, *args, check=False)
    except Exception:
        pass


def _forget_checkout(repo_dir: Path, wt_path: Path) -> None:
    """Drop ONE gone checkout's registration (a global prune drops the source's own missing worktrees too)."""
    if not Path(wt_path).exists():
        _git_quiet(repo_dir, "worktree", "remove", "--force", "--force", str(wt_path))


def _remove_paths(repo_dir: Path, wt_path: Path, branch: str, *, allowed_root: Optional[Any] = None) -> bool:
    """Forget a deleted checkout's Git registration and branch (the caller deleted its files; a
    surviving directory returns False, so its record and pin stay: ``removal_failed``). With ``allowed_root``,
    a path not strictly inside it is never touched: while it exists it is kept like a survivor, else forgotten."""
    wt_path = Path(wt_path)
    if allowed_root is not None and not _deletable(wt_path, Path(allowed_root)):
        return not os.path.lexists(wt_path)  # not ours: no git runs on a path a record names (#1239)
    if wt_path.exists():
        return False
    _forget_checkout(repo_dir, wt_path)
    if branch:
        _git_quiet(repo_dir, "branch", "-D", branch)
    return True


def _mark_removal_failed(match: Dict[str, Any], data_dir: Optional[Any]) -> None:
    """Keep the first failure time and the surviving directory identity (record lock held)."""
    row = _record_at(match["path"], data_dir, op="mark_removal_failed")
    if row is not None:
        failed = worktree_removal.record_failure(row)
        if failed != row:
            _put_record(failed, data_dir)


def _unpin_snapshot(repo: Path, row: Dict[str, Any]) -> None:
    """Drop a snapshot's baseline and result pins (only refs this module owns)."""
    for key in ("baseline_ref", "result_ref"):
        ref = str(row.get(key) or "")
        if ref.startswith((_BASELINE_REF_PREFIX, _RESULT_REF_PREFIX)):
            _git_quiet(repo, "update-ref", "-d", ref)


def _discard_snapshot_checkout(row: Dict[str, Any], *, root: Path, data_dir: Optional[Any]) -> None:
    """Undo a provisioning that failed after its record and pin may exist: files,
    admin dir, pins, record (the postcondition ``tests/test_private_copy_primitive.py``
    asserts: no record, no ref, no ``dlg_*`` directory). Best-effort: custody GC
    reconciles crash leftovers once the owning task is proven over."""
    wt_path = Path(str(row.get("path") or ""))
    try:
        if _deletable(wt_path, root) and wt_path.exists():
            _force_rmtree(wt_path)
        repo = Path(str(row.get("git_dir") or row.get("target_root") or "."))
        _forget_checkout(repo, wt_path)
        _unpin_snapshot(repo, row)
        _drop_record(wt_path, data_dir)
    except Exception:
        log.warning("Failed to discard execution snapshot %s after a provisioning failure",
                    row.get("snapshot_id"), exc_info=True)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WorktreeHandle:
    task_id: str
    path: str
    branch: str
    base_sha: str
    repo_dir: str
    created_at: float
    parent_task_id: str = ""
    source_is_system_repo: bool = True
    target_head: str = ""
    source_index_clean: bool = False
    git_dir: str = ""


def provision_worktree(
    *,
    repo_dir: Any,
    task_id: Any,
    base_sha: str = "",
    parent_task_id: str = "",
    worktree_root: Optional[Any] = None,
    data_dir: Optional[Any] = None,
    source_is_system_repo: bool = True,
) -> WorktreeHandle:
    """Give a child the source's current eligible tree, under task-owned GC.

    The branch pins the synthetic current-tree baseline, not just source HEAD.
    Unlike delegated execution snapshots, this checkout belongs to the child
    task and is removed by worktree retention, never by delegated-run custody.
    """
    target = Path(repo_dir).resolve()
    if base_sha:
        expected = _git(target, "rev-parse", "--verify", f"{base_sha}^{{commit}}").stdout.strip()
        if _git(target, "rev-parse", "HEAD").stdout.strip() != expected:
            raise ValueError("isolated child source HEAD differs from the requested base_sha")
    return _provision_git_copy(
        target_root=target, task_id=task_id, snapshot_id=f"task_{_safe_name(task_id)}",
        worktree_root=worktree_root, data_dir=data_dir,
        task_copy=True, parent_task_id=parent_task_id,
        source_is_system_repo=source_is_system_repo,
    )


@dataclass(frozen=True)
class ExecutionSnapshotHandle:
    """A private, disposable execution root for ONE delegated run.

    The snapshot is a detached ``git worktree`` of the AUTHORITY TARGET's
    repository, checked out at a snapshot commit B that git made from the
    target's real current files (tracked edits, staged changes and every
    untracked file Git does not ignore). The run writes ONLY here;
    its contribution is the diff from B to the captured result commit C, and
    the nanny applies or rejects that result into the target explicitly —
    never automatically. ``excluded_untracked`` names what git could not take
    (nested repositories, SQLite WAL indexes, special files).
    """

    snapshot_id: str
    task_id: str
    path: str
    target_root: str
    baseline_ref: str
    baseline_sha: str
    baseline_tree: str
    manifest_digest: str
    target_head: str
    created_at: float
    entry_count: int = 0
    excluded_untracked: tuple = ()
    # Standalone payload snapshots (R1): the snapshot owns its OWN .git (the
    # target is a non-Git skill payload), so cleanup touches no target-repo
    # worktree/ref command. ``payload_hash`` is the pre-copy skill-loader
    # content hash — the whole-payload CAS baseline for the explicit apply.
    standalone: bool = False
    payload_hash: str = ""
    # Wall-clock seconds the provision took; a disclosure on the start receipt and
    # the baseline manifest (a heavy tree is visible the moment it is snapshotted).
    provisioning_sec: float = 0.0
    git_dir: str = ""  # Stable shared Git metadata survives deletion of a source worktree.


def _git_env_index(index_path: Path) -> Dict[str, str]:
    return _host_env(GIT_INDEX_FILE=str(index_path))


def _git_env(repo_dir: Path, *args: str, env: Dict[str, str],
             check: bool = True, input_bytes: bytes = b"") -> subprocess.CompletedProcess:
    proc = subprocess.run(["git", *args], cwd=str(repo_dir), capture_output=True, check=False, env=env,
                          input=input_bytes if input_bytes else None)
    if check and proc.returncode != 0:
        raise GitFailed(proc.returncode, proc.args, proc.stdout, proc.stderr)
    return proc


def _gitlink_held(folder: Path) -> str:
    """Why a gitlink's folder keeps a writer's checkout, or "" (empty in a fresh copy; a link there is staged)."""
    try:
        with os.scandir(folder) as entries:
            return "populated_submodule" if not folder.is_symlink() and next(entries, None) is not None else ""
    except (FileNotFoundError, NotADirectoryError):
        return ""
    except OSError:  # what it holds is unknown, never "empty"
        return "" if folder.is_symlink() else "unlistable_directory"


def _nul_paths(raw: bytes) -> List[str]:
    return list(dict.fromkeys(p for p in raw.decode("utf-8", errors="surrogateescape").split("\0") if p))


def _nul_input(paths: List[str]) -> bytes:
    return b"\0".join(p.encode("utf-8", errors="surrogateescape") for p in paths) + b"\0"


_UNLISTABLE = re.compile(r"warning: could not open directory '(.*)': ")
# What a capture names but cannot hold: the copy is kept for recovery until disposition.
UNSAVED_REASONS = frozenset({"nested_repository", "populated_submodule", "unlistable_directory"})
_SNAPSHOT_IDENTITY = {"GIT_AUTHOR_NAME": "Ouroboros", "GIT_AUTHOR_EMAIL": "ouroboros@localhost",
                      "GIT_COMMITTER_NAME": "Ouroboros", "GIT_COMMITTER_EMAIL": "ouroboros@localhost"}


def snapshot_commit(source: Path, *, message: str, parent: str = "",
                    held_out: str = "", held_keep: frozenset = frozenset(),
                    ) -> Tuple[str, str, str, List[Dict[str, Any]]]:
    """Commit ``source``'s current files without touching its index, HEAD or files.

    A private index copied from the source's real index (its stat cache makes an
    unchanged file cost one ``lstat``) stages every tracked path and every
    untracked file Git does not ignore. ``assume-unchanged`` is cleared,
    and ``skip-worktree`` too for a path that is on disk, so neither hides an
    edit; an absent ``skip-worktree`` entry keeps its indexed blob (sparse
    checkouts stay correct). An untracked nested repository, a folder Git cannot
    open, a SQLite WAL index (shared memory) and a special file are left out and
    named, and so, at a result capture, is a gitlink folder the run filled. The commit's
    parent is ``parent`` or the source's HEAD (none on an unborn branch).
    ``held_out`` names a folder (a run's ``.review-evidence/``) whose files stay out
    of the commit unless listed in ``held_keep`` (what the baseline already holds):
    named evidence is retained beside the capture, never hashed into the source's
    object database. Returns ``(commit, tree, head, excluded)``."""
    from ouroboros.store import is_wal_index

    head_proc = _git(source, "rev-parse", "--verify", "HEAD", check=False)
    head = head_proc.stdout.strip() if head_proc.returncode == 0 else ""
    located = Path(_git(source, "rev-parse", "--git-path", "index").stdout.strip())
    real_index = located if located.is_absolute() else source / located
    private = Path(tempfile.mkdtemp(prefix="ouroboros-snapshot-index-"))  # never among the copies
    index = private / "index"
    env = _git_env_index(index)
    excluded: List[Dict[str, Any]] = []
    try:
        if real_index.is_file():
            # The index's own time stays: Git re-reads an entry written in that same tick
            # (racily clean), so a same-size edit is never trusted as unchanged.
            shutil.copy2(real_index, index)
        else:
            _git_env(source, "read-tree", "--empty", env=env)
        listed = _git_env(source, "ls-files", "-v", "-z", env=env).stdout.split(b"\0")
        assumed, skipped, absent = [], [], set()
        for item in listed:
            tag, _, raw = item.partition(b" ")
            if not raw:
                continue
            rel = raw.decode("utf-8", errors="surrogateescape")
            if tag[:1].islower():  # `ls-files -v` lowercases an assume-unchanged entry's tag
                assumed.append(rel)
            if tag[:1] in (b"S", b"s"):
                if os.path.lexists(source / rel):
                    skipped.append(rel)
                else:
                    absent.add(rel)
        if assumed:
            _git_env(source, "update-index", "-z", "--no-assume-unchanged", "--stdin", env=env,
                     input_bytes=_nul_input(assumed))
        if skipped:
            _git_env(source, "update-index", "-z", "--no-skip-worktree", "--stdin", env=env,
                     input_bytes=_nul_input(skipped))
        stage: List[str] = []
        dropped: List[str] = []
        prefix = held_out.strip("/") + "/" if held_out.strip("/") else ""

        def held(rel: str) -> bool:  # the folder's files, or an entry standing where the folder belongs
            return bool(prefix) and (rel == prefix.rstrip("/") or rel.startswith(prefix)) and rel not in held_keep

        modes: Dict[str, set] = {}  # every stage's mode: a conflict may name a gitlink in any of them
        for item in _git_env(source, "ls-files", "-s", "-z", env=env).stdout.split(b"\0"):
            meta, _, raw = item.partition(b"\t")
            if raw:
                modes.setdefault(raw.decode("utf-8", errors="surrogateescape"), set()).add(meta[:6])
        for rel, stages in modes.items():
            if rel in absent:
                continue
            if parent and b"160000" in stages and (reason := _gitlink_held(source / rel)):
                # A result holds a gitlink, never what a checked-out submodule or a staged repository keeps.
                excluded.append({"path": rel, "reason": reason})
            if held(rel):
                dropped.append(rel)
            elif is_wal_index(source / rel):
                excluded.append({"path": rel, "reason": "sqlite_wal_index"})
                dropped.append(rel)
            else:
                stage.append(rel)
        listing = _git_env(source, "ls-files", "-z", "--others", "--exclude-standard",
                           env={**env, "LC_ALL": "C", "LANGUAGE": "C"})
        # Git lists past a folder it cannot open with exit 0 and only this warning (fixed locale).
        excluded += [{"path": found.group(1).rstrip("/"), "reason": "unlistable_directory"} for found in
                     map(_UNLISTABLE.match, listing.stderr.decode("utf-8", "surrogateescape").splitlines()) if found]
        for rel in _nul_paths(listing.stdout):
            try:
                mode = os.lstat(source / rel).st_mode
            except FileNotFoundError:
                continue  # removed while listed: not part of this snapshot
            if stat.S_ISDIR(mode):
                excluded.append({"path": rel.rstrip("/"), "reason": "nested_repository"})
            elif not (stat.S_ISREG(mode) or stat.S_ISLNK(mode)):
                excluded.append({"path": rel, "reason": "special_file"})
            elif is_wal_index(source / rel):
                excluded.append({"path": rel, "reason": "sqlite_wal_index"})
            elif not held(rel):
                stage.append(rel)
        if prefix:
            excluded.append({"path": prefix.rstrip("/"), "reason": "named_evidence"})
        if dropped:
            _git_env(source, "update-index", "-z", "--force-remove", "--stdin", env=env,
                     input_bytes=_nul_input(dropped))
        if stage:
            # Each named path's CURRENT bytes (an absent one leaves the index), and nothing else.
            _git_env(source, "update-index", "-z", "--add", "--remove", "--stdin", env=env,
                     input_bytes=_nul_input(stage))
        tree = _git_env(source, "write-tree", env=env).stdout.decode("utf-8").strip()
        base = parent or head
        commit = _git_env(source, "commit-tree", tree, *(["-p", base] if base else []), "-m", message,
                          env=_host_env(**_SNAPSHOT_IDENTITY)).stdout.decode("utf-8").strip()
    finally:
        shutil.rmtree(private, ignore_errors=True)
    return commit, tree, head, excluded


def provision_execution_snapshot(
    *, target_root: Any, task_id: Any, snapshot_id: str,
    worktree_root: Optional[Any] = None, data_dir: Optional[Any] = None,
) -> ExecutionSnapshotHandle:
    """Provision a run-owned current-tree snapshot; custody owns its lifetime.
    Writers and evidence-only investigations get the same linked worktree."""
    return _provision_git_copy(
        target_root=target_root, task_id=task_id, snapshot_id=snapshot_id,
        worktree_root=worktree_root, data_dir=data_dir,
    )


def _check_out(target: Path, wt_path: Path, commit: str, *, branch: str = "") -> None:
    """A linked worktree at ``commit`` (or ``branch``), populated by ``reset --hard``:
    the checkout git's own ``worktree add`` performs, without the target's hooks
    and without submodule recursion."""
    wt_path.parent.mkdir(parents=True, exist_ok=True)
    _forget_checkout(target, wt_path)  # a crashed earlier attempt's registration of this path
    if branch:
        _git(target, "worktree", "add", "--no-checkout", "--force", str(wt_path), branch)
    else:
        _git(target, "worktree", "add", "--detach", "--no-checkout", str(wt_path), commit)
    _git(wt_path, "reset", "--hard", "--quiet", "--no-recurse-submodules")


def _provision_git_copy(
    *,
    target_root: Any,
    task_id: Any,
    snapshot_id: str,
    worktree_root: Optional[Any] = None,
    data_dir: Optional[Any] = None,
    task_copy: bool = False,
    parent_task_id: str = "",
    source_is_system_repo: bool = True,
) -> ExecutionSnapshotHandle | WorktreeHandle:
    """Snapshot ``target_root``'s REAL current tree into a private linked worktree.

    The snapshot commit is pinned (``refs/ouroboros/delegated/<snap>``, or the
    task copy's ``subagent/<task>`` branch) and the copy's record is written
    BEFORE the worktree exists, so a crash at any later point leaves a copy that
    custody GC can name. A failure undoes everything it made and raises with
    git's own words.
    """
    target = Path(target_root).resolve()
    if not (target / ".git").exists():
        raise ValueError(f"delegated execution snapshot target {target} is not a git working tree")
    root = _resolve_root(worktree_root)
    data_root = _data_dir(data_dir)
    if _is_within(root, data_root) or _is_within(data_root, root):
        raise ValueError(f"subagent worktree root {root} overlaps runtime data {data_root}")
    snap = str(snapshot_id or "").strip()
    if not snap:
        raise ValueError("snapshot_id is required for a delegated execution snapshot")
    safe_snap = _safe_name(snap)
    task = str(task_id or "")
    started = time.monotonic()
    name = _safe_name(task_id) if task_copy else f"dlg_{_safe_name(task_id)}_{safe_snap[:16]}"
    wt_path = copy_path(root, name).resolve()
    _assert_root_isolated(wt_path, target, data_root)
    branch = f"{_BRANCH_PREFIX}{_safe_name(task_id)}" if task_copy else ""
    baseline_ref = f"refs/heads/{branch}" if task_copy else f"{_BASELINE_REF_PREFIX}{safe_snap}"
    common = Path(_git(target, "rev-parse", "--git-common-dir").stdout.strip())
    git_dir = str((common if common.is_absolute() else target / common).resolve())
    root.mkdir(parents=True, exist_ok=True)
    _record_at(wt_path, data_dir, op="provision")  # an unreadable record of this copy refuses here
    # A stale checkout of the SAME copy (a crashed earlier attempt) is plain files;
    # its admin dir and pin are replaced below.
    if _deletable(wt_path, root) and wt_path.exists():
        _force_rmtree(wt_path)
    baseline_sha, tree_sha, target_head, excluded = snapshot_commit(
        target, message=f"ouroboros: delegated-run baseline {snap}")
    manifest_raw = _git_env(target, "ls-tree", "-r", "-z", tree_sha, env=_host_env()).stdout
    if task_copy and target_head and tree_sha == _git(target, "rev-parse", f"{target_head}^{{tree}}").stdout.strip():
        baseline_sha = target_head
    source_index_clean = bool(target_head and _git(
        target, "diff-index", "--cached", "--quiet", target_head, "--", check=False).returncode == 0)
    manifest_digest = hashlib.sha256(manifest_raw).hexdigest()
    entry_count = sum(1 for chunk in manifest_raw.split(b"\0") if chunk)

    def recorded(provisioning_sec: float) -> Tuple[Any, Dict[str, Any]]:
        if task_copy:
            owned = WorktreeHandle(
                task_id=task, path=str(wt_path), branch=branch, base_sha=baseline_sha,
                repo_dir=str(target), created_at=time.time(), parent_task_id=parent_task_id,
                source_is_system_repo=source_is_system_repo, target_head=target_head,
                source_index_clean=source_index_clean, git_dir=git_dir)
            return owned, asdict(owned)
        handle = ExecutionSnapshotHandle(
            snapshot_id=snap, task_id=task, path=str(wt_path), target_root=str(target),
            baseline_ref=baseline_ref, baseline_sha=baseline_sha, baseline_tree=tree_sha,
            manifest_digest=manifest_digest, target_head=target_head, created_at=time.time(),
            entry_count=entry_count, excluded_untracked=tuple(excluded),
            provisioning_sec=provisioning_sec, git_dir=git_dir)
        return handle, {**asdict(handle), "kind": _KIND_DELEGATED_EXEC}

    _, row = recorded(0.0)
    claimed = False
    try:
        _put_record(row, data_dir)  # claim before effect: everything after it is nameable by custody GC
        claimed = True
        _git(target, "update-ref", baseline_ref, baseline_sha)
        _check_out(target, wt_path, baseline_sha, branch=branch)
        handle, row = recorded(round(time.monotonic() - started, 3))
        _put_record(row, data_dir)
    except Exception:
        if claimed:
            if task_copy:
                try:
                    remove_worktree(path=str(wt_path), worktree_root=root, data_dir=data_dir)
                except Exception:
                    log.warning("Failed to discard task copy %s after provisioning failure", task, exc_info=True)
            else:
                _discard_snapshot_checkout(row, root=root, data_dir=data_dir)
        raise
    return handle


def capture_result_commit(snapshot_id: str, patch_path: Path, *, held_out: str = "",
                          data_dir: Optional[Any] = None) -> Dict[str, Any]:
    """Commit a delegated run's final files as C (parent B), pin it, and stream
    the patch B..C into ``patch_path``.

    The run's own worktree index seeds the private index, so the capture costs
    what the run changed; committed work counts by its files. ``held_out`` (the
    run's ``.review-evidence/`` folder, relative to the copy root) keeps new named
    evidence out of C. The patch is plumbing ``git diff-tree -p --binary
    --full-index``: a user's porcelain diff config (``diff.noprefix``, renames,
    external drivers) never reshapes what ``git apply`` reads back. Both pins stay
    until disposition. Returns ``result_sha``, ``result_tree``, ``head``,
    ``result_ref``, ``patch_size`` and the ``excluded`` rows."""
    row = find_execution_snapshot(snapshot_id, data_dir)
    if row is None:
        raise FileNotFoundError(f"no private copy record for snapshot {snapshot_id}")
    clone = Path(str(row.get("path") or ""))
    baseline = str(row.get("baseline_sha") or "")
    keep: frozenset = frozenset()
    if held_out.strip("/"):
        keep = frozenset(_nul_paths(_git_env(clone, "ls-tree", "-r", "-z", "--name-only", baseline, "--",
                                             held_out.strip("/"), env=_host_env()).stdout))
    commit, tree, head, excluded = snapshot_commit(
        clone, message=f"ouroboros: delegated-run result {snapshot_id}", parent=baseline,
        held_out=held_out, held_keep=keep)
    repo = Path(str(row.get("git_dir") or row.get("target_root") or clone))
    result_ref = f"{_RESULT_REF_PREFIX}{_safe_name(snapshot_id)}"
    with _record_lock(row["path"], data_dir, op="capture_result", task_id=str(row.get("task_id") or "")):
        _git(repo, "update-ref", result_ref, commit)
        current = _record_at(row["path"], data_dir, op="capture_result") or row
        _put_record({**current, "result_ref": result_ref, "result_sha": commit, "result_tree": tree}, data_dir)
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    with open(patch_path, "wb") as sink:
        proc = subprocess.run(["git", "diff-tree", "-r", "-p", "--binary", "--full-index", "--no-renames",
                               "--no-ext-diff", "--no-textconv", "--no-color", baseline, commit],
                              cwd=str(repo), stdout=sink, stderr=subprocess.PIPE, check=False, env=_host_env())
    if proc.returncode != 0:
        raise GitFailed(proc.returncode, proc.args, b"", proc.stderr)
    return {"result_sha": commit, "result_tree": tree, "head": head, "result_ref": result_ref,
            "baseline_ref": str(row.get("baseline_ref") or ""), "patch_size": patch_path.stat().st_size,
            "excluded": excluded}


def release_snapshot_checkout(snapshot_id: str, *, worktree_root: Optional[Any] = None,
                              data_dir: Optional[Any] = None) -> bool:
    """Remove a captured run's worktree, keeping its record and pins (the result is the saved patch and C;
    a continuation recreates the checkout). The record says ``checkout_released`` BEFORE the files go,
    so a remnant is never taken for a whole checkout. True when no checkout remains."""
    row = find_execution_snapshot(snapshot_id, data_dir)
    if row is None or row.get("standalone"):
        return False
    root = _resolve_root(worktree_root)
    wt_path = Path(str(row.get("path") or ""))
    with _record_lock(wt_path, data_dir, op="release_checkout", task_id=str(row.get("task_id") or "")), \
            owner_keep.gate(_data_dir(data_dir), row.get("task_id")) as why:
        if why or (wt_path.exists() and not _deletable(wt_path, root)):
            return False  # kept by the owner, or not under this root: the checkout stays whole and unmarked
        row = _record_at(wt_path, data_dir, op="release_checkout")
        if row is None or row.get("snapshot_id") != snapshot_id:
            return False  # no stale row decides: this record went or became another's
        _put_record({**row, "checkout_released": row.get("checkout_released") or utc_now_iso()}, data_dir)
        if wt_path.exists():
            _force_rmtree(wt_path)
        if wt_path.exists():
            _mark_removal_failed(row, data_dir)
            return False
        _forget_checkout(Path(str(row.get("git_dir") or row.get("target_root") or ".")), wt_path)
        worktree_removal.retired(row, _data_dir(data_dir))
    return True


def checkout_whole(row: Dict[str, Any]) -> bool:
    """The checkout is there and was never released (a remnant of a release or restore is not whole)."""
    return Path(str(row.get("path") or "")).exists() and not row.get("checkout_released")


def recreate_snapshot_checkout(snapshot_id: str, *, worktree_root: Optional[Any] = None,
                               data_dir: Optional[Any] = None) -> Dict[str, Any]:
    """Check a released copy out again for a continuation: files and index of its result C, HEAD at B, so
    the next capture measures the whole run from B (a file C holds stays tracked, ignored or not) and a
    moved HEAD still means the run committed. A remnant is removed first; the mark clears once whole.
    Without a captured result it refuses: a checkout goes only after its result was saved."""
    row = find_execution_snapshot(snapshot_id, data_dir)
    if row is None:
        raise FileNotFoundError(f"no private copy record for snapshot {snapshot_id}")
    if checkout_whole(row):
        return row
    wt_path = Path(str(row.get("path") or ""))
    result, baseline = str(row.get("result_sha") or ""), str(row.get("baseline_sha") or "")
    if not result or not baseline:
        raise FileNotFoundError(f"snapshot {snapshot_id} has no checkout and no captured result to restore")
    repo = Path(str(row.get("git_dir") or row.get("target_root") or "."))
    with _record_lock(wt_path, data_dir, op="recreate_checkout", task_id=str(row.get("task_id") or "")):
        current = _record_at(wt_path, data_dir, op="recreate_checkout") or row
        _put_record({**current, "checkout_released": current.get("checkout_released") or utc_now_iso()}, data_dir)
        if _deletable(wt_path, _resolve_root(worktree_root)) and wt_path.exists():
            _force_rmtree(wt_path)
        _check_out(repo, wt_path, result)
        _git(wt_path, "reset", "--soft", "--quiet", baseline)
        current = {key: value for key, value in current.items() if key != "checkout_released"}
        _put_record(current, data_dir)
    return current


def isolated_git_env() -> Dict[str, str]:
    """A git environment no host or child configuration can shape: parent-side git over a payload
    snapshot never consults the system or global config (diff drivers, textconv, excludes, hook
    templates), only `-c` overrides and the repo-local config, which payload paths reset first."""
    return _host_env(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT="0")


def payload_git_metadata_refusal(exec_root: Path) -> str:
    """lstat-trust check on a child-writable snapshot's Git metadata, or "".

    The child held a shell inside the snapshot: a ``.git`` (or ``.git/config``,
    ``.git/objects``) replaced by a symlink would make the parent's capture
    read — or a naive fix WRITE — through a child-chosen path outside the
    snapshot. Checked with lstat semantics BEFORE any parent git operation;
    a refusal is a typed capture failure and touches nothing.
    """
    git_dir = Path(exec_root) / ".git"
    if git_dir.is_symlink() or not git_dir.is_dir():
        return ".git is not a real directory (symlinked or replaced by the run)"
    config = git_dir / "config"
    if os.path.lexists(config) and (config.is_symlink() or not config.is_file()):
        return ".git/config is not a regular file (symlinked or replaced by the run)"
    objects = git_dir / "objects"
    if objects.is_symlink() or not objects.is_dir():
        return ".git/objects is not a real directory (symlinked or replaced by the run)"
    return ""


class _RawBlobWriter:
    """ONE ``git fast-import`` storing the raw blobs of one staging call.

    A Git process per blob would cost a process start per file. Each blob is
    framed by its exact length (``data <n>``), so payload bytes are never parsed
    as commands, and fast-import opens no path and applies no attribute, filter
    or eol conversion. Each id is read back (``get-mark``) before the next blob
    is sent and stderr goes to a file, so no unread pipe can fill and stall the
    writer. The ids name stored objects only after the explicit ``done``
    (``--done``) and exit 0 (``finish``); any other end kills the process.
    By default fast-import holds a blob, its predecessor and a delta of them in
    memory; ``core.bigFileThreshold`` streams every blob over 1 MiB (the
    ``--big-file-threshold`` option reaches only the first blob in Git 2.53),
    and the written pack is kept rather than exploded into loose objects by a
    second process; large payloads need not stay in Git's delta search.
    """

    def __init__(self, root: Path, env: Dict[str, str]):
        self._root, self._env, self._errors = root, env, None
        self._proc: Optional[subprocess.Popen] = None

    def __enter__(self) -> "_RawBlobWriter":
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.kill()  # it never acknowledged ``done``: nothing it wrote is staged
            self._proc.wait()
            for pipe in (self._proc.stdin, self._proc.stdout):
                with contextlib.suppress(OSError):
                    pipe.close()
        if self._errors is not None:
            self._errors.close()

    def _failure(self, detail: bytes) -> GitFailed:
        if self._proc.poll() is None:
            self._proc.kill()
        self._errors.seek(0)
        return GitFailed(self._proc.wait(), ["git", "fast-import"], detail, self._errors.read() or detail)

    def write(self, blob: bytes) -> str:
        """Store one blob; its object id."""
        if self._proc is None:
            self._errors = tempfile.TemporaryFile()
            self._proc = subprocess.Popen(
                ["git", "-c", "core.bigFileThreshold=1m", "-c", "fastimport.unpackLimit=0",
                 "fast-import", "--quiet", "--done"], cwd=str(self._root), env=self._env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._errors)
        try:
            self._proc.stdin.write(b"blob\nmark :1\ndata %d\n" % len(blob))
            self._proc.stdin.write(blob)
            self._proc.stdin.write(b"\nget-mark :1\n")
            self._proc.stdin.flush()
        except OSError as exc:  # the process is gone: its status and stderr say why
            raise self._failure(str(exc).encode()) from exc
        answer = self._proc.stdout.readline().strip()
        if not re.fullmatch(rb"[0-9a-f]{40}|[0-9a-f]{64}", answer):
            raise self._failure(b"no object id from fast-import: " + answer[:80])
        return answer.decode("ascii")

    def finish(self) -> None:
        """End the stream; raises unless every written blob is now a stored object."""
        if self._proc is None:
            return
        try:
            self._proc.stdin.write(b"done\n")
            self._proc.stdin.close()
        except OSError as exc:
            raise self._failure(str(exc).encode()) from exc
        trailing = self._proc.stdout.read()
        if self._proc.wait() != 0 or trailing:
            raise self._failure(trailing[:200])


class RawStagingIncomplete(subprocess.CalledProcessError):
    """``update-index --index-info`` exited 0, but the index lacks records as sent."""

    def __str__(self) -> str:
        return self.stderr.decode("utf-8", "replace")


def stage_raw_payload_inventory(
    worktree: Path, rel_paths: Any, env: Dict[str, str],
    baseline_modes: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Stage exact RAW bytes into the CURRENT index — no .gitattributes, no filters.

    ``git add``/worktree ``update-index`` run clean/eol filters from a child- or
    payload-authored ``.gitattributes`` (CRLF staged as LF — reproduced by
    review), so each file's raw bytes are read here, one file at a time, stored
    by ONE ``git fast-import`` for the whole inventory (``_RawBlobWriter``) and
    staged via ``--index-info``, both in the caller's ``env`` (its index, object
    database and config); ``.gitattributes`` is just content, a symlink stages
    as 120000 over its raw link-target bytes. Modes:
    with ``baseline_modes`` (capture) regular files pin to the baseline mode for
    existing paths / 100644 for new ones so an executable-bit flip can never
    ride a patch; without it (provisioning/reviewer capture) on-disk modes ride.
    ``--index-info`` exits 0 even when it skips a path Git refuses (``..``, a
    ``.git`` component, a Windows device name), so each record is sent in its
    ``ls-files --stage`` shape and ONE such read must list it back verbatim.
    Returns the paths whose on-disk executable bit diverges from the staged mode.
    """
    root = Path(worktree)
    lines: List[bytes] = []
    divergent: List[str] = []
    with _RawBlobWriter(root, env) as writer:
        for rel in sorted(str(p) for p in rel_paths):
            path = root / rel
            if path.is_symlink():
                blob, mode = os.readlink(os.fsencode(str(path))), "120000"
            else:
                blob = path.read_bytes()
                executable = bool(path.stat().st_mode & 0o111)
                if baseline_modes is None:
                    mode = "100755" if executable else "100644"
                else:
                    base = baseline_modes.get(rel, "")
                    mode = base if base in ("100644", "100755") else "100644"
                    if executable != (mode == "100755"):
                        divergent.append(rel)
            sha = writer.write(blob)
            lines.append(f"{mode} {sha} 0\t{rel}".encode("utf-8", errors="surrogateescape"))
        writer.finish()
    update = subprocess.run(["git", "update-index", "-z", "--index-info"], cwd=str(root), capture_output=True,
                            env=env, input=b"\0".join(lines) + b"\0" if lines else b"", check=False)
    if update.returncode:
        raise GitFailed(update.returncode, update.args, update.stdout, update.stderr)
    listed = set(_git_env(root, "ls-files", "--stage", "-z", env=env).stdout.split(b"\0"))
    unstaged = [line.partition(b"\t")[2] for line in lines if line not in listed]
    if unstaged:  # names at most five; Git's own stderr is carried as evidence, never parsed
        detail = b"update-index exited 0, but the index lacks %d of %d records as sent (%s); git: %s" % (
            len(unstaged), len(lines), b", ".join(unstaged[:5]), update.stderr.strip()[:300])
        raise RawStagingIncomplete(update.returncode, update.args, update.stdout, detail)
    return divergent


@contextlib.contextmanager
def payload_capture_git_env(exec_root: Path, *, objects: Optional[Path] = None):
    """A PARENT-OWNED throwaway Git control dir over the child-writable snapshot.

    Yields a git env whose GIT_DIR, config, hooks template and GIT_INDEX_FILE
    all live in a host-managed temp directory (mkdtemp, 0700; the index file is
    pre-created 0600) — never inside the snapshot the child could write. The
    child's object database (``objects`` — a linked worktree's COMMON objects
    directory — or ``<exec_root>/.git/objects``) is attached READ-ONLY via the alternates mechanism
    (objects are content-addressed, so the recorded baseline commit/tree can be
    read but not silently substituted), while new blobs staged by the capture
    land in the parent-owned control ODB. Child ``.git/index`` and
    ``.git/config`` are never read and never written: an index-only blob forged
    by the child simply does not exist for this environment, and no
    child-controlled diff driver / filter / hook can execute in the parent.
    """
    resolved = Path(exec_root).resolve()
    control = Path(tempfile.mkdtemp(prefix="obo-payload-capture-"))
    try:
        env = isolated_git_env()
        _git(control, "init", "--template=", env=env)
        alternates = control / ".git" / "objects" / "info" / "alternates"
        alternates.parent.mkdir(parents=True, exist_ok=True)
        # Byte-exact: text mode would emit CRLF on Windows -> git seeks "objects\r".
        source_objects = Path(objects).resolve() if objects else resolved / ".git" / "objects"
        alternates.write_bytes((str(source_objects) + "\n").encode("utf-8"))
        index = control / "index"
        os.close(os.open(index, os.O_CREAT | os.O_WRONLY, 0o600))
        env["GIT_DIR"] = str(control / ".git")
        env["GIT_WORK_TREE"] = str(resolved)
        env["GIT_INDEX_FILE"] = str(index)
        yield env
    finally:
        _force_rmtree(control)  # the capture's git objects are read-only; a plain rmtree leaves them on Windows


def _reproduce_confined_link(target_root: Path, src_link: Path, dest_link: Path) -> bool:
    """Reproduce one payload symlink in the snapshot, or refuse an escape (R1 item 4).

    A confined RELATIVE link is copied verbatim (``copytree(symlinks=True)``
    semantics); a confined ABSOLUTE link is rewritten to the equivalent relative
    link so the snapshot stays self-contained; a link resolving outside the
    payload is NOT copied (it is already outside the loader inventory).
    """
    try:
        raw = os.readlink(src_link)
        resolved = src_link.resolve(strict=False)
        rel_target = resolved.relative_to(target_root)
    except (OSError, ValueError):
        return False
    if os.path.isabs(raw):
        raw = os.path.relpath(target_root / rel_target, src_link.parent)
    dest_link.parent.mkdir(parents=True, exist_ok=True)
    if not os.path.lexists(dest_link):
        os.symlink(raw, dest_link)
    return True


def _copy_payload_inventory(target: Path, dest: Path, *, manifest_entry: str = "",
                            manifest_scripts: Optional[List[Dict[str, Any]]] = None,
                            preserve_links: bool = True) -> int:
    """Copy the exact skill-loader-visible inventory of ``target`` into ``dest``.

    The loader inventory is the SSOT walk (cache/control-dir exclusions, symlink
    escape exclusion, credential-shape refusal) — no second filesystem walk is
    invented; a declared manifest entry or script joins it even under an excluded
    folder, exactly as it joins the content hash. A file reached through a
    symlinked ancestor directory reproduces the ancestor LINK once instead of
    materializing a second copy under it. Read-only review views set
    ``preserve_links=False`` to retain each inventory path's bytes even when a
    link points into an excluded dependency folder; they copy no extra files.
    """
    from ouroboros.skill_loader import _iter_payload_files

    resolved_target = target.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    copied = 0
    for path in _iter_payload_files(resolved_target, manifest_entry=manifest_entry,
                                    manifest_scripts=manifest_scripts):
        rel = path.relative_to(resolved_target)
        # A symlinked ANCESTOR directory is reproduced as the link itself; the
        # linked content is copied at its real (confined) location by its own
        # inventory entry, exactly like copytree(symlinks=True).
        ancestor = resolved_target
        via_link = False
        for part in rel.parts[:-1]:
            ancestor = ancestor / part
            if preserve_links and ancestor.is_symlink():
                link_rel = ancestor.relative_to(resolved_target)
                if _reproduce_confined_link(resolved_target, ancestor, dest / link_rel):
                    copied += 1
                via_link = True
                break
        if via_link:
            continue
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if preserve_links and path.is_symlink():
            if _reproduce_confined_link(resolved_target, path, out):
                copied += 1
            continue
        shutil.copy2(path, out)
        copied += 1
    return copied


def provision_payload_snapshot(
    *,
    target_root: Any,
    task_id: Any,
    snapshot_id: str,
    worktree_root: Optional[Any] = None,
    data_dir: Optional[Any] = None,
) -> ExecutionSnapshotHandle:
    """Snapshot ONE non-Git skill payload into a private STANDALONE Git repo (R1 §9.2).

    The live payload is NEVER initialized as Git and never touched: the
    loader-visible inventory is copied into a private directory under the
    delegated snapshot root (outside repo/data), Git is initialized only THERE,
    and the copied payload becomes the synthetic baseline commit. The pre-copy
    skill-loader content hash is recomputed after the copy — a changed hash
    means the snapshot raced another writer, so it is removed and the run must
    not start. Registered durably (``standalone=true``) BEFORE any start intent.
    """
    from ouroboros.tools.delegate_integration import payload_content_hash

    target = Path(target_root).resolve()
    if not target.is_dir():
        raise ValueError(f"skill payload {target} does not exist")
    if (target / ".git").exists():
        raise ValueError(
            f"skill payload {target} unexpectedly contains .git; refusing to snapshot")
    root = _resolve_root(worktree_root)
    if _is_within(root, target) or _is_within(target, root):
        raise ValueError(f"subagent worktree root {root} overlaps the snapshot target {target}")
    # The TARGET legitimately lives inside runtime data (data/skills/...), but
    # the snapshot ROOT itself must stay outside BOTH the system repo and the
    # runtime data root — a snapshot under data/ would put a child-writable
    # Git repo inside live state (reproduced by review).
    _assert_root_isolated(root, Path(__file__).resolve().parents[1], _data_dir(data_dir))
    snap = str(snapshot_id or "").strip()
    if not snap:
        raise ValueError("snapshot_id is required for a delegated payload snapshot")
    safe_snap = _safe_name(snap)
    task = str(task_id or "")
    started = time.monotonic()
    wt_path = copy_path(root, f"dlgp_{_safe_name(task_id)}_{safe_snap[:16]}").resolve()
    _record_at(wt_path, data_dir, op="provision_payload_snapshot")  # an unreadable record refuses before the copy
    root.mkdir(parents=True, exist_ok=True)
    if _deletable(wt_path, root) and wt_path.exists():
        _force_rmtree(wt_path)  # idempotent re-provision of the SAME snapshot id
    source_hash = payload_content_hash(target)
    env = isolated_git_env()
    try:
        # The standalone snapshot's .git is private, so its only shared metadata
        # is its own record, written at the end.
        entry_count = _copy_payload_inventory(target, wt_path)
        # Fully config-isolated baseline: empty template dir (no user hooks),
        # no global/system config (Fable F1). RAW staging instead of `git
        # add` (Sol P1 modes/filters): a payload .gitattributes (eol/clean)
        # must not normalize the recorded baseline away from the live raw
        # bytes, and the baseline records the REAL file modes.
        _git(wt_path, "init", "--template=", env=env)
        from ouroboros.skill_loader import _iter_payload_files

        stage_raw_payload_inventory(
            wt_path,
            (p.relative_to(wt_path).as_posix() for p in _iter_payload_files(wt_path)),
            env)
        _git(
            wt_path,
            "-c", "user.email=ouroboros@localhost", "-c", "user.name=Ouroboros",
            "commit", "--allow-empty", "-m",
            f"ouroboros: delegated payload baseline {snap}", env=env,
        )
        baseline_sha = _git(wt_path, "rev-parse", "HEAD", env=env).stdout.strip()
        baseline_tree = _git(wt_path, "rev-parse", "HEAD^{tree}", env=env).stdout.strip()
        manifest_raw = _git(wt_path, "ls-tree", "-r", "-z", baseline_tree, env=env).stdout
        import hashlib

        manifest_digest = hashlib.sha256(
            manifest_raw.encode("utf-8", errors="surrogateescape")).hexdigest()
        if payload_content_hash(target) != source_hash:
            raise RuntimeError(
                "the live payload changed while it was being snapshotted "
                "(another writer raced the copy); retry the delegation")
        handle = ExecutionSnapshotHandle(
            snapshot_id=snap,
            task_id=task,
            path=str(wt_path),
            target_root=str(target),
            baseline_ref="",
            baseline_sha=baseline_sha,
            baseline_tree=baseline_tree,
            manifest_digest=manifest_digest,
            target_head="",
            created_at=time.time(),
            entry_count=entry_count,
            standalone=True,
            payload_hash=source_hash,
            provisioning_sec=round(time.monotonic() - started, 3),
        )
        # Record write INSIDE the cleanup scope: an unrecorded snapshot
        # directory would be invisible to disposal/retention (orphan leak).
        _put_record({**asdict(handle), "kind": _KIND_DELEGATED_EXEC}, data_dir)
    except Exception:
        _force_rmtree(wt_path)
        raise
    return handle


def find_execution_snapshot(snapshot_id: str, data_dir: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    """The record of a delegated execution snapshot, or None."""
    snap = str(snapshot_id or "").strip()
    if not snap:
        return None
    return next((entry for entry in _records(data_dir, op="find_execution_snapshot")
                 if entry.get("kind") == _KIND_DELEGATED_EXEC and entry.get("snapshot_id") == snap), None)


def remove_execution_snapshot(
    snapshot_id: str,
    *,
    worktree_root: Optional[Any] = None,
    data_dir: Optional[Any] = None,
    holds: Optional[Callable[[Dict[str, Any]], bool]] = None,
    reasons: Optional[Dict[str, str]] = None,
) -> bool:
    """Remove a disposed snapshot: its checkout (if still there), its baseline and
    result pins and its record. Call only after explicit disposition or GC's
    ownership/custody proof. A surviving tree, or one outside the worktree root,
    keeps its record and pins; unchanged failures wait until a new process. The record read under
    the lock decides, with ``holds`` (GC's proof) and the owner's keep, also after apply or reject.
    ``reasons`` receives the keep gate's refusal by the copy's path, as in ``prune_orphans``.
    """
    entry = find_execution_snapshot(snapshot_id, data_dir)
    if entry is None or worktree_removal.unchanged(entry):
        return False
    root = _resolve_root(worktree_root)
    wt_path = Path(str(entry.get("path") or ""))
    with _record_lock(wt_path, data_dir, op="remove", task_id=str(entry.get("task_id") or "")):
        entry = _record_at(wt_path, data_dir, op="remove")
        if entry is None or entry.get("snapshot_id") != snapshot_id or (holds is not None and not holds(entry)):
            return False
        with owner_keep.gate(_data_dir(data_dir), entry.get("task_id")) as why:
            if why:
                if reasons is not None:
                    reasons[str(wt_path)] = why
                return False
            if _deletable(wt_path, root) and wt_path.exists():
                _force_rmtree(wt_path)
            if wt_path.exists() if _deletable(wt_path, root) else os.path.lexists(wt_path):
                # It survived its removal, or lies outside this worktree root: record and pins stay.
                _mark_removal_failed(entry, data_dir)
                return False
            if not entry.get("standalone"):
                # A standalone payload snapshot (R1 §10.4) keeps its .git INSIDE the directory just
                # deleted; a Git snapshot also owns an admin dir and pins in the TARGET repository.
                target = Path(str(entry.get("git_dir") or entry.get("target_root") or "."))
                _forget_checkout(target, wt_path)
                _unpin_snapshot(target, entry)
            _drop_record(wt_path, data_dir)
    worktree_removal.retired(entry, _data_dir(data_dir))
    return True


def _task_proven_over(task_id: str, data_dir: Any, live: Any) -> bool:
    """C10: terminal record, a negative supervisor probe, and no lineage process."""
    from ouroboros import body_candidate
    try:
        return bool(task_id) and live is not None and body_candidate._recorded_terminal(task_id, data_dir) \
            and live(task_id) is False and not body_candidate.live_lineage_processes(data_dir, task_id)
    except Exception:
        return False  # unreadable or unknown ownership never authorizes GC


def snapshot_released(entry: Dict[str, Any], open_snapshot_ids: set, *, data_dir: Any = None, live: Any = None) -> bool:
    """The one proof a delegated snapshot goes by, for the custody GC and the owner's free action
    alike (its preview offers exactly what this answers): the keep-set does not name it, its task is
    proven over (``live``: the supervisor's probe unless given), fresh readable custody does not hold it."""
    from ouroboros import delegate_custody as custody

    if live is None:
        from supervisor.queue import task_settlement_liveness as live
    snap = str(entry.get("snapshot_id") or "")
    try:
        return snap not in open_snapshot_ids and _task_proven_over(str(entry.get("task_id") or ""), data_dir, live) \
            and not owner_keep.is_kept(_data_dir(data_dir), entry.get("task_id")) \
            and not custody.custody_log_unreadable(_data_dir(data_dir)) \
            and snap not in custody.open_snapshot_ids(_data_dir(data_dir))
    except Exception:
        return False


def prune_execution_snapshots(
    open_snapshot_ids: set,
    *,
    worktree_root: Optional[Any] = None,
    data_dir: Optional[Any] = None,
    only: Optional[Any] = None,
    live: Optional[Any] = None,
    reasons: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Retire snapshots only after their task is proven over and custody releases them
    (``snapshot_released``, ``live`` as there; the keep-set is conservative, fresh custody is checked
    before each removal). A provisional snapshot can precede START_REQUESTED, so absent custody is not
    proof its owner is over; unknown task/process/custody state keeps even a missing checkout's row
    and pin. ``only`` restricts the pass; surviving trees keep ``removal_failed``. ``reasons``: as in ``prune_orphans``."""
    open_ids = {str(s) for s in (open_snapshot_ids or set())}
    chosen = None if only is None else {str(s) for s in only}
    removed: List[str] = []
    kept: List[str] = []
    for entry in _records(data_dir, op="prune_execution_snapshots"):
        if entry.get("kind") != _KIND_DELEGATED_EXEC:
            continue
        snap = str(entry.get("snapshot_id") or "")
        if chosen is not None and snap not in chosen:
            continue
        if not snapshot_released(entry, open_ids, data_dir=data_dir, live=live):
            kept.append(snap)
            continue
        (removed if remove_execution_snapshot(snap, worktree_root=worktree_root, data_dir=data_dir, holds=lambda row: snapshot_released(
            row, open_ids, data_dir=data_dir, live=live), reasons=reasons) else kept).append(snap)
    return {"removed": removed, "kept": kept}


def remove_worktree(
    *,
    task_id: str = "",
    path: str = "",
    worktree_root: Optional[Any] = None,
    data_dir: Optional[Any] = None,
    lock_wait_sec: Optional[float] = None,
) -> bool:
    """Tear down a worktree by task_id or path and drop its record. Returns success."""
    want_path = str(Path(path).resolve()) if path else ""
    match: Optional[Dict[str, Any]] = None
    for entry in _records(data_dir, op="remove_worktree"):
        if entry.get("kind") in (_KIND_DELEGATED_EXEC, _KIND_BODY_CANDIDATE):
            continue
        if (task_id and entry.get("task_id") == str(task_id)) or (want_path and entry.get("path") == want_path):
            match = entry
            break
    root = _resolve_root(worktree_root)
    if match is None:
        # Unrecorded path: best-effort directory removal, but ONLY inside the
        # configured worktree root (never an arbitrary path supplied by a caller).
        if want_path and Path(want_path).exists() and _is_within(Path(want_path), root):
            _force_rmtree(Path(want_path))
            return True
        return False
    return _retire_worktree(match, root, data_dir, match.get("branch") or "", lock_wait_sec)


def _retire_worktree(match: Dict[str, Any], root: Path, data_dir: Any, branch: str,
                     lock_wait_sec: Optional[float] = None, holds: Optional[Callable[[Dict[str, Any], str], Optional[str]]] = None,
                     reasons: Optional[Dict[str, str]] = None) -> bool:
    """Retire one copy under its own record lock, only while its record still equals ``match`` and
    ``holds`` (the deciding owner's current proof, asked before the keep gate) still answers the branch to
    drop, not ``None``. An own-body candidate is marked retiring inside the gate, at the effect: a keep or a
    busy gate leaves its record as it was."""
    if worktree_removal.unchanged(match):
        return False
    wt_path = Path(match.get("path") or "")
    with _record_lock(wt_path, data_dir, op="remove_worktree", task_id=str(match.get("task_id") or ""),
                      timeout_sec=lock_wait_sec):
        if _record_at(wt_path, data_dir, op="remove_worktree") != match:
            if reasons is not None:
                reasons[str(wt_path)] = "changed"
            return False  # another writer changed ownership since the decision
        branch = holds(match, branch) if holds else branch
        if branch is None:
            return False
        with owner_keep.gate(_data_dir(data_dir), match.get("task_id")) as why:
            if why:
                if reasons is not None:
                    reasons[str(wt_path)] = why
                return False
            if match.get("kind") == _KIND_BODY_CANDIDATE and not match.get("retiring"):
                _put_record({**match, "retiring": True}, data_dir)  # no resume takes it now, even if its removal fails
            if _deletable(wt_path, root) and wt_path.exists():
                _force_rmtree(wt_path)
            from ouroboros import body_candidate
            body_candidate.discard_scratch(match)  # an own-body copy's sibling process environment, unless retained
            if not _remove_paths(Path(match.get("git_dir") or match.get("repo_dir") or "."), wt_path,
                                 branch, allowed_root=root):
                _mark_removal_failed(match, data_dir)  # the directory survived: record and pin stay
                return False
            _drop_record(wt_path, data_dir)
    worktree_removal.retired(match, _data_dir(data_dir))
    return True


def prune_orphans(
    *,
    worktree_root: Optional[Any] = None,
    data_dir: Optional[Any] = None,
    retention_days: Optional[int] = None,
    live: Optional[Any] = None,
    only: Optional[Any] = None,
    only_rows: Optional[Dict[str, Dict[str, Any]]] = None,
    reasons: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Reconcile worktrees on the custody cadence and the owner's free action.

    A plain copy missing or past retention goes only with proof its task is over: its record is
    terminal, ``live(task_id)`` (the supervisor's probe: True live, False proven not live, None
    unknown) answers False, no lineage process lives, and its unique work is proven saved
    (``copy_capture_proof``). Without a probe or with an unknown answer, age alone deletes nothing. ``only`` restricts the pass to
    those task ids (the free action's previewed rows), leaving every other copy untouched.
    ``only_rows`` (path -> the record previewed) names exact objects: one whose record changed stays.
    Each decision is rechecked under that copy's record lock; a checkout that survives
    removal keeps its first failure stamp and pin; a new process or directory identity retries it.
    ``reasons``, when given, receives by copy path the typed answer a kept copy got from this pass
    (a body verdict's reason, a busy or unreadable record, the keep gate's refusal); a proof that
    simply no longer holds names none: the caller reads that fresh.
    """
    reasons = {} if reasons is None else reasons
    retention = retention_days if retention_days is not None else get_gc_retention_days()
    cutoff = age_cutoff(retention)
    root = _resolve_root(worktree_root)
    chosen = None if only is None else {str(task_id) for task_id in only}
    entries = _records(data_dir, op="prune_orphans")
    selected = [entry for entry in entries if entry.get("kind") != _KIND_DELEGATED_EXEC
                and (chosen is None or str(entry.get("task_id") or "") in chosen)
                and (only_rows is None or only_rows.get(str(entry.get("path") or "")) == entry)
                and not worktree_removal.unchanged(entry)]
    for entry in entries:
        previewed = (only_rows or {}).get(str(entry.get("path") or ""))
        if previewed is not None and previewed != entry:
            reasons[str(entry["path"])] = "changed"
    if not selected:
        return {"removed": 0, "kept": len(entries)}
    # Candidate evidence comes without any lock; each decision is rechecked under that copy's own lock.
    from functools import partial
    from ouroboros import body_candidate
    from ouroboros.copy_capture_proof import plain_copy_branch
    verdicts: Dict[str, Dict[str, Any]] = {}
    for entry in selected:
        if entry.get("kind") == _KIND_BODY_CANDIDATE:
            expired = max(float(entry.get("created_at") or 0), float(entry.get("used_at") or 0)) < cutoff
            # A kept task's candidate is not captured: no pin or event for what stays (the gate reads the keep again).
            verdicts[str(entry.get("path") or "")] = {"remove": False, "reason": owner_keep.OWNER_KEEP} if expired and \
                owner_keep.is_kept(_data_dir(data_dir), entry.get("task_id")) else body_candidate.retention_verdict(
                    entry, data_dir=data_dir, expired=expired)

    def body_branch(verdict: Dict[str, Any], match: Dict[str, Any], branch: str) -> Optional[str]:
        if body_candidate.verdict_holds(match, verdict, data_dir=data_dir):
            return "" if verdict.get("keep_branch") else branch
        reasons[str(match.get("path") or "")] = "changed"
        return None

    removed = 0
    for preview in selected:
        path = str(preview.get("path") or "")
        branch = str(preview.get("branch") or "")
        try:
            with _record_lock(path, data_dir, op="prune", task_id=str(preview.get("task_id") or "")):
                entry = _record_at(path, data_dir, op="prune_orphans")
                if entry != preview:
                    reasons[path] = "changed"
                    continue
                if entry.get("kind") == _KIND_BODY_CANDIDATE:
                    if not verdicts[path]["remove"]:
                        reasons[path] = verdicts[path]["reason"]
                        continue
                else:
                    path_exists = bool(entry.get("path")) and Path(entry["path"]).exists()
                    if (path_exists and float(entry.get("created_at") or 0) >= cutoff) \
                            or not _task_proven_over(str(entry.get("task_id") or ""), data_dir, live):
                        continue
        except (WorktreeOpsLockBusy, SubagentWorktreeRegistryCorrupt) as exc:
            reasons[path] = "record_busy" if isinstance(exc, WorktreeOpsLockBusy) else "record_unreadable"
            continue  # busy or unreadable: the next pass decides
        removed += bool(_retire_worktree(entry, root, data_dir, branch, holds=partial(body_branch, verdicts[path]) if entry.get(
            "kind") == _KIND_BODY_CANDIDATE else partial(plain_copy_branch, root=root, data_dir=data_dir, live=live), reasons=reasons))
    return {"removed": removed, "kept": len(_records(data_dir, op="prune_orphans"))}


def list_worktrees(data_dir: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Every readable copy record (for UI / inspection)."""
    return _records(data_dir)


def registered_checkout(path: Any, data_dir: Optional[Any] = None) -> bool:
    """Is ``path`` one of the recorded copies? An unreadable own record answers yes: it was never read as absent."""
    from ouroboros.private_copy_read import checkout_registration

    return checkout_registration(path, data_dir) != "absent"
