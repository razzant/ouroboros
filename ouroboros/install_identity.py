"""Installation identity and mutation authority for stored data-root pointers.

Only the configured canonical root may mint an identity. A copied/moved root
gets its own identity; a stored path never acquires authority by existing.
Identity-free isolated roots remain usable by tools and hermetic tests.

Canonical paths and ownership verdicts are process-local memos; copy/move happens
before admission, never beneath a running install. Identity records use a stat
witness, and minting clears verdicts. No steady-state check reads JSON or resolves
ancestors again.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import pathlib
import uuid
from typing import Any

from ouroboros.store import StoreError, name_lock
from ouroboros.utils import atomic_write_json, utc_now_iso

log = logging.getLogger(__name__)
IDENTITY_PATH = pathlib.Path("state/install.json")
TASK_ROOT_FIELDS = ("budget_drive_root", "drive_root", "child_drive_root", "headless_child_drive_root")
_CANONICAL_ROOTS: dict[Any, pathlib.Path] = {}
_IDENTITIES: dict[pathlib.Path, tuple[str, tuple | None, dict | None]] = {}
_INITIALIZED: dict[pathlib.Path, dict] = {}
_VERDICTS: dict[tuple, tuple[pathlib.Path, pathlib.Path | None]] = {}


class StoreForeign(StoreError):
    """A stored pointer names another installation; no mutation was attempted."""

    reason = reason_code = "foreign_install_root"
    rollback_confirmed = True

    def __init__(self, path: Any, install_root: Any) -> None:
        self.path, self.install_root = pathlib.Path(path), pathlib.Path(install_root)
        super().__init__(f"{self.path} belongs to another Ouroboros install at {self.install_root}; "
                         "move it explicitly to work on it here")


def layout_root(path: Any) -> pathlib.Path:
    """Canonical parent of the existing headless/task-drive layouts, recursively."""
    root = pathlib.Path(path).resolve(strict=False)
    while True:
        parents = root.parents
        if (root.name == "data" and len(parents) > 3 and parents[1].name == "headless_tasks"
                and parents[2].name == "state"):
            root = parents[3]
        elif len(parents) > 1 and root.parent.name == "task_drives":
            root = parents[1]
        else:
            return root


def _same_root(recorded: Any, root: pathlib.Path) -> bool:
    """One directory under two spellings (letter case on a case-insensitive disk) is one install."""
    if str(recorded) == str(root):
        return True
    if str(recorded).casefold() != str(root).casefold():
        return False  # no stat for an ordinary different path
    try:
        return os.path.samefile(recorded, root)
    except OSError:
        return False


_ACTING = contextvars.ContextVar("ouroboros_acting_install", default=None)


@contextlib.contextmanager
def acting_as(root: Any):
    """Act as the install ROOT names for this context: tooling that launched an isolated install
    (a benchmark isolate) reads its store from outside, as that install's own process would."""
    token = _ACTING.set(layout_root(root))
    try:
        yield
    finally:
        _ACTING.reset(token)


def canonical_root() -> pathlib.Path:
    acting = _ACTING.get()
    if acting is not None:
        return acting
    configured = os.environ.get("OUROBOROS_DATA_DIR")
    if configured is None:
        configured = (None, os.environ.get("OUROBOROS_APP_ROOT"), os.path.expanduser("~"))
    if configured not in _CANONICAL_ROOTS:
        from ouroboros.config import resolve_data_dir
        _CANONICAL_ROOTS[configured] = layout_root(resolve_data_dir())
    return _CANONICAL_ROOTS[configured]


def _read(root: pathlib.Path) -> dict | None:
    cached = _IDENTITIES.get(root)
    path = cached[0] if cached else str(root / IDENTITY_PATH)
    try:
        stat = os.stat(path)
        witness = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
        if cached and cached[1] == witness:
            return cached[2]
        record = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        _IDENTITIES[root] = (path, None, None)
        return None
    except (OSError, ValueError) as exc:
        raise StoreError(f"install identity unreadable at {root / IDENTITY_PATH}: {exc}") from exc
    if (not isinstance(record, dict) or not record.get("id") or not record.get("root")
            or not isinstance(record.get("copied_from", []), list)
            or any(not isinstance(path, str) or not path for path in record.get("copied_from", []))):
        raise StoreError(f"install identity invalid at {root / IDENTITY_PATH}")
    _IDENTITIES[root] = (path, witness, record)
    return record


def initialize_install() -> dict:
    """Atomically establish this process's canonical identity at its first write/boot.

    The existing startup owned-process check reaches this before recovery. The
    same gate covers standalone writers. Successful initialization runs once per
    canonical root in this process; the lock serializes worker first touch.
    """
    root = canonical_root()
    if root in _INITIALIZED:
        return _INITIALIZED[root]
    record = _read(root)
    if record and _same_root(record["root"], root):
        _INITIALIZED[root] = record
        return record
    from ouroboros.store import bounded_wait
    wait = bounded_wait(5.0)
    with name_lock(root, "install.lock", timeout_sec=wait):
        record = _read(root)
        if record and _same_root(record["root"], root):
            _INITIALIZED[root] = record
            return record
        previous = record.get("root") if record else None
        origins = record.get("copied_from", []) if record else []
        record = {"id": uuid.uuid4().hex, "root": str(root), "created_at": utc_now_iso()}
        if previous:
            record["copied_from"] = list(dict.fromkeys([*origins, previous]))
        atomic_write_json(root / IDENTITY_PATH, record, trailing_newline=True, fsync=True)
        _IDENTITIES.pop(root, None)
        _VERDICTS.clear()
        _INITIALIZED[root] = record
        if previous:
            log.warning("This data directory moved from %s to %s and now has a separate Ouroboros "
                        "install identity. Saved paths keep their original owners; move tasks "
                        "explicitly to work on them here.", previous, root)
        return record


def recorded_id(path: Any) -> str:
    """The identity PATH's install has minted for itself, read-only: ``""`` before its first boot or
    write mints one, or while a copy still carries its original's record. A record that cannot be read
    again keeps the identity this process last read there (copy/move never happens beneath a running
    install); one never read gives ``""``. Never mints and never raises; a stat witness keeps reads cheap."""
    root = layout_root(path)
    try:
        record = _read(root)
    except StoreError:
        record = (_IDENTITIES.get(root) or (None, None, None))[2]
    return str(record["id"]) if record and _same_root(record["root"], root) else ""


def _ownership(path: Any, *, initialize: bool = True) -> tuple[pathlib.Path, pathlib.Path | None]:
    current = canonical_root()
    if initialize and current not in _INITIALIZED:
        initialize_install()
    if str(pathlib.Path(path)) == str(current):
        return current, None  # this install's own root, spelled canonically: no stat, no walk
    record = _read(current)
    # The root separates creation-free reads of copies that still share an id.
    key = (str(path), current, (record or {}).get("id"))
    if key in _VERDICTS:
        return _VERDICTS[key]
    origins = (record or {}).get("copied_from", [])
    root, foreign = pathlib.Path(path).resolve(strict=False), None
    # Stop at the configured root: its ancestors do not own it. This walk runs
    # only on first use of a path in this identity's lifetime.
    for candidate in (root, *root.parents):
        candidate = layout_root(candidate)
        if candidate == current or _same_root(candidate, current):
            break
        if str(candidate) in origins or _read(candidate) is not None:
            foreign = candidate
            break
    _VERDICTS[key] = (root, foreign)
    return root, foreign


def owned_root(path: Any) -> bool:
    """Whether this process may mutate PATH. An unreadable identity raises, never grants, for every
    path but this install's own canonical root, which needs no identity read."""
    return _ownership(path)[1] is None


def require_owned_root(path: Any) -> pathlib.Path:
    """Refuse before any lock, transaction, file copy or process registration."""
    root, foreign = _ownership(path)
    if foreign is not None:
        raise StoreForeign(root, foreign)
    return root


def check_store_root(path: Any, *, write: bool) -> None:
    """Even a read transaction uses rw SQLite/journal/name-lock IO: refuse foreign roots.

    An own read stays creation-free; copied history in the current install is
    readable before its first boot/write establishes the identity.
    """
    root, foreign = _ownership(path, initialize=write)
    if foreign is not None:
        raise StoreForeign(root, foreign)


def require_task_roots(task: dict) -> None:
    """Validate TASK_ROOT_FIELDS before restored-task admission or stored-root effects.

    Call before file preparation, process launch, mailbox or accounting work.
    Raises StoreForeign with reason_code and plain detail in str(error); it
    never rewrites pointers. Own layout drives and identity-free roots remain usable.
    """
    for key in TASK_ROOT_FIELDS:
        if task.get(key):
            require_owned_root(task[key])


def foreign_task_refusal(task: dict) -> StoreError | None:
    """The typed refusal when a stored task names another install's root, else None; an unreadable
    identity is returned too: unknown authority grants no launch here."""
    try:
        require_task_roots(task)
    except StoreError as error:
        return error
    return None


def disclose_foreign_task(root: Any, task_id: str, error: StoreError) -> None:
    """Name the refusal once on this install's own task row, status unchanged (best effort, off the queue lock)."""
    from ouroboros.task_results import write_task_result

    fields = {"reason_code": getattr(error, "reason_code", None) or "install_identity_unreadable",
              "reason_detail": str(error)}

    def disclose(current: dict, _patch: dict) -> dict | None:
        if not current or all(current.get(key) == value for key, value in fields.items()):
            return None
        return {"status": current["status"], **fields}

    try:
        write_task_result(root, task_id, "", strict_existing_dict=True, _field_projector=disclose)
    except Exception:
        log.debug("Foreign-root reason for %s not recorded; the row stays unlaunched", task_id, exc_info=True)


def rest_foreign_member(rest: Any, task_id: str, stored: dict, error: StoreForeign) -> None:
    """Disclose on this install's task, then rest its unchanged pointer until an explicit move.

    No foreign row is read. The reason write advances the member signature, so
    take its new witness once; a concurrent relocation prevents this rest.
    """
    from ouroboros import store
    from ouroboros.task_results import write_task_result

    fields = {"reason_code": error.reason, "reason_detail": str(error)}
    pointers = {key: stored.get(key) for key in TASK_ROOT_FIELDS}

    def unchanged(row: dict) -> bool:
        return bool(row) and all(row.get(key) == value for key, value in pointers.items())

    def disclose(current: dict, _patch: dict) -> dict | None:
        if not unchanged(current) or all(current.get(key) == value for key, value in fields.items()):
            return None
        return {"status": current["status"], **fields}

    write_task_result(rest.root, task_id, stored["status"], strict_existing_dict=True, _field_projector=disclose)
    with store.open_store(rest.root, write=False) as txn:
        current = txn.load(task_id) or {}
        if not unchanged(current) or any(current.get(key) != value for key, value in fields.items()):
            return
        rest.signatures[task_id] = list(txn.row(task_id, "write_seq", "updated_at"))
    rest.wait(task_id, error.reason, report=fields)
