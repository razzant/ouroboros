"""The installation's ownership set and the one bounded stop of a server generation (ARCHITECTURE §9).

``state/owned_processes.json`` under the canonical data root names every process this
installation started and has not forgotten. Executor and tracked-command records enter
through ``workspace_executor._register_process`` and leave through ``_forget_process``;
ledgered spawns enter through ``process_custody.record_process`` and leave when the ledger
compaction drops their row or a stop confirms their exit. The per-drive records and the PID
ledger stay the custody they were (the reaper reads the ledger); the set is the address the
exit reads instead of walking ``data/state``. One document, atomic replace, written under
the canonical ledger's append lock. A registration that cannot take that lock is never lost:
it lands as one lock-free file in ``state/owned_processes.pending/``, which every read merges
and the next locked update folds in (the directory holds only such contended registrations).

``begin_owned_stop`` is called first by every shutdown door (owner Restart, the lifespan
teardown, the emergency exit): it starts the generation's one grace deadline and stamps
``stop_requested_at`` on every target before any other owner waits, so a launcher kill at the
grace still leaves every pending stop recorded. ``stop_owned_work`` is the generation's one
stop. The first caller starts it; a later caller joins it until completion or the shared
deadline and never starts a second one. It stops through the existing typed kill helpers,
forgets what the existing liveness predicates confirm, stamps the rest ``unconfirmed_since``
without waiting for a lock once the deadline passed, and returns: the exit owner exits. A target recorded
after the stop began (the owner Restart stops before the server exits) is born stamped. The
next start's ``finish_unconfirmed_stops`` retries every stamped record under the same bound
before admission and before any extension or replacement process starts; what it cannot
confirm stays recorded and counted. Records an older release left on disk are indexed once by
``start_inherited_import`` on a background thread, so no start waits on that walk.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

from ouroboros.utils import append_jsonl, atomic_write_json, jsonl_append_lock_path, utc_now_iso

log = logging.getLogger(__name__)

OWNED_PROCESSES_FILENAME = "owned_processes.json"
PENDING_DIRNAME = "owned_processes.pending"
_SCHEMA_VERSION = 1
_EXECUTOR_KINDS = ("foreground", "service")
_POLL_SEC = 0.05  # liveness re-check granularity of process_custody.stop_ledgered_processes


class _Stop:
    """One server generation's stop: started once, then joined until done or its deadline."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.deadline: Optional[float] = None  # set by the first shutdown door (begin or stop)
        self.started = False
        self.outcome: Optional[Dict[str, Any]] = None


_GENERATION_STOP = _Stop()  # one per server process; a test starts a fresh one


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def installation_root(drive_root: Any) -> pathlib.Path:
    """The canonical data root whose set names ``drive_root``'s processes.

    A drive inside the configured data root belongs to it. Outside it (an isolated
    root), an own child drive maps to its parent by the host layout of
    ``task_custody.own_child_drives``; any other drive is its own root."""
    from ouroboros.config import resolve_data_dir

    drive = pathlib.Path(drive_root).resolve(strict=False)
    data = resolve_data_dir().resolve(strict=False)
    if drive == data or data in drive.parents:
        return data
    while True:
        parents = drive.parents
        if (len(parents) > 3 and drive.name == "data" and parents[1].name == "headless_tasks"
                and parents[2].name == "state"):
            drive = parents[3]
        elif len(parents) > 1 and parents[0].name == "task_drives":
            drive = parents[1]
        else:
            return drive


def owned_processes_path(root: Any) -> pathlib.Path:
    return pathlib.Path(root) / "state" / OWNED_PROCESSES_FILENAME


def _pending_files(root: Any, *, strict: bool = False) -> Dict[pathlib.Path, Dict[str, Any]]:
    """The contended registrations not folded yet (normally none): file -> entry."""
    found: Dict[pathlib.Path, Dict[str, Any]] = {}
    try:
        paths = sorted((pathlib.Path(root) / "state" / PENDING_DIRNAME).glob("*.json"))
    except OSError:
        if strict:
            raise
        return found
    for path in paths:
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            if strict:
                raise
            continue  # a torn write never names a process; its registrant reported the failure
        if isinstance(entry, dict) and entry.get("record_id"):
            found[path] = entry
        elif strict:
            raise ValueError(f"invalid pending owned-process record: {path}")
    return found


def _read_document(root: Any, *, pending: Optional[Dict[pathlib.Path, Dict[str, Any]]] = None,
                   strict: bool = False) -> Dict[str, Any]:
    """The stored set plus the contended registrations not folded yet. Absent or unreadable
    reads as an empty set without the import mark, so the next start indexes the records on
    disk again (an unreadable file is logged)."""
    from ouroboros.utils import read_text_across_replace

    if pending is None:  # before the document: a concurrent fold-and-delete cannot hide an entry
        pending = _pending_files(root, strict=strict)
    path = owned_processes_path(root)
    try:
        document = json.loads(read_text_across_replace(path))
    except FileNotFoundError:
        document = None
    except (OSError, ValueError):
        log.critical("Owned-process set %s is unreadable; it names nothing until rewritten", path, exc_info=True)
        if strict:
            raise
        document = None
    if not isinstance(document, dict) or not isinstance(document.get("records"), dict):
        if strict and document is not None:
            raise ValueError(f"invalid owned-process set: {path}")
        document = {"schema_version": _SCHEMA_VERSION, "records": {}}
    for entry in pending.values():
        _put(document, entry)
    return document


def owned_records(drive_root: Any, *, strict: bool = False) -> List[Dict[str, Any]]:
    """Every record the set names; strict consumers distinguish unreadable from empty."""
    return [dict(entry) for entry in _read_document(installation_root(drive_root), strict=strict)["records"].values()]


def executor_record_paths(drive_root: Any, kind: str) -> List[pathlib.Path]:
    return [pathlib.Path(entry["record_path"]) for entry in owned_records(drive_root)
            if entry.get("kind") == kind and entry.get("record_path")]


def _update(root: Any, change: Callable[[Dict[str, Any]], bool], *, timeout_sec: float = 2.0) -> bool:
    """Apply ``change(document)`` under the canonical ledger's append lock; write when it changed."""
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
    from ouroboros.process_custody import ledger_path

    root = pathlib.Path(root)
    lock_path = jsonl_append_lock_path(ledger_path(root))
    lock_fd = acquire_exclusive_file_lock(lock_path, timeout_sec=timeout_sec, stale_sec=10.0, owner_aware_stale=True)
    if lock_fd is None:
        log.warning("Owned-process set of %s unchanged: the process-custody lock is unavailable", root)
        return False
    try:
        pending = _pending_files(root)
        document = _read_document(root, pending=pending)
        if change(document) or pending:
            document["schema_version"] = _SCHEMA_VERSION
            atomic_write_json(owned_processes_path(root), document, trailing_newline=True)
        for path in pending:  # folded into the document just written
            path.unlink(missing_ok=True)
        return True
    except Exception:
        log.warning("Owned-process set of %s could not be updated", root, exc_info=True)
        return False
    finally:
        release_exclusive_file_lock(lock_path, lock_fd)


def _born_stamped(entry: Dict[str, Any]) -> Dict[str, Any]:
    """A target recorded after this generation's shutdown began: the next start finishes it."""
    if _GENERATION_STOP.deadline is not None and _is_stop_target(entry):
        return {**entry, "stop_requested_at": entry.get("stop_requested_at") or utc_now_iso()}
    return entry


def _put(document: Dict[str, Any], entry: Dict[str, Any]) -> bool:
    records = document.setdefault("records", {})
    previous = records.get(entry["record_id"])
    entry = _born_stamped(entry)
    if isinstance(previous, dict) and previous.get("birth") == entry.get("birth"):
        # The same process recorded again keeps a stop already requested for it.
        entry = {**entry, "stop_requested_at": previous.get("stop_requested_at") or entry.get("stop_requested_at"),
                 "unconfirmed_since": previous.get("unconfirmed_since")}
    if previous == entry:
        return False
    records[entry["record_id"]] = entry
    return True


def _forget(root: Any, record_ids: Iterable[str], *, drive_root: str = "") -> bool:
    ids = set(record_ids)

    def change(document: Dict[str, Any]) -> bool:
        records = document.get("records", {})
        gone = [rid for rid in ids if rid in records
                and (not drive_root or records[rid].get("drive_root") == drive_root)]
        for rid in gone:
            records.pop(rid)
        return bool(gone)

    return _update(root, change) if ids else True


def _executor_entry(path: Any, record: Dict[str, Any]) -> Dict[str, Any]:
    path = pathlib.Path(path)
    entry = {
        "record_id": str(record.get("id") or path.stem),
        "kind": str(record.get("record_type") or ""),
        "executor_type": str(record.get("executor_type") or ""),
        "host_pid": _int(record.get("host_pid")),
        "birth": str(record.get("created_at") or ""),
        "task_id": str(record.get("task_id") or ""),
        "root_task_id": str(record.get("root_task_id") or ""),
        "drive_root": str(path.parents[2]),
        "record_path": str(path),
        "stop_requested_at": None,
        "unconfirmed_since": None,
    }
    # Docker records may carry host_pid=0: the backend identity names the process.
    entry.update({key: str(record[key]) for key in ("container_name", "backend_pid", "backend_pidfile")
                  if record.get(key)})
    return entry


def _write_pending(root: Any, entry: Dict[str, Any]) -> bool:
    """One lock-free pending file: every read merges it, the next locked update folds it."""
    import uuid

    try:
        pending_dir = pathlib.Path(root) / "state" / PENDING_DIRNAME
        pending_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(pending_dir / f"{entry['record_id']}.{uuid.uuid4().hex}.json", entry, trailing_newline=True)
        return True
    except Exception:
        log.warning("Owned process %s is not in the ownership set", entry["record_id"], exc_info=True)
        return False


def _publish(root: Any, entry: Dict[str, Any]) -> bool:
    """Add ``entry`` to the set; under contention, as a lock-free pending file instead."""
    return _update(root, lambda document: _put(document, entry)) or _write_pending(root, _born_stamped(entry))


def record_executor_process(path: Any, record: Dict[str, Any]) -> bool:
    """Add one executor/tracked-command record (``_register_process``); False when not indexed."""
    entry = _executor_entry(path, record)
    return _publish(installation_root(entry["drive_root"]), entry)


def forget_executor_process(path: Any) -> bool:
    """Remove the record ``_forget_process`` just deleted."""
    path = pathlib.Path(path)
    return _forget(installation_root(path.parents[2]), [path.stem])


def record_ledgered_process(drive_root: Any, entry: Dict[str, Any]) -> bool:
    """Add one ledger row ``record_process`` just appended (both spawn funnels)."""
    from ouroboros.process_custody import ledger_path

    pid = _int(entry.get("pid"))
    if pid <= 0:
        return False
    row = {key: value for key, value in entry.items() if key != "ts"}
    fingerprint = row.get("fingerprint") if isinstance(row.get("fingerprint"), dict) else {}
    purpose = str(row.get("purpose") or "")
    drive = pathlib.Path(drive_root).resolve(strict=False)
    record = {
        "record_id": f"pid-{pid}",
        "kind": "companion" if purpose.startswith("companion:") else "supervised",
        "host_pid": pid,
        "birth": str(fingerprint.get("start_time_boot") or fingerprint.get("start_time") or ""),
        "purpose": purpose,
        "scope": str(row.get("scope") or ""),
        "owner_task": str(row.get("owner_task") or ""),
        "drive_root": str(drive),
        "record_path": str(ledger_path(drive)),
        "ledger_entry": row,
        "stop_requested_at": None,
        "unconfirmed_since": None,
    }
    return _publish(installation_root(drive), record)


def forget_ledgered_pids(drive_root: Any, pids: Iterable[int]) -> bool:
    """Remove the rows a ledger compaction dropped from ``drive_root``'s ledger."""
    drive = pathlib.Path(drive_root).resolve(strict=False)
    return _forget(installation_root(drive), {f"pid-{pid}" for pid in pids if _int(pid) > 0},
                   drive_root=str(drive))


def _stamp(root: Any, field: str, select: Callable[[Dict[str, Any]], bool], *,
           timeout_sec: float = 2.0) -> List[Dict[str, Any]]:
    """Stamp ``field`` (first time only) on the selected records; return them as stored."""
    now, selected = utc_now_iso(), []

    def change(document: Dict[str, Any]) -> bool:
        selected.clear()
        changed = False
        for entry in document.get("records", {}).values():
            if not select(entry):
                continue
            if not entry.get(field):
                entry[field], changed = now, True
            selected.append(dict(entry))
        return changed

    if not _update(root, change, timeout_sec=timeout_sec):
        # Contended: the stop acts on the last readable set, and every stop request lands through
        # the pending protocol (a stamp seen on a pending registration may exist only in this
        # read), so a launcher kill leaves it recorded for the next start.
        selected = []
        for entry in _read_document(root)["records"].values():
            if select(entry):
                entry = {**entry, field: entry.get(field) or now}
                if field == "stop_requested_at":
                    _write_pending(root, entry)
                selected.append(dict(entry))
    return selected


def _is_stop_target(entry: Dict[str, Any]) -> bool:
    """Executor records and the ledgered processes a task started. Server machinery
    (workers, event bus, local model, audit), companions and installation daemons keep
    their lifecycle owners in the teardown."""
    if entry.get("kind") in _EXECUTOR_KINDS:
        return True
    return (entry.get("kind") == "supervised" and bool(entry.get("owner_task"))
            and entry.get("scope") in ("task", "session"))


def _exit_proven(pid: int) -> bool:
    from ouroboros.platform_layer import pid_provably_gone
    from ouroboros.process_containment import pid_is_zombie

    return pid <= 0 or pid_provably_gone(pid) or pid_is_zombie(pid)


def _stop_executor_record(root: Any, entry: Dict[str, Any], deadline: float) -> bool:
    from ouroboros import workspace_executor as executor

    path = pathlib.Path(entry["record_path"])
    record = executor._load_process_record(path)
    if record is None:
        _forget(root, [entry["record_id"]])  # the custody record is gone: nothing left to stop
        return True
    if not executor._valid_process_record(path, record, check_identity=False):
        return False  # never acted on; stays recorded
    if record.get("executor_type") == "local":
        pid = _int(record.get("host_pid"))
        if not _exit_proven(pid):
            if not executor._host_pid_matches_record(record):
                return False  # alive under an identity the record cannot prove: no right to signal
            executor._kill_host_pid(pid)
            while not _exit_proven(pid):
                if time.monotonic() >= deadline:
                    return False
                time.sleep(_POLL_SEC)
    elif not executor._stop_record_process(path, record, wait=True):
        return False  # Docker: the backend receipt is the proof
    executor._forget_process(path)
    return True


def _stop_ledgered(root: Any, entry: Dict[str, Any], deadline: float, retained: set) -> bool:
    from ouroboros.platform_layer import kill_pid_tree, kill_process_group_id, pid_is_alive
    from ouroboros.process_custody import _fingerprint_matches, _service_group_survives_leader

    row = entry.get("ledger_entry") if isinstance(entry.get("ledger_entry"), dict) else {}

    def alive() -> bool:
        return _fingerprint_matches(row) or _service_group_survives_leader(row)

    if alive():
        # Liveness may tolerate an unmeasurable start time (retention); a signal needs the
        # explicit stop's measured identity, or a recorded group that still has members.
        # Anything else stays recorded and is never signalled. Installation daemon
        # subtrees are spared, as by the reaper.
        pid, pgid = _int(row.get("pid")), _int(row.get("pgid"))
        measured = _fingerprint_matches(row, require_measured=True)
        group = pgid > 0 and (measured or _service_group_survives_leader(row))
        if not (measured or group):
            return False
        if group:
            kill_process_group_id(pgid, **({"exclude_pids": retained} if retained else {}))
        if measured and (pgid <= 0 or (retained and pid_is_alive(pid))):
            kill_pid_tree(pid, exclude_pids=retained or None)
        while alive():
            if time.monotonic() >= deadline:
                return False
            time.sleep(_POLL_SEC)
    _forget(root, [entry["record_id"]])
    return True


def _stop_entry(root: Any, entry: Dict[str, Any], deadline: float, retained: set,
                skip_paths: frozenset = frozenset()) -> bool:
    """One record's bounded stop: True once its exit is confirmed and it left the set."""
    if entry.get("kind") in _EXECUTOR_KINDS:
        if entry.get("record_path") in skip_paths:
            return False  # its in-memory owner holds the Popen/backend handle and settles it
        return _stop_executor_record(root, entry, deadline)
    return _stop_ledgered(root, entry, deadline, retained)


def _start(name: str, fn: Callable[[], Any]) -> threading.Thread:
    def run() -> None:
        try:
            fn()
        except Exception:
            log.warning("Owned-work step %s failed; its records stay", name, exc_info=True)

    thread = threading.Thread(target=run, name=name, daemon=True)
    thread.start()
    return thread


def _stop_records(root: Any, entries: List[Dict[str, Any]], deadline: float, *,
                  owners: Iterable[tuple] = (), skip_paths: frozenset = frozenset()) -> List[str]:
    """Run every owner and record stop in parallel until ``deadline``; return the ids still named."""
    retained: set = set()
    if any(entry.get("kind") not in _EXECUTOR_KINDS for entry in entries):
        from ouroboros.process_custody import live_daemon_root_pids

        retained = live_daemon_root_pids(root)
    threads = [_start(f"owned-stop-{name}", fn) for name, fn in owners]
    threads += [_start(f"owned-stop-{entry['record_id']}",
                       lambda entry=entry: _stop_entry(root, entry, deadline, retained, skip_paths))
                for entry in entries]
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    named = _read_document(root)["records"]
    remaining = sorted(entry["record_id"] for entry in entries if entry["record_id"] in named)
    if remaining:
        # After the deadline no fresh lock wait: one attempt. Every pending stop already carries
        # ``stop_requested_at``, which the next start retries whether or not this stamp lands.
        _stamp(root, "unconfirmed_since", lambda entry: entry.get("record_id") in remaining,
               timeout_sec=max(0.0, deadline - time.monotonic()))
    return remaining


def _stop_budget_sec() -> float:
    """The launcher's force-exit grace less Uvicorn's bounded drain (runtime_limits: one budget)."""
    from ouroboros.runtime_limits import LAUNCHER_STOP_GRACE_SEC, SERVER_GRACEFUL_SHUTDOWN_TIMEOUT_SEC

    return LAUNCHER_STOP_GRACE_SEC - SERVER_GRACEFUL_SHUTDOWN_TIMEOUT_SEC


def _memory_owners(root: pathlib.Path) -> List[tuple]:
    """This process's in-memory owners: Popen handles and service logs no record carries."""
    def commands() -> None:
        from ouroboros.tools.shell import kill_all_tracked_subprocesses

        kill_all_tracked_subprocesses()

    def services() -> None:
        from ouroboros.tools.services import kill_all_services

        kill_all_services(root, wait=True, durable=False)

    return [("commands", commands), ("services", services)]


def _run_stop(root: pathlib.Path, deadline: float) -> Dict[str, Any]:
    started = time.monotonic()
    targets = _stamp(root, "stop_requested_at", _is_stop_target,  # persisted before any wait
                     timeout_sec=max(0.0, min(2.0, deadline - started)))
    from ouroboros.workspace_executor import _services_snapshot

    held = frozenset(str(record.durable_record_path) for record in _services_snapshot()
                     if record.durable_record_path is not None)
    remaining = _stop_records(root, targets, deadline, owners=_memory_owners(root), skip_paths=held)
    if remaining:
        log.critical("Owned-work stop left %d record(s) unconfirmed at the deadline; the next start "
                     "retries them", len(remaining))
        try:
            append_jsonl(root / "logs" / "supervisor.jsonl", {
                "ts": utc_now_iso(), "type": "owned_stop_unconfirmed", "records": remaining})
        except Exception:
            log.debug("Failed to record the unconfirmed owned stop", exc_info=True)
    return {"state": "unconfirmed" if remaining else "completed", "targets": len(targets),
            "confirmed": len(targets) - len(remaining), "unconfirmed": remaining,
            "elapsed_sec": round(time.monotonic() - started, 3)}


def begin_owned_stop(drive_root: Any = None) -> None:
    """A shutdown door was entered: start the generation's one grace deadline (if no door did
    yet) and persist ``stop_requested_at`` on every target before any other owner waits."""
    stop = _GENERATION_STOP
    with stop.lock:
        if stop.deadline is None:
            stop.deadline = time.monotonic() + _stop_budget_sec()
        deadline = stop.deadline
    try:
        if drive_root is None:
            from ouroboros.config import resolve_data_dir

            drive_root = resolve_data_dir()
        _stamp(installation_root(drive_root), "stop_requested_at", _is_stop_target,
               timeout_sec=max(0.0, min(2.0, deadline - time.monotonic())))
    except Exception:
        log.warning("Owned-work stop requests not stamped at shutdown entry; the stop stamps them",
                    exc_info=True)


def stop_owned_work(drive_root: Any = None) -> Dict[str, Any]:
    """Start/join the one stop; completed evidence includes any subsequently registered targets."""
    if drive_root is None:
        from ouroboros.config import resolve_data_dir

        drive_root = resolve_data_dir()
    root = installation_root(drive_root)
    stop = _GENERATION_STOP
    with stop.lock:
        starting = not stop.started
        if starting:
            stop.started = True
            if stop.deadline is None:
                stop.deadline = time.monotonic() + _stop_budget_sec()
        deadline = stop.deadline
    if not starting:
        stop.done.wait(max(0.0, deadline - time.monotonic()))
        return _current_stop_outcome(root, stop.outcome or {"state": "in_progress", "joined": True})
    outcome: Dict[str, Any] = {"state": "failed"}
    try:
        outcome = _run_stop(root, deadline)
    except Exception:
        log.critical("Owned-work stop failed; the set keeps its records for the next start", exc_info=True)
    finally:
        stop.outcome = outcome
        stop.done.set()
    return _current_stop_outcome(root, outcome)


def _current_stop_outcome(root: Any, outcome: Dict[str, Any]) -> Dict[str, Any]:
    """Late targets keep their next-start recovery; no second stop or wait is added."""
    result = dict(outcome)
    if result.get("state") == "completed":
        try:
            remaining = [entry["record_id"] for entry in owned_records(root, strict=True) if _is_stop_target(entry)]
        except Exception:
            remaining = ["ownership_set_unreadable"]
        if remaining:
            result.update(state="unconfirmed", unconfirmed=remaining)
    return result


def import_inherited_records(drive_root: Any) -> Optional[int]:
    """Index executor records left on disk before the set existed, once.

    Runs while the set lacks its import mark (a missing or unreadable file), from the
    boot's background thread only, never on exit: one walk of ``data/state`` and
    ``data/task_drives`` that matches named directory symlinks without descending through them."""
    from ouroboros import workspace_executor as executor

    root = installation_root(drive_root)
    if _read_document(root).get("inherited_import"):
        return None
    entries: Dict[str, Dict[str, Any]] = {}
    for base in (root / "state", root / "task_drives"):
        for parent, dirs, _files in os.walk(base):
            if executor._PROCESS_STATE_DIR not in dirs:
                continue
            for path in sorted((pathlib.Path(parent) / executor._PROCESS_STATE_DIR).glob("*.json")):
                record = executor._load_process_record(path)
                if record is not None and executor._valid_process_record(path, record, check_identity=False):
                    entry = _executor_entry(path, record)
                    entries[entry["record_id"]] = entry

    def change(document: Dict[str, Any]) -> bool:
        records = document.setdefault("records", {})
        for record_id, entry in entries.items():  # landing after a stop began, a target is born stamped
            records.setdefault(record_id, _born_stamped(entry))
        document["inherited_import"] = {"at": utc_now_iso(), "records": len(entries)}
        return True

    return len(entries) if _update(root, change) else None


def finish_unconfirmed_stops(drive_root: Any) -> Dict[str, Any]:
    """The next start's bounded retry of every stop a previous generation requested.

    Retries each record stamped ``stop_requested_at`` or ``unconfirmed_since`` under the
    stop's own deadline, forgets confirmed exits, keeps the rest stamped and writes one
    ``owned_stops_finished`` supervisor row when there was work. It never walks the disk."""
    counts: Dict[str, Any] = {"retried": 0, "confirmed": 0, "unconfirmed": 0}
    try:
        root = installation_root(drive_root)
        pending = [dict(entry) for entry in _read_document(root)["records"].values()
                   if entry.get("stop_requested_at") or entry.get("unconfirmed_since")]
        if pending:
            remaining = _stop_records(root, pending, time.monotonic() + _stop_budget_sec())
            counts.update(retried=len(pending), confirmed=len(pending) - len(remaining),
                          unconfirmed=len(remaining))
            append_jsonl(root / "logs" / "supervisor.jsonl", {
                "ts": utc_now_iso(), "type": "owned_stops_finished", **counts})
    except Exception:
        log.warning("Finishing unconfirmed owned stops failed; their records stay for the next start",
                    exc_info=True)
    return counts


def start_inherited_import(drive_root: Any) -> Optional[threading.Thread]:
    """Index inherited records on a background thread; None when the set already carries the mark.

    The walk can take minutes on a cold disk and no stop at this start waits on an inherited
    record, so the ready path never waits on it; one that lands after a stop began is stored
    stamped. An exit before it lands leaves the mark unset: those processes stay running until
    the next start indexes them and a later exit stops them."""
    root = installation_root(drive_root)
    if _read_document(root).get("inherited_import"):
        return None
    began = time.monotonic()

    def run() -> None:
        records = import_inherited_records(root)
        if records is not None:
            append_jsonl(root / "logs" / "supervisor.jsonl", {
                "ts": utc_now_iso(), "type": "owned_records_imported", "records": records,
                "seconds": round(time.monotonic() - began, 3)})

    return _start("owned-records-import", run)
