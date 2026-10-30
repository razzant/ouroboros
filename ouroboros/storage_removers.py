"""Who retires or shrinks what each counted place holds, and how its last pass ended.

Dashboard -> Storage shows one line per place from ``place_removers`` and ``query_data(kind="storage")``
returns the same facts; both read them on every request, with or without a count. A place names its
removers (``PLACE_REMOVERS``): the maintenance child's job rows (``maintenance_jobs``: the row's state,
``last_error`` while its last run threw, ``record_compression``'s totals for the plane in that place) and,
for the engine place, the engine's last retention pass (the read-only ``GET /v2/maintenance/gc``, which
never starts one; answered within a wall bound, connect and read together, the read itself ended by a
longer one, a projection polled every second reusing a record read in the last few seconds). An engine
that cannot be asked, or not within the bound, is ``unavailable`` and one that keeps no such record
``unsupported``: unknown, never zero. The runtime ``storage`` fact reads the job rows alone
(``engine=False``: no engine call while a turn is prepared); ``not_ok`` names what it lists.
"""
from __future__ import annotations

import pathlib
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

try:  # Python 3.11+
    from typing import NotRequired, TypedDict  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - CI supports Python 3.10.
    from typing_extensions import NotRequired, TypedDict  # type: ignore[assignment]

from ouroboros import maintenance_jobs, store

# place id -> its removers: the maintenance child's jobs that retire or shrink its bytes, and ``engine``.
PLACE_REMOVERS: Dict[str, tuple] = {
    "model_calls": ("evidence_packing",),
    "engine": ("engine",),
    "task_results": ("evidence_packing", "record_compression", "runtime_retirement"),
    "archives": ("evidence_packing",),
    "work_folders": ("custody_trash",),
    "app_state": ("runtime_retirement", "record_compression", "custody_trash"),
    "data_other": ("runtime_retirement",),
}
# record_compression's plane in a place: task evidence in Task results, reviewer seat homes in App state.
_COMPRESSION_PLANES = {"task_results": "evidence", "app_state": "seats"}
ENGINE_READ_SEC = 2.0  # the answer: connect and read together, in wall time (a later record fills the reuse slot)
ENGINE_REUSE_SEC = 5.0  # a projection polled every second reuses the record read this recently; no read outlives it
_ENGINE_OUTCOMES = ("completed", "held", "truncated", "failed")
_recent: Dict[str, Tuple[float, Optional["StorageRemover"]]] = {}  # data root -> (monotonic time, engine fact)
_recent_lock = threading.Lock()


class StorageRemover(TypedDict):
    """One remover of a place: a maintenance job's row or the engine's last retention pass."""

    id: str  # engine | runtime_retirement | custody_trash | evidence_packing | record_compression
    state: str  # a job: pending | running | done | failed | never | unsupported | unavailable; the engine: completed | held | truncated | failed | none | unavailable | unsupported
    at: Optional[str]  # the job row's last write; the engine pass's end
    last_error: NotRequired[Dict[str, str]]  # a job whose last run threw: {class, at}, until a run returns normally
    compressed: NotRequired[int]  # record_compression: files compressed so far in this place's plane
    saved_bytes: NotRequired[Optional[int]]  # record_compression: that plane's allocation saved; null where unreported
    freed_bytes: NotRequired[int]  # the engine pass
    held: NotRequired[int]  # the engine pass: folders it held (project roots and native homes)
    held_since: NotRequired[str]  # the earliest of those holds
    errors: NotRequired[int]  # the engine pass's errors
    first_error: NotRequired[Optional[str]]


def place_removers(data_root: Any, *, engine: bool = True, recent: bool = False) -> Dict[str, List[StorageRemover]]:
    """Each place's removers as read now (module docstring); a place nothing removes is absent. ``engine``
    False reads the local job rows alone; ``recent`` reuses an engine record read in ``ENGINE_REUSE_SEC``."""
    root = pathlib.Path(data_root)
    names = sorted({name for names in PLACE_REMOVERS.values() for name in names if name != "engine"})
    rows: Optional[Dict[str, Dict[str, Any]]] = {}  # no store file yet: no job has recorded a fact (and a
    if store.store_path(root).is_file():            # passive read never lists a folder to learn that)
        try:
            rows = maintenance_jobs.jobs(root, names)
        except Exception:  # an unreadable store: each job's outcome is unknown, never "not run"
            rows = None
    engine_fact = _engine(root, recent=recent) if engine else None
    found = {place: [engine_fact if name == "engine" else _job(name, rows, place) for name in removers]
             for place, removers in PLACE_REMOVERS.items()}
    return {place: [fact for fact in removers if fact is not None]  # type: ignore[misc]
            for place, removers in found.items() if any(fact is not None for fact in removers)}


def _job(name: str, rows: Optional[Dict[str, Dict[str, Any]]], place: str) -> StorageRemover:
    if rows is None:
        return {"id": name, "state": "unavailable", "at": None}
    row = rows.get(name)
    if not row:
        return {"id": name, "state": "never", "at": None}
    fact: StorageRemover = {"id": name, "state": "unsupported" if row.get("unsupported") else str(row.get("state") or "never"),
                            "at": row.get("updated_at")}
    error = row.get("last_error")
    if isinstance(error, dict):
        fact["last_error"] = {"class": str(error.get("class") or ""), "at": str(error.get("at") or "")}
    totals = row.get("totals") if name == "record_compression" and isinstance(row.get("totals"), dict) else {}
    total = totals.get(_COMPRESSION_PLANES.get(place, "")) or {}
    if "compressed" in total:  # this place's own plane, never a job-wide sum under two places
        fact["compressed"], fact["saved_bytes"] = int(total.get("compressed") or 0), total.get("saved_bytes")
    return fact


def _engine(root: pathlib.Path, *, recent: bool = False) -> Optional[StorageRemover]:
    """The engine's last retention pass; None where no engine was ever provisioned under ROOT. ``recent``
    reuses the record read in the last ``ENGINE_REUSE_SEC``; every answer fills that reuse slot."""
    key = str(root)
    if recent:
        with _recent_lock:
            seen = _recent.get(key)
        if seen is not None and time.monotonic() - seen[0] < ENGINE_REUSE_SEC:
            return seen[1]
    return _engine_read(root, key)


def _remember(key: str, fact: Optional[StorageRemover]) -> Optional[StorageRemover]:
    with _recent_lock:
        _recent[key] = (time.monotonic(), fact)
    return fact


def _engine_read(root: pathlib.Path, key: str) -> Optional[StorageRemover]:
    """The record answered within ``ENGINE_READ_SEC`` of wall time, connect and read together: httpx bounds
    each phase alone, so the read runs on a daemon thread (the gateway imported and built first: the bound
    is the engine's, never a cold import's). A read still running may finish until ``ENGINE_REUSE_SEC``;
    its record then replaces the ``unavailable`` answered meanwhile, so the next poll shows it. At that
    bound the waiting side closes the gateway's client: the thread ends even against an engine that
    trickles bytes."""
    from ouroboros.gateways import claudexor
    from ouroboros.storage import ENGINE_DIR

    home = root / ENGINE_DIR
    if not (home / "daemon" / "control-api.json").is_file():
        return _remember(key, None)
    try:
        gateway = claudexor.ClaudexorGateway(claudexor.discover_daemon_at(home))
    except Exception as exc:
        return _remember(key, _engine_failed(claudexor, exc))
    outcome: Dict[str, Any] = {}
    done = threading.Event()
    closer = threading.Timer(ENGINE_REUSE_SEC, gateway.close)  # the read's own bound, from the waiting side
    closer.name, closer.daemon = "storage-engine-read-bound", True

    def read() -> None:
        try:
            fact = _engine_ask(claudexor, gateway)
        finally:
            closer.cancel()
        with _recent_lock:
            outcome["fact"] = fact
            if outcome.get("late") and fact["state"] != "unavailable":
                _recent[key] = (time.monotonic(), fact)  # answered after the bound: the next poll shows it
        done.set()

    closer.start()
    threading.Thread(target=read, name="storage-engine-read", daemon=True).start()
    done.wait(ENGINE_READ_SEC)
    with _recent_lock:
        fact = outcome.get("fact") or {"id": "engine", "state": "unavailable", "at": None}
        outcome["late"] = "fact" not in outcome
        _recent[key] = (time.monotonic(), fact)
    return fact


def _engine_ask(claudexor: Any, gateway: Any) -> StorageRemover:
    try:
        with gateway:
            last = gateway.maintenance_gc_last(timeout_sec=ENGINE_REUSE_SEC)["last"]
    except Exception as exc:
        return _engine_failed(claudexor, exc)
    return {"id": "engine", "state": "none", "at": None} if last is None else _engine_pass(last)


def _engine_failed(claudexor: Any, exc: Exception) -> StorageRemover:
    """A read that got no record: ``unsupported`` from an engine without the route, else ``unavailable``."""
    route = isinstance(exc, claudexor.ClaudexorUnavailable) and exc.code == "maintenance_unavailable"
    return {"id": "engine", "state": "unsupported" if route else "unavailable", "at": None}


def _engine_pass(last: Dict[str, Any]) -> StorageRemover:
    """The engine's record (its ``GcLastPass``) as one remover; a record of another shape is unavailable."""
    try:
        held, errors = last["held"], last["errors"]
        fact: StorageRemover = {
            "id": "engine", "state": str(last["outcome"]), "at": str(last["finished_at"]),
            "freed_bytes": int(last["freed_bytes"]), "held": int(held.get("roots") or 0) + int(held.get("native_homes") or 0),
            "errors": int(errors["count"]), "first_error": errors.get("first")}
        since = sorted(str(hold["since"]) for hold in last.get("holds") or () if hold.get("since"))
    except (KeyError, TypeError, ValueError, AttributeError):
        return {"id": "engine", "state": "unavailable", "at": None}
    if fact["state"] not in _ENGINE_OUTCOMES:
        return {"id": "engine", "state": "unavailable", "at": None}
    if since:
        fact["held_since"] = since[0]
    return fact


def not_ok(found: Dict[str, List[StorageRemover]]) -> List[str]:
    """One phrase per remover whose last outcome is not clean, each once though it serves several places:
    a job that failed, whose last run threw or whose facts are unreadable; an engine pass that held,
    stopped early, failed or reported errors, or an engine record that is absent or cannot be read."""
    phrases, seen = [], set()
    for removers in found.values():
        for fact in removers:
            if fact["id"] not in seen:
                seen.add(fact["id"])
                phrase = _not_ok_phrase(fact)
                if phrase:
                    phrases.append(phrase)
    return phrases


def _not_ok_phrase(fact: StorageRemover) -> str:
    state = fact["state"]
    if fact["id"] != "engine":
        error = fact.get("last_error")
        if error:
            return f"{fact['id']}: last run threw {error['class']} at {error['at']}"
        return f"{fact['id']}: {state}" if state in ("failed", "unavailable") else ""
    if state in ("none", "unavailable", "unsupported"):
        return f"engine retention: {state}"
    if state == "completed" and not fact.get("errors"):
        return ""
    detail = [f"last pass {state} at {fact['at']}"]
    if fact.get("held"):
        detail.append(f"holding {fact['held']} folders since {fact.get('held_since') or 'unknown'}")
    if fact.get("errors"):
        detail.append(f"{fact['errors']} errors, first: {fact.get('first_error')}")
    return "engine retention: " + "; ".join(detail)
