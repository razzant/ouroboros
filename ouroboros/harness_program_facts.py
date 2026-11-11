"""Event-fed vendor program observation. Context reads one local snapshot only.

An observation is replaceable evidence, not an operation ledger or an updater.
Dates describe the observations, including newest reads retained by the engine.
"""
from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import threading
from contextlib import contextmanager
from datetime import datetime

log = logging.getLogger(__name__)
OBSERVATION_FILE = "state/harness_programs.json"
_events_lock = threading.Lock()
_pending = set()


def _root(root=None):
    if root is None:
        from ouroboros.config import DATA_DIR
        root = DATA_DIR
    return pathlib.Path(root)


def _read(root):
    try:
        value = json.loads((root / OBSERVATION_FILE).read_text(encoding="utf-8"))
        if (isinstance(value, dict) and isinstance(value.get("programs"), dict)
                and all(isinstance(row, dict) for row in value["programs"].values())
                and isinstance(value.get("events", {}), dict)):
            return value
        return {}
    except (OSError, ValueError):
        return {}


def _stamp(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError, OverflowError):
        return 0


@contextmanager
def _observation_lock(root):
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
    from ouroboros.utils import assert_test_data_path

    path = root / OBSERVATION_FILE
    assert_test_data_path(path)
    lock_path = path.with_suffix(".lock")
    fd = acquire_exclusive_file_lock(lock_path, timeout_sec=2, owner_aware_stale=True)
    try:
        yield fd is not None
    finally:
        if fd is not None:
            release_exclusive_file_lock(lock_path, fd)


def _record_inventory(inventory, complete, root):
    """Caller owns the lock for local file work only."""
    from ouroboros.utils import atomic_write_json

    if not isinstance(inventory, dict) or not isinstance(inventory.get("harnesses"), list):
        return
    saved = _read(root)
    observed = inventory.get("observedAt") or ""
    boundary = saved.get("completeObservedAt")
    newest_complete = complete and _stamp(observed) >= _stamp(boundary)
    programs = dict(saved.get("programs", {}))
    for row in inventory["harnesses"]:
        at = row.get("observedAt") or observed
        previous = programs.get(row["harness"])
        prior = previous.get("observedAt") if previous else None if newest_complete else boundary
        if _stamp(prior) > _stamp(at):
            continue
        selected, newest = row.get("selection") or {}, row.get("newest") or {}
        programs[row["harness"]] = {"path": selected.get("path"), "version": selected.get("version"),
            "updateAvailable": row.get("updateAvailable", "cannot_tell"),
            "newestStatus": newest.get("status", "not_read"),
            "newestVersion": newest.get("version") if newest.get("status") == "available" else None,
            "newestObservedAt": newest.get("observedAt"), "observedAt": at}
    if newest_complete:
        named = {row["harness"] for row in inventory["harnesses"]}
        programs = {name: row for name, row in programs.items()
                    if name in named or _stamp(row.get("observedAt")) >= _stamp(observed)}
        boundary = observed
    saved = {"programs": programs, "completeObservedAt": boundary, "events": saved.get("events", {})}
    events = saved.get("events", {})
    if events:
        saved["events"] = {key: value for key, value in events.items()
                           if key == "engineGeneration" or key.removeprefix("settled:") in programs}
    atomic_write_json(root / OBSERVATION_FILE, saved)


def record_inventory(inventory, *, complete=True, root=None):
    """Merge current evidence; no previous version, receipt, command or credential is retained."""
    root = _root(root)
    with _observation_lock(root) as acquired:
        if acquired:
            _record_inventory(inventory, complete, root)


def observe_inventory(inventory, *, complete=True):
    """Observation failure never changes an engine response or an owner's operation."""
    try:
        record_inventory(inventory, complete=complete)
    except Exception:
        log.debug("Vendor program observation could not be recorded", exc_info=True)
    return inventory


def vendor_programs_line(root=None):
    """Local file read only: no gateway imports, subprocess, network, clock or wake."""
    saved = _read(_root(root))
    if not saved.get("programs") and not saved.get("completeObservedAt"):
        return "Harness programs: not observed."
    date = lambda value: str(value)[:10] if _stamp(value) else "date unknown"
    programs = saved["programs"]
    observed_dates = {date(row.get("observedAt")) for row in programs.values()}
    newest_dates = {date(row.get("newestObservedAt")) for row in programs.values()
                    if row.get("path") and (row.get("newestVersion") or row.get("updateAvailable") == "no")}
    shared_observed = len(observed_dates) <= 1
    shared_newest = len(newest_dates) == 1
    age = "observed " + next(iter(observed_dates), date(saved.get("completeObservedAt"))) if shared_observed else "observations dated below"
    if shared_newest:
        age += "; newest read " + next(iter(newest_dates))
    parts, absent = [], []
    for name, row in sorted(programs.items()):
        seen = "" if shared_observed else ", observed " + date(row.get("observedAt"))
        if not row.get("path"):
            absent.append(name if shared_observed else name + " observed " + date(row.get("observedAt")))
            continue
        installed = row.get("version") or "version unknown"
        newest, verdict = row.get("newestVersion"), row.get("updateAvailable")
        fact = (f"{newest} available" if verdict == "yes" and newest else
                "up to date" if verdict == "no" else
                f"newest {newest}, order unknown" if newest else "newest not read")
        if row.get("newestStatus") == "unavailable":
            fact = "newest unavailable " + date(row.get("newestObservedAt"))
        if not shared_newest and (newest or verdict == "no"):
            fact += ", newest read " + date(row.get("newestObservedAt"))
        parts.append(f"{name} {installed} ({fact}{seen})")
    if absent:
        parts.append("not installed: " + ", ".join(absent))
    if not saved.get("completeObservedAt"):
        parts.append("other programs not observed")
    return f"Harness programs ({age}): " + ("; ".join(parts) or "none listed") + "."


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _generation(endpoint):
    from ouroboros.gateways.claudexor_metadata import endpoint_generation

    generation = endpoint_generation(endpoint)
    return _fingerprint(generation) if generation is not None else None


def _refresh(endpoint, root, event):
    """Spend an event before I/O; failed reads wait for the next real event."""
    from ouroboros.gateways.claudexor import ClaudexorGateway
    from ouroboros.utils import atomic_write_json

    try:
        with _observation_lock(root) as acquired:
            saved = _read(root)
            if not acquired or saved.get("events", {}).get(event[0]) == event[1]:
                return
            if event[0] == "engineGeneration" and _generation(endpoint) != event[1]:
                return
            saved.setdefault("programs", {})
            saved.setdefault("events", {})[event[0]] = event[1]
            atomic_write_json(root / OBSERVATION_FILE, saved)
        with ClaudexorGateway(endpoint) as gateway:
            gateway.handshake()
            inventory = gateway.maintenance_harnesses(fresh=True, check_latest=False)
        record_inventory(inventory, root=root)
    except Exception:
        log.debug("Vendor program event observation unavailable", exc_info=True)
    finally:
        with _events_lock:
            _pending.discard((root, event))


def _launch(endpoint, root, event):
    try:
        threading.Thread(target=_refresh, args=(endpoint, root, event), name="vendor-program-observation", daemon=True).start()
    except RuntimeError:
        with _events_lock:
            _pending.discard((root, event))
        log.debug("Vendor program observation could not start", exc_info=True)


def _request(endpoint, event):
    root = _root()
    if _read(root).get("events", {}).get(event[0]) == event[1]:
        return
    with _events_lock:
        if (root, event) in _pending:
            return
        _pending.add((root, event))
    _launch(endpoint, root, event)


def engine_ready(gateway):
    """Once per engine generation across the install; a current observation does nothing."""
    from ouroboros.gateways.claudexor_maintenance import supports_program_maintenance

    if not supports_program_maintenance(gateway):
        return
    endpoint = getattr(gateway, "_endpoint", None)
    generation = _generation(endpoint)
    if generation is not None:
        _request(endpoint, ("engineGeneration", generation))


def operation_observed(gateway, operation):
    """A newly settled receipt, including failed/cancelled work; unresolved custody waits."""
    if operation.get("state") not in {"succeeded", "failed", "cancelled", "interrupted"} or operation.get("termination") == "unconfirmed":
        return operation
    endpoint = getattr(gateway, "_endpoint", None)
    if endpoint is not None:
        _request(endpoint, (f"settled:{operation.get('harness')}", _fingerprint((operation.get("id"), operation.get("finishedAt")))))
    return operation
