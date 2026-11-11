"""Event-fed vendor program observation. Context reads one local snapshot only.

An observation is replaceable evidence, not an operation ledger or an updater.
Identical semantic facts keep their original date, including across host restarts.
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


def _facts(programs):
    return {name: {key: row.get(key) for key in ("path", "version", "updateAvailable", "newestVersion")}
            for name, row in programs.items()}


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


def _record_inventory(inventory, complete, root, event=None):
    """Caller owns the observation lock across acquisition, merge and event publication."""
    from ouroboros.utils import atomic_write_json

    if not isinstance(inventory, dict) or not isinstance(inventory.get("harnesses"), list):
        return
    saved = _read(root)
    programs = dict(saved.get("programs", {}))
    observed = inventory.get("observedAt") or ""
    stamp = _stamp(observed)
    complete_at = _stamp(saved.get("completeAt"))
    full = complete and stamp >= complete_at
    incoming = {row["harness"]: row for row in inventory["harnesses"] if isinstance(row, dict) and row.get("harness")}
    if full:
        programs = {name: row for name, row in programs.items()
                    if name in incoming or max(_stamp(row.get("observedAt")), _stamp(row.get("readAt"))) > stamp}
        saved["completeAt"] = observed
    for name, row in incoming.items():
        seen = row.get("observedAt") or observed
        previous = programs.get(name, {})
        if (_stamp(seen) < _stamp(previous.get("observedAt"))
                or stamp < _stamp(previous.get("readAt"))
                or not full and name not in programs and stamp < complete_at):
            continue
        selected, newest = row.get("selection") or {}, row.get("newest") or {}
        programs[name] = {"path": selected.get("path"), "version": selected.get("version"),
            "updateAvailable": row.get("updateAvailable", "cannot_tell"),
            "newestVersion": newest.get("version") if newest.get("status") == "available" else None,
            "newestObservedAt": newest.get("observedAt"), "observedAt": seen, "readAt": observed}
    if _facts(programs) != _facts(saved.get("programs", {})) or "factsSince" not in saved:
        saved["factsSince"] = observed[:10] or "date unknown"
    saved["programs"] = programs
    if stamp >= _stamp(saved.get("observedAt")):
        saved["observedAt"] = observed
    events = saved.get("events", {})
    if event:
        events[event[0]] = event[1]
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
    if not saved:
        return "Vendor programs: not observed."
    parts = []
    for name, row in sorted(saved["programs"].items()):
        installed = (row.get("version") or "version unknown") if row.get("path") else "not installed"
        newest = f"; newest {row['newestVersion']}" if row.get("newestVersion") else ""
        parts.append(f"{name} {installed} ({row.get('updateAvailable', 'cannot_tell')}{newest})")
    return f"Vendor programs (facts unchanged since {saved.get('factsSince', 'date unknown')}): " + ("; ".join(parts) or "none listed") + "."


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _generation(endpoint):
    from ouroboros.gateways.claudexor_metadata import endpoint_generation

    generation = endpoint_generation(endpoint)
    return _fingerprint(generation) if generation is not None else None


def _refresh(endpoint, root, event):
    """The shared lock deduplicates acquisition too, not just publication."""
    from ouroboros.gateways.claudexor import ClaudexorGateway

    try:
        with _observation_lock(root) as acquired:
            if not acquired or _read(root).get("events", {}).get(event[0]) == event[1]:
                return
            if event[0] == "engineGeneration" and _generation(endpoint) != event[1]:
                return
            with ClaudexorGateway(endpoint) as gateway:
                gateway.handshake()
                inventory = gateway.maintenance_harnesses(fresh=True, check_latest=False)
            _record_inventory(inventory, True, root, event)
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
