"""Event-fed vendor program observation. Context reads one local snapshot only.

An observation is replaceable evidence, not an operation ledger or an updater.
Identical semantic facts keep their original date, including across host restarts.
"""
from __future__ import annotations

import json
import logging
import pathlib
import threading
from datetime import datetime

log = logging.getLogger(__name__)
OBSERVATION_FILE = "state/harness_programs.json"
_lock = threading.Lock()
_events_lock = threading.Lock()
_ready = {}
_settled = {}


def _root(root=None):
    if root is None:
        from ouroboros.config import DATA_DIR
        root = DATA_DIR
    return pathlib.Path(root)


def _read(root):
    try:
        value = json.loads((root / OBSERVATION_FILE).read_text(encoding="utf-8"))
        if (isinstance(value, dict) and isinstance(value.get("programs"), dict)
                and all(isinstance(row, dict) for row in value["programs"].values())):
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


def record_inventory(inventory, *, complete=True, root=None):
    """Merge current evidence; no previous version, receipt, command or credential is retained."""
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
    from ouroboros.utils import assert_test_data_path, atomic_write_json

    root = _root(root)
    path = root / OBSERVATION_FILE
    assert_test_data_path(path)
    if not isinstance(inventory, dict) or not isinstance(inventory.get("harnesses"), list):
        return
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(".lock")
        fd = acquire_exclusive_file_lock(lock_path, timeout_sec=2, owner_aware_stale=True)
        if fd is None:
            return
        try:
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
            atomic_write_json(path, saved)
        finally:
            release_exclusive_file_lock(lock_path, fd)


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


def _refresh(endpoint, root):
    """A passive event acquisition, in its own thread so Connect's ceiling stays bounded."""
    from ouroboros.gateways.claudexor import ClaudexorGateway

    try:
        with ClaudexorGateway(endpoint) as gateway:
            gateway.handshake()
            inventory = gateway.maintenance_harnesses(fresh=True, check_latest=False)
        record_inventory(inventory, root=root)
    except Exception:
        log.debug("Vendor program event observation unavailable", exc_info=True)


def _launch(endpoint, root):
    try:
        threading.Thread(target=_refresh, args=(endpoint, root), name="vendor-program-observation", daemon=True).start()
    except RuntimeError:
        log.debug("Vendor program observation could not start", exc_info=True)


def engine_ready(gateway):
    """Once per descriptor generation in this host process, never once per status poll."""
    from ouroboros.gateways.claudexor_metadata import endpoint_generation

    endpoint = getattr(gateway, "_endpoint", None)
    generation = endpoint_generation(endpoint)
    if generation is None:
        return
    root = _root()
    with _events_lock:
        if _ready.get(root) == generation:
            return
        _ready[root] = generation
    _launch(endpoint, root)


def operation_observed(gateway, operation):
    """A newly settled receipt, including failed/cancelled work; unresolved custody waits."""
    if operation.get("state") not in {"succeeded", "failed", "cancelled", "interrupted"} or operation.get("termination") == "unconfirmed":
        return operation
    endpoint = getattr(gateway, "_endpoint", None)
    if endpoint is None:
        return operation
    root = _root()
    key, stamp = (root, operation.get("harness")), (operation.get("id"), operation.get("finishedAt"))
    with _events_lock:
        if _settled.get(key) == stamp:
            return operation
        _settled[key] = stamp
    _launch(endpoint, root)
    return operation
