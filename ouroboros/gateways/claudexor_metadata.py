"""Immutable protocol metadata for each engine's currently selected descriptor generation.

Liveness/admission probes remain fresh. Only model-call preparation reuses a
successful normal handshake; the route catalog is immutable within a generation.
Each descriptor path (the owned engine's, a managed deployment's work engine's)
keeps only its current generation, so one engine's handshake never evicts the
other's. No client, token or historical generation is retained here.
"""
from __future__ import annotations

import copy
import hashlib
import pathlib
import threading
from concurrent.futures import Future, TimeoutError

STORE_IDENTITY = {"X-Claudexor-Store-Identity": "1"}  # opt-in ``storeId``: handshake, run start, thread turn
_lock = threading.Lock()
_generations = {}  # descriptor path -> [generation key, handshake body, catalog future]


def descriptor_stamp(path):
    try:
        stat = pathlib.Path(path).stat()
        return stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size
    except OSError:
        return None


def endpoint_generation(endpoint):
    path = getattr(endpoint, "descriptor_path", "")
    stamp = getattr(endpoint, "descriptor_identity", None)
    if not path or not stamp or descriptor_stamp(path) != stamp:
        return None
    return path, stamp, hashlib.sha256(endpoint.token.encode()).hexdigest()


def _select(key):
    """This descriptor's current generation, replacing an older one of the same path (lock held)."""
    entry = _generations.get(key[0])
    if entry is None or entry[0] != key:
        entry = _generations[key[0]] = [key, None, None]
    return entry


def _current(key):
    entry = _generations.get(key[0]) if key is not None else None
    return entry if entry is not None and entry[0] == key else None


def remember_handshake(gateway, body):
    gateway._store_identity = body.get("storeId") if isinstance(body.get("storeId"), str) else ""
    key = endpoint_generation(gateway._endpoint)
    if key is not None:
        with _lock:
            _select(key)[1] = copy.deepcopy(body) if body.get("servingMode") != "recovery_only" else None


def remembered_engine(gateway):
    """``(version, build sha)`` a remembered handshake proved for THIS gateway's descriptor
    generation (one daemon process, one engine build), or ``None``: no new request is made."""
    key = endpoint_generation(getattr(gateway, "_endpoint", None))
    with _lock:
        entry = _current(key)
        engine = (entry[1] or {}).get("engine") if entry is not None else None
    return (str(engine.get("version") or ""), str(engine.get("sha") or "")) if isinstance(engine, dict) else None


def model_gateway(factory):
    """Use a proven generation, or let the existing owner ensure/probe it."""
    from ouroboros.claudexor_daemon import attach_home, attached_endpoint, get_owned_daemon, owned_config_dir
    from ouroboros.gateways.claudexor import ClaudexorGateway, ClaudexorUnavailable, discover_daemon_at

    try:
        home = attach_home()
        endpoint = attached_endpoint(home) if home is not None else discover_daemon_at(owned_config_dir())
    except ClaudexorUnavailable:
        return factory()
    key = endpoint_generation(endpoint)
    with _lock:
        entry = _current(key)
        body = copy.deepcopy(entry[1]) if entry is not None else None
    if body is None:
        return factory()
    gateway = ClaudexorGateway(endpoint)
    gateway._engine_version = str(body["engine"]["version"])
    gateway._engine_build_sha = str(body["engine"].get("sha") or "")
    gateway._store_identity = str(body.get("storeId") or "")
    daemon = get_owned_daemon()
    daemon._proven_engine_version = gateway.engine_version
    if home is None:
        daemon.reconcile_rotation(gateway)
    return gateway


def operations(gateway, fetch, *, timeout_sec=None):
    """Coalesce concurrent catalog reads; never hold the cache lock during I/O."""
    key = endpoint_generation(gateway._endpoint)
    if key is None:
        return fetch()
    with _lock:
        entry = _select(key)
        owner = entry[2] is None
        if owner:
            entry[2] = Future()
        pending = entry[2]
    if owner:
        try:
            pending.set_result(fetch())
        except BaseException as exc:
            with _lock:
                current = _current(key)
                if current is not None and current[2] is pending:
                    current[2] = None
            pending.set_exception(exc)
    try:
        return copy.deepcopy(pending.result(timeout=timeout_sec))
    except TimeoutError as exc:
        from ouroboros.gateways.claudexor import ClaudexorUnavailable
        raise ClaudexorUnavailable("daemon_unreachable", "Catalog read is still unanswered",
                                  observation_timeout=True) from exc
