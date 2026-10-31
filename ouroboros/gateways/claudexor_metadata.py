"""Immutable protocol metadata for the currently selected descriptor generation.

Liveness/admission probes remain fresh. Only model-call preparation reuses a
successful normal handshake; the route catalog is immutable within a generation.
No client, token or historical generation is retained here.
"""
from __future__ import annotations

import copy
import hashlib
import pathlib
import threading
from concurrent.futures import Future, TimeoutError

_lock = threading.Lock()
_key = None
_handshake = None
_operations = None


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
    global _key, _handshake, _operations
    if key != _key:
        _key, _handshake, _operations = key, None, None


def remember_handshake(gateway, body):
    gateway._store_identity = body.get("storeId") if isinstance(body.get("storeId"), str) else ""
    key = endpoint_generation(gateway._endpoint)
    if key is not None:
        global _handshake
        with _lock:
            _select(key)
            _handshake = copy.deepcopy(body) if body.get("servingMode") != "recovery_only" else None


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
        body = copy.deepcopy(_handshake) if key is not None and key == _key else None
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
    global _operations
    with _lock:
        _select(key)
        owner = _operations is None
        if owner:
            _operations = Future()
        pending = _operations
    if owner:
        try:
            pending.set_result(fetch())
        except BaseException as exc:
            with _lock:
                if _key == key and _operations is pending:
                    _operations = None
            pending.set_exception(exc)
    try:
        return copy.deepcopy(pending.result(timeout=timeout_sec))
    except TimeoutError as exc:
        from ouroboros.gateways.claudexor import ClaudexorUnavailable
        raise ClaudexorUnavailable("daemon_unreachable", "Catalog read is still unanswered",
                                  observation_timeout=True) from exc
