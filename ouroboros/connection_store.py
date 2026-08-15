"""Owner-state store for remote SSH connection metadata (RWS v2, D6).

ONE store module owns the path (``data/state/remote_connections.json``),
schema, lock, and atomic writer for remote connection records. Every reader
and writer imports it — the owner-only gateway handlers, the file/data-tool
denial predicates, and (stage B) the browser self-call denial — the same
owner-state discipline as settings and skill trust.

Trust boundary (D6, documented — not enforced by command blacklists): the
store lives inside the operator's OS account; a process running AS that
account (including agent shell commands) can physically read any file the
account can. The v1 command-blacklist detectors ("cat remote_connections.json",
"python -c" bypass hunting in the shell guard) were security theater over that
boundary and are deliberately NOT part of RWS v2. What this module and its
consumers DO guarantee are the deterministic accidental-access guards: the
structured file/data tools refuse the store and its lock/temp/hardlink aliases
(``tools/core.py``), the gateway file browser treats it as owner-only
(``gateway/files.py``), and the HTTP surface for it is owner-authenticated
even on loopback (``server_auth.py``). The store itself holds NO secrets —
ssh aliases, pinned host identities, and lifecycle only; key material stays in
the operator's own OpenSSH configuration.

Persistence goes through the SSOT helpers ``utils.atomic_write_json`` /
``utils.read_json_dict`` plus the ``platform_layer`` exclusive-lock primitive
(ALPHA RWS-04) — no private JSON writer.
"""

from __future__ import annotations

import pathlib
import secrets
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from ouroboros.platform_layer import (
    acquire_exclusive_file_lock,
    release_exclusive_file_lock,
)
from ouroboros.utils import atomic_write_json, read_json_dict, utc_now_iso

CONNECTION_STORE_SCHEMA_VERSION = 1
CONNECTION_LIFECYCLES = frozenset({"active", "retired"})
_LOCK_TIMEOUT_SEC = 4.0
_LOCK_STALE_SEC = 90.0


def connection_store_path() -> pathlib.Path:
    from ouroboros.config import REMOTE_CONNECTIONS_PATH

    return pathlib.Path(REMOTE_CONNECTIONS_PATH)


def is_connection_store_path(
    candidate: pathlib.Path,
    *,
    store_path: pathlib.Path | None = None,
) -> bool:
    """Match the owner store and its lock/atomic aliases without following writes."""

    target = pathlib.Path(store_path or connection_store_path()).resolve(strict=False)
    path = pathlib.Path(candidate)
    try:
        if path.exists() and target.exists() and path.samefile(target):
            return True
    except OSError:
        pass
    try:
        if path.parent.resolve(strict=False) != target.parent:
            return False
    except OSError:
        return False
    name = path.name.casefold()
    target_name = target.name.casefold()
    return (
        name == target_name
        or name == f"{target_name}.lock"
        or name.startswith(f".{target_name}.tmp.")
    )


def normalize_ssh_alias(value: Any) -> str:
    alias = str(value or "").strip()
    if not alias or alias.startswith("-"):
        raise ValueError("ssh_alias must be a non-empty host token without a leading dash")
    if len(alias) > 255:
        raise ValueError("ssh_alias must be at most 255 characters")
    if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in alias):
        raise ValueError("ssh_alias must not contain whitespace or control characters")
    return alias


def _clean_text(value: Any, field: str, *, maximum: int) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required")
    if len(text) > maximum or any(ord(char) < 32 or ord(char) == 127 for char in text):
        raise ValueError(f"{field} is invalid")
    return text


@contextmanager
def _owner_store_lock(path: pathlib.Path) -> Iterator[None]:
    """Serialize owner-store updates with the shared owner-only lock primitive."""

    lock_path = path.with_name(path.name + ".lock")
    fd = acquire_exclusive_file_lock(
        lock_path,
        timeout_sec=_LOCK_TIMEOUT_SEC,
        stale_sec=_LOCK_STALE_SEC,
        mode=0o600,
    )
    if fd is None:
        raise TimeoutError(f"could not lock remote connection store {path}")
    try:
        yield
    finally:
        release_exclusive_file_lock(lock_path, fd)


def _read_store(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "_schema_version": CONNECTION_STORE_SCHEMA_VERSION,
            "connections": [],
        }
    payload = read_json_dict(path)
    if payload is None:
        # An EXISTING but unreadable/non-object file is a loud failure, never an
        # empty store: silently starting fresh would drop pinned host identities.
        raise ValueError("remote connection store is unreadable or malformed")
    if (
        payload.get("_schema_version") != CONNECTION_STORE_SCHEMA_VERSION
        or not isinstance(payload.get("connections"), list)
        or any(not isinstance(item, dict) for item in payload["connections"])
    ):
        raise ValueError("remote connection store has an unsupported schema")
    return payload


def _write_store(path: pathlib.Path, payload: dict[str, Any]) -> None:
    atomic_write_json(
        path,
        payload,
        fsync=True,
        mode=0o600,
        fsync_directory=True,
    )


def _mutate_store(
    path: pathlib.Path,
    mutator: Callable[[list[dict[str, Any]]], Any],
) -> Any:
    with _owner_store_lock(path):
        payload = _read_store(path)
        connections = [dict(item) for item in payload["connections"]]
        result = mutator(connections)
        _write_store(
            path,
            {
                "_schema_version": CONNECTION_STORE_SCHEMA_VERSION,
                "connections": connections,
            },
        )
        return result


def list_connections(
    path: pathlib.Path | None = None,
    *,
    include_retired: bool = False,
) -> list[dict[str, Any]]:
    rows = [
        dict(item)
        for item in _read_store(pathlib.Path(path or connection_store_path()))[
            "connections"
        ]
    ]
    if include_retired:
        return rows
    return [row for row in rows if row.get("lifecycle", "active") == "active"]


def get_connection(
    connection_id: str,
    path: pathlib.Path | None = None,
    *,
    include_retired: bool = True,
) -> dict[str, Any] | None:
    wanted = str(connection_id or "").strip()
    for row in list_connections(path, include_retired=include_retired):
        if str(row.get("id") or "") == wanted:
            return row
    return None


def connection_trust_index(path: pathlib.Path | None = None) -> dict[str, str]:
    """``{connection_id: trust identity}`` for every ACTIVE connection.

    The admission door's single read (D6). The value is the pinned host identity,
    and it is deliberately handed out as an OPAQUE string: admission compares it
    for equality to decide "still the same trusted host" and never interprets it,
    so this module stays the only place that decides what host trust MEANS. A
    retired connection is absent rather than present-and-untrusted — admission
    must refuse it as unknown, not weigh it.

    An unpinned active connection maps to ``""``: it exists (so admission's
    membership check passes and the request is not "unknown connection"), but it
    can never equal a fenced trust identity, so anything fenced against it stays
    refused.
    """

    return {
        str(row.get("id") or ""): str(row.get("expected_host_id") or "")
        for row in list_connections(path)
        if str(row.get("id") or "").strip()
    }


def add_connection(
    *,
    name: str,
    ssh_alias: str,
    path: pathlib.Path | None = None,
) -> dict[str, Any]:
    display_name = _clean_text(name, "name", maximum=80)
    alias = normalize_ssh_alias(ssh_alias)
    target = pathlib.Path(path or connection_store_path())

    def add(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if any(
            row.get("lifecycle", "active") == "active"
            and row.get("ssh_alias") == alias
            for row in rows
        ):
            raise ValueError("an active connection already uses this ssh_alias")
        now = utc_now_iso()
        row = {
            "id": f"conn_{secrets.token_hex(8)}",
            "name": display_name,
            "ssh_alias": alias,
            "expected_host_id": "",
            "host_id_history": [],
            "lifecycle": "active",
            "retired_at": None,
            "bootstrapped_at": None,
            "bootstrap_build": "",
            "bootstrap_contract_set": 0,
            "created_at": now,
            "updated_at": now,
        }
        rows.append(row)
        return dict(row)

    return _mutate_store(target, add)


def record_bootstrap(
    connection_id: str,
    *,
    build: str = "",
    contract_set: int = 0,
    path: pathlib.Path | None = None,
) -> dict[str, Any]:
    """Persist that THIS connection has been bootstrapped, with which build and
    which Home↔execd CONTRACT SET.

    Bootstrap compatibility is OWNER STATE, not live state: "the compatible remote
    executor is installed on that host" is a fact about the host, established by an
    owner action, and it does not stop being true because Ouroboros restarted. It
    used to live only in a process-local dict, so after a restart the New Project
    picker was empty and the UI copy ("run Bootstrap, or Test to refresh health")
    described a recovery path that could not work — Test never restored the flag,
    because the flag only existed if a bootstrap had happened in the same process.

    Health FRESHNESS stays process-local on purpose (``gateway/connections.py``): it
    is a claim about the last few minutes, which no durable record can make.

    Clearing is the business of the two operations that invalidate the claim:
    :func:`retrust_connection` (a new host identity has never been proven
    compatible) and :func:`retire_connection`.

    ``contract_set`` is what makes the claim CHECKABLE rather than merely present.
    The release id alone cannot answer "can this install still serve me?" — most
    Ouroboros releases change no shared contract, so comparing release ids would
    report every connection as stale after every upgrade, and comparing nothing (the
    previous behaviour) reported none of them ever. The contract set moves only when a
    shared contract does, so `bootstrap_contract_set != CONTRACT_SET_VERSION` is a
    true statement about compatibility. A record written before this field existed
    reads as 0, which is exactly right: those installs predate the versioning.
    """
    wanted = _clean_text(connection_id, "connection_id", maximum=80)
    executor_build = str(build or "").strip()[:200]
    declared_contract_set = (
        int(contract_set)
        if isinstance(contract_set, int) and not isinstance(contract_set, bool)
        else 0
    )

    def record(rows: list[dict[str, Any]]) -> dict[str, Any]:
        for row in rows:
            if row.get("id") != wanted:
                continue
            if row.get("lifecycle", "active") != "active":
                raise ValueError("connection_retired")
            now = utc_now_iso()
            row["bootstrapped_at"] = now
            if executor_build:
                row["bootstrap_build"] = executor_build
            row["bootstrap_contract_set"] = declared_contract_set
            row["updated_at"] = now
            return dict(row)
        raise KeyError(wanted)

    return _mutate_store(pathlib.Path(path or connection_store_path()), record)


def _clear_bootstrap(row: dict[str, Any]) -> None:
    """Drop the durable bootstrap claim — ONE spelling for both invalidators."""
    row["bootstrapped_at"] = None
    row["bootstrap_build"] = ""
    row["bootstrap_contract_set"] = 0


def pin_connection_host(
    connection_id: str,
    host_id: str,
    *,
    path: pathlib.Path | None = None,
) -> dict[str, Any]:
    wanted = _clean_text(connection_id, "connection_id", maximum=80)
    observed = _clean_text(host_id, "host_id", maximum=256)

    def pin(rows: list[dict[str, Any]]) -> dict[str, Any]:
        for row in rows:
            if row.get("id") != wanted:
                continue
            if row.get("lifecycle", "active") != "active":
                raise ValueError("connection_retired")
            expected = str(row.get("expected_host_id") or "")
            if expected and expected != observed:
                raise ValueError("remote host identity changed; explicit owner retrust is required")
            if not expected:
                now = utc_now_iso()
                row["expected_host_id"] = observed
                row["host_id_history"] = [
                    {"host_id": observed, "trusted_at": now, "superseded_at": None}
                ]
                row["updated_at"] = now
            return dict(row)
        raise KeyError(wanted)

    return _mutate_store(pathlib.Path(path or connection_store_path()), pin)


def retrust_connection(
    connection_id: str,
    new_host_id: str,
    *,
    path: pathlib.Path | None = None,
    has_active_lease: bool = False,
) -> dict[str, Any]:
    if has_active_lease:
        raise ValueError("connection has an active task or lease")
    wanted = _clean_text(connection_id, "connection_id", maximum=80)
    replacement = _clean_text(new_host_id, "host_id", maximum=256)

    def retrust(rows: list[dict[str, Any]]) -> dict[str, Any]:
        for row in rows:
            if row.get("id") != wanted:
                continue
            if row.get("lifecycle", "active") != "active":
                raise ValueError("connection_retired")
            current = str(row.get("expected_host_id") or "")
            if not current:
                raise ValueError("connection has no pinned host identity")
            if current == replacement:
                return dict(row)
            now = utc_now_iso()
            history = [dict(item) for item in row.get("host_id_history", [])]
            for item in history:
                if item.get("host_id") == current and not item.get("superseded_at"):
                    item["superseded_at"] = now
            history.append(
                {"host_id": replacement, "trusted_at": now, "superseded_at": None}
            )
            row["host_id_history"] = history
            row["expected_host_id"] = replacement
            # A DIFFERENT host has never been proven compatible with this build,
            # whatever the previous one had installed.
            _clear_bootstrap(row)
            row["updated_at"] = now
            return dict(row)
        raise KeyError(wanted)

    return _mutate_store(pathlib.Path(path or connection_store_path()), retrust)


def retire_connection(
    connection_id: str,
    *,
    path: pathlib.Path | None = None,
    has_active_lease: bool = False,
) -> dict[str, Any]:
    if has_active_lease:
        raise ValueError("connection has an active task or lease")
    wanted = _clean_text(connection_id, "connection_id", maximum=80)

    def retire(rows: list[dict[str, Any]]) -> dict[str, Any]:
        for row in rows:
            if row.get("id") != wanted:
                continue
            if row.get("lifecycle", "active") == "retired":
                return dict(row)
            now = utc_now_iso()
            row["lifecycle"] = "retired"
            row["retired_at"] = now
            _clear_bootstrap(row)
            row["updated_at"] = now
            return dict(row)
        raise KeyError(wanted)

    return _mutate_store(pathlib.Path(path or connection_store_path()), retire)


__all__ = [
    "CONNECTION_LIFECYCLES",
    "CONNECTION_STORE_SCHEMA_VERSION",
    "add_connection",
    "connection_store_path",
    "connection_trust_index",
    "get_connection",
    "is_connection_store_path",
    "list_connections",
    "normalize_ssh_alias",
    "pin_connection_host",
    "record_bootstrap",
    "retire_connection",
    "retrust_connection",
]
