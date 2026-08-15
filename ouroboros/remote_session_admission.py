"""Opening, reusing and identity-checking ONE remote project session.

Split out of ``remote_workspace`` at the module size gate (§5 pre-split rule), and
the cut follows a real seam rather than a line count: the broker owns LIFECYCLE —
the request queue, the worker channels, panic, leases — while everything here
answers a single narrower question, "may this placement use a session, and which
one". That question has its own vocabulary (the four-part session key, the
per-key admission lock, the identity comparisons against the handshake) and its
own failure mode: admitting a target that is not the one the ref names.

These are FUNCTIONS over a broker rather than methods on it, so the coupling is
visible in every signature: nothing here holds broker state of its own, and the
broker keeps three thin delegating methods because they are what its dispatch
table names.

Identity is checked against the handshake and NOTHING is trusted because it
arrived: host id, workspace id, canonical root, capability hash and execd artifact
must all match what Home expects, and a mismatch is a typed refusal that demands
explicit owner action rather than an import of somebody else's target.
"""

from __future__ import annotations

import dataclasses
import pathlib
import re
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ouroboros.remote_worker_proxy import (
    json_copy as _json_copy,
    opaque as _opaque,
)
from ouroboros.workspace_diagnostics import RemoteWorkspaceError
from ouroboros.workspace_ref import SshWorkspaceRef, normalize_workspace_ref

_SSH_ALIAS_RE = re.compile(r"^[^\s\x00-\x20\x7f-][^\s\x00-\x1f\x7f]*$")


@dataclass(frozen=True)
class SessionOpenRequest:
    connection: dict[str, Any]
    remote_root: str
    project_id: str
    workspace_id: str
    server_generation: str
    capability_manifest: dict[str, Any]
    drive_root: pathlib.Path
    bundle_dir: pathlib.Path | None = None
    ssh_binary: str | None = None


@dataclass
class Session:
    key: tuple[str, str, str, str]
    connection: dict[str, Any]
    remote_root: str
    transport: Any  # a RemoteTransport; typed Any to keep this module below the broker
    handshake: dict[str, Any]
    opened_at: float = field(default_factory=time.monotonic)
    last_used_at: float = field(default_factory=time.monotonic)


def session_admission_result(session: Session) -> dict[str, Any]:
    """The nonsecret session facts a caller needs to seal a placement.

    Deliberately narrow: identities and target-native facts, never a transport
    handle, never a connection secret, never Home task state. Whatever the Home
    admission authority wants to record, it records itself.
    """

    facts = dict(session.handshake)
    return {
        "ok": True,
        # The placement descriptor DERIVED from the session identity — not a
        # second authority: `workspace_admission` still seals what is persisted.
        "workspace_ref": SshWorkspaceRef(
            connection_id=session.key[0],
            remote_root=session.remote_root,
            workspace_id=session.key[2],
        ).to_payload(),
        "connection_id": session.key[0],
        "project_id": session.key[1],
        "workspace_id": session.key[2],
        "server_generation": session.key[3],
        "remote_root": session.remote_root,
        "host_id": str(facts.get("host_id") or ""),
        "canonical_root": str(facts.get("canonical_root") or ""),
        "capability_hash": str(facts.get("capability_hash") or ""),
        "release_id": str(facts.get("release_id") or ""),
        "artifact_sha256": str(facts.get("artifact_sha256") or ""),
        "handshake": facts,
    }


def session_request(
    broker: Any,
    connection: Mapping[str, Any],
    remote_root: str,
    project_id: str,
    workspace_id: str,
) -> SessionOpenRequest:
    row = _json_copy(connection, "connection")
    connection_id = _opaque(row.get("id"), "connection_id")
    alias = str(row.get("ssh_alias") or "")
    if not _SSH_ALIAS_RE.fullmatch(alias):
        raise ValueError("ssh_alias is invalid")
    row["id"] = connection_id
    return SessionOpenRequest(
        connection=row,
        remote_root=str(remote_root),
        project_id=str(project_id),
        workspace_id=str(workspace_id),
        server_generation=broker.server_generation,
        capability_manifest=dict(broker.capability_projection),
        drive_root=broker.drive_root,
        bundle_dir=broker.bundle_dir,
        ssh_binary=broker.ssh_binary,
    )

def admit_on_broker(broker: Any, payload: dict[str, Any]) -> dict[str, Any]:
    connection = payload.get("connection")
    connection = connection if isinstance(connection, Mapping) else {}
    lock_key = (
        _opaque(connection.get("id"), "connection_id"),
        _opaque(payload.get("project_id"), "project_id"),
        str(payload.get("workspace_id") or payload.get("remote_root") or ""),
    )
    with broker._state_lock:
        lock = broker._admission_key_locks.setdefault(lock_key, threading.Lock())
    with lock:
        return _admit_locked(broker, payload)


def _abort_if_cancelled(
    broker: Any,
    session: Session,
    task_id: str,
    cancel: threading.Event,
    external_cancel: threading.Event | None,
) -> None:
    """Kill this task's remote work if admission was cancelled mid-flight.

    Checked AFTER the session is usable, because a cancel that arrives while
    the handshake is in progress must still reach the target — the honest
    outcome is a killed process group, not a session that quietly proceeds.
    """

    if not (
        cancel.is_set()
        or (external_cancel is not None and external_cancel.is_set())
    ):
        return
    try:
        session.transport.cancel(
            {"task_id": task_id, "request_id": "", "operation_id": ""}
        )
    except Exception:
        pass
    raise RemoteWorkspaceError(
        "admission_cancelled",
        "Remote admission was cancelled.",
        phase="connect",
    )


def _admit_locked(broker: Any, payload: dict[str, Any]) -> dict[str, Any]:
    request = session_request(
        broker,
        payload["connection"],
        payload["remote_root"],
        payload["project_id"],
        payload.get("workspace_id") or "",
    )
    raw_task_id = str(payload.get("task_id") or "")
    task_id = "" if raw_task_id.startswith("project:") else raw_task_id
    cancel = payload["cancel"]
    external_cancel = payload.get("external_cancel")
    if cancel.is_set() or (external_cancel is not None and external_cancel.is_set()):
        raise RemoteWorkspaceError("admission_cancelled", "Remote admission was cancelled.", phase="connect")
    if not request.workspace_id:
        with broker._state_lock:
            root_match = next(
                (
                    session
                    for key, session in broker._sessions.items()
                    if key[0] == request.connection["id"]
                    and key[1] == request.project_id
                    and key[3] == broker.server_generation
                    and session.remote_root.rstrip("/") == request.remote_root.rstrip("/")
                ),
                None,
            )
        if root_match is not None:
            request = dataclasses.replace(
                request,
                workspace_id=root_match.key[2],
            )
    if request.workspace_id:
        existing_key = (
            request.connection["id"],
            request.project_id,
            request.workspace_id,
            broker.server_generation,
        )
        with broker._state_lock:
            existing = broker._sessions.get(existing_key)
        if existing is not None:
            if task_id:
                with broker._state_lock:
                    broker._admission_transports[task_id] = (existing.transport, False)
            _abort_if_cancelled(broker, existing, task_id, cancel, external_cancel)
            if task_id:
                with broker._state_lock:
                    broker._task_sessions[task_id] = existing_key
            return session_admission_result(existing)
    transport = broker._new_transport(request)
    if task_id:
        with broker._state_lock:
            broker._admission_transports[task_id] = (transport, True)
    try:
        facts = transport.handshake()
        if cancel.is_set() or (external_cancel is not None and external_cancel.is_set()):
            raise RemoteWorkspaceError("admission_cancelled", "Remote admission was cancelled.", phase="connect")
        observed_host = _opaque(facts.get("host_id"), "host_id")
        expected_host = str(request.connection.get("expected_host_id") or "")
        if expected_host and expected_host != observed_host:
            raise RemoteWorkspaceError(
                "host_identity_mismatch",
                "Remote host identity changed; explicit re-trust is required.",
                phase="bootstrap",
            )
        observed_workspace = _opaque(facts.get("workspace_id"), "workspace_id")
        if request.workspace_id and request.workspace_id != observed_workspace:
            raise RemoteWorkspaceError(
                "workspace_identity_mismatch", "Remote workspace identity changed.", phase="bootstrap"
            )
        if facts.get("capability_hash") != broker.capability_projection["manifest_sha256"]:
            raise RemoteWorkspaceError(
                "capability_mismatch", "Remote execd capabilities differ from Home.", phase="bootstrap"
            )
        if str(facts.get("canonical_root") or "") != request.remote_root.rstrip("/"):
            raise RemoteWorkspaceError(
                "workspace_root_mismatch",
                "Remote canonical git root differs from the selected path.",
                phase="bootstrap",
            )
        identity = getattr(transport, "artifact_identity", None)
        expected_artifact = identity() if callable(identity) else {}
        if expected_artifact and any(
            facts.get(field) != value
            for field, value in expected_artifact.items()
            if field != "artifact_size" and value
        ):
            raise RemoteWorkspaceError(
                "execd_artifact_mismatch",
                "Remote execd artifact identity differs from Home selection.",
                phase="bootstrap",
            )
        key = (request.connection["id"], request.project_id, observed_workspace, broker.server_generation)
        session = Session(key, request.connection, request.remote_root, transport, dict(facts))
        inserted = False
        with broker._state_lock:
            existing = broker._sessions.get(key)
            if existing is None:
                broker._sessions[key] = session
                selected = session
                inserted = True
            else:
                selected = existing
        if selected is not session:
            # A concurrent admission of the same key won; this transport is surplus.
            # Retired through the broker's ONE door so its panic custody ends with it.
            broker._retire_transport(transport)
            session = selected
            if task_id:
                with broker._state_lock:
                    broker._admission_transports[task_id] = (session.transport, False)
        elif task_id:
            with broker._state_lock:
                broker._admission_transports[task_id] = (session.transport, False)
        try:
            _abort_if_cancelled(broker, session, task_id, cancel, external_cancel)
        except BaseException:
            if inserted:
                with broker._state_lock:
                    if broker._sessions.get(key) is session:
                        broker._sessions.pop(key, None)
                broker._retire_transport(session.transport)
            raise
        if task_id:
            with broker._state_lock:
                broker._task_sessions[task_id] = key
        return session_admission_result(session)
    except BaseException:
        broker._retire_transport(transport)
        raise


def session_for_ref(
    broker: Any,
    raw_ref: Mapping[str, Any],
    *,
    task_id: str = "",
    parent_task_id: str = "", project_id: str = "",
) -> Session:
    ref = normalize_workspace_ref(dict(raw_ref))
    if ref is None or ref.kind != "ssh":
        raise ValueError("remote broker requires an SSH workspace ref")
    connection_id = str(ref.connection_id)
    workspace_id = str(ref.workspace_id)
    with broker._state_lock:
        key = broker._task_sessions.get(task_id) if task_id else None
        if key is None and task_id and parent_task_id:
            key = broker._task_sessions.get(parent_task_id)
        if key is None and task_id and project_id:
            key = (
                connection_id,
                project_id,
                workspace_id,
                broker.server_generation,
            )
        if key is None and task_id:
            raise RemoteWorkspaceError(
                "task_session_unbound",
                "Task is not bound to a remote workspace session.",
                phase="authorize",
            )
        if key is not None and ((key[0], key[2]) != (connection_id, workspace_id)
                                or (project_id and key[1] != project_id)):
            raise RemoteWorkspaceError(
                "task_session_mismatch",
                "Task is bound to another remote workspace session.",
                phase="authorize",
            )
        candidates = [candidate for candidate in broker._sessions
                      if (candidate[0], candidate[2]) == (connection_id, workspace_id)]
        if key is None:
            if len(candidates) > 1:
                raise RemoteWorkspaceError(
                    "remote_session_ambiguous",
                    "Remote workspace is bound to multiple projects; task identity is required.",
                    phase="authorize",
                )
            key = candidates[0] if candidates else None
        session = broker._sessions.get(key) if key is not None else None
        if session is not None and task_id:
            broker._task_sessions.setdefault(task_id, key)
    if session is None:
        raise RemoteWorkspaceError(
            "remote_session_disconnected",
            "Remote workspace session is not connected.",
            phase="connect",
            retryable=True,
            details={"connection_id": connection_id, "workspace_id": workspace_id},
        )
    session.last_used_at = time.monotonic()
    return session
