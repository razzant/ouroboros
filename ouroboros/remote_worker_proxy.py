"""Pickle-safe worker Pipe proxy for the server-owned SSH broker (RWS v2 §3.1).

A worker never owns a transport: it owns one bidirectional `Pipe` endpoint and
speaks a small request/reply protocol over it.  The endpoint is bound to the
server generation that minted it, and every call carries that generation — so a
proxy inherited across a restart or belonging to a replaced worker is answered
with a typed `BROKER_GENERATION_STALE` instead of being served by the wrong
broker or hanging on a pipe nobody reads.  Every failure mode here is typed and
deadline-bounded: closed pipe, protocol mismatch, timeout, stale generation.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import multiprocessing
import pathlib
import re
import threading
import time
import uuid
from collections.abc import Mapping
from multiprocessing.connection import Connection
from typing import Any

from ouroboros.remote_protocol import canonical_json
from ouroboros.remote_refusal_actions import ACTION_READMIT_PROJECT, ACTION_REPORT_DEFECT
from ouroboros.workspace_diagnostics import (
    ExecutionDiagnostic,
    ProcessExecutionResult,
    RemoteWorkspaceError,
    ToolExecutionEnvelope,
)

# One typed code for "you are talking to a broker generation that no longer
# exists". It is a refusal, never a wait: a hang here would stall a worker for
# its whole task deadline for no possible benefit.
BROKER_GENERATION_STALE = "BROKER_GENERATION_STALE"

_REQUEST_TIMEOUT_SEC = 120.0
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_OPAQUE_RE = re.compile(r"^[A-Za-z0-9_:@-](?:[A-Za-z0-9_.:@-]{0,254}[A-Za-z0-9_:@-])?$")


def opaque(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not _OPAQUE_RE.fullmatch(text):
        raise ValueError(f"{field_name} must be a file-safe opaque ID")
    return text


def optional_opaque(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    return opaque(text, field_name) if text else ""


def json_copy(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    try:
        copied = json.loads(canonical_json(dict(value)).decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"{label} must be bounded canonical JSON: {exc}") from exc
    if not isinstance(copied, dict):
        raise ValueError(f"{label} must be an object")
    return copied


def capability_projection(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the strict, JSON-safe native-capability proof sent to execd."""

    hashes = {
        "manifest_sha256": manifest.get("manifest_sha256"),
        "public_schema_sha256": manifest.get("public_schema_sha256"),
    }
    if any(not isinstance(value, str) or not _HASH_RE.fullmatch(value) for value in hashes.values()):
        raise ValueError("capability manifest hashes are invalid")
    operations = manifest.get("native_operations")
    if not isinstance(operations, list):
        raise ValueError("capability manifest native_operations must be a list")
    names = [str(row.get("name") or "") for row in operations if isinstance(row, dict)]
    if len(names) != len(operations) or any(not name for name in names):
        raise ValueError("capability manifest native_operations are invalid")
    projection = {
        "schema_version": int(manifest.get("schema_version") or 0),
        **hashes,
        "native_operations": sorted(names),
        "native_operations_sha256": hashlib.sha256(canonical_json(sorted(names))).hexdigest(),
        "native_kernel_modules_sha256": hashlib.sha256(
            canonical_json(sorted(str(item) for item in manifest.get("native_kernel_modules") or []))
        ).hexdigest(),
        "native_import_modules_sha256": hashlib.sha256(
            canonical_json(sorted(str(item) for item in manifest.get("native_import_modules") or []))
        ).hexdigest(),
        "native_import_edges_sha256": hashlib.sha256(
            canonical_json(manifest.get("native_import_edges") or {})
        ).hexdigest(),
    }
    return json.loads(canonical_json(projection).decode("utf-8"))


class RemoteWorkspacePipeProxy:
    """Worker client containing only one Pipe endpoint, never an SSH handle."""

    def __init__(self, endpoint: Connection, server_generation: str = "") -> None:
        self._endpoint = endpoint
        self._server_generation = str(server_generation or "")
        self._lock = threading.Lock()
        self._closed = False

    @property
    def server_generation(self) -> str:
        return self._server_generation

    def __getstate__(self) -> dict[str, Any]:
        return {
            "endpoint": self._endpoint,
            "closed": self._closed,
            "server_generation": self._server_generation,
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self._endpoint = state["endpoint"]
        self._server_generation = str(state.get("server_generation") or "")
        self._lock = threading.Lock()
        self._closed = bool(state.get("closed"))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._endpoint.close()
        except (OSError, EOFError):
            pass

    close_parent_copy = close

    def prepare(self, workspace_ref: Mapping[str, Any], **kwargs: Any) -> Any:
        result = self._call("prepare", {"workspace_ref": dict(workspace_ref), **kwargs})
        return prepared_from_dict(result)

    def execute_prepared(
        self,
        workspace_ref: Mapping[str, Any],
        prepared: Any,
        *,
        canonical_args: Mapping[str, Any],
        task_id: str = "",
        timeout_sec: float | None = None,
        import_kind: str = "",
        import_context: Mapping[str, Any] | None = None,
    ) -> ToolExecutionEnvelope:
        result = self._call(
            "execute_prepared",
            {
                "workspace_ref": dict(workspace_ref),
                "prepared": dataclasses.asdict(prepared),
                "canonical_args": dict(canonical_args),
                "task_id": task_id,
                "timeout_sec": timeout_sec,
                # The declared import channel travels the proxy too: a worker-side
                # Home channel must reach the same closed-registry check as a
                # server-side one, or which process happens to hold the broker would
                # decide whether the boundary is enforced.
                "import_kind": str(import_kind or ""),
                "import_context": dict(import_context or {}),
            },
            timeout_sec=execution_wait_timeout(canonical_args, timeout_sec),
        )
        return envelope_from_dict(result)

    def abort_prepared(
        self,
        workspace_ref: Mapping[str, Any],
        prepared: Any,
        *,
        task_id: str = "",
        reason: str = "denied",
    ) -> bool:
        return bool(
            self._call(
                "abort_prepared",
                {
                    "workspace_ref": dict(workspace_ref),
                    "prepared": dataclasses.asdict(prepared),
                    "task_id": task_id,
                    "reason": reason,
                },
            )
        )

    def fetch_blob(
        self,
        workspace_ref: Mapping[str, Any],
        blob_id: str,
        *,
        max_bytes: int,
        task_id: str,
    ) -> bytes:
        result = self._call(
            "fetch_blob",
            {
                "workspace_ref": dict(workspace_ref),
                "blob_id": blob_id,
                "max_bytes": max_bytes,
                "task_id": task_id,
            },
        )
        if not isinstance(result, bytes):
            raise self._error(
                "remote_blob_invalid",
                "Remote blob response was not bytes.",
                phase="import",
            )
        return result

    def cancel(self, workspace_ref: Mapping[str, Any], **kwargs: Any) -> bool:
        return bool(self._call("cancel", {"workspace_ref": dict(workspace_ref), **kwargs}))

    def open_browser_forward(
        self,
        workspace_ref: Mapping[str, Any],
        *,
        remote_port: int,
        task_id: str,
    ) -> dict[str, Any]:
        return dict(
            self._call(
                "open_browser_forward",
                {
                    "workspace_ref": dict(workspace_ref),
                    "remote_port": int(remote_port),
                    "task_id": task_id,
                },
            )
        )

    def close_browser_forward(self, forward_id: str) -> bool:
        return bool(
            self._call(
                "close_browser_forward",
                {"forward_id": str(forward_id)},
            )
        )

    def _call(
        self,
        method: str,
        payload: dict[str, Any],
        *,
        timeout_sec: float | None = None,
    ) -> Any:
        if self._closed:
            raise self._error(
                "broker_pipe_closed",
                "Remote workspace worker channel is closed.",
                phase="stream",
            )
        correlation_id = uuid.uuid4().hex
        message = {
            "correlation_id": correlation_id,
            "method": method,
            "payload": payload,
            "server_generation": self._server_generation,
        }
        wait_sec = _REQUEST_TIMEOUT_SEC if timeout_sec is None else max(1.0, float(timeout_sec))
        with self._lock:
            try:
                self._endpoint.send(message)
                deadline = time.monotonic() + wait_sec
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not self._endpoint.poll(remaining):
                        raise self._error(
                            "broker_pipe_timeout",
                            "Remote workspace broker did not answer before the deadline.",
                            phase="stream",
                            completion="unknown",
                            retryable=True,
                        )
                    response = self._endpoint.recv()
                    if not isinstance(response, dict):
                        break
                    # Requests are serialized by _lock, but a timed-out request
                    # may finish later. Drain only those stale replies instead
                    # of poisoning the next correlation on this durable pipe.
                    if response.get("correlation_id") == correlation_id:
                        break
            except (EOFError, OSError) as exc:
                raise self._error(
                    "broker_pipe_closed",
                    "Remote workspace broker channel closed.",
                    phase="stream",
                    completion="unknown",
                    retryable=True,
                ) from exc
        if not isinstance(response, dict):
            raise self._error(
                "broker_pipe_protocol",
                "Remote workspace broker returned a mismatched response.",
                phase="stream",
                completion="unknown",
            )
        if not response.get("ok"):
            error = response.get("error") if isinstance(response.get("error"), dict) else {}
            raise self._error(
                str(error.get("code") or "remote_workspace_error"),
                str(error.get("message") or "Remote workspace request failed."),
                phase=str(error.get("phase") or "stream"),
                completion=str(error.get("completion") or "unknown"),
                retryable=bool(error.get("retryable")),
                details=error.get("details") if isinstance(error.get("details"), dict) else {},
            )
        return response.get("result")

    @staticmethod
    def _error(code: str, message: str, **kwargs: Any) -> Exception:
        return RemoteWorkspaceError(code, message, **kwargs)


class WorkerChannels:
    """The BROKER side of the worker channels — both ends live in one module.

    Keyed by OWNER (``"worker:3"``), never a list: a respawned worker must
    REPLACE its channel, and a leftover endpoint for a dead worker would keep the
    broker's poll loop answering into a pipe nobody reads. Each minted proxy is
    stamped with the server generation, which is what lets the broker refuse a
    stale call instead of serving or hanging on it.
    """

    def __init__(self, server_generation: str) -> None:
        self._server_generation = str(server_generation or "")
        self._lock = threading.Lock()
        self._endpoints: dict[str, Connection] = {}
        self._send_locks: dict[int, threading.Lock] = {}

    def mint(self, owner: str = "") -> tuple[Connection, RemoteWorkspacePipeProxy]:
        """Return `(broker endpoint, worker proxy)`, retiring `owner`'s previous one."""

        broker_endpoint, worker_endpoint = multiprocessing.Pipe(duplex=True)
        owner_key = str(owner or f"anonymous:{uuid.uuid4().hex}")
        with self._lock:
            previous = self._endpoints.pop(owner_key, None)
            if previous is not None:
                self._send_locks.pop(id(previous), None)
            self._endpoints[owner_key] = broker_endpoint
            self._send_locks[id(broker_endpoint)] = threading.Lock()
        close_endpoint(previous)
        return broker_endpoint, RemoteWorkspacePipeProxy(
            worker_endpoint, self._server_generation
        )

    def close_owner(self, owner: str) -> bool:
        with self._lock:
            endpoint = self._endpoints.pop(str(owner), None)
            if endpoint is not None:
                self._send_locks.pop(id(endpoint), None)
        close_endpoint(endpoint)
        return endpoint is not None

    def close_all(self) -> int:
        with self._lock:
            endpoints = list(self._endpoints.values())
            self._endpoints.clear()
            self._send_locks.clear()
        for endpoint in endpoints:
            close_endpoint(endpoint)
        return len(endpoints)

    def live(self) -> list[Connection]:
        with self._lock:
            return list(self._endpoints.values())

    def send_lock(self, endpoint: Connection) -> threading.Lock | None:
        with self._lock:
            return self._send_locks.get(id(endpoint))

    def drop(self, endpoints: list[Connection]) -> None:
        """Forget endpoints the poll loop found dead."""

        with self._lock:
            self._endpoints = {
                owner: endpoint
                for owner, endpoint in self._endpoints.items()
                if endpoint not in endpoints
            }
            for endpoint in endpoints:
                self._send_locks.pop(id(endpoint), None)
        for endpoint in endpoints:
            close_endpoint(endpoint)

    def detach_after_fork(self) -> None:
        """Drop inherited descriptor copies in a forked child."""

        for endpoint in self.live():
            close_endpoint(endpoint)
        with self._lock:
            self._endpoints = {}
            self._send_locks = {}


def close_endpoint(endpoint: Connection | None) -> None:
    """Close a Pipe endpoint; an already-dead pipe is not an error."""

    if endpoint is None:
        return
    try:
        endpoint.close()
    except (OSError, EOFError):
        pass


def prepared_from_dict(raw: Any) -> Any:
    # Deliberately function-local: the prepared-call token is the broker's own
    # contract, and importing it at module scope would make the proxy and the
    # broker a cycle. Nothing else here reaches upward.
    from ouroboros.remote_workspace import PreparedRemoteCall

    if isinstance(raw, PreparedRemoteCall):
        return raw
    values = validated_prepared(raw)
    diagnostic = values.get("diagnostic")
    return PreparedRemoteCall(
        **{
            key: values[key]
            for key in (
                "request_id",
                "operation_id",
                "tool",
                "prepared_token",
                "prepared_hash",
                "expires_at_ms",
                "execution_args",
                "native_facts",
            )
        },
        diagnostic=diagnostic_from_dict(diagnostic) if isinstance(diagnostic, dict) else None,
    )


def validated_prepared(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise RemoteWorkspaceError(
            "prepared_response_invalid",
            "Execd returned an invalid prepared response.",
            phase="prepare",
        )
    result = {
        "request_id": opaque(raw.get("request_id"), "request_id"),
        "operation_id": opaque(raw.get("operation_id"), "operation_id"),
        "tool": str(raw.get("tool") or ""),
        "prepared_token": opaque(raw.get("prepared_token"), "prepared_token"),
        "prepared_hash": str(raw.get("prepared_hash") or ""),
        "expires_at_ms": int(raw.get("expires_at_ms") or 0),
        "execution_args": json_copy(raw.get("execution_args"), "execution_args"),
        "native_facts": json_copy(raw.get("native_facts"), "native_facts"),
    }
    if not result["tool"] or re.fullmatch(r"[0-9a-f]{64}", result["prepared_hash"]) is None:
        raise RemoteWorkspaceError(
            "prepared_response_invalid",
            "Execd returned invalid prepared identity.",
            phase="prepare",
        )
    if result["expires_at_ms"] <= int(time.time() * 1000):
        raise RemoteWorkspaceError(
            "prepared_call_expired",
            "Execd prepared call is already expired.",
            phase="prepare",
        )
    if isinstance(raw.get("diagnostic"), Mapping):
        result["diagnostic"] = dict(raw["diagnostic"])
    return result


def diagnostic_from_dict(raw: Mapping[str, Any]) -> ExecutionDiagnostic:
    domains = {"transport", "protocol", "policy", "filesystem", "process", "artifact"}
    completions = {"not_started", "completed", "unknown"}
    domain = str(raw.get("domain") or "protocol")
    completion = str(raw.get("completion") or "unknown")
    return ExecutionDiagnostic(
        domain=domain if domain in domains else "protocol",  # type: ignore[arg-type]
        code=str(raw.get("code") or "remote_error"),
        message=str(raw.get("message") or "Remote operation failed."),
        phase=str(raw.get("phase") or "execute"),
        request_id=str(raw.get("request_id") or ""),
        operation_id=str(raw.get("operation_id") or ""),
        completion=completion if completion in completions else "unknown",  # type: ignore[arg-type]
        retryable=bool(raw.get("retryable")),
        errno=int(raw["errno"]) if isinstance(raw.get("errno"), int) else None,
        details=dict(raw.get("details") or {}) if isinstance(raw.get("details"), dict) else {},
    )


def envelope_from_dict(raw: Any) -> ToolExecutionEnvelope:
    values = validated_envelope_dict(raw)
    diagnostic = diagnostic_from_dict(values["diagnostic"]) if isinstance(values.get("diagnostic"), dict) else None
    process_raw = values.get("process")
    process = None
    if isinstance(process_raw, dict):
        process = ProcessExecutionResult(
            returncode=int(process_raw.get("returncode") or 0),
            stdout=str(process_raw.get("stdout") or ""),
            stderr=str(process_raw.get("stderr") or ""),
            backend_trace=dict(process_raw.get("backend_trace") or {}),
            args=[str(item) for item in list(process_raw.get("args") or [])],
        )
    return ToolExecutionEnvelope(
        text=str(values.get("text") or ""),
        diagnostic=diagnostic,
        process=process,
        artifacts=tuple(dict(item) for item in list(values.get("artifacts") or []) if isinstance(item, dict)),
        trace=dict(values.get("trace") or {}),
    )


def validated_envelope_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, ToolExecutionEnvelope):
        raw = dataclasses.asdict(raw)
    if not isinstance(raw, Mapping):
        raise RemoteWorkspaceError(
            "remote_result_invalid",
            "Execd returned an invalid operation envelope.",
            phase="finalize",
            completion="unknown",
        )
    copied = json_copy(raw, "operation envelope")
    if not isinstance(copied.get("text", ""), str):
        raise RemoteWorkspaceError(
            "remote_result_invalid",
            "Execd operation envelope text is invalid.",
            phase="finalize",
            completion="unknown",
        )
    return copied


def execution_wait_timeout(canonical_args: Mapping[str, Any], supplied: Any) -> float:
    value = supplied
    if value is None:
        value = canonical_args.get("timeout_sec", canonical_args.get("timeout"))
    if value is None:
        return _REQUEST_TIMEOUT_SEC
    try:
        execution_sec = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("remote execution timeout must be numeric") from exc
    if not 1.0 <= execution_sec <= 86_400.0:
        raise ValueError("remote execution timeout must be in 1..86400 seconds")
    return execution_sec + 30.0


def reconnect_failure(
    connection_id: str,
    error: BaseException | None = None,
) -> dict[str, Any]:
    """A reconnect that did not reach ``ready``, in the CONTRACT's own vocabulary.

    ``status`` is ``degraded``/``disconnected`` and never a sixth word: the
    ``connection_state`` contract (``gateway/contracts.py::ConnectionStateOutbound``)
    knows five states, and every consumer keys off them — ``_record_runtime_health``
    only accepts those, and the browser reducer treats an unknown word as *no typed
    status at all* and falls back to derivation, which hides the Reconnect button at
    exactly the moment a reconnect is what the owner needs. A raised reconnect is
    ``degraded`` for the same reason the gateway's own exception projection is: the
    connection answered enough to fail, and the impairment is health, not absence.

    ``diagnostic`` is the typed detail MAPPING, not its message string. The contract
    types it ``Dict[str, Any]`` and all three layers that carry it
    (``gateway/connections.py``, ``connections_ui.js``, ``remote_task_state.js``)
    drop a non-object — so a reason that was computed and shipped used to be thrown
    away one hop before the owner could read it.
    """
    detail = (
        error_dict(error)
        if error is not None
        else {
            "code": "remote_session_absent",
            "message": "No admitted project session is available to reconnect.",
            "phase": "connect",
            "completion": "not_started",
            "retryable": False,
            "action": ACTION_READMIT_PROJECT,
        }
    )
    return {
        "status": "degraded" if error is not None else "disconnected",
        "phase": detail["phase"],
        "completion": detail["completion"],
        "error_code": detail["code"],
        # One vocabulary (`remote_refusal_actions`), not a local spelling: this used
        # to say `retry_reconnect`, which named the same button the rest of the surface
        # calls `reconnect_connection` — and an owner cannot tell two names for one
        # action apart from two different actions.
        #
        # And it is READ from the refusal, never recomputed here. It used to be
        # `ACTION_RECONNECT if retryable else ACTION_READMIT_PROJECT`, which is a
        # local opinion about every condition this function can be handed: a
        # `host_identity_changed` is not retryable, so a different machine answering
        # the alias told the owner to READMIT THE PROJECT — an action that succeeds
        # and leaves the block exactly where it was, the dead-end shape the Refusal
        # Action Rule forbids. `error_dict` carries the code's own answer now.
        "action": detail["action"],
        "diagnostic": dict(detail),
        "log_refs": [],
        "connection_id": connection_id,
        "sessions": [],
        "reconciliation": [],
    }


def error_dict(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, RemoteWorkspaceError):
        diagnostic = exc.diagnostic()
        try:
            from ouroboros.observability import redact_projection

            redacted = redact_projection(dict(diagnostic.details)).value
            details = dict(redacted) if isinstance(redacted, Mapping) else {}
        except Exception:
            redacted_diagnostic = ExecutionDiagnostic(
                domain="transport",
                code="redacted_details",
                message="Remote transport details.",
                phase="stream",
                details=dict(diagnostic.details),
            )
            details = dict(redacted_diagnostic.details)
        return {
            "code": exc.code,
            "message": safe_error_text(diagnostic.message),
            "phase": exc.phase,
            "completion": exc.completion,
            "retryable": exc.retryable,
            # The refusal's OWN action, at the top level as well as inside `details`:
            # a consumer of this wire dict must not have to re-derive it (the one that
            # did, `reconnect_failure`, derived it from `retryable` and advised the
            # wrong button). `details` carries it too, because that is the slot the
            # browser and `--json` already read.
            "action": exc.action,
            "details": details,
        }
    return {
        "code": type(exc).__name__,
        "message": safe_error_text(exc),
        "phase": "stream",
        "completion": "unknown",
        "retryable": False,
        # An UNTYPED exception at the broker↔worker seam is not a policy answer at
        # all — it is a crash in Home's own wiring, and no owner button moves it.
        # `retry` here would be advice that cannot work; the register already names
        # this condition honestly.
        "action": ACTION_REPORT_DEFECT,
        "details": {},
    }


def safe_error_text(exc: Any) -> str:
    text = str(exc).replace("\x00", "")
    try:
        from ouroboros.observability import redact_projection

        text = str(redact_projection(text).value)
    except Exception:
        pass
    home = str(pathlib.Path.home())
    if home and home != "/":
        text = text.replace(home, "<home>")
    return " ".join(text.split())[:2000] or type(exc).__name__
