"""The server-generation owner of every SSH session (RWS v2 §3.1).

EXACTLY ONE broker per server generation. Workers never create transports: they
get one Pipe endpoint each and submit prepared calls through it, while this
object alone owns OpenSSH processes, protocol sessions, leases, bootstrap and
reconciliation. That is what makes the emergency-stop invariant enforceable —
there is one place that knows every live remote child.

Three properties are load-bearing and each has named tests:

* **Proxy lifecycle is bound to worker generation.** Each worker gets a distinct
  endpoint registered under its owner key; `respawn_worker` replaces that
  worker's endpoint (closing the dead one) instead of accumulating a second live
  channel; `kill_workers` closes all. Every proxy call carries the generation it
  was minted for, so a call from a stale generation is answered with a typed
  `BROKER_GENERATION_STALE` — refused, never hung.
* **Panic never waits for a lock.** It snapshots the custody registers
  (`_panic_transports`, `_panic_events`), takes the state lock only
  opportunistically (`blocking=False`), and calls each transport's own
  non-waiting panic. A panic that could block on the ordinary broker lock would
  be a software delay in the one path that may not have one. Custody ENDS on the
  ordinary exit too, through the one `_retire_transport` door — see the registers'
  own note in `__init__` for what an append-only version of them cost.
* **Admission here is SESSION admission only.** Home task-admission policy
  (attachment staging, task-acceptance evidence) belongs to
  `workspace_admission` and the transfer service. This module verifies target
  IDENTITY — host, workspace, canonical root, capability hash, execd artifact —
  opens or reuses the session, and binds the task to it. Nothing else.
"""

from __future__ import annotations
import concurrent.futures
import dataclasses
import os
import pathlib
import queue
import threading
import uuid
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from typing import Any, Protocol, runtime_checkable

from ouroboros.config import get_ssh_timeout_sec
from ouroboros.remote_protocol import canonical_json
from ouroboros.remote_service_leases import (
    RemoteServiceLeaseBook,
    refresh_connection_leases,
)
from ouroboros.remote_ssh import OpenSSHExecdTransport
from ouroboros.remote_worker_proxy import (
    RemoteWorkspacePipeProxy,
    capability_projection as _capability_projection,
    envelope_from_dict as _envelope_from_dict,
    error_dict as _error_dict,
    execution_wait_timeout as _execution_wait_timeout,
    json_copy as _json_copy,
    opaque as _opaque,
    optional_opaque as _optional_opaque,
    prepared_from_dict as _prepared_from_dict,
    reconnect_failure as _reconnect_failure,
    validated_envelope_dict as _validated_envelope_dict,
    validated_prepared as _validated_prepared,
)
from ouroboros.remote_worker_proxy import (
    BROKER_GENERATION_STALE,
    WorkerChannels,
)
from ouroboros.workspace_diagnostics import (
    ExecutionDiagnostic,
    RemoteWorkspaceError,
    ToolExecutionEnvelope,
)
from ouroboros.remote_session_admission import (
    SessionOpenRequest,
    Session as _Session,
    admit_on_broker,
    session_for_ref,
    session_request,
)
from ouroboros.workspace_ref import normalize_workspace_ref

_PIPE_QUEUE_LIMIT = 128
_BROKER_MAX_INFLIGHT = 32
_BROKER_IO_WORKERS = 8
_BROKER_POLL_SEC = 0.02
_DEFAULT_REQUEST_TIMEOUT_SEC = 120.0


# `RemoteWorkspaceError` is re-exported from `workspace_diagnostics`, where it
# lives beside the `ExecutionDiagnostic` it projects into. Keeping the class
# here would force the transport and the journal to import the broker to raise
# it — the exact upward arrow this rebuild removes.
__all__ = [
    "PreparedRemoteCall",
    "RemoteSessionBroker",
    "RemoteWorkspaceError",
    "SessionOpenRequest",
    "finish_remote_task",
    "get_remote_workspace_service",
    "set_remote_workspace_service",
]


@dataclass(frozen=True)
class PreparedRemoteCall:
    """Target-canonical facts awaiting one Home safety authorization."""

    request_id: str
    operation_id: str
    tool: str
    prepared_token: str
    prepared_hash: str
    expires_at_ms: int
    execution_args: dict[str, Any]
    native_facts: dict[str, Any]
    diagnostic: ExecutionDiagnostic | None = None


@runtime_checkable
class RemoteTransport(Protocol):
    """One broker-owned project session; never passed into a worker."""

    handshake: Callable[[], dict[str, Any]]
    prepare: Callable[[Mapping[str, Any], Mapping[str, bytes]], dict[str, Any]]
    execute_prepared: Callable[[Mapping[str, Any]], dict[str, Any]]
    abort_prepared: Callable[[Mapping[str, Any]], bool]
    fetch_blob: Callable[[str, int], bytes]
    reconcile: Callable[[], list[dict[str, Any]]]
    cancel: Callable[[Mapping[str, Any]], bool]
    task_lease: Callable[..., bool]
    panic: Callable[[], None]
    close: Callable[[], None]


class RemoteTransportFactory(Protocol):
    __call__: Callable[[SessionOpenRequest], RemoteTransport]


@runtime_checkable
class RemoteWorkspaceService(Protocol):
    """Small public contract consumed by gateway and workspace dispatch lanes."""

    prepare: Callable[..., PreparedRemoteCall]
    execute_prepared: Callable[..., ToolExecutionEnvelope]
    abort_prepared: Callable[..., bool]
    close_project_session: Callable[..., bool]
    fetch_blob: Callable[..., bytes]
    cancel: Callable[..., bool]
    open_browser_forward: Callable[..., dict[str, Any]]
    close_browser_forward: Callable[[str], bool]
    finish_task: Callable[..., bool]


@dataclass(order=True)
class _BrokerRequest:
    priority: int
    sequence: int
    method: str = field(compare=False)
    payload: dict[str, Any] = field(compare=False)
    future: concurrent.futures.Future[Any] = field(compare=False)


_SERVICE_LOCK = threading.RLock()
_REMOTE_WORKSPACE_SERVICE: RemoteWorkspaceService | None = None
_LIVE_BROKERS: "weakref.WeakSet[RemoteSessionBroker]" = weakref.WeakSet()


def set_remote_workspace_service(service: RemoteWorkspaceService | None) -> None:
    global _REMOTE_WORKSPACE_SERVICE
    with _SERVICE_LOCK:
        _REMOTE_WORKSPACE_SERVICE = service


def get_remote_workspace_service() -> RemoteWorkspaceService:
    with _SERVICE_LOCK:
        service = _REMOTE_WORKSPACE_SERVICE
    if service is None:
        raise RemoteWorkspaceError(
            "remote_workspace_unavailable",
            "Remote workspace broker is not configured.",
            phase="connect",
        )
    return service


def finish_remote_task(subject: Any, task_id: str) -> bool:
    """End one SSH task lease and drop the task's Home import staging.

    Two halves, in this order. The LEASE is the broker's: once it is released the
    target may reap the task's process groups. The STAGING directory is Home's, and
    it belongs to the transfer service rather than to the session broker — the
    broker owns remote sessions, not Home disk. Cleanup runs even when the lease
    release fails, because leftover pre-publication temp files are not evidence and
    keeping them would make a failed release also a disk leak.
    """

    from ouroboros.remote_transfer import discard_task_import_staging
    from ouroboros.workspace_ref import workspace_ref_for

    ref = workspace_ref_for(subject)
    if ref is None or ref.kind != "ssh":
        return False
    task = _opaque(task_id, "task_id")
    try:
        released = bool(
            get_remote_workspace_service().finish_task(ref.to_payload(), task_id=task)
        )
    finally:
        drive_root = (
            subject.get("drive_root")
            if isinstance(subject, Mapping)
            else getattr(subject, "drive_root", "")
        )
        discard_task_import_staging(drive_root, task)
    return released


class RemoteSessionBroker:
    """Server-generation owner of all OpenSSH sessions and worker proxies."""

    def __init__(
        self,
        drive_root: pathlib.Path,
        server_generation: str,
        capability_manifest: Mapping[str, Any],
        *,
        transport_factory: RemoteTransportFactory | None = None,
        bundle_dir: pathlib.Path | None = None,
        ssh_binary: str | None = None,
        home_importer: Any = None,
        pending_recovery: Callable[["RemoteSessionBroker"], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.drive_root = pathlib.Path(drive_root).resolve(strict=False)
        self.server_generation = _opaque(server_generation, "server_generation")
        # Public model schemas may legitimately contain JSON numbers such as
        # ``5.0`` defaults.  They are hashed on Home, but never cross the execd
        # wire.  Canonicalize only the integer/string proof that is uploaded.
        self.capability_projection = _capability_projection(capability_manifest)
        self.bundle_dir = pathlib.Path(bundle_dir).resolve(strict=False) if bundle_dir is not None else None
        self.ssh_binary = str(ssh_binary or "").strip() or None
        # The two Home seams, injected once here and never looked up: the
        # transport carries the importer, and recovery is a hook. Nothing in the
        # transport/broker path imports a Home authority to obtain either.
        #
        # The DEFAULT importer is constructed here, at the top of the graph, rather
        # than reached for from below — a function-local import of the transfer
        # service inside the transport would be exactly the upward arrow that forced
        # the donor to mirror Home guards inside its remote path. An explicit
        # `home_importer=` still wins, which is what lets a test drive the boundary
        # without a Home authority behind it.
        if home_importer is None or pending_recovery is None:
            from ouroboros.remote_transfer import (
                RemoteTransferService,
                recover_pending_scopes,
            )

            home_importer = home_importer or RemoteTransferService()
            pending_recovery = pending_recovery or recover_pending_scopes
        self._home_importer = home_importer
        self._pending_recovery = pending_recovery
        self._transport_factory = transport_factory or OpenSSHExecdTransport
        from ouroboros.remote_browser_forward import SSHBrowserForwardManager

        self._browser_forwards = SSHBrowserForwardManager(
            self.drive_root,
            ssh_binary=self.ssh_binary or "ssh",
        )
        self._requests: queue.PriorityQueue[_BrokerRequest] = queue.PriorityQueue(maxsize=_PIPE_QUEUE_LIMIT)
        self._request_sequence = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._sessions: dict[tuple[str, str, str, str], _Session] = {}
        self._task_sessions: dict[str, tuple[str, str, str, str]] = {}
        self._service_leases = RemoteServiceLeaseBook()
        self._worker_channels = WorkerChannels(self.server_generation)
        self._admission_cancels: dict[str, threading.Event] = {}
        self._admission_transports: dict[str, tuple[RemoteTransport, bool]] = {}
        # The PRE-QUEUE admission window, by admission identity -> connection. A task
        # being admitted right now holds no task session and no service lease yet, so
        # without this an owner retire/retrust between "admission started" and
        # "session bound" would read the connection as idle and pull it out from under
        # a task that is already committed to it. It lives here because the broker is
        # the only thing that knows an admission is in flight; a second registry
        # elsewhere would be a second answer to "is this connection busy".
        self._admitting: dict[str, str] = {}
        # The PANIC CUSTODY registers. Panic reads them as a snapshot without waiting
        # for the ordinary broker lock, which is the property that makes panic
        # software-delay-free — see `panic()`.
        #
        # They are keyed by `id()` and pruned on the ORDINARY exit path, which is the
        # correction to their first shape. They were append-only lists whose single
        # reset was `_detach_after_fork_child`, so every transport ever minted — every
        # session, plus one per Test, Bootstrap and directory listing — and one
        # `threading.Event` per admission stayed reachable for the whole life of the
        # server generation, holding a dead `subprocess.Popen` and its stderr buffer
        # with them. Custody has to end when custody ends; a register that only grows
        # is not a custody register, it is a log.
        #
        # Keyed by `id()` rather than held in a list because that makes DISCARD a
        # single expression at each exit (no O(n) `remove`, no ValueError when a path
        # runs twice) and makes panic's de-duplication structural instead of a `seen`
        # set it maintained by hand. A key cannot be stale: the register holds a strong
        # reference, so the id stays reserved for exactly as long as the entry does.
        self._panic_transports: dict[int, RemoteTransport] = {}
        self._panic_events: dict[int, threading.Event] = {}
        # NOT the same shape, deliberately: this is idempotent per admission SCOPE
        # (`setdefault` on one connection × project × workspace triple), so it is
        # bounded by the owner's own set of scopes rather than by traffic through them.
        self._admission_key_locks: dict[tuple[str, str, str], threading.Lock] = {}
        self._state_lock = threading.RLock()
        self._io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=_BROKER_IO_WORKERS,
            thread_name_prefix="remote-broker-io",
        )
        self._inflight = threading.BoundedSemaphore(_BROKER_MAX_INFLIGHT)
        self._started = False
        _LIVE_BROKERS.add(self)

    def start(self) -> None:
        with self._state_lock:
            if self._started:
                return
            if self._stop.is_set():
                raise RemoteWorkspaceError(
                    "broker_closed",
                    "Remote workspace broker cannot restart after close.",
                    phase="connect",
                )
            self._started = True
            self._thread = threading.Thread(
                target=self._run,
                name=f"remote-session-broker-{self.server_generation[:12]}",
                daemon=True,
            )
            self._thread.start()

    def create_worker_pipe_proxy(
        self,
        owner: str = "",
    ) -> RemoteWorkspacePipeProxy:
        """Mint one worker channel, retiring any channel `owner` already had."""

        self.start()
        _endpoint, proxy = self._worker_channels.mint(owner)
        return proxy

    def close_worker_pipe_proxy(self, owner: str) -> bool:
        """Close exactly one worker's channel (a dead worker's, on respawn)."""

        return self._worker_channels.close_owner(owner)

    def close_worker_pipe_proxies(self) -> int:
        """Close every worker channel (pool teardown), keeping the broker alive."""

        return self._worker_channels.close_all()

    def recover(self) -> list[dict[str, Any]]:
        return list(self._submit("recover", {}, priority=5))

    def recover_scope(
        self,
        *,
        connection_id: str,
        project_id: str,
        workspace_id: str,
    ) -> list[dict[str, Any]]:
        """Walk ONE already-reopened scope's operation ledger to a conclusion.

        Split out from `recover` because the Home recovery hook decides scope by
        scope whether reopening is even allowed (a retired connection is not), and
        it needs to reconcile exactly the scope it just admitted — not everything the
        broker happens to hold.
        """
        return list(
            self._submit(
                "recover_scope",
                {
                    "connection_id": _opaque(connection_id, "connection_id"),
                    "project_id": _opaque(project_id, "project_id"),
                    "workspace_id": _opaque(workspace_id, "workspace_id"),
                },
                priority=5,
            )
        )

    def status(self, connection_id: str | None = None) -> dict[str, Any]:
        with self._state_lock:
            rows = []
            for key, session in self._sessions.items():
                if connection_id is not None and key[0] != connection_id:
                    continue
                health = getattr(session.transport, "health", None)
                if not callable(health):
                    transport_state = {"status": "ready", "phase": "ready"}
                else:
                    try:
                        observed = health()
                        transport_state = (
                            dict(observed)
                            if isinstance(observed, Mapping)
                            else {"status": "unknown", "phase": "connect"}
                        )
                    except Exception:
                        transport_state = {
                            "status": "disconnected",
                            "phase": "connect",
                        }
                rows.append(
                    {
                        "id": key[0],
                        "connection_id": key[0],
                        "project_id": key[1],
                        "workspace_id": key[2],
                        "server_generation": key[3],
                        **transport_state,
                        "opened_at_monotonic": session.opened_at,
                        "last_used_at_monotonic": session.last_used_at,
                        "active_task_count": sum(1 for task_key in self._task_sessions.values() if task_key == key),
                    }
                )
        return {"connections": rows}

    health = status

    def reconnect_connection(
        self,
        connection: Mapping[str, Any],
        *,
        timeout_sec: float = _DEFAULT_REQUEST_TIMEOUT_SEC,
    ) -> dict[str, Any]:
        row = _json_copy(connection, "connection")
        connection_id = _opaque(row.get("id"), "connection_id")
        try:
            return dict(
                self._submit(
                    "reconnect_connection",
                    {"connection": row, "timeout_sec": max(1.0, float(timeout_sec))},
                    priority=0,
                    timeout_sec=max(1.0, float(timeout_sec)) + 5.0,
                )
            )
        except Exception as exc:
            return _reconnect_failure(connection_id, exc)

    def test_connection(
        self,
        connection: Mapping[str, Any],
        *,
        timeout_sec: float = 10.0,
    ) -> dict[str, Any]:
        request = self._session_request(connection, "", "", "")
        transport = self._new_transport(request)
        try:
            probe = getattr(transport, "probe", None)
            return dict(probe(timeout_sec=timeout_sec) if callable(probe) else transport.handshake())
        finally:
            self._retire_transport(transport)

    def bootstrap(
        self,
        connection: Mapping[str, Any],
        *,
        timeout_sec: float = 30.0,
    ) -> dict[str, Any]:
        request = self._session_request(connection, "", "", "")
        transport = self._new_transport(request)
        try:
            bootstrap = getattr(transport, "bootstrap", None)
            if not callable(bootstrap):
                raise RemoteWorkspaceError(
                    "bootstrap_unsupported",
                    "Remote transport does not expose bootstrap.",
                    phase="bootstrap",
                )
            return dict(bootstrap(timeout_sec=timeout_sec))
        finally:
            self._retire_transport(transport)

    def list_directories(
        self,
        connection: Mapping[str, Any],
        *,
        remote_root: str = "",
        timeout_sec: float = 10.0,
    ) -> dict[str, Any]:
        request = self._session_request(connection, remote_root, "", "")
        transport = self._new_transport(request)
        try:
            list_directories = getattr(transport, "list_directories", None)
            if not callable(list_directories):
                raise RemoteWorkspaceError(
                    "directory_listing_unsupported",
                    "Remote transport does not expose directory listing.",
                    phase="connect",
                )
            return dict(list_directories(remote_root=remote_root, timeout_sec=timeout_sec))
        finally:
            self._retire_transport(transport)
    def admit_workspace(
        self,
        connection: Mapping[str, Any],
        *,
        remote_root: str,
        project_id: str,
        workspace_id: str = "",
        task_id: str = "",
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        """Open or reuse ONE project session and bind `task_id` to it.

        SESSION admission only. Home task-admission policy — attachment staging,
        acceptance evidence, the durable task record — belongs to
        `workspace_admission` and the transfer service, which call this and then
        do their own work. Two admission authorities is how the donor ended up
        mirroring Home guards inside the broker.
        """

        project_id = _opaque(project_id, "project_id")
        task_id = _optional_opaque(task_id, "task_id")
        owned_cancel = threading.Event()
        # Panic custody for the duration of THIS admission, and no longer. It used to
        # be an append with no matching removal anywhere on the ordinary path, so one
        # `threading.Event` per admission — every task and every task-less
        # verification probe — accumulated for the life of the server generation while
        # the sibling registers below were correctly cleaned in the same `finally`.
        self._panic_events[id(owned_cancel)] = owned_cancel
        # A distinct admission identity even for a task-less verification probe, so
        # concurrent admissions of the same connection cannot clear each other's
        # in-flight marker.
        admitting_key = task_id or f"probe:{uuid.uuid4().hex[:16]}"
        with self._state_lock:
            self._admitting[admitting_key] = str(connection.get("id") or "")
            if task_id:
                self._admission_cancels[task_id] = owned_cancel
        try:
            result = self._submit(
                "admit",
                {
                    "connection": dict(connection),
                    "remote_root": str(remote_root),
                    "project_id": project_id,
                    "workspace_id": workspace_id,
                    "task_id": task_id,
                    "cancel": owned_cancel,
                    "external_cancel": cancel_event,
                },
                priority=5,
                timeout_sec=float(get_ssh_timeout_sec("admission")),
            )
            return dict(result)
        except BaseException:
            owned_cancel.set()
            raise
        finally:
            self._panic_events.pop(id(owned_cancel), None)
            with self._state_lock:
                self._admitting.pop(admitting_key, None)
                if task_id:
                    self._admission_cancels.pop(task_id, None)
                    self._admission_transports.pop(task_id, None)
    def prepare(
        self,
        workspace_ref: Mapping[str, Any],
        *,
        request_id: str,
        operation_id: str,
        tool: str,
        args: Mapping[str, Any],
        blobs: Mapping[str, bytes] | None = None,
        deadline_ms: int | None = None,
        task_id: str = "",
        parent_task_id: str = "", project_id: str = "",
    ) -> PreparedRemoteCall:
        result = self._submit(
            "prepare",
            {
                "workspace_ref": dict(workspace_ref),
                "request_id": request_id,
                "operation_id": operation_id,
                "tool": tool,
                "args": dict(args),
                "blobs": dict(blobs or {}),
                "deadline_ms": deadline_ms,
                "task_id": task_id,
                "parent_task_id": parent_task_id, "project_id": project_id,
            },
            priority=10,
        )
        return _prepared_from_dict(result)
    def execute_prepared(
        self,
        workspace_ref: Mapping[str, Any],
        prepared: PreparedRemoteCall,
        *,
        canonical_args: Mapping[str, Any],
        task_id: str = "",
        timeout_sec: float | None = None,
        import_kind: str = "",
        import_context: Mapping[str, Any] | None = None,
    ) -> ToolExecutionEnvelope:
        return _envelope_from_dict(
            self._submit(
                "execute_prepared",
                {
                    "workspace_ref": dict(workspace_ref),
                    "prepared": dataclasses.asdict(prepared),
                    "canonical_args": dict(canonical_args),
                    "task_id": task_id,
                    "timeout_sec": timeout_sec,
                    "import_kind": str(import_kind or ""),
                    "import_context": dict(import_context or {}),
                },
                priority=10,
                timeout_sec=_execution_wait_timeout(canonical_args, timeout_sec),
            )
        )
    def abort_prepared(
        self,
        workspace_ref: Mapping[str, Any],
        prepared: PreparedRemoteCall,
        *,
        task_id: str = "",
        reason: str = "denied",
    ) -> bool:
        return bool(
            self._submit(
                "abort_prepared",
                {
                    "workspace_ref": dict(workspace_ref),
                    "prepared": dataclasses.asdict(prepared),
                    "task_id": task_id,
                    "reason": str(reason)[:1000],
                },
                priority=0,
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
        return bytes(
            self._submit(
                "fetch_blob",
                {
                    "workspace_ref": dict(workspace_ref),
                    "blob_id": blob_id,
                    "max_bytes": int(max_bytes),
                    "task_id": task_id,
                },
                priority=20,
            )
        )
    def open_browser_forward(
        self,
        workspace_ref: Mapping[str, Any],
        *,
        remote_port: int,
        task_id: str,
    ) -> dict[str, Any]:
        session = self._session_for_ref(workspace_ref, task_id=_opaque(task_id, "task_id"))
        return dataclasses.asdict(
            self._browser_forwards.open(
                session.connection,
                remote_port=int(remote_port),
                task_id=task_id,
            )
        )
    def close_browser_forward(self, forward_id: str) -> bool:
        return self._browser_forwards.close(str(forward_id))
    def cancel(
        self,
        workspace_ref: Mapping[str, Any],
        *,
        task_id: str = "",
        request_id: str = "",
        operation_id: str = "",
    ) -> bool:
        if not task_id and not (request_id and operation_id):
            raise ValueError("cancel requires task_id or request_id+operation_id")
        # Cancellation must not sit behind a blocked ordinary request on the
        # broker queue.  Transport implementations provide an independent
        # control writer and must kill the selected group before ACK.
        session = self._session_for_ref(workspace_ref, task_id=task_id)
        cancelled = bool(
            session.transport.cancel(
                {
                    "task_id": _optional_opaque(task_id, "task_id"),
                    "request_id": _optional_opaque(request_id, "request_id"),
                    "operation_id": _optional_opaque(operation_id, "operation_id"),
                }
            )
        )
        if task_id:
            with self._state_lock:
                self._task_sessions.pop(task_id, None)
            self._browser_forwards.close_task(task_id)
        return cancelled
    def cancel_admission(self, task_id: str) -> bool:
        task_id = _opaque(task_id, "task_id")
        with self._state_lock:
            event = self._admission_cancels.get(task_id)
        if event is None:
            return False
        event.set()
        with self._state_lock:
            ownership = self._admission_transports.get(task_id)
        if ownership is not None:
            transport, exclusive = ownership
            try:
                if exclusive:
                    self._retire_transport(transport)
                else:
                    transport.cancel({"task_id": task_id, "request_id": "", "operation_id": ""})
            except Exception:
                pass
        self._browser_forwards.close_task(task_id)
        return True
    def finish_task(
        self,
        workspace_ref: Mapping[str, Any],
        *,
        task_id: str,
    ) -> bool:
        """Idempotently end a task lease while preserving its project session."""

        task_id = _opaque(task_id, "task_id")
        ref = normalize_workspace_ref(dict(workspace_ref))
        if ref is None or ref.kind != "ssh":
            raise ValueError("finish_task requires an SSH workspace ref")
        with self._state_lock:
            key = self._task_sessions.get(task_id)
            if key is not None and (key[0], key[2]) != (
                ref.connection_id,
                ref.workspace_id,
            ):
                raise RemoteWorkspaceError(
                    "task_session_mismatch",
                    "Task completion refers to another remote workspace.",
                    phase="authorize",
                )
            session = self._sessions.get(key) if key is not None else None
        if session is None:
            return False
        try:
            return bool(
                session.transport.cancel(
                    {"task_id": task_id, "request_id": "", "operation_id": ""}
                )
            )
        finally:
            task_lease = getattr(session.transport, "task_lease", None)
            if callable(task_lease):
                task_lease(task_id, forget=True)
            with self._state_lock:
                if self._task_sessions.get(task_id) == key:
                    self._task_sessions.pop(task_id, None)
            self._browser_forwards.close_task(task_id)
    def close_project_session(
        self,
        workspace_ref: Mapping[str, Any],
        *,
        project_id: str,
    ) -> bool:
        """Close only the exact project/workspace admission session."""

        payload = {
            "workspace_ref": dict(workspace_ref),
            "project_id": _opaque(project_id, "project_id"),
        }
        return bool(self._submit("close_project_session", payload, priority=0))
    def cancel_connection(self, connection_id: str) -> int:
        return int(
            self._submit(
                "cancel_connection",
                {"connection_id": _opaque(connection_id, "connection_id")},
                priority=0,
            )
        )
    def has_active_lease(self, connection_id: str) -> bool:
        """Whether an owner mutation would pull this connection out from under work.

        Three states count, and the third is the one a naive check misses: a bound
        task session, an active service lease, and an admission that is IN FLIGHT —
        already committed to this connection but not yet holding either.
        """
        connection_id = _opaque(connection_id, "connection_id")
        with self._state_lock:
            has_task = any(
                key[0] == connection_id for key in self._task_sessions.values()
            ) or connection_id in set(self._admitting.values())
        if has_task:
            return True
        refresh_connection_leases(
            self._service_leases,
            connection_id,
            session_for_key=self._session_by_key,
        )
        return self._service_leases.active_for_connection(connection_id)
    def panic(self) -> None:
        self._stop.set()
        panic_forwards = getattr(self._browser_forwards, "panic_close_all", None)
        if callable(panic_forwards):
            panic_forwards()
        # ONE atomic snapshot each, and no lock: `tuple(dict.values())` completes inside
        # a single C call, so a concurrent register or discard cannot be observed
        # half-done and panic still never waits on anything. De-duplication is
        # structural — the register is keyed by identity, so a transport reached
        # through two paths appears once.
        transports = tuple(self._panic_transports.values())
        admission_events = tuple(self._panic_events.values())
        if self._state_lock.acquire(blocking=False):
            try:
                self._sessions.clear()
                self._task_sessions.clear()
                self._service_leases = RemoteServiceLeaseBook()
                self._admission_transports.clear()
                self._admission_cancels.clear()
                self._admitting.clear()
            finally:
                self._state_lock.release()
        for event in admission_events:
            event.set()
        for transport in transports:
            try:
                transport.panic()
            except Exception:
                pass
        # Panic is terminal for this broker, so the custody registers are released with
        # everything else. Cleared AFTER the loop above, never before: a register
        # emptied first would be a panic that reached nothing.
        self._panic_transports.clear()
        self._panic_events.clear()
    @classmethod
    def panic_close_all(cls) -> None:
        for broker in list(_LIVE_BROKERS):
            try:
                broker.panic()
            except Exception:
                pass

    def close(self, timeout_sec: float | None = None) -> None:
        if self._stop.is_set() and not self._started:
            return
        self._stop.set()
        self._browser_forwards.close_all()
        self.panic()
        self.close_worker_pipe_proxies()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, float(timeout_sec or 0.0)))
        self._started = False
        self._io_executor.shutdown(wait=False, cancel_futures=True)
        _LIVE_BROKERS.discard(self)
    def _detach_after_fork_child(self) -> None:
        """Drop inherited broker/SSH descriptors without signalling the parent."""

        self._stop.set()
        self._worker_channels.detach_after_fork()
        for transport in tuple(self._panic_transports.values()):
            detach = getattr(transport, "detach_after_fork", None)
            if callable(detach):
                try:
                    detach()
                except Exception:
                    pass
        self._sessions = {}
        self._task_sessions = {}
        self._service_leases.clear()
        self._admission_transports = {}
        self._admitting = {}
        self._panic_transports = {}
        self._panic_events = {}
        self._thread = None
        self._started = False
    def _submit(
        self,
        method: str,
        payload: dict[str, Any],
        *,
        priority: int,
        timeout_sec: float = _DEFAULT_REQUEST_TIMEOUT_SEC,
    ) -> Any:
        self.start()
        if self._stop.is_set():
            raise RemoteWorkspaceError(
                "broker_closed",
                "Remote workspace broker is closed.",
                phase="stream",
            )
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        with self._state_lock:
            self._request_sequence += 1
            sequence = self._request_sequence
        try:
            self._requests.put_nowait(_BrokerRequest(priority, sequence, method, payload, future))
        except queue.Full as exc:
            raise RemoteWorkspaceError(
                "broker_overloaded",
                "Remote workspace broker queue is full.",
                phase="stream",
                retryable=True,
            ) from exc
        try:
            return future.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError as exc:
            raise RemoteWorkspaceError(
                "remote_request_timeout",
                "Remote workspace request exceeded its Home deadline.",
                phase="stream",
                completion="unknown",
                retryable=True,
            ) from exc
    def _new_transport(self, request: SessionOpenRequest) -> RemoteTransport:
        try:
            transport = self._transport_factory(
                request, home_importer=self._home_importer
            )
        except TypeError:
            # A test double or an alternative transport may not take the seam.
            transport = self._transport_factory(request)
        # Custody BEGINS here. It ends at `_retire_transport`, which every ordinary
        # close goes through, or at `panic()`.
        self._panic_transports[id(transport)] = transport
        if self._stop.is_set():
            self.panic()
            raise RemoteWorkspaceError("broker_closed", "Remote workspace broker is closed.", phase="stream")
        return transport
    def _retire_transport(self, transport: RemoteTransport) -> None:
        """Close a transport AND end its panic custody — the ONE ordinary exit.

        It is one door rather than a `pop` beside each `transport.close()` for the
        reason the register existed in its broken form for as long as it did: there
        are six close sites (a session's, a connection-wide cancel's, a cancelled
        admission's, and the three probe-shaped calls Test/Bootstrap/list-directories
        make), and the next one added would have had to remember. Discard FIRST, so a
        `close()` that raises still ends custody: the transport is condemned either
        way, and a panic that later reaches a half-closed child would try to kill
        streams whose owner already released them.
        """

        self._panic_transports.pop(id(transport), None)
        transport.close()
    def _run(self) -> None:
        while not self._stop.is_set():
            self._poll_worker_endpoints()
            try:
                request = self._requests.get(timeout=_BROKER_POLL_SEC)
            except queue.Empty:
                continue
            if request.future.cancelled():
                continue
            if not self._inflight.acquire(blocking=False):
                request.future.set_exception(
                    RemoteWorkspaceError(
                        "broker_overloaded",
                        "Remote workspace broker has too many in-flight requests.",
                        phase="stream",
                        retryable=True,
                    )
                )
                continue
            submitted = self._io_executor.submit(
                self._dispatch,
                request.method,
                request.payload,
            )
            submitted.add_done_callback(
                lambda completed, target=request.future: self._complete_request(target, completed)
            )
    def _poll_worker_endpoints(self) -> None:
        # The READ side takes no lock, and the reason is CONFINEMENT rather than
        # symmetry: this method runs only on the single broker thread (`_run`), so
        # there is exactly one reader per endpoint. Said here because the write side
        # DOES lock, and an asymmetry nobody explained is how an unlocked
        # `broker_overloaded` write came to look acceptable — a second poller would
        # desync a channel silently, so one would have to bring a read lock with it.
        endpoints = self._worker_channels.live()
        dead: list[Connection] = []
        for endpoint in endpoints:
            try:
                if not endpoint.poll(0):
                    continue
                message = endpoint.recv()
                if not self._inflight.acquire(blocking=False):
                    self._send_to_worker(
                        endpoint,
                        {
                            "correlation_id": (
                                str(message.get("correlation_id") or "") if isinstance(message, dict) else ""
                            ),
                            "ok": False,
                            "error": _error_dict(
                                RemoteWorkspaceError(
                                    "broker_overloaded",
                                    "Remote broker has too many in-flight requests.",
                                    phase="stream",
                                    retryable=True,
                                )
                            ),
                        },
                    )
                    continue
                submitted = self._io_executor.submit(
                    self._dispatch_pipe_message,
                    message,
                )
                submitted.add_done_callback(lambda completed, target=endpoint: self._complete_pipe(target, completed))
            except (EOFError, OSError):
                dead.append(endpoint)
        if dead:
            self._worker_channels.drop(dead)
    def _complete_request(
        self,
        target: concurrent.futures.Future[Any],
        completed: concurrent.futures.Future[Any],
    ) -> None:
        self._inflight.release()
        if target.cancelled():
            return
        try:
            target.set_result(completed.result())
        except BaseException as exc:
            target.set_exception(exc)
    def _complete_pipe(
        self,
        endpoint: Connection,
        completed: concurrent.futures.Future[dict[str, Any]],
    ) -> None:
        self._inflight.release()
        try:
            response = completed.result()
        except BaseException as exc:
            response = {
                "correlation_id": "",
                "ok": False,
                "error": _error_dict(exc),
            }
        self._send_to_worker(endpoint, response)
    def _send_to_worker(self, endpoint: Connection, response: dict[str, Any]) -> None:
        """The ONE write to a worker channel: every response goes through ITS lock.

        Two writers share each broker endpoint — the broker thread, from
        `_poll_worker_endpoints`, and the `remote-broker-io` pool threads, from
        `_complete_pipe`. A `multiprocessing.Connection` emits the 4-byte length
        header and the payload as TWO writes once the payload passes 16 KiB, which a
        `prepared` or a `result` response routinely does, so an unlocked write lands
        BETWEEN another frame's header and its body. The worker's next `recv()` then
        reads a length that no longer matches its bytes — and the channel is durable
        per worker, so that desyncs the worker for the rest of its life rather than
        losing one request.

        The `broker_overloaded` reply was the one unlocked write, and being SMALL is
        what made it dangerous: it fits entirely inside the gap in a large frame.

        A missing lock means the endpoint was retired between the poll and here
        (`WorkerChannels.drop`/`close_owner`/`mint`), and each of those closes the
        endpoint, so the worker's own side is already EOF and gets a typed
        `broker_pipe_closed` rather than waiting: there is nothing to answer into.
        """

        lock = self._worker_channels.send_lock(endpoint)
        if lock is None:
            return
        try:
            with lock:
                endpoint.send(response)
        except (EOFError, OSError):
            return
    def _dispatch_pipe_message(self, message: Any) -> dict[str, Any]:
        correlation_id = str(message.get("correlation_id") or "") if isinstance(message, dict) else ""
        try:
            if not isinstance(message, dict):
                raise ValueError("worker broker message must be an object")
            claimed_generation = str(message.get("server_generation") or "")
            if claimed_generation and claimed_generation != self.server_generation:
                # A proxy that outlived its generation (inherited across a
                # restart, or a replaced worker) is REFUSED, not served and not
                # left waiting: serving it would let a dead generation touch
                # live sessions, and hanging would stall the worker for its
                # whole deadline.
                raise RemoteWorkspaceError(
                    BROKER_GENERATION_STALE,
                    "Remote broker generation no longer matches this worker channel.",
                    phase="authorize",
                    completion="not_started",
                    details={"expected_generation": self.server_generation},
                )
            method = str(message.get("method") or "")
            payload = message.get("payload")
            if not isinstance(payload, dict):
                raise ValueError("worker broker payload must be an object")
            result = self._dispatch(method, payload)
            if isinstance(result, (PreparedRemoteCall, ToolExecutionEnvelope)):
                result = dataclasses.asdict(result)
            return {"correlation_id": correlation_id, "ok": True, "result": result}
        except Exception as exc:
            return {
                "correlation_id": correlation_id,
                "ok": False,
                "error": _error_dict(exc),
            }
    def _dispatch(self, method: str, payload: dict[str, Any]) -> Any:
        handlers: dict[str, Callable[[dict[str, Any]], Any]] = {
            "prepare": self._prepare_on_broker,
            "execute_prepared": self._execute_on_broker,
            "abort_prepared": self._abort_on_broker,
            "fetch_blob": self._fetch_blob_on_broker,
            "cancel": self._cancel_on_broker,
            "cancel_connection": self._cancel_connection_on_broker,
            "close_project_session": self._close_project_session_on_broker,
            "recover": self._recover_on_broker,
            "recover_scope": self._recover_scope_on_broker,
            "reconnect_connection": self._reconnect_connection_on_broker,
            "admit": self._admit_on_broker,
            "open_browser_forward": self._open_browser_forward_on_broker,
            "close_browser_forward": self._close_browser_forward_on_broker,
        }
        handler = handlers.get(method)
        if handler is None:
            raise ValueError(f"unsupported broker method: {method}")
        return handler(payload)
    def _open_browser_forward_on_broker(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.open_browser_forward(
            payload["workspace_ref"],
            remote_port=int(payload["remote_port"]),
            task_id=_opaque(payload["task_id"], "task_id"),
        )

    def _close_browser_forward_on_broker(self, payload: dict[str, Any]) -> bool:
        return self.close_browser_forward(str(payload.get("forward_id") or ""))
    def _session_request(
        self,
        connection: Mapping[str, Any],
        remote_root: str,
        project_id: str,
        workspace_id: str,
    ) -> SessionOpenRequest:
        return session_request(self, connection, remote_root, project_id, workspace_id)

    def _admit_on_broker(self, payload: dict[str, Any]) -> dict[str, Any]:
        return admit_on_broker(self, payload)

    def _session_for_ref(
        self,
        workspace_ref: Mapping[str, Any],
        *,
        task_id: str = "",
        parent_task_id: str = "",
        project_id: str = "",
    ) -> _Session:
        return session_for_ref(
            self,
            workspace_ref,
            task_id=task_id,
            parent_task_id=parent_task_id,
            project_id=project_id,
        )

    def _prepare_on_broker(self, payload: dict[str, Any]) -> dict[str, Any]:
        task_id = _optional_opaque(payload.get("task_id"), "task_id")
        parent_task_id = _optional_opaque(payload.get("parent_task_id"), "parent_task_id")
        project_id = _optional_opaque(payload.get("project_id"), "project_id")
        session = self._session_for_ref(
            payload["workspace_ref"], task_id=task_id,
            parent_task_id=parent_task_id, project_id=project_id,
        )
        request_id = _opaque(payload.get("request_id"), "request_id")
        operation_id = _opaque(payload.get("operation_id"), "operation_id")
        tool = str(payload.get("tool") or "")
        args = _json_copy(payload.get("args"), "args")
        blobs = payload.get("blobs") if isinstance(payload.get("blobs"), dict) else {}
        bounded_blobs = {
            _opaque(blob_id, "blob_id"): bytes(value)
            for blob_id, value in blobs.items()
            if isinstance(value, (bytes, bytearray, memoryview))
        }
        response = session.transport.prepare(
            {
                "request_id": request_id,
                "operation_id": operation_id,
                "tool": tool,
                "args": args,
                "task_id": task_id,
                "workspace_id": session.key[2],
                "deadline_ms": payload.get("deadline_ms"),
            },
            bounded_blobs,
        )
        return _validated_prepared(response)

    def _execute_on_broker(self, payload: dict[str, Any]) -> dict[str, Any]:
        task_id = _optional_opaque(payload.get("task_id"), "task_id")
        session = self._session_for_ref(payload["workspace_ref"], task_id=task_id)
        prepared = _prepared_from_dict(payload.get("prepared"))
        canonical_args = _json_copy(payload.get("canonical_args"), "canonical_args")
        if canonical_json(canonical_args) != canonical_json(prepared.execution_args):
            raise RemoteWorkspaceError(
                "prepared_arguments_mismatch",
                "Home authorization does not match target-prepared arguments.",
                phase="authorize",
            )
        response_timeout = _execution_wait_timeout(
            canonical_args,
            payload.get("timeout_sec"),
        )

        response = _validated_envelope_dict(
            session.transport.execute_prepared(
                {
                    "request_id": prepared.request_id,
                    "operation_id": prepared.operation_id,
                    "prepared_hash": prepared.prepared_hash,
                    "prepared_token": prepared.prepared_token,
                    "task_id": task_id,
                    # The caller's declared channel, defaulting to the ordinary tool
                    # result. The broker does not choose it and does not read it: the
                    # closed registry is checked where the intent is persisted
                    # (`remote_pending_operations`), so an undeclared channel fails
                    # there rather than being silently accepted as a tool result.
                    "_home_import_kind": str(payload.get("import_kind") or "") or "task_result_v1",
                    # The EXACT document the target was prepared with, so the Home
                    # import validates the returned manifest against the policy that
                    # actually ran rather than one recomputed after the fact (§3.2).
                    # It rides in the import context because the broker is transport:
                    # it forwards the document, it never reads it or decides it.
                    "_home_import_context": {
                        "export_policy": dict(prepared.native_facts.get("export_policy") or {}),
                        **(
                            dict(payload["import_context"])
                            if isinstance(payload.get("import_context"), Mapping)
                            else {}
                        ),
                    },
                    "_response_timeout_sec": response_timeout,
                }
            )
        )
        self._service_leases.observe(session.key, prepared, response, task_id=task_id)
        return response

    def _abort_on_broker(self, payload: dict[str, Any]) -> bool:
        task_id = _optional_opaque(payload.get("task_id"), "task_id")
        session = self._session_for_ref(payload["workspace_ref"], task_id=task_id)
        prepared = _prepared_from_dict(payload.get("prepared"))
        return bool(
            session.transport.abort_prepared(
                {
                    "request_id": prepared.request_id,
                    "operation_id": prepared.operation_id,
                    "prepared_hash": prepared.prepared_hash,
                    "prepared_token": prepared.prepared_token,
                    "reason": str(payload.get("reason") or "denied")[:1000],
                }
            )
        )

    def _fetch_blob_on_broker(self, payload: dict[str, Any]) -> bytes:
        session = self._session_for_ref(
            payload["workspace_ref"],
            task_id=_opaque(payload.get("task_id"), "task_id"),
        )
        blob_id = _opaque(payload.get("blob_id"), "blob_id")
        max_bytes = int(payload.get("max_bytes") or 0)
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        return session.transport.fetch_blob(blob_id, max_bytes)

    def _cancel_on_broker(self, payload: dict[str, Any]) -> bool:
        task_id = _optional_opaque(payload.get("task_id"), "task_id")
        session = self._session_for_ref(payload["workspace_ref"], task_id=task_id)
        cancelled = bool(
            session.transport.cancel(
                {
                    "task_id": task_id,
                    "request_id": _optional_opaque(payload.get("request_id"), "request_id"),
                    "operation_id": _optional_opaque(payload.get("operation_id"), "operation_id"),
                }
            )
        )
        if task_id:
            with self._state_lock:
                self._task_sessions.pop(task_id, None)
            self._browser_forwards.close_task(task_id)
        return cancelled

    def _cancel_connection_on_broker(self, payload: dict[str, Any]) -> int:
        connection_id = _opaque(payload.get("connection_id"), "connection_id")
        with self._state_lock:
            victims = [(key, session) for key, session in self._sessions.items() if key[0] == connection_id]
            for key, _session in victims:
                self._sessions.pop(key, None)
                for task_id, task_key in list(self._task_sessions.items()):
                    if task_key == key:
                        self._task_sessions.pop(task_id, None)
                self._service_leases.discard_session(key)
        for _key, session in victims:
            try:
                session.transport.cancel({"task_id": "", "request_id": "", "operation_id": ""})
            except Exception:
                pass
            self._retire_transport(session.transport)
        self._browser_forwards.close_connection(connection_id)
        return len(victims)

    def _close_project_session_on_broker(
        self,
        payload: dict[str, Any],
    ) -> bool:
        ref = normalize_workspace_ref(dict(payload.get("workspace_ref") or {}))
        if ref is None or ref.kind != "ssh":
            raise ValueError("close_project_session requires an SSH workspace ref")
        key = (
            str(ref.connection_id),
            _opaque(payload.get("project_id"), "project_id"),
            str(ref.workspace_id),
            self.server_generation,
        )
        with self._state_lock:
            session = self._sessions.pop(key, None)
            task_ids = [task_id for task_id, task_key
                        in self._task_sessions.items() if task_key == key]
            for task_id in task_ids:
                self._task_sessions.pop(task_id, None)
            self._service_leases.discard_session(key)
        if session is None:
            return False
        for task_id in task_ids:
            task_lease = getattr(session.transport, "task_lease", None)
            if callable(task_lease):
                task_lease(task_id, forget=True)
            self._browser_forwards.close_task(task_id)
        self._retire_transport(session.transport)
        return True

    def _session_by_key(self, key: tuple[str, str, str, str]) -> _Session | None:
        with self._state_lock:
            return self._sessions.get(key)

    def _recover_scope_on_broker(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Reconcile the session the recovery hook just reopened for this scope."""

        key = (
            str(payload["connection_id"]),
            str(payload["project_id"]),
            str(payload["workspace_id"]),
            self.server_generation,
        )
        session = self._session_by_key(key)
        if session is None:
            raise RemoteWorkspaceError(
                "remote_session_disconnected",
                "The scope was not reopened before reconciliation.",
                phase="finalize",
                completion="unknown",
                retryable=True,
            )
        reconcile = getattr(session.transport, "reconcile", None)
        if not callable(reconcile):
            raise RemoteWorkspaceError(
                "reconnect_unsupported",
                "Remote transport does not expose reconciliation.",
                phase="finalize",
            )
        return list(reconcile())

    def _recover_on_broker(self, _payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Reconcile durable pending scopes through the injected recovery hook.

        Reopening a scope needs the connection store and the project registry to
        prove the scope still means what it meant — Home authorities the broker does
        not import. So recovery is INJECTED, and the hook decides scope by scope
        whether reopening is even allowed. The hook reports every scope it did NOT
        reconcile, with a reason: an empty list would read as "all clean" for
        operations that are still outstanding.
        """

        return list(self._pending_recovery(self))

    def _reconnect_connection_on_broker(self, payload: dict[str, Any]) -> dict[str, Any]:
        connection = _json_copy(payload.get("connection"), "connection")
        connection_id = _opaque(connection.get("id"), "connection_id")
        timeout_sec = max(1.0, float(payload.get("timeout_sec") or 0))
        with self._state_lock:
            sessions = [session for key, session in self._sessions.items() if key[0] == connection_id]
        if not sessions:
            return _reconnect_failure(connection_id)
        recovered: list[dict[str, Any]] = []
        reconciliation: list[dict[str, Any]] = []
        for session in sessions:
            reconnect = getattr(session.transport, "reconnect", None)
            if not callable(reconnect):
                raise RemoteWorkspaceError(
                    "reconnect_unsupported",
                    "Remote transport does not support reconnect.",
                    phase="connect",
                )
            row = dict(reconnect(timeout_sec=timeout_sec))
            facts = row.get("handshake")
            if isinstance(facts, dict):
                prior = session.handshake
                stable = ("host_id", "workspace_id", "canonical_root", "capability_hash")
                if any(facts.get(field) != prior.get(field) for field in stable):
                    raise RemoteWorkspaceError(
                        "reconnect_identity_mismatch",
                        "Reconnected execd session changed its admitted identity.",
                        phase="bootstrap",
                    )
                session.handshake = dict(facts)
            recovered.append({"workspace_id": session.key[2], **row})
            reconciliation.extend(item for item in list(row.get("reconciliation") or []) if isinstance(item, dict))
        return {
            "status": "ready",
            "phase": "ready",
            "completion": "completed",
            "error_code": "",
            "action": "",
            "diagnostic": "",
            "log_refs": [],
            "connection_id": connection_id,
            "sessions": recovered,
            "reconciliation": reconciliation,
        }

RemoteWorkspaceBroker = RemoteSessionBroker


def _after_fork_child() -> None:
    global _REMOTE_WORKSPACE_SERVICE
    for broker in list(_LIVE_BROKERS):
        broker._detach_after_fork_child()
    _REMOTE_WORKSPACE_SERVICE = None
    _LIVE_BROKERS.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_child)
