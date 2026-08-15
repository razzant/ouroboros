"""Session transport: one framed execd session over a supervised OpenSSH child.

Session-transport ONLY (RWS v2 §3.2/§5).  The module moves frames, owns the
OpenSSH child and its threads, and knows nothing about Home policy:

* it does not import ``artifacts``, ``observability``, ``remote_transfer`` or any
  Home policy authority — at module level or otherwise.  The single Home
  interaction is ``self.home_importer``, the ``HomeImporter`` protocol injected
  at construction, reached only through :mod:`ouroboros.remote_reconciliation`.
  The postmortem's dominant failure was exactly the reverse arrow: a transport
  that imported Home to finish a result, so every Home guard had to be mirrored
  here;
* effective OpenSSH configuration validation lives in
  :mod:`ouroboros.remote_ssh_config` (mandated pre-split — the donor module was
  AT the 1600-line gate before adaptation);
* durability and reconciliation live in
  :mod:`ouroboros.remote_pending_operations` / :mod:`ouroboros.remote_reconciliation`.

Two structural properties are worth stating because tests pin them.  Control
traffic and bulk traffic use separate locks, so lease renewal, cancel and panic
stay live while a blob upload is backpressured.  And panic never waits: it
best-effort writes one non-blocking panic frame, then tears down its children
without awaiting any acknowledgement — the remote custodian owns the
unreachable case.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import logging
import os
import pathlib
import re
import shlex
import subprocess
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any

from ouroboros.platform_layer import (
    best_effort_nonblocking_pipe_write,
    kill_process_group_id,
    kill_process_tree,
)
from ouroboros.process_custody import spawn_supervised
from ouroboros.remote_contract_admission import admit_home_contract_set
from ouroboros.remote_pending_operations import (
    bind_transport_intent,
    restore_transport_tracking,
    validate_transport_session_identity,
)
from ouroboros.remote_protocol import (
    MAX_BULK_BYTES,
    MAX_PREAMBLE_BYTES,
    PREAMBLE_MAGIC,
    PROTOCOL_MAJOR,
    PROTOCOL_MINOR,
    ControlSequence,
    ProtocolError,
    canonical_json,
    encode_bulk,
    encode_control,
    lease_answer_id,
    parse_session_preamble,
    read_frame,
)
from ouroboros.remote_reconciliation import (
    complete_transport_import,
    prefetch_remote_result_import,
    reconcile_remote_operations,
    remove_transport_pending,
)
from ouroboros.remote_ssh_bootstrap import (
    SelectedRelease,
    select_and_install,
)
from ouroboros.remote_ssh_bootstrap import (
    _validate_archive as _bootstrap_validate_archive,
)
from ouroboros.remote_ssh_config import (
    minimal_ssh_env,
    safe_text,
    transport_error,
    validated_ssh_base_command,
    validated_ssh_config,
)

_REMOTE_BASE = ".local/share/ouroboros/execd"
_SESSION_TIMEOUT_SEC = 120.0
# A protocol/safety contract, not configuration: the lost-lease ceiling is the
# physical failure-detection bound for a true partition, never a grace period.
_LEASE_TTL_MS = 15_000
_LEASE_RENEW_SEC = 5.0
_STDERR_LIMIT = 64 * 1024
_MAX_PENDING_MESSAGES = 512
_MAX_KNOWN_OPERATIONS = 512
# Owner-facing display bounds. Both are reported with an exact total beside the
# bounded list — a bare slice makes a shortened answer read as a complete one.
_MAX_DISCLOSED_DIRECTIVES = 4
_MAX_LISTED_DIRS = 1000
_ACK_TIMEOUT_SEC = 5.0
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_LOG = logging.getLogger(__name__)

# Compatibility export for existing bootstrap security tests.
_validate_archive = _bootstrap_validate_archive


def _release_child_streams(process: subprocess.Popen[bytes] | None) -> None:
    """Release a child's pipes WITHOUT flushing and WITHOUT stealing the fd.

    OPEN-6 (PR 79). The donor released these with ``os.close(stream.fileno())``,
    which closes the descriptor behind the back of the object that owns it: the
    ``FileIO`` ``subprocess`` created with ``closefd=True``.  That object still
    believes it is open, the OS immediately reuses the descriptor NUMBER for the
    next child, and when the stale object is finalized it closes a descriptor
    that now belongs to a LIVE process — after which closing that live process's
    own ``stdout`` raises ``OSError: [Errno 9] Bad file descriptor``.  That is
    exactly the CI symptom: the SSH probe *after* ``broker.panic()`` failed while
    Python closed ``subprocess.stdout``.

    Closing the RAW stream fixes both halves at once.  The raw object is the
    descriptor's owner, so nothing is closed twice and no live descriptor can be
    clobbered; and closing the raw does NOT flush the buffered wrapper, so a
    forked child never pushes bytes into the parent's SSH pipe and panic never
    waits on a flush.  With ``bufsize=0`` the stream IS the raw object, so the
    same call is correct there too.
    """

    if process is None:
        return
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is None:
            continue
        try:
            getattr(stream, "raw", stream).close()
        except (OSError, ValueError):
            pass


class OpenSSHExecdTransport:
    """One multiplexed framed execd session over a supervised OpenSSH child."""

    def __init__(self, request: Any, *, home_importer: Any = None) -> None:
        self.request = request
        # The ONE Home object this transport may touch, injected by the broker
        # (RWS v2 §3.2). It is not looked up, not imported and not defaulted to
        # a Home authority: a missing importer makes a completed result fail to
        # ACK, which keeps the remote evidence, rather than letting the
        # transport decide anything about Home state on its own.
        self.home_importer = home_importer
        self.alias = str(request.connection["ssh_alias"])
        self._ssh_base, resolved_config = validated_ssh_config(
            self.alias,
            request.ssh_binary,
            forwarding=False,
        )
        warning_directives = resolved_config.get(
            "_ouroboros_warning_directives",
            [],
        )
        # The count is exact even though the list is bounded: this warning tells the
        # owner which forwarding directives their alias declared, and "4 of them"
        # rendered as four with no note reads as "that was all of them".
        self._warnings = (
            [
                {
                    "code": "ssh_alias_forwarding_neutralized",
                    "directives": list(warning_directives[:_MAX_DISCLOSED_DIRECTIVES]),
                    "directives_total": len(warning_directives),
                    "directives_truncated": (
                        len(warning_directives) > _MAX_DISCLOSED_DIRECTIVES
                    ),
                }
            ]
            if warning_directives
            else []
        )
        self._process: subprocess.Popen[bytes] | None = None
        self._session_lock = threading.RLock()
        self._selected_release: SelectedRelease | None = None
        self._nonce = os.urandom(24)
        self._sequence = 0
        self._send_lock = threading.RLock()
        # Execd accepts control traffic while receiving a blob, but only one
        # prepare may own the manifest/bulk stream at a time.  Keep this lock
        # separate from _send_lock so cancel/panic/lease control stays live.
        self._upload_lock = threading.Lock()
        self._condition = threading.Condition()
        self._messages: deque[dict[str, Any]] = deque()
        self._receive_sequence = ControlSequence()
        self._reader_error: BaseException | None = None
        self._lease_refusal: dict[str, Any] = {}
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._lease_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._stderr = bytearray()
        self._stderr_carry = ""
        self._handshake: dict[str, Any] | None = None
        self._last_reconciliation: list[dict[str, Any]] = []
        self._active_tasks: set[str] = set()
        (
            self._known_operations,
            self._operation_contexts,
        ) = restore_transport_tracking(request)
        self._downloads: dict[str, dict[str, Any]] = {}
        self._download_current = ""
        # The blob Home walked away from while the target may still be streaming it.
        # Only one download runs at a time (`_download_lock`), so one name is enough.
        self._download_draining = ""
        self._download_lock = threading.Lock()
        self._helper_lock = threading.RLock()
        self._helper_process: subprocess.Popen[bytes] | None = None

    def probe(self, *, timeout_sec: float = 10.0) -> dict[str, Any]:
        facts = self._platform_probe(timeout_sec)
        host_id = self._installed_host_id(
            timeout_sec,
            required=False,
            selected=None,
        )
        return {
            "ok": True,
            "status": "ready",
            "phase": "connect",
            **({"host_id": host_id} if host_id else {}),
            **facts,
            "warnings": [
                dict(row) for row in getattr(self, "_warnings", [])
            ],
        }

    def bootstrap(self, *, timeout_sec: float = 30.0) -> dict[str, Any]:
        selected, result = select_and_install(
            self.request,
            run_remote=self._run_remote,
            platform_probe=self._platform_probe,
            timeout_sec=timeout_sec,
        )
        self._selected_release = selected
        self._upload_capability_manifest(timeout_sec)
        if not str(self.request.connection.get("expected_host_id") or ""):
            self._initialize_host_id(timeout_sec, selected)
        host_id = self._installed_host_id(
            timeout_sec,
            required=True,
            selected=selected,
        )
        return {
            **result,
            "host_id": host_id,
            "release_id": selected.release,
            "artifact_sha256": selected.archive_sha256,
            "artifact_size": selected.archive_size,
            "warnings": [
                dict(row) for row in getattr(self, "_warnings", [])
            ],
        }

    def list_directories(
        self,
        *,
        remote_root: str = "",
        timeout_sec: float = 10.0,
    ) -> dict[str, Any]:
        requested = str(remote_root or "")
        script = (
            'set -eu; p="${1:-$HOME}"; cd -- "$p"; root=$(pwd -P); '
            'printf "ROOT\\t%s\\n" "$root"; '
            'for d in ./*; do [ -d "$d" ] || continue; '
            'name=${d#./}; git=0; [ -e "$d/.git" ] && git=1; '
            'printf "DIR\\t%s\\t%s\\n" "$name" "$git"; done'
        )
        completed = self._run_remote(
            ["sh", "-c", script, "sh", requested],
            timeout_sec=timeout_sec,
        )
        root = ""
        rows: list[dict[str, Any]] = []
        for line in completed.stdout.decode("utf-8", errors="replace").splitlines():
            parts = line.split("\t")
            if len(parts) == 2 and parts[0] == "ROOT":
                root = parts[1]
            elif len(parts) == 3 and parts[0] == "DIR" and root:
                rows.append(
                    {
                        "name": parts[1],
                        "path": f"{root.rstrip('/')}/{parts[1]}",
                        "is_git": parts[2] == "1",
                    }
                )
        # A directory listing is BROWSED: an owner picking a remote project folder
        # who is shown 1000 entries with no note concludes the 1001st does not exist.
        # The bound stays, the exact count travels with it.
        #
        # `truncated` is spelled exactly that way because the owner-facing handler
        # (`gateway/connections.py::api_connection_dirs`) ALREADY reads
        # `result.get("truncated")` — the transport simply never set it, so that read
        # was dead and the handler fell back to re-deriving truncation from its own
        # 500-row slice. It could not see a cut this side had already made.
        return {
            "ok": True,
            "path": root,
            "parent": str(pathlib.PurePosixPath(root).parent) if root else "",
            "dirs": rows[:_MAX_LISTED_DIRS],
            "dirs_total": len(rows),
            "truncated": len(rows) > _MAX_LISTED_DIRS,
        }

    def handshake(self) -> dict[str, Any]:
        self._ensure_session()
        assert self._handshake is not None
        return dict(self._handshake)

    def artifact_identity(self) -> dict[str, Any]:
        selected = self._selected_release
        return {
            "release_id": selected.release if selected else "",
            "artifact_sha256": selected.archive_sha256 if selected else "",
            "artifact_size": selected.archive_size if selected else 0,
        }

    def prepare(
        self,
        message: Mapping[str, Any],
        blobs: Mapping[str, bytes],
    ) -> dict[str, Any]:
        self._ensure_session()
        request_id = str(message["request_id"])
        operation_id = str(message["operation_id"])
        task_id = str(message.get("task_id") or "")
        key = (request_id, operation_id)
        try:
            response_timeout = max(
                0.1,
                float(
                    message.get("_response_timeout_sec")
                    or _SESSION_TIMEOUT_SEC
                ),
            )
        except (TypeError, ValueError):
            response_timeout = _SESSION_TIMEOUT_SEC
        if (
            key not in self._known_operations
            and len(self._known_operations) >= _MAX_KNOWN_OPERATIONS
        ):
            raise transport_error(
                "operation_tracking_capacity",
                "Home pending-operation tracking capacity is exhausted.",
                phase="prepare",
                completion="not_started",
                retryable=True,
            )
        self._renew_lease(task_id)
        fields = {
            "request_id": request_id,
            "operation_id": operation_id,
            "tool": str(message["tool"]),
            "args": dict(message["args"]),
        }
        for name in ("task_id", "workspace_id", "deadline_ms"):
            if message.get(name) not in (None, ""):
                fields[name] = message[name]
        if not self._upload_lock.acquire(timeout=response_timeout):
            raise transport_error(
                "blob_upload_busy",
                "Another remote input upload did not finish in time.",
                phase="import",
                completion="not_started",
                retryable=True,
            )
        try:
            for blob_id, payload in blobs.items():
                self._upload_blob(
                    request_id,
                    operation_id,
                    str(blob_id),
                    bytes(payload),
                )
            # Keep prepare in the transaction: it consumes exactly the blobs
            # staged under this request/operation before the next uploader runs.
            self._send("prepare", **fields)
        finally:
            self._upload_lock.release()
        prepared_response = lambda row: (
                row.get("kind") in {"prepared", "diagnostic"}
                and row.get("request_id") == request_id
                and row.get("operation_id") == operation_id
            )
        if "_response_timeout_sec" in message:
            response = self._wait_control(
                prepared_response,
                timeout_sec=response_timeout,
            )
        else:
            response = self._wait_control(prepared_response)
        self._raise_diagnostic(response)
        prepared = response["prepared"]
        result = {
            "request_id": request_id,
            "operation_id": operation_id,
            "tool": prepared["tool"],
            "prepared_token": prepared["prepared_token"],
            "prepared_hash": response["prepared_hash"],
            "expires_at_ms": response["expires_ms"],
            "execution_args": prepared["execution_args"],
            "native_facts": prepared["native_facts"],
        }
        self._known_operations[key] = response["prepared_hash"]
        contexts = getattr(self, "_operation_contexts", None)
        if contexts is None:
            contexts = {}
            self._operation_contexts = contexts
        contexts[key] = {
            "task_id": task_id,
            "operation_id": operation_id,
            "tool": str(message["tool"]),
            "import_kind": "",
            "import_context": {},
            "validator": None,
            "pending_record": None,
        }
        return result

    def execute_prepared(self, message: Mapping[str, Any]) -> dict[str, Any]:
        self._ensure_session()
        request_id = str(message["request_id"])
        operation_id = str(message["operation_id"])
        prepared_hash = str(message["prepared_hash"])
        key = (request_id, operation_id)
        known = getattr(self, "_known_operations", None)
        if known is None:
            known = {key: prepared_hash}
            self._known_operations = known
        if known.get(key) != prepared_hash:
            raise transport_error(
                "prepared_identity_mismatch",
                "Home lost the exact prepared operation identity.",
                phase="authorize",
                completion="not_started",
            )
        context = bind_transport_intent(
            self,
            message,
            request_id=request_id,
            operation_id=operation_id,
            prepared_hash=prepared_hash,
        )
        self._renew_lease(str(message.get("task_id") or ""))
        self._send(
            "continue",
            request_id=request_id,
            operation_id=operation_id,
            prepared_hash=prepared_hash,
            optional={"prepared_token": str(message["prepared_token"])},
        )
        try:
            response_timeout = float(message.get("_response_timeout_sec") or _SESSION_TIMEOUT_SEC)
        except (TypeError, ValueError):
            response_timeout = _SESSION_TIMEOUT_SEC

        if "_response_timeout_sec" in message:
            response = self._wait_control(
                lambda row: (
                    row.get("kind") in {"result", "diagnostic"}
                    and row.get("request_id") == request_id
                    and row.get("operation_id") == operation_id
                ),
                timeout_sec=max(0.1, response_timeout),
            )
        else:
            response = self._wait_control(
                lambda row: (
                    row.get("kind") in {"result", "diagnostic"}
                    and row.get("request_id") == request_id
                    and row.get("operation_id") == operation_id
                )
            )
        self._raise_diagnostic(response)
        result = response.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("envelope"), dict):
            raise transport_error(
                "remote_result_invalid",
                "Execd returned an invalid result envelope.",
                phase="finalize",
                completion="unknown",
            )
        try:
            envelope, fetched = prefetch_remote_result_import(
                result,
                self.fetch_blob,
            )
            envelope = complete_transport_import(
                self,
                context,
                result,
                envelope,
                fetched,
            )
        except Exception as exc:
            raise transport_error(
                "remote_result_import_failed",
                "Completed remote result could not be verified and imported on Home.",
                phase="import",
                completion="completed",
                details={"error_type": type(exc).__name__},
            ) from exc
        ack_sequence = self._send(
            "ack",
            ack_seq=int(response["seq"]),
            request_id=request_id,
            operation_id=operation_id,
            optional={"prepared_hash": prepared_hash},
        )
        try:
            ack = self._wait_control(
                lambda row: (
                    row.get("kind") in {"ack", "diagnostic"}
                    and (
                        row.get("ack_seq") == ack_sequence
                        or (
                            row.get("request_id") == request_id
                            and row.get("operation_id") == operation_id
                        )
                    )
                ),
                timeout_sec=_ACK_TIMEOUT_SEC,
            )
        except Exception:
            pass
        else:
            if ack.get("kind") == "ack":
                if remove_transport_pending(context):
                    self._known_operations.pop(key, None)
                    self._operation_contexts.pop(key, None)
                else:
                    _LOG.warning(
                        "Remote operation %s was ACKed but Home pending "
                        "cleanup remains for reconciliation.",
                        operation_id,
                    )
        return envelope

    def abort_prepared(self, message: Mapping[str, Any]) -> bool:
        sequence = self._send(
            "abort",
            request_id=str(message["request_id"]),
            operation_id=str(message["operation_id"]),
            reason=str(message.get("reason") or "denied")[:4096],
            optional={"prepared_token": str(message.get("prepared_token") or "")},
        )
        self._wait_control(lambda row: row.get("kind") == "ack" and row.get("ack_seq") == sequence)
        key = (str(message["request_id"]), str(message["operation_id"]))
        contexts = getattr(self, "_operation_contexts", None)
        context = contexts.get(key, {}) if contexts is not None else {}
        if not isinstance(context.get("pending_record"), Mapping):
            self._known_operations.pop(key, None)
        if contexts is not None and not isinstance(
            context.get("pending_record"),
            Mapping,
        ):
            contexts.pop(key, None)
        return True

    def fetch_blob(self, blob_id: str, max_bytes: int) -> bytes:
        self._ensure_session()
        self._download_lock.acquire()
        request_id = f"blob_{uuid.uuid4().hex}"
        try:
            with self._condition:
                self._downloads[blob_id] = {
                    "request_id": request_id,
                    "max_bytes": int(max_bytes),
                    "event": threading.Event(),
                    "data": bytearray(),
                    "error": None,
                }
            self._send(
                "blob_fetch",
                request_id=request_id,
                blob_id=blob_id,
                size=int(max_bytes),
            )
            state = self._downloads[blob_id]
            if not state["event"].wait(_SESSION_TIMEOUT_SEC):
                raise transport_error(
                    "remote_blob_timeout",
                    "Remote blob transfer timed out.",
                    phase="import",
                    completion="unknown",
                    retryable=True,
                )
            if state["error"] is not None:
                raise state["error"]
            return bytes(state["data"])
        finally:
            with self._condition:
                self._downloads.pop(blob_id, None)
                if self._download_current == blob_id:
                    # The latch has to come down with the download it names. Left set,
                    # a blob that timed out mid-transfer made the NEXT, unrelated fetch
                    # raise "unexpected or overlapping blob download manifest" — inside
                    # the READER THREAD, where it becomes `_reader_error` and turns
                    # every later wait on this session into `ssh_session_disconnected`.
                    # One slow blob then took the whole transport down, along with every
                    # other operation riding it, and the caller that lost its result had
                    # already COMPLETED on the target.
                    self._download_current = ""
                    self._download_draining = blob_id
            self._download_lock.release()

    def reconcile(self) -> list[dict[str, Any]]:
        reconnected = self._ensure_session()
        if reconnected:
            return [dict(row) for row in self._last_reconciliation]
        rows = reconcile_remote_operations(
            self,
            ack_timeout_sec=_ACK_TIMEOUT_SEC,
            retention_cap=_MAX_KNOWN_OPERATIONS,
        )
        self._last_reconciliation = rows
        return [dict(row) for row in rows]

    def reconnect(self, *, timeout_sec: float = _SESSION_TIMEOUT_SEC) -> dict[str, Any]:
        with self._session_lock:
            if self._stop.is_set():
                raise transport_error(
                    "ssh_session_closed",
                    "SSH execd session is closed.",
                    phase="connect",
                )
            process = self._process
            live = (
                process is not None
                and process.poll() is None
                and self._reader_error is None
            )
            if not live:
                self._reset_wire_state()
                self.bootstrap(timeout_sec=timeout_sec)
            try:
                if not live:
                    self._start_session(timeout_sec=timeout_sec)
                    validate_transport_session_identity(self)
                rows = reconcile_remote_operations(
                    self,
                    ack_timeout_sec=_ACK_TIMEOUT_SEC,
                    retention_cap=_MAX_KNOWN_OPERATIONS,
                )
            except BaseException:
                process = self._process
                if (
                    not live
                    or process is None
                    or process.poll() is not None
                    or self._reader_error is not None
                ):
                    self._reset_wire_state()
                raise
            self._last_reconciliation = rows
            return {
                "status": "ready",
                "phase": "reconcile",
                "completion": "completed",
                "handshake": dict(self._handshake or {}),
                "reconciliation": [dict(row) for row in rows],
            }

    def health(self) -> dict[str, Any]:
        process = self._process
        live = not self._stop.is_set() and process is not None and process.poll() is None and self._reader_error is None
        return {
            "status": "ready" if live else "disconnected",
            "phase": "stream" if live else "connect",
            "completion": "completed" if live else "unknown",
            "reconnectable": not self._stop.is_set(),
            "child_pid": int(process.pid) if live else None,
            "error": safe_text(self._reader_error) if self._reader_error else "",
            # A refused lease never surfaces on a caller's thread, so health() is where
            # the owner can see that this Home no longer owns its remote process groups.
            "lease_refusal": dict(getattr(self, "_lease_refusal", {})),
            "warnings": [
                dict(row) for row in getattr(self, "_warnings", [])
            ],
        }

    def cancel(self, message: Mapping[str, Any]) -> bool:
        self._ensure_session()
        request_id = str(message.get("request_id") or "") or f"cancel_{uuid.uuid4().hex}"
        operation_id = str(message.get("operation_id") or "") or f"cancel_{uuid.uuid4().hex}"
        sequence = self._send(
            "cancel",
            request_id=request_id,
            operation_id=operation_id,
            task_id=str(message.get("task_id") or "") or None,
        )
        self._wait_control(lambda row: row.get("kind") == "ack" and row.get("ack_seq") == sequence)
        task_id = str(message.get("task_id") or "")
        if task_id:
            self.task_lease(task_id, forget=True)
        return True

    def task_lease(self, task_id: str, *, forget: bool = False) -> bool:
        tracked = bool(task_id) and str(task_id) in self._active_tasks
        if forget and task_id:
            self._active_tasks.discard(str(task_id))
        return tracked

    def panic(self) -> None:
        self._stop.set()
        process = self._process
        self._process = None
        helper = self._helper_process
        self._helper_process = None
        try:
            if (
                process is not None
                and process.poll() is None
                and process.stdin is not None
                and self._send_lock.acquire(blocking=False)
            ):
                try:
                    sequence = self._sequence
                    self._sequence += 1
                    payload = encode_control(
                        {
                            "kind": "panic",
                            "seq": sequence,
                            "server_generation": self.request.server_generation,
                        }
                    )
                    try:
                        best_effort_nonblocking_pipe_write(
                            process.stdin,
                            payload,
                        )
                    except Exception:
                        # The helper's contract is never-raising; retain panic
                        # teardown even if a test double or platform anomaly
                        # violates it.
                        pass
                finally:
                    self._send_lock.release()
        finally:
            self._panic_discard_process(helper)
            self._panic_discard_process(process)

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        with self._helper_lock:
            helper = self._helper_process
            self._helper_process = None
        if helper is not None and helper.poll() is None:
            kill_process_tree(helper)
        process = self._process
        # DROPPED, the way `panic()` and `_reset_wire_state()` already drop it. This
        # method killed the child and left `self._process` pointing at the corpse, so
        # a closed transport still held a `subprocess.Popen` and its buffered stderr
        # for as long as anything held the transport — and until the broker's custody
        # register learned to forget a retired transport, that was the whole server
        # generation.
        self._process = None
        self._discard_process(process)
        # The re-lease set is WIRE state: kept across a reconnect on purpose (a
        # reopened session must re-declare its live tasks) and meaningless after a
        # close, since there is no session left to lease them on.
        self._active_tasks.clear()
        with self._condition:
            self._condition.notify_all()

    def detach_after_fork(self) -> None:
        """Close only inherited descriptor copies; never signal the parent SSH."""

        self._stop.set()
        process = self._process
        helper = self._helper_process
        if helper is not None:
            _release_child_streams(helper)
            self._helper_process = None
        _release_child_streams(process)
        self._process = None

    @staticmethod
    def _discard_process(process: subprocess.Popen[bytes] | None) -> None:
        if process is None:
            return
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
        if process.poll() is None:
            try:
                kill_process_tree(process)
                process.wait(timeout=2)
            except Exception:
                try:
                    process.kill()
                except OSError:
                    pass

    @staticmethod
    def _panic_discard_process(
        process: subprocess.Popen[bytes] | None,
    ) -> None:
        if process is None:
            return
        if process.poll() is None:
            if not kill_process_group_id(process.pid):
                try:
                    process.kill()
                except OSError:
                    pass
        _release_child_streams(process)

    def _reset_wire_state(self) -> None:
        process = self._process
        self._process = None
        self._discard_process(process)
        current = threading.current_thread()
        for thread in (self._reader, self._stderr_reader, self._lease_thread):
            if thread is not None and thread is not current and thread.is_alive():
                thread.join(timeout=2)
        with self._condition:
            error = transport_error(
                "ssh_session_disconnected",
                "SSH execd session was replaced.",
                phase="stream",
                completion="unknown",
                retryable=True,
            )
            for state in self._downloads.values():
                state["error"] = error
                state["event"].set()
            self._nonce = os.urandom(24)
            self._sequence = 0
            self._messages.clear()
            self._receive_sequence = ControlSequence()
            self._reader_error = None
            self._reader = None
            self._stderr_reader = None
            self._lease_thread = None
            self._stderr.clear()
            self._stderr_carry = ""
            self._handshake = None
            self._downloads.clear()
            self._download_current = ""
            self._download_draining = ""
            self._condition.notify_all()

    def _ensure_session(self) -> bool:
        with self._session_lock:
            if self._process is not None and self._process.poll() is None and self._reader_error is None:
                return False
            if self._stop.is_set():
                raise transport_error(
                    "ssh_session_closed",
                    "SSH execd session is closed.",
                    phase="connect",
                )
            self._reset_wire_state()
            self.bootstrap()
            try:
                self._start_session()
                validate_transport_session_identity(self)
                rows = reconcile_remote_operations(
                    self,
                    ack_timeout_sec=_ACK_TIMEOUT_SEC,
                    retention_cap=_MAX_KNOWN_OPERATIONS,
                )
            except BaseException:
                self._reset_wire_state()
                raise
            self._last_reconciliation = rows
            return True

    def _admit_contract_set(
        self,
        peer_contract_set: Any,
        selected: SelectedRelease,
    ) -> None:
        """This session's half of the contract-set admission (both seams live in
        :mod:`ouroboros.remote_contract_admission`); it only supplies the identities."""

        admit_home_contract_set(
            peer_contract_set,
            release=selected.release,
            artifact_sha256=selected.archive_sha256,
            connection_id=str(self.request.connection.get("id") or ""),
        )

    def _start_session(self, *, timeout_sec: float = _SESSION_TIMEOUT_SEC) -> None:
        self._upload_capability_manifest(_SESSION_TIMEOUT_SEC)
        selected = self._selected_release
        if selected is None:
            raise transport_error(
                "execd_release_unselected",
                "No verified execd release was selected.",
                phase="bootstrap",
            )
        capability_hash = str(self.request.capability_manifest["manifest_sha256"])
        script = (
            'set -eu; base="$HOME/' + _REMOTE_BASE + '"; '
            'target="$HOME/$1"; self="$target/bin/ouroboros-execd"; '
            'OUROBOROS_EXECD_SELF="$self" exec "$self" '
            '--state-root "$base/state" --workspace-root "$2" '
            '--workspace-id "$3" --connection-id "$4" --project-id "$5" '
            '--server-generation "$6" --release-id "$7" --artifact-sha256 "$8" '
            '--capability-manifest "$base/manifests/$9.json" --session-nonce "${10}"'
        )
        remote = shlex.join(
            [
                str(item)
                for item in [
                    "sh",
                    "-c",
                    script,
                    "sh",
                    selected.target_rel,
                    self.request.remote_root,
                    self.request.workspace_id,
                    self.request.connection["id"],
                    self.request.project_id,
                    self.request.server_generation,
                    selected.release,
                    selected.archive_sha256,
                    capability_hash,
                    self._nonce.hex(),
                ]
            ]
        )
        ssh_base = validated_ssh_base_command(
            self.alias,
            self.request.ssh_binary,
        )
        process = spawn_supervised(
            [*ssh_base, remote],
            drive_root=self.request.drive_root,
            purpose=f"remote_ssh:{self.request.connection['id']}",
            scope="session",
            new_process_group=True,
            required_custody=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=minimal_ssh_env(),
        )
        self._process = process
        assert process.stdout is not None
        self._stderr_reader = threading.Thread(
            target=self._stderr_loop,
            args=(process,),
            daemon=True,
            name=f"execd-stderr-{self.request.connection['id']}",
        )
        self._stderr_reader.start()
        prefix = bytearray()
        deadline = time.monotonic() + timeout_sec
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="execd-preamble",
        ) as preamble_reader:
            while len(prefix) < MAX_PREAMBLE_BYTES:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise transport_error(
                        "execd_preamble_timeout",
                        "Execd did not return its authenticated preamble.",
                        phase="bootstrap",
                        details={"stderr": self._stderr_text()},
                    )
                try:
                    chunk = preamble_reader.submit(
                        process.stdout.read,
                        1,
                    ).result(timeout=remaining)
                except concurrent.futures.TimeoutError as exc:
                    self._discard_process(process)
                    raise transport_error(
                        "execd_preamble_timeout",
                        "Execd did not return its authenticated preamble.",
                        phase="bootstrap",
                        details={"stderr": self._stderr_text()},
                    ) from exc
                if not chunk:
                    raise transport_error(
                        "execd_start_failed",
                        "SSH closed before execd returned its preamble.",
                        phase="bootstrap",
                        details={"stderr": self._stderr_text()},
                    )
                prefix.extend(chunk)
                if PREAMBLE_MAGIC not in prefix:
                    continue
                magic_at = prefix.find(PREAMBLE_MAGIC)
                authenticated_prefix_size = magic_at + len(PREAMBLE_MAGIC) + len(self._nonce.hex()) + 1
                if len(prefix) < authenticated_prefix_size:
                    continue
                try:
                    consumed, _major, peer_contract_set = parse_session_preamble(
                        prefix, self._nonce
                    )
                except ProtocolError as exc:
                    if "exceeds the bounded prefix" in str(exc):
                        continue
                    raise transport_error(
                        "execd_preamble_invalid",
                        "Execd returned a malformed or unauthenticated preamble.",
                        phase="bootstrap",
                        details={"reason": safe_text(exc)},
                    ) from exc
                if consumed == len(prefix):
                    break
            else:
                raise transport_error(
                    "execd_preamble_invalid",
                    "Execd preamble exceeded its strict bound.",
                    phase="bootstrap",
                )
        # CONTRACT-SET ADMISSION, at the earliest byte that can carry the answer.
        # The preamble already announces the target's contract set (it is the wire
        # minor — see `remote_contracts`), so a build pair that cannot safely
        # cooperate is refused HERE: before a single frame is written, before any
        # tool call has borrowed the session, and with the release identity of what
        # is actually installed in hand. Doing it later was the defect — the
        # disagreement then surfaced inside an unrelated operation's PREPARE.
        #
        # It is also what keeps the OLD direction from failing mysteriously: a
        # target built before this check refuses our handshake frame on its own
        # protocol-version rule and dies, which Home would have reported as a
        # disconnect. We never send that frame.
        self._admit_contract_set(peer_contract_set, selected)
        self._reader = threading.Thread(
            target=self._read_loop,
            args=(process,),
            daemon=True,
            name=f"execd-reader-{self.request.connection['id']}",
        )
        self._reader.start()
        self._send(
            "handshake",
            nonce=self._nonce.hex(),
            protocol_major=PROTOCOL_MAJOR,
            protocol_minor=PROTOCOL_MINOR,
            client_build="ouroboros-home",
            capability_hash=capability_hash,
        )
        response = self._wait_control(
            lambda row: row.get("kind") == "handshake_ok",
            timeout_sec=timeout_sec,
        )
        # The preamble is scanned out of a byte stream; the handshake frame is the
        # peer's structured statement of the same number. Checking both costs one
        # comparison and means a target cannot announce one contract set in its
        # preamble and act on another.
        self._admit_contract_set(response.get("protocol_minor"), selected)
        admission = response.get("optional", {}).get("admission", {})
        attestation = response.get("optional", {}).get("artifact", {})
        if not isinstance(attestation, dict):
            attestation = {}
        if attestation.get("release_id") != selected.release or attestation.get("sha256") != selected.archive_sha256:
            raise transport_error(
                "execd_artifact_mismatch",
                "Execd did not attest the selected immutable artifact.",
                phase="bootstrap",
            )
        self._handshake = {
            "protocol_major": response["protocol_major"],
            "protocol_minor": response["protocol_minor"],
            "host_id": response["host_id"],
            "server_generation": response["server_generation"],
            "build": response.get("build", ""),
            "capability_hash": response.get("capability_hash", ""),
            "release_id": selected.release,
            "artifact_sha256": selected.archive_sha256,
            "warnings": [dict(row) for row in self._warnings],
            **(dict(admission) if isinstance(admission, dict) else {}),
        }
        self._renew_lease("")
        for task_id in list(self._active_tasks):
            self._renew_lease(task_id)
        self._lease_thread = threading.Thread(
            target=self._lease_loop,
            args=(process,),
            daemon=True,
            name=f"execd-lease-{self.request.connection['id']}",
        )
        self._lease_thread.start()

    def _send(self, kind: str, **fields: Any) -> int:
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            raise transport_error(
                "ssh_session_disconnected",
                "SSH execd session is disconnected.",
                phase="stream",
                completion="unknown",
                retryable=True,
            )
        fields = {key: value for key, value in fields.items() if value is not None}
        with self._send_lock:
            sequence = self._sequence
            self._sequence += 1
            process.stdin.write(encode_control({"kind": kind, "seq": sequence, **fields}))
            process.stdin.flush()
            return sequence

    def _wait_control(
        self,
        predicate: Callable[[dict[str, Any]], bool],
        timeout_sec: float = _SESSION_TIMEOUT_SEC,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_sec
        with self._condition:
            while True:
                for index, row in enumerate(self._messages):
                    if predicate(row):
                        del self._messages[index]
                        return row
                if self._reader_error is not None:
                    raise transport_error(
                        "ssh_session_disconnected",
                        "SSH execd session ended while awaiting a response.",
                        phase="stream",
                        completion="unknown",
                        retryable=True,
                        details={
                            "reason": safe_text(self._reader_error),
                            "stderr": self._stderr_text(),
                        },
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise transport_error(
                        "remote_request_timeout",
                        "Execd response exceeded the transport deadline.",
                        phase="stream",
                        completion="unknown",
                        retryable=True,
                    )
                self._condition.wait(min(remaining, 1.0))

    def _read_loop(self, process: subprocess.Popen[bytes]) -> None:
        assert process.stdout is not None
        try:
            while not self._stop.is_set() and self._process is process:
                label, payload = read_frame(process.stdout)
                if label == "bulk":
                    self._receive_bulk(bytes(payload))
                else:
                    assert isinstance(payload, dict)
                    self._receive_sequence.observe(payload)
                    if lease_answer_id(payload):
                        self._observe_lease_answer(payload)
                        continue
                    if payload["kind"] == "blob_manifest":
                        self._receive_manifest(payload)
                        continue
                    if payload["kind"] == "diagnostic":
                        with self._condition:
                            download = next(
                                (
                                    state
                                    for state in self._downloads.values()
                                    if state["request_id"] == payload.get("request_id")
                                ),
                                None,
                            )
                            if download is not None:
                                diagnostic = payload.get("diagnostic")
                                diagnostic = diagnostic if isinstance(diagnostic, dict) else {}
                                download["error"] = transport_error(
                                    str(diagnostic.get("code") or "remote_blob_error"),
                                    str(diagnostic.get("message") or "Remote blob fetch failed."),
                                    phase=str(diagnostic.get("phase") or "import"),
                                )
                                download["event"].set()
                                continue
                    with self._condition:
                        if len(self._messages) >= _MAX_PENDING_MESSAGES:
                            raise ProtocolError("too many unconsumed execd control messages")
                        self._messages.append(payload)
                        self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                if self._process is process:
                    self._reader_error = exc
                    self._condition.notify_all()

    def _stderr_loop(self, process: subprocess.Popen[bytes]) -> None:
        assert process.stderr is not None
        try:
            while not self._stop.is_set() and self._process is process:
                chunk = process.stderr.read(4096)
                if not chunk:
                    return
                with self._condition:
                    self._stderr_carry += chunk.decode("utf-8", errors="replace")
                    retained = _STDERR_LIMIT + 4096
                    if len(self._stderr_carry) > retained:
                        self._stderr_carry = self._stderr_carry[-retained:]
        except OSError:
            return

    def _receive_manifest(self, message: dict[str, Any]) -> None:
        blob_id = str(message["blob_id"])
        with self._condition:
            state = self._downloads.get(blob_id)
            if (
                state is None
                or state["request_id"] != message["request_id"]
                or int(message["size"]) > state["max_bytes"]
                or self._download_current
            ):
                raise ProtocolError("unexpected or overlapping blob download manifest")
            state.update(
                {
                    "size": int(message["size"]),
                    "sha256": str(message["sha256"]),
                    "chunk_seq": 0,
                }
            )
            # A manifest Home asked for and accepted ends the drain: anything still in
            # flight from the abandoned transfer would now be indistinguishable from
            # this one's bytes, and the size and SHA-256 checks below are what judge that.
            self._download_draining = ""
            if state["size"] == 0:
                state["event"].set()
            else:
                self._download_current = blob_id

    def _receive_bulk(self, payload: bytes) -> None:
        with self._condition:
            blob_id = self._download_current
            state = self._downloads.get(blob_id)
            if not blob_id or state is None:
                if self._download_draining:
                    # Chunks for a download Home abandoned: its own timeout fired while
                    # the target was still sending. That is a race Home created, not a
                    # desynced wire, so the bytes are dropped and the session lives. A
                    # bulk frame with nothing abandoned behind it IS a desync and still
                    # raises.
                    return
                raise ProtocolError("unexpected execd bulk frame")
            state["data"].extend(payload)
            if len(state["data"]) > state["size"]:
                raise ProtocolError("execd bulk data exceeds manifest size")
            complete = len(state["data"]) == state["size"]
            chunk_seq = int(state["chunk_seq"])
            state["chunk_seq"] = chunk_seq + 1
        self._send(
            "blob_ack",
            request_id=state["request_id"],
            blob_id=blob_id,
            chunk_seq=chunk_seq,
            complete=complete,
        )
        if complete:
            digest = hashlib.sha256(state["data"]).hexdigest()
            if digest != state["sha256"]:
                state["error"] = transport_error(
                    "blob_hash_mismatch",
                    "Remote blob failed SHA-256 verification.",
                    phase="import",
                )
            with self._condition:
                self._download_current = ""
                state["event"].set()

    def _upload_blob(
        self,
        request_id: str,
        operation_id: str,
        blob_id: str,
        payload: bytes,
    ) -> None:
        digest = hashlib.sha256(payload).hexdigest()
        self._send(
            "blob_manifest",
            request_id=request_id,
            operation_id=operation_id,
            blob_id=blob_id,
            size=len(payload),
            sha256=digest,
        )
        chunks = list(range(0, len(payload), MAX_BULK_BYTES)) or [0]
        for chunk_seq, offset in enumerate(chunks):
            if payload:
                process = self._process
                assert process is not None and process.stdin is not None
                with self._send_lock:
                    process.stdin.write(encode_bulk(payload[offset : offset + MAX_BULK_BYTES]))
                    process.stdin.flush()
            self._wait_control(
                lambda row: (
                    row.get("kind") == "blob_ack"
                    and row.get("request_id") == request_id
                    and row.get("operation_id") == operation_id
                    and row.get("blob_id") == blob_id
                    and row.get("chunk_seq") == chunk_seq
                )
            )

    def _renew_lease(self, task_id: str) -> None:
        if task_id:
            self._active_tasks.add(task_id)
        fields: dict[str, Any] = {
            "server_generation": self.request.server_generation,
            "lease_id": f"lease_{uuid.uuid4().hex}",
            "ttl_ms": _LEASE_TTL_MS,
        }
        if task_id:
            fields["task_id"] = task_id
        self._send("lease", **fields)

    def _observe_lease_answer(self, message: dict[str, Any]) -> None:
        """Consume a lease answer IN THE READER, so no sender ever waits for one.

        ``_renew_lease`` stays exactly as unblocking as it was — it writes and returns
        — because the lease path also carries panic and cancel and may never acquire a
        wait.  The answer is therefore handled out of band by the always-running
        control reader instead of being matched by a waiter, which is also what keeps
        it out of ``_messages``: one frame every renewal would otherwise fill the
        bounded pending queue and kill a healthy session.

        A refusal is kept — one slot, last one wins — because it is a safety fact the
        owner may need: it means this Home's generation is not the one that owns the
        remote process groups, so its remote work is already condemned to the lease
        ceiling.  It is disclosed through ``health()`` rather than raised, since it
        arrives on no caller's thread.
        """

        if message.get("kind") != "diagnostic":
            return
        diagnostic = message.get("diagnostic")
        with self._condition:
            self._lease_refusal = dict(diagnostic) if isinstance(diagnostic, dict) else {}

    def _lease_loop(self, process: subprocess.Popen[bytes]) -> None:
        while self._process is process and not self._stop.wait(_LEASE_RENEW_SEC):
            try:
                self._renew_lease("")
                for task_id in list(self._active_tasks):
                    self._renew_lease(task_id)
            except Exception:
                return

    def _raise_diagnostic(self, response: Mapping[str, Any]) -> None:
        if response.get("kind") != "diagnostic":
            return
        diagnostic = response.get("diagnostic")
        diagnostic = diagnostic if isinstance(diagnostic, dict) else {}
        raise transport_error(
            str(diagnostic.get("code") or "remote_error"),
            str(diagnostic.get("message") or "Remote execd operation failed."),
            phase=str(diagnostic.get("phase") or "execute"),
            completion=str(diagnostic.get("completion") or "unknown"),
            retryable=bool(diagnostic.get("retryable")),
            details=diagnostic.get("details") if isinstance(diagnostic.get("details"), dict) else {},
        )

    def _platform_probe(self, timeout_sec: float) -> dict[str, Any]:
        script = (
            "set -eu; system=$(uname -s); machine=$(uname -m); "
            "libc=$(getconf GNU_LIBC_VERSION 2>/dev/null || true); "
            'printf "%s\\t%s\\t%s\\n" "$system" "$machine" "$libc"'
        )
        completed = self._run_remote(
            ["sh", "-c", script, "sh"],
            timeout_sec=timeout_sec,
        )
        fields = completed.stdout.decode("utf-8", errors="replace").rstrip("\r\n").split("\t")
        if len(fields) != 3 or fields[0] != "Linux":
            raise transport_error(
                "remote_platform_unsupported",
                "Execd supports Linux remote hosts only.",
                phase="connect",
            )
        machine = {"x86_64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}.get(fields[1])
        if machine is None:
            raise transport_error(
                "remote_platform_unsupported",
                f"Unsupported remote architecture: {safe_text(fields[1])}",
                phase="connect",
            )
        libc_parts = fields[2].split()
        return {
            "system": "Linux",
            "machine": machine,
            "libc": "glibc" if libc_parts[:1] == ["glibc"] else "unknown",
            "libc_version": libc_parts[1] if len(libc_parts) > 1 else "",
        }

    def _installed_host_id(
        self,
        timeout_sec: float,
        *,
        required: bool,
        selected: SelectedRelease | None,
    ) -> str:
        executable = (
            f"$HOME/{selected.target_rel}/bin/ouroboros-execd"
            if selected is not None
            else f"$HOME/{_REMOTE_BASE}/current/bin/ouroboros-execd"
        )
        script = (
            'set -eu; base="$HOME/' + _REMOTE_BASE + '"; '
            f'exec "{executable}" '
            '--state-root "$base/state" --print-host-id'
        )
        try:
            completed = self._run_remote(
                ["sh", "-c", script, "sh"],
                timeout_sec=timeout_sec,
            )
        except Exception:
            if required:
                raise
            return ""
        host_id = completed.stdout.decode("utf-8", errors="replace").strip()
        if not host_id or any(char.isspace() for char in host_id):
            if required:
                raise transport_error(
                    "host_identity_invalid",
                    "Installed execd returned an invalid continuity identity.",
                    phase="bootstrap",
                )
            return ""
        return host_id

    def _initialize_host_id(
        self,
        timeout_sec: float,
        selected: SelectedRelease,
    ) -> None:
        script = (
            'set -eu; base="$HOME/' + _REMOTE_BASE + '"; '
            'exec "$HOME/$1/bin/ouroboros-execd" '
            '--state-root "$base/state" --initialize-host-id'
        )
        self._run_remote(
            ["sh", "-c", script, "sh", selected.target_rel],
            timeout_sec=timeout_sec,
        )

    def _upload_capability_manifest(self, timeout_sec: float) -> None:
        digest = str(self.request.capability_manifest["manifest_sha256"])
        payload = canonical_json(dict(self.request.capability_manifest))
        script = (
            'set -eu; base="$HOME/' + _REMOTE_BASE + '/manifests"; '
            'mkdir -p "$base"; umask 077; tmp="$base/.$1.$$"; '
            'cat > "$tmp"; mv "$tmp" "$base/$1.json"'
        )
        self._run_remote(
            ["sh", "-c", script, "sh", digest],
            input_bytes=payload,
            timeout_sec=timeout_sec,
        )

    def _run_remote(
        self,
        command: list[str],
        *,
        input_bytes: bytes | None = None,
        input_path: pathlib.Path | None = None,
        timeout_sec: float,
    ) -> subprocess.CompletedProcess[bytes]:
        if input_bytes is not None and input_path is not None:
            raise ValueError("remote input bytes and path are mutually exclusive")
        ssh_base = validated_ssh_base_command(
            self.alias,
            self.request.ssh_binary,
        )
        input_stream = input_path.open("rb") if input_path is not None else None
        try:
            process = spawn_supervised(
                [*ssh_base, shlex.join([str(item) for item in command])],
                drive_root=self.request.drive_root,
                purpose=f"remote_ssh_helper:{self.request.connection['id']}",
                scope="session",
                new_process_group=True,
                required_custody=True,
                stdin=(
                    subprocess.PIPE
                    if input_bytes is not None
                    else input_stream
                    if input_stream is not None
                    else subprocess.DEVNULL
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=minimal_ssh_env(),
            )
        except BaseException:
            if input_stream is not None:
                input_stream.close()
            raise
        with self._helper_lock:
            if self._stop.is_set():
                kill_process_tree(process)
                raise transport_error(
                    "ssh_operation_cancelled",
                    "SSH operation was cancelled.",
                    phase="connect",
                )
            self._helper_process = process
        try:
            stdout, stderr = process.communicate(input=input_bytes, timeout=timeout_sec)
        except subprocess.TimeoutExpired as exc:
            kill_process_tree(process)
            raise transport_error(
                "ssh_timeout",
                "SSH operation timed out.",
                phase="connect",
                retryable=True,
            ) from exc
        finally:
            if input_stream is not None:
                input_stream.close()
            with self._helper_lock:
                if self._helper_process is process:
                    self._helper_process = None
        completed = subprocess.CompletedProcess(
            process.args,
            int(process.returncode or 0),
            stdout,
            stderr,
        )
        if completed.returncode != 0:
            raise transport_error(
                "ssh_command_failed",
                "Remote SSH operation failed.",
                phase="connect",
                details={
                    "returncode": completed.returncode,
                    "stderr": safe_text(completed.stderr.decode("utf-8", errors="replace")),
                },
            )
        return completed

    def _stderr_text(self) -> str:
        with self._condition:
            return safe_text(self._stderr_carry)
