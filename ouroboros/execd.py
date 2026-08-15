"""Restricted remote workspace executor served over framed OpenSSH stdio.

Execd owns no model, Home policy, registry, credentials, task queue or review
state.  It accepts only the explicit native workspace kernel, binds every call
through PREPARE -> CONTINUE|ABORT, journals handler start before effects, and
keeps remote process groups under expiring Home-generation/task leases.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import os
import pathlib
import secrets
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, BinaryIO

from ouroboros import execd_state as _state
from ouroboros.export_policy_contract import (
    QUESTION_EXPORT,
    export_disclosure_block,
)
from ouroboros.execd_state import (
    MAX_NATIVE_RESULT_BLOBS,
    MAX_STAGED_BLOB_BYTES,
    CASBlobStore,
    ExecdError,
    LeaseCustody,
    OperationJournal,
    continuity_host_id,
    initialize_continuity_host_id,
)
from ouroboros.execd_spool import ProcessLogSpool
from ouroboros.remote_contract_admission import admit_execd_contract_set
from ouroboros.remote_protocol import (
    MAX_BULK_BYTES,
    MAX_CONTROL_BYTES,
    MAX_REMOTE_EXTERNAL_ENVELOPE_BYTES,
    PROTOCOL_MAJOR,
    PROTOCOL_MINOR,
    ControlSequence,
    ProtocolEOF,
    ProtocolError,
    canonical_json,
    canonical_prepared_hash,
    encode_bulk,
    encode_control,
    lease_answer_marker,
    read_frame,
    session_preamble,
)
from ouroboros.execd_task_files import (
    ATTACHMENT_STAGE_OPERATION,
    INTERNAL_TASK_FILE_OPERATIONS,
    MEDIA_EXPORT_OPERATION,
    RemoteTaskFileCache,
    RemoteTaskFileError,
    attachment_blob_map,
    media_export_artifact_row,
    media_export_execution_args,
    media_export_policy_facts,
)
from ouroboros.platform_layer import subprocess_new_group_kwargs
from ouroboros.workspace_diagnostics import ExecutionDiagnostic, ToolExecutionEnvelope
from ouroboros.workspace_native_contract import bundle_prepared_facts
from ouroboros.workspace_native import (
    MANDATORY_REMOTE_NATIVE_OPERATIONS,
    NativeExecutionControl,
    NativeOperationResult,
    NativePreparedOperation,
    execute_native_operation,
    prepare_native_operation,
)

EXECD_BUILD = "1"
PREPARED_CALL_TTL_MS = 30_000
# How many artifact ROWS an externalized envelope lists inline. The bound exists so
# a remote envelope cannot size a Home record; it is always reported alongside the
# exact produced count, never applied as a bare slice.
_MAX_ENVELOPE_ARTIFACT_ROWS = 128


@dataclass
class _PreparedState:
    token: str
    prepared_hash: str
    prepared: dict[str, Any]
    prepare_args: dict[str, Any]
    execution_args: dict[str, Any]
    native_facts: dict[str, Any]
    input_blobs: dict[str, bytes]
    expires_at_ms: int


class ExecdNativeControl(NativeExecutionControl):
    def __init__(self, custody: LeaseCustody, task_id: str) -> None:
        self.custody = custody
        self.task_id = task_id
        # Bound per operation by `execute`; the native kernel reads it through
        # the optional `process_spool` slot of NativeExecutionControl.
        self.process_spool: Any = None
        self._cancelled = threading.Event()

    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def register_process(
        self,
        *,
        pgid: int,
        keep_alive: bool = False,
        service_id: str = "",
    ) -> None:
        self.custody.register(
            pgid=pgid,
            task_id=self.task_id,
            keep_alive=keep_alive,
            service_id=service_id,
        )

    def release_process(self, *, pgid: int, service_id: str = "") -> None:
        self.custody.release(pgid=pgid, service_id=service_id)

    def recover_service(self, *, service_id: str, name: str = "") -> dict[str, Any] | None:
        _ = str(name)
        return self.custody.recover_service(
            service_id=service_id,
            task_id=self.task_id,
        )

    def stop_service(self, *, service_id: str) -> bool:
        return bool(self.custody.stop_service(service_id))

    def cancel(self) -> None:
        self._cancelled.set()
        if self.task_id:
            self.custody.cancel_task(self.task_id)


class ExecdService:
    """In-process authority used by both stdio server and deterministic tests."""

    def __init__(
        self,
        state_root: pathlib.Path,
        workspace_root: pathlib.Path,
        *,
        connection_id: str,
        project_id: str,
        server_generation: str,
        capability_manifest: Mapping[str, Any],
        release_id: str,
        artifact_sha256: str,
        requested_workspace_id: str = "",
        start_custodian: bool = False,
    ) -> None:
        self.state_root = pathlib.Path(state_root).expanduser().resolve(strict=False)
        self.state_root.mkdir(parents=True, exist_ok=True, mode=_state.MODE_PRIVATE_DIR)
        os.chmod(self.state_root, _state.MODE_PRIVATE_DIR)
        self.connection_id = _state.opaque(connection_id, "connection_id")
        self.project_id = _state.opaque(project_id, "project_id")
        self.server_generation = _state.opaque(server_generation, "server_generation")
        self.release_id, self.artifact_sha256 = _state.release_attestation(release_id, artifact_sha256)
        self.capability_manifest = _state.json_object(capability_manifest, "capability_manifest")
        self.capability_hash = str(
            self.capability_manifest.get("manifest_sha256")
            or hashlib.sha256(canonical_json(self.capability_manifest)).hexdigest()
        )
        if not _state.HASH_RE.fullmatch(self.capability_hash):
            raise ExecdError(
                "capability_manifest_invalid",
                "Capability manifest hash is invalid.",
                phase="bootstrap",
            )
        operations = self.capability_manifest.get("native_operations")
        if not isinstance(operations, list) or set(str(item) for item in operations) != set(
            MANDATORY_REMOTE_NATIVE_OPERATIONS
        ):
            raise ExecdError(
                "capability_manifest_invalid",
                "Capability manifest does not match the exact native allowlist.",
                phase="bootstrap",
            )
        self.host_id = continuity_host_id(self.state_root)
        self.workspace_root, self.git_facts = _admit_git_workspace(workspace_root)
        self.workspace_id = _workspace_identity(
            self.host_id,
            self.workspace_root,
            self.git_facts,
        )
        if requested_workspace_id and requested_workspace_id != self.workspace_id:
            raise ExecdError(
                "workspace_identity_mismatch",
                "Workspace path no longer identifies the admitted worktree.",
                phase="bootstrap",
            )
        workspace_state = self.state_root / "workspaces" / self.workspace_id
        project_state = workspace_state / "connections" / self.connection_id / "projects" / self.project_id
        self.cas = CASBlobStore(project_state / "blobs")
        self.spool = CASBlobStore(project_state / "spool")
        # D8: process logs get their own quota-bounded spool with sealed
        # content-addressed blobs, so an unbounded stdout terminates its
        # process group instead of silently losing the trace past the
        # in-memory capture bound.
        self.process_logs = ProcessLogSpool(project_state / "process_logs")
        self.journal = OperationJournal(
            project_state / "operations",
            connection_id=self.connection_id,
            workspace_id=self.workspace_id,
            spool=self.spool,
            blobs=self.cas,
        )
        self.task_files = RemoteTaskFileCache(
            self.state_root,
            connection_id=self.connection_id,
            server_generation=self.server_generation,
        )
        self.custody = LeaseCustody(
            self.state_root / "custody" / self.server_generation / self.project_id / f"{self.workspace_id}.json",
            self.server_generation,
        )
        self.session_id = secrets.token_hex(16)
        self._prepared: dict[tuple[str, str], _PreparedState] = {}
        self._running: dict[tuple[str, str], ExecdNativeControl] = {}
        self._lock = threading.RLock()
        self._custodian_process: subprocess.Popen[bytes] | None = None
        self._custodian_id = ""
        if start_custodian:
            self._custodian_process = self._spawn_custodian()

    def handshake(self, client_capability_hash: str = "") -> dict[str, Any]:
        self._revalidate_workspace()
        if client_capability_hash and client_capability_hash != self.capability_hash:
            raise ExecdError(
                "capability_mismatch",
                "Home and execd capability manifests differ.",
                phase="bootstrap",
            )
        return {
            "protocol_major": PROTOCOL_MAJOR,
            "protocol_minor": PROTOCOL_MINOR,
            "build": EXECD_BUILD,
            "release_id": self.release_id,
            "artifact_sha256": self.artifact_sha256,
            "host_id": self.host_id,
            "server_generation": self.server_generation,
            "session_id": self.session_id,
            "workspace_id": self.workspace_id,
            "canonical_root": str(self.workspace_root),
            "git": dict(self.git_facts),
            "capability_hash": self.capability_hash,
            "capability_manifest": dict(self.capability_manifest),
            "platform": {
                "system": sys.platform,
                "machine": __import__("platform").machine(),
                "python": sys.version.split()[0],
                "libc": list(__import__("platform").libc_ver()),
            },
        }

    def prepare(
        self,
        *,
        request_id: str,
        operation_id: str,
        tool: str,
        args: Mapping[str, Any],
        task_id: str = "",
        blobs: Mapping[str, bytes] | None = None,
        deadline_ms: int | None = None,
    ) -> dict[str, Any]:
        request_id = _state.opaque(request_id, "request_id")
        operation_id = _state.opaque(operation_id, "operation_id")
        task_id = _state.opaque(task_id, "task_id", optional=True)
        if tool not in MANDATORY_REMOTE_NATIVE_OPERATIONS and tool not in INTERNAL_TASK_FILE_OPERATIONS:
            raise ExecdError(
                "native_operation_forbidden",
                f"Native operation is not allowlisted: {tool}",
                phase="prepare",
            )
        self._revalidate_workspace()
        now_ms = int(time.time() * 1000)
        expires_at_ms = now_ms + PREPARED_CALL_TTL_MS
        if deadline_ms is not None:
            if not isinstance(deadline_ms, int) or isinstance(deadline_ms, bool):
                raise ExecdError("deadline_invalid", "deadline_ms is invalid.", phase="prepare")
            expires_at_ms = min(expires_at_ms, deadline_ms)
        if expires_at_ms <= now_ms:
            raise ExecdError("prepared_call_expired", "Call deadline has expired.", phase="prepare")
        staged: dict[str, bytes] = {}
        blob_hashes: dict[str, str] = {}
        for supplied_id, data in dict(blobs or {}).items():
            blob_id = _state.opaque(supplied_id, "blob_id")
            payload = bytes(data)
            digest = self.cas.put(payload)
            staged[blob_id] = payload
            blob_hashes[blob_id] = digest
        prepare_args = _state.json_object(args, "args")
        prepared_native = self._prepare_operation(
            tool, prepare_args, task_id=task_id, blobs=staged
        )
        execution_args = _state.json_object(prepared_native.execution_args, "execution_args")
        native_facts = _state.json_object(prepared_native.native_facts, "native_facts")
        prepared_object = {
            "protocol_major": PROTOCOL_MAJOR,
            "protocol_minor": PROTOCOL_MINOR,
            "execd_build": EXECD_BUILD,
            "release_id": self.release_id,
            "artifact_sha256": self.artifact_sha256,
            "capability_hash": self.capability_hash,
            "host_id": self.host_id,
            "connection_id": self.connection_id,
            "project_id": self.project_id,
            "workspace_id": self.workspace_id,
            "canonical_root": str(self.workspace_root),
            "server_generation": self.server_generation,
            "session_id": self.session_id,
            "task_id": task_id,
            "request_id": request_id,
            "operation_id": operation_id,
            "tool": tool,
            "execution_args": execution_args,
            "native_facts": native_facts,
            "blob_hashes": blob_hashes,
            "expires_at_ms": expires_at_ms,
        }
        prepared_hash = canonical_prepared_hash(prepared_object)
        token = secrets.token_hex(16)
        key = (request_id, operation_id)
        with self._lock:
            existing = self._prepared.get(key)
            if existing is not None:
                if existing.prepared_hash != prepared_hash:
                    raise ExecdError(
                        "prepared_identity_conflict",
                        "Request/operation identity was reused with different arguments.",
                        phase="prepare",
                    )
                state = existing
            else:
                state = _PreparedState(
                    token=token,
                    prepared_hash=prepared_hash,
                    prepared=prepared_object,
                    prepare_args=prepare_args,
                    execution_args=execution_args,
                    native_facts=native_facts,
                    input_blobs=staged,
                    expires_at_ms=expires_at_ms,
                )
                self._prepared[key] = state
        return {
            "request_id": request_id,
            "operation_id": operation_id,
            "tool": tool,
            "prepared_token": state.token,
            "prepared_hash": state.prepared_hash,
            "expires_at_ms": state.expires_at_ms,
            "execution_args": dict(state.execution_args),
            "native_facts": dict(state.native_facts),
        }

    def continue_prepared(
        self,
        *,
        request_id: str,
        operation_id: str,
        prepared_hash: str,
        prepared_token: str,
    ) -> dict[str, Any]:
        key = (_state.opaque(request_id, "request_id"), _state.opaque(operation_id, "operation_id"))
        with self._lock:
            state = self._prepared.pop(key, None)
        if state is None:
            raise ExecdError(
                "prepared_call_stale",
                "Prepared call is absent, expired, aborted or already consumed.",
                phase="authorize",
            )
        if (
            state.prepared_hash != prepared_hash
            or state.token != prepared_token
            or int(time.time() * 1000) >= state.expires_at_ms
        ):
            raise ExecdError(
                "prepared_call_mismatch",
                "Prepared authorization identity is invalid or expired.",
                phase="authorize",
            )
        self._revalidate_workspace()
        task_id = str(state.prepared.get("task_id") or "")
        self._revalidate_prepared_target(state, task_id=task_id)
        status, stored = self.journal.begin(
            task_id=task_id,
            operation_id=key[1],
            request_hash=state.prepared_hash,
            binding=state.prepared,
        )
        if status == "completed":
            if stored is None:
                return _missing_result(key, state.prepared_hash, unavailable=True)
            return stored
        if status == "unknown":
            return _missing_result(key, state.prepared_hash, unavailable=False)
        control = ExecdNativeControl(self.custody, task_id)
        control.process_spool = self.process_logs.bind(
            task_id=task_id, operation_id=key[1]
        )
        with self._lock:
            self._running[key] = control
        try:
            tool = str(state.prepared["tool"])
            if tool in INTERNAL_TASK_FILE_OPERATIONS:
                native_result = self._execute_task_file_operation(
                    tool,
                    state.execution_args,
                    native_facts=state.native_facts,
                    blobs=state.input_blobs,
                    task_id=task_id,
                )
            else:
                native_result = execute_native_operation(
                    self.workspace_root,
                    tool,
                    state.execution_args,
                    native_facts=state.native_facts,
                    blobs=state.input_blobs,
                    task_id=task_id,
                    control=control,
                )
            if (
                len(native_result.blobs) > MAX_NATIVE_RESULT_BLOBS
                or sum(len(payload) for payload in native_result.blobs.values()) > MAX_STAGED_BLOB_BYTES
            ):
                raise ExecdError(
                    "native_blob_set_too_large",
                    "Native operation output exceeds the atomic blob-set limit.",
                    phase="finalize",
                    completion="completed",
                )
            output_blobs: dict[str, str] = {}
            for blob_id, payload in native_result.blobs.items():
                digest = str(blob_id)
                if not _state.HASH_RE.fullmatch(digest):
                    raise ExecdError(
                        "native_blob_id_invalid",
                        "Native operation returned a non-CAS blob ID.",
                        phase="finalize",
                        completion="completed",
                    )
                output_blobs[digest] = self.cas.put(
                    bytes(payload),
                    expected_sha256=digest,
                )
            envelope = _wire_json(native_result.envelope)
            envelope = _bound_envelope(envelope, self.cas)
            result = {
                "completion": "completed",
                "prepared_hash": state.prepared_hash,
                "envelope": envelope,
                "output_blobs": output_blobs,
            }
        except BaseException as exc:
            diagnostic = _state.exception_diagnostic(
                exc,
                request_id=key[0],
                operation_id=key[1],
                phase="execute",
                completion="completed",
                domain="process",
            )
            result = {
                "completion": "completed",
                "prepared_hash": state.prepared_hash,
                "envelope": _wire_json(
                    ToolExecutionEnvelope(
                        text=_state.safe_error_text(exc),
                        diagnostic=_diagnostic(diagnostic),
                    )
                ),
                "output_blobs": {},
            }
        finally:
            with self._lock:
                self._running.pop(key, None)
        return self.journal.complete(
            task_id=task_id,
            operation_id=key[1],
            request_hash=state.prepared_hash,
            result=result,
        )

    def abort(self, request_id: str, operation_id: str, prepared_token: str = "") -> bool:
        key = (_state.opaque(request_id, "request_id"), _state.opaque(operation_id, "operation_id"))
        with self._lock:
            state = self._prepared.get(key)
            if state is None or (prepared_token and state.token != prepared_token):
                return False
            self._prepared.pop(key, None)
            return True

    def reconcile(self, request_id: str, operation_id: str, prepared_hash: str) -> dict[str, Any]:
        key = (_state.opaque(request_id, "request_id"), _state.opaque(operation_id, "operation_id"))
        task_id = ""
        with self._lock:
            prepared = self._prepared.get(key)
            if prepared is not None:
                task_id = str(prepared.prepared.get("task_id") or "")
        return self.journal.reconcile(task_id, key[1], prepared_hash)

    def cancel(
        self,
        *,
        task_id: str = "",
        request_id: str = "",
        operation_id: str = "",
    ) -> bool:
        task_id = _state.opaque(task_id, "task_id", optional=True)
        exact = (
            (_state.opaque(request_id, "request_id"), _state.opaque(operation_id, "operation_id"))
            if request_id and operation_id
            else None
        )
        cancelled = False
        with self._lock:
            for key, control in list(self._running.items()):
                if exact == key or (task_id and control.task_id == task_id):
                    control.cancel()
                    cancelled = True
            for key, state in list(self._prepared.items()):
                if exact == key or (task_id and state.prepared.get("task_id") == task_id):
                    self._prepared.pop(key, None)
                    cancelled = True
        if task_id:
            cancelled = bool(self.custody.cancel_task(task_id)) or cancelled
            cancelled = self.task_files.cleanup_task(task_id) or cancelled
            # D8 retention, TERMINAL half: a sealed log's quota was reserved and never
            # handed back, so the host-wide 8 GiB was a one-way ratchet.
            self.process_logs.release_task(task_id)
        return cancelled

    def renew_lease(self, ttl_ms: int, task_id: str = "", server_generation: str = "") -> None:
        self.custody.renew(
            ttl_ms=ttl_ms,
            task_id=task_id,
            server_generation=server_generation,
        )
        # ...and the AGE half, on the custody tick: the terminal half needs a cancel a
        # Home that died mid-task never sends. Bounded work on a path already writing.
        self.process_logs.expire_retained()

    def fetch_blob(self, blob_id: str, max_bytes: int) -> bytes:
        return self.cas.read(blob_id, max_bytes=max_bytes)

    def acknowledge(self, task_id: str, operation_id: str, prepared_hash: str) -> None:
        self.journal.acknowledge(task_id, operation_id, prepared_hash)

    def close(self, *, kill_owned: bool = True) -> None:
        with self._lock:
            controls = list(self._running.values())
            self._prepared.clear()
        for control in controls:
            control.cancel()
        if kill_owned:
            self.custody.kill_generation(self._custodian_id)
        if self._custodian_id:
            self.custody.request_custodian_close(self._custodian_id)
        process = self._custodian_process
        retained = bool(self.custody.refresh_snapshot().get("groups"))
        if process is not None and process.poll() is None and not retained:
            try:
                process.terminate()
            except OSError:
                pass

    def _prepare_task_file_operation(
        self,
        tool: str,
        args: Mapping[str, Any],
        *,
        task_id: str,
        blobs: Mapping[str, bytes],
    ) -> NativePreparedOperation:
        if not task_id:
            raise ExecdError(
                "remote_task_id_required",
                "Internal task-file operations require a task identity.",
                phase="prepare",
            )
        try:
            if tool == ATTACHMENT_STAGE_OPERATION:
                manifest, _verified = attachment_blob_map(
                    args.get("manifest"),
                    blobs,
                )
                return NativePreparedOperation(
                    execution_args={"manifest": manifest},
                    native_facts={"attachment_count": len(manifest)},
                )
            if tool == MEDIA_EXPORT_OPERATION:
                facts, _payload = self.task_files.export_media(
                    self.workspace_root, args, task_id=task_id
                )
                return NativePreparedOperation(
                    execution_args=media_export_execution_args(args, facts),
                    native_facts={**facts, **media_export_policy_facts(args)},
                )
        except RemoteTaskFileError as exc:
            raise ExecdError(
                exc.code,
                str(exc),
                phase="prepare",
            ) from exc
        raise ExecdError(
            "native_operation_forbidden",
            "Internal task-file operation is not allowlisted.",
            phase="prepare",
        )

    def _execute_task_file_operation(
        self,
        tool: str,
        args: Mapping[str, Any],
        *,
        native_facts: Mapping[str, Any],
        blobs: Mapping[str, bytes],
        task_id: str,
    ) -> NativeOperationResult:
        try:
            if tool == ATTACHMENT_STAGE_OPERATION:
                manifest = self.task_files.stage_attachments(
                    task_id,
                    args.get("manifest"),
                    blobs,
                )
                return NativeOperationResult(
                    ToolExecutionEnvelope(
                        text="Remote task attachments staged.",
                        trace={"attachment_manifest": manifest},
                    )
                )
            if tool == MEDIA_EXPORT_OPERATION:
                facts, payload = self.task_files.export_media(
                    self.workspace_root,
                    args,
                    task_id=task_id,
                    expected_sha256=str(native_facts.get("sha256") or ""),
                    expected_size=int(native_facts.get("size") or 0),
                )
                digest = facts["sha256"]
                return NativeOperationResult(
                    ToolExecutionEnvelope(
                        text="Remote task media exported.",
                        artifacts=[media_export_artifact_row(facts)],
                        trace={
                            "remote_media": facts,
                            # Manifest for Home's judge; why at `MANIFEST_TRACE_KEYS`.
                            **export_disclosure_block(
                                native_facts, [], [str(facts.get("relative_path") or "")],
                                question=QUESTION_EXPORT),
                        },
                    ),
                    blobs={digest: payload},
                )
        except RemoteTaskFileError as exc:
            raise ExecdError(
                exc.code,
                str(exc),
                phase="execute",
                completion="completed",
            ) from exc
        raise ExecdError(
            "native_operation_forbidden",
            "Internal task-file operation is not allowlisted.",
            phase="execute",
            completion="completed",
        )

    def _revalidate_workspace(self) -> None:
        root, facts = _admit_git_workspace(self.workspace_root)
        identity = _workspace_identity(self.host_id, root, facts)
        if root != self.workspace_root or identity != self.workspace_id:
            raise ExecdError(
                "workspace_identity_mismatch",
                "Workspace was replaced or rebound after admission.",
                phase="prepare",
            )

    def _prepare_operation(
        self,
        tool: str,
        prepare_args: Mapping[str, Any],
        *,
        task_id: str,
        blobs: Mapping[str, bytes],
    ) -> NativePreparedOperation:
        """The ONE prepare door: resolve the operation, then bundle its facts.

        Both the prepare RPC and the post-authorization revalidation come through
        here, so the bundled fact block is part of the SAME `native_facts` both
        compare — bundling at one call site only would make every replay look like
        a changed target.
        """

        if tool in INTERNAL_TASK_FILE_OPERATIONS:
            prepared = self._prepare_task_file_operation(
                tool, prepare_args, task_id=task_id, blobs=blobs
            )
        else:
            prepared = prepare_native_operation(
                self.workspace_root,
                tool,
                prepare_args,
                task_id=task_id,
                **({"blobs": blobs} if tool == "execute_reviewed_payload" else {}),
            )
        bundle_prepared_facts(
            prepared.native_facts,
            root=self.workspace_root,
            run_git=functools.partial(
                subprocess.run,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                timeout=10,
                check=False,
            ),
        )
        return prepared

    def _revalidate_prepared_target(self, state: _PreparedState, *, task_id: str) -> None:
        """Re-resolve target facts after authorization and before journaling."""

        tool = str(state.prepared["tool"])
        try:
            refreshed = self._prepare_operation(
                tool, state.prepare_args, task_id=task_id, blobs=state.input_blobs
            )
            execution_args = _state.json_object(refreshed.execution_args, "execution_args")
            native_facts = _state.json_object(refreshed.native_facts, "native_facts")
        except Exception as exc:
            raise ExecdError(
                "prepared_target_revalidation_failed",
                "Prepared target facts are no longer valid.",
                phase="authorize",
                details={"error_type": type(exc).__name__},
            ) from exc
        if execution_args != state.execution_args or native_facts != state.native_facts:
            raise ExecdError(
                "prepared_target_changed",
                "Prepared target facts changed before execution.",
                phase="authorize",
            )

    def _spawn_custodian(self) -> subprocess.Popen[bytes]:
        configured_self = os.environ.get("OUROBOROS_EXECD_SELF", "").strip()
        if configured_self:
            cmd = [configured_self]
        elif getattr(sys, "frozen", False):
            cmd = [sys.executable]
        else:
            cmd = [sys.executable, "-m", "ouroboros.execd"]
        cmd.extend(["--custodian", str(self.custody.state_path), "--server-generation", self.server_generation])
        deadline = time.monotonic() + (_state.MAX_LEASE_TTL_MS / 1000.0) + 2.0
        while True:
            try:
                custodian_id = self.custody.claim_custodian()
                break
            except ExecdError as exc:
                if exc.code not in {"generation_active", "generation_closing"}:
                    raise
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
        self._custodian_id = custodian_id
        cmd.extend(["--custodian-id", custodian_id])
        try:
            return subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **subprocess_new_group_kwargs(),
            )
        except BaseException:
            self.custody.request_custodian_close(custodian_id)
            raise


class ExecdProtocolServer:
    """Strict framed stdio adapter; ExecdService remains the state authority."""

    def __init__(self, service: ExecdService, reader: BinaryIO, writer: BinaryIO) -> None:
        self.service = service
        self.reader = reader
        self.writer = writer
        self._receive_sequence = ControlSequence()
        self._send_sequence = 0
        self._write_lock = threading.Lock()
        self._operation_threads: set[threading.Thread] = set()
        self._incoming_blob: dict[str, Any] | None = None
        self._uploaded_blobs: dict[tuple[str, str], dict[str, str]] = {}
        self._outgoing_blob: dict[str, Any] | None = None
        self._blob_credit = threading.Condition()
        self._serving_stop = False

    def serve(self) -> None:
        try:
            while True:
                label, payload = read_frame(self.reader)
                if label == "bulk":
                    self._receive_bulk(bytes(payload))
                    continue
                assert isinstance(payload, dict)
                self._receive_sequence.observe(payload)
                try:
                    self._receive_control(payload)
                except ProtocolError:
                    raise
                except Exception as exc:
                    if "request_id" not in payload:
                        raise
                    diagnostic = _state.exception_diagnostic(
                        exc,
                        request_id=str(payload["request_id"]),
                        operation_id=str(payload.get("operation_id") or payload["request_id"]),
                        phase="execute",
                        completion="not_started",
                    )
                    self._send(
                        "diagnostic",
                        request_id=str(payload["request_id"]),
                        operation_id=str(payload.get("operation_id") or payload["request_id"]),
                        diagnostic=diagnostic,
                    )
                if self._serving_stop:
                    return
        except ProtocolEOF:
            pass
        finally:
            for staged in self._uploaded_blobs.values():
                for digest in staged.values():
                    self.service.cas.unpin(digest)
            self._uploaded_blobs.clear()
            self.service.close(kill_owned=True)

    def _receive_control(self, message: dict[str, Any]) -> None:
        kind = str(message["kind"])
        if kind == "handshake":
            # CONTRACT-SET ADMISSION at the first frame (rationale and both seams:
            # `remote_contract_admission`). This is the direction Home cannot cover,
            # because the Home in question may predate the check entirely.
            admit_execd_contract_set(
                message.get("protocol_minor"),
                release=self.service.release_id,
            )
            facts = self.service.handshake(str(message.get("capability_hash") or ""))
            self._send(
                "handshake_ok",
                protocol_major=PROTOCOL_MAJOR,
                protocol_minor=PROTOCOL_MINOR,
                host_id=facts["host_id"],
                server_generation=self.service.server_generation,
                platform=json.dumps(facts["platform"], sort_keys=True)[:4096],
                build=EXECD_BUILD,
                capability_hash=self.service.capability_hash,
                optional={
                    "artifact": {
                        "release_id": facts["release_id"],
                        "sha256": facts["artifact_sha256"],
                    },
                    "admission": {
                        key: facts[key]
                        for key in (
                            "workspace_id",
                            "canonical_root",
                            "git",
                            "session_id",
                        )
                    },
                },
            )
            return
        if kind == "blob_manifest":
            if self._incoming_blob is not None:
                raise ProtocolError("another blob upload is already active")
            size = int(message["size"])
            if size < 0 or size > MAX_STAGED_BLOB_BYTES:
                raise ProtocolError("blob manifest size exceeds execd limit")
            self._incoming_blob = {
                "request_id": message["request_id"],
                "operation_id": str(message.get("operation_id") or ""),
                "blob_id": message["blob_id"],
                "size": size,
                "sha256": str(message["sha256"]),
                "data": bytearray(),
                "chunk_seq": 0,
            }
            if size == 0:
                self._finish_incoming_blob()
                self._send(
                    "blob_ack",
                    request_id=message["request_id"],
                    operation_id=str(message.get("operation_id") or ""),
                    blob_id=message["blob_id"],
                    chunk_seq=0,
                    complete=True,
                )
            return
        if kind == "blob_fetch":
            if self._outgoing_blob is not None:
                raise ProtocolError("another blob download is already active")
            payload = self.service.fetch_blob(
                str(message["blob_id"]),
                int(message["size"]),
            )
            thread = threading.Thread(
                target=self._send_blob,
                args=(str(message["request_id"]), str(message["blob_id"]), payload),
                daemon=True,
                name="execd-blob-download",
            )
            self._operation_threads.add(thread)
            thread.start()
            return
        if kind == "prepare":
            blob_key = (str(message["request_id"]), str(message["operation_id"]))
            uploaded = self._uploaded_blobs.pop(blob_key, {})
            try:
                response = self.service.prepare(
                    request_id=message["request_id"],
                    operation_id=message["operation_id"],
                    tool=message["tool"],
                    args=message["args"],
                    task_id=str(message.get("task_id") or ""),
                    blobs={
                        blob_id: self.service.cas.read(
                            digest,
                            max_bytes=MAX_STAGED_BLOB_BYTES,
                        )
                        for blob_id, digest in uploaded.items()
                    },
                    deadline_ms=(int(message["deadline_ms"]) if "deadline_ms" in message else None),
                )
            finally:
                for digest in uploaded.values():
                    self.service.cas.unpin(digest)
            self._send(
                "prepared",
                request_id=response["request_id"],
                operation_id=response["operation_id"],
                prepared_hash=response["prepared_hash"],
                prepared={
                    "prepared_token": response["prepared_token"],
                    "tool": response["tool"],
                    "execution_args": response["execution_args"],
                    "native_facts": response["native_facts"],
                },
                expires_ms=response["expires_at_ms"],
            )
            return
        if kind == "continue":
            thread = threading.Thread(
                target=self._continue_and_send,
                args=(dict(message),),
                daemon=True,
                name=f"execd-operation-{str(message['operation_id'])[:24]}",
            )
            self._operation_threads.add(thread)
            thread.start()
            return
        if kind == "abort":
            self.service.abort(
                message["request_id"],
                message["operation_id"],
                str(message.get("optional", {}).get("prepared_token") or ""),
            )
            self._send(
                "ack",
                ack_seq=int(message["seq"]),
                request_id=message["request_id"],
                operation_id=message["operation_id"],
            )
            return
        if kind == "reconcile":
            result = self.service.reconcile(
                message["request_id"],
                message["operation_id"],
                message["prepared_hash"],
            )
            self._send(
                "reconcile_result",
                request_id=message["request_id"],
                operation_id=message["operation_id"],
                completion=result["completion"],
                result=_bounded_wire_result(result),
            )
            return
        if kind == "lease":
            self._answer_lease(message)
            return
        if kind == "cancel":
            self.service.cancel(
                task_id=str(message.get("task_id") or ""),
                request_id=message["request_id"],
                operation_id=message["operation_id"],
            )
            self._send(
                "ack",
                ack_seq=int(message["seq"]),
                request_id=message["request_id"],
                operation_id=message["operation_id"],
            )
            return
        if kind == "panic":
            self.service.close(kill_owned=True)
            self._serving_stop = True
            return
        if kind == "ack":
            if message.get("operation_id") and message.get("request_id"):
                self.service.acknowledge(
                    "",
                    str(message["operation_id"]),
                    str(message.get("optional", {}).get("prepared_hash") or ""),
                )
                self._send(
                    "ack",
                    ack_seq=int(message["seq"]),
                    request_id=str(message["request_id"]),
                    operation_id=str(message["operation_id"]),
                )
            return
        if kind == "blob_ack":
            with self._blob_credit:
                outgoing = self._outgoing_blob
                if (
                    outgoing is not None
                    and outgoing["blob_id"] == message["blob_id"]
                    and outgoing["chunk_seq"] == message["chunk_seq"]
                ):
                    outgoing["acked"] = True
                    self._blob_credit.notify_all()
            return
        raise ProtocolError(f"unsupported execd control kind: {kind}")

    def _receive_bulk(self, payload: bytes) -> None:
        state = self._incoming_blob
        if state is None:
            raise ProtocolError("bulk frame has no manifest")
        state["data"].extend(payload)
        if len(state["data"]) > state["size"]:
            raise ProtocolError("bulk payload exceeds declared size")
        complete = len(state["data"]) == state["size"]
        self._send(
            "blob_ack",
            request_id=state["request_id"],
            operation_id=state["operation_id"],
            blob_id=state["blob_id"],
            chunk_seq=state["chunk_seq"],
            complete=complete,
        )
        state["chunk_seq"] += 1
        if complete:
            self._finish_incoming_blob()

    def _finish_incoming_blob(self) -> None:
        state = self._incoming_blob
        if state is None:
            raise ProtocolError("blob manifest state is absent")
        digest = self.service.cas.put(
            bytes(state["data"]),
            expected_sha256=state["sha256"],
        )
        key = (str(state["request_id"]), str(state["operation_id"]))
        staged = self._uploaded_blobs.setdefault(key, {})
        blob_id = str(state["blob_id"])
        if blob_id in staged and staged[blob_id] != digest:
            raise ProtocolError("blob ID was reused with different content")
        if blob_id not in staged:
            self.service.cas.pin(digest)
            staged[blob_id] = digest
        self._incoming_blob = None

    def _send_blob(self, request_id: str, blob_id: str, payload: bytes) -> None:
        try:
            digest = hashlib.sha256(payload).hexdigest()
            self._send(
                "blob_manifest",
                request_id=request_id,
                blob_id=blob_id,
                size=len(payload),
                sha256=digest,
            )
            for chunk_seq, offset in enumerate(range(0, len(payload), MAX_BULK_BYTES)):
                with self._blob_credit:
                    self._outgoing_blob = {
                        "blob_id": blob_id,
                        "chunk_seq": chunk_seq,
                        "acked": False,
                    }
                with self._write_lock:
                    self.writer.write(encode_bulk(payload[offset : offset + MAX_BULK_BYTES]))
                    self.writer.flush()
                with self._blob_credit:
                    if not self._blob_credit.wait_for(
                        lambda: bool(self._outgoing_blob and self._outgoing_blob.get("acked")),
                        timeout=30,
                    ):
                        raise ProtocolError("blob receiver did not return credit")
        finally:
            with self._blob_credit:
                self._outgoing_blob = None
                self._blob_credit.notify_all()
            self._operation_threads.discard(threading.current_thread())

    def _continue_and_send(self, message: dict[str, Any]) -> None:
        try:
            token = str(message.get("optional", {}).get("prepared_token") or "")
            result = self.service.continue_prepared(
                request_id=message["request_id"],
                operation_id=message["operation_id"],
                prepared_hash=message["prepared_hash"],
                prepared_token=token,
            )
            self._send(
                "result",
                request_id=message["request_id"],
                operation_id=message["operation_id"],
                completion=str(result.get("completion") or "unknown"),
                result=_bounded_wire_result(result),
                prepared_hash=message["prepared_hash"],
            )
        except Exception as exc:
            diagnostic = _state.exception_diagnostic(
                exc,
                request_id=str(message["request_id"]),
                operation_id=str(message["operation_id"]),
                phase="execute",
                completion="unknown",
            )
            self._send(
                "diagnostic",
                request_id=message["request_id"],
                operation_id=message["operation_id"],
                diagnostic=diagnostic,
            )
        finally:
            self._operation_threads.discard(threading.current_thread())

    def _answer_lease(self, message: dict[str, Any]) -> None:
        """Ack an honored lease, refuse a foreign one with a typed diagnostic.

        A lease frame carries no ``request_id``, so the generic handler in ``serve``
        cannot answer for it: an exception escaping here would end the SESSION instead
        of refusing the one lease.  Renewal is the most frequent frame on the wire, so
        this adds no I/O of its own — acceptance rides the renewal custody already
        performs, a refusal is decided before the state file is touched, and repeating
        an identical lease produces the same answer and no extra state.
        """

        lease_id = str(message["lease_id"])
        try:
            self.service.renew_lease(
                int(message["ttl_ms"]),
                str(message.get("task_id") or ""),
                server_generation=str(message["server_generation"]),
            )
        except ExecdError as exc:
            self._send(
                "diagnostic",
                request_id=lease_id,
                operation_id=lease_id,
                diagnostic=exc.diagnostic(lease_id, lease_id),
                optional=lease_answer_marker(lease_id),
            )
            return
        self._send(
            "ack",
            ack_seq=int(message["seq"]),
            optional=lease_answer_marker(lease_id, self.service.server_generation),
        )

    def _send(self, kind: str, **fields: Any) -> None:
        with self._write_lock:
            message = {"kind": kind, "seq": self._send_sequence, **fields}
            self._send_sequence += 1
            self.writer.write(encode_control(message))
            self.writer.flush()


def _admit_git_workspace(path: pathlib.Path) -> tuple[pathlib.Path, dict[str, Any]]:
    root = pathlib.Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ExecdError("workspace_not_directory", "Workspace root is not a directory.", phase="bootstrap")

    run_git = functools.partial(
        subprocess.run,
        cwd=str(root),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        timeout=10,
        check=False,
    )
    toplevel_result = run_git(["git", "rev-parse", "--show-toplevel"])
    if toplevel_result.returncode != 0:
        raise ExecdError(
            "workspace_not_git",
            "Remote project admission requires a git worktree root.",
            phase="bootstrap",
            details={"stderr": toplevel_result.stderr[-2000:]},
        )
    common_result = run_git(["git", "rev-parse", "--git-common-dir"])
    common_text = common_result.stdout.strip()
    common_candidate = pathlib.Path(common_text) if pathlib.Path(common_text).is_absolute() else root / common_text
    if common_result.returncode != 0 or not common_text or common_text.startswith("-") or not common_candidate.exists():
        common_result = run_git(["git", "rev-parse", "--git-dir"])
    toplevel_text = toplevel_result.stdout.strip()
    common_text = common_result.stdout.strip()
    if common_result.returncode != 0 or not toplevel_text or not common_text:
        raise ExecdError(
            "git_facts_invalid",
            "Git returned incomplete workspace facts.",
            phase="bootstrap",
        )
    toplevel = pathlib.Path(toplevel_text).resolve(strict=True)
    common = (
        pathlib.Path(common_text) if pathlib.Path(common_text).is_absolute() else (toplevel / common_text)
    ).resolve(strict=True)
    if toplevel != root:
        raise ExecdError(
            "workspace_not_git_root",
            "Remote project must attach at the canonical git toplevel.",
            phase="bootstrap",
            details={"canonical_root": str(toplevel)},
        )
    stat = root.stat()
    admission = _admission_git_state(root)
    return root, {
        "toplevel": str(toplevel),
        "common_dir": str(common),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        **admission,
    }


def _admission_git_state(root: pathlib.Path) -> dict[str, Any]:
    """Capture the immutable base fence without exposing worktree path names."""

    run_git = functools.partial(
        subprocess.run,
        cwd=str(root),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )
    head_result = run_git(["git", "rev-parse", "--verify", "HEAD"])
    head = head_result.stdout.decode("ascii", errors="ignore").strip()
    branch_result = run_git(["git", "symbolic-ref", "-q", "--short", "HEAD"])
    branch = branch_result.stdout.decode("utf-8", errors="replace").strip()
    status_result = run_git(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ]
    )
    if status_result.returncode:
        status_result = run_git(
            [
                "git",
                "-c",
                "diff.ignoreSubmodules=none",
                "status",
                "--porcelain",
                "--untracked-files=all",
            ]
        )
    if status_result.returncode:
        raise ExecdError(
            "git_status_unavailable",
            "Git status could not be captured at workspace admission.",
            phase="bootstrap",
        )
    index_result = run_git(["git", "rev-parse", "--git-path", "index"])
    index_text = index_result.stdout.decode("utf-8", errors="strict").strip()
    if index_result.returncode or not index_text or index_text.startswith("-") or "\n" in index_text:
        index_result = run_git(["git", "rev-parse", "--git-dir"])
        if index_result.returncode:
            raise ExecdError(
                "git_index_unavailable",
                "Git index identity could not be captured at workspace admission.",
                phase="bootstrap",
            )
        index_text = os.path.join(
            index_result.stdout.decode("utf-8", errors="strict").strip(),
            "index",
        )
    index_path = pathlib.Path(index_text)
    if not index_path.is_absolute():
        index_path = root / index_path
    try:
        index_bytes = index_path.read_bytes()
        index_present = True
    except FileNotFoundError:
        index_bytes = b""
        index_present = False
    status_lines = [line for line in status_result.stdout.splitlines() if line.strip()]
    return {
        "head": head,
        "head_present": bool(head_result.returncode == 0 and head),
        "branch": branch,
        "index_present": index_present,
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "status_sha256": hashlib.sha256(status_result.stdout).hexdigest(),
        "dirty": bool(status_lines),
        "status_count": len(status_lines),
    }


def _workspace_identity(
    host_id: str,
    root: pathlib.Path,
    git_facts: Mapping[str, Any],
) -> str:
    identity = canonical_json(
        {
            "host_id": host_id,
            "canonical_root": str(root),
            "git_toplevel": str(git_facts["toplevel"]),
            "git_common_dir": str(git_facts["common_dir"]),
            "device": int(git_facts["device"]),
            "inode": int(git_facts["inode"]),
        }
    )
    return hashlib.sha256(identity).hexdigest()


def _diagnostic(raw: Mapping[str, Any]) -> ExecutionDiagnostic:
    domain = str(raw.get("domain") or "protocol")
    completion = str(raw.get("completion") or "unknown")
    return ExecutionDiagnostic(
        domain=domain
        if domain in {"transport", "protocol", "policy", "filesystem", "process", "artifact"}
        else "protocol",  # type: ignore[arg-type]
        code=str(raw.get("code") or "remote_error"),
        message=str(raw.get("message") or "Remote operation failed."),
        phase=str(raw.get("phase") or "execute"),
        request_id=str(raw.get("request_id") or ""),
        operation_id=str(raw.get("operation_id") or ""),
        completion=completion if completion in {"not_started", "completed", "unknown"} else "unknown",  # type: ignore[arg-type]
        retryable=bool(raw.get("retryable")),
        # `errno`, read the same way the HOME-side parser reads it
        # (`remote_worker_proxy.diagnostic_from_dict`). This arm dropped it, so a
        # diagnostic that round-tripped through the target lost the one field that
        # says WHICH filesystem refusal it was — a derived value erased by the
        # PARSER rather than by whoever produced it, which is the same class as a
        # projection that drops a field the constructor computed.
        errno=int(raw["errno"]) if isinstance(raw.get("errno"), int) else None,
        details=dict(raw.get("details") or {}),
    )


def _wire_json(value: Any) -> Any:
    """Normalize dataclass/tuple values without weakening the strict codec."""

    if dataclasses.is_dataclass(value):
        return _wire_json(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _wire_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_wire_json(item) for item in value]
    return value


def _bounded_wire_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Keep full durable results while omitting an oversized CAS index on wire."""

    projected = {str(key): value for key, value in result.items()}
    nested = projected.get("result")
    if isinstance(nested, Mapping):
        projected["result"] = _bounded_wire_result(nested)
    output_blobs = projected.get("output_blobs")
    if not isinstance(output_blobs, Mapping):
        return projected
    try:
        encoded = canonical_json(projected)
        if len(encoded) <= MAX_CONTROL_BYTES - 32_768:
            return projected
    except ProtocolError:
        pass
    # Omission is intentional: Home then trusts only SHA/size declarations in
    # the verified (possibly externalized) envelope artifacts.
    projected.pop("output_blobs", None)
    return projected


def _bound_envelope(
    envelope: dict[str, Any],
    cas: CASBlobStore,
) -> dict[str, Any]:
    """Externalize an otherwise oversized successful result before framing."""

    encoded = json.dumps(
        envelope,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    try:
        if len(canonical_json(envelope)) <= MAX_CONTROL_BYTES // 2:
            return envelope
    except ProtocolError:
        pass
    if len(encoded) > MAX_REMOTE_EXTERNAL_ENVELOPE_BYTES:
        raise ExecdError(
            "native_envelope_too_large",
            "Native operation envelope exceeds the import limit.",
            phase="finalize",
            completion="completed",
        )
    digest = cas.put(encoded)
    ref = {
        "name": "operation-envelope.json",
        "blob_id": digest,
        "sha256": digest,
        "size": len(encoded),
        "mime": "application/json",
        "truncated": False,
    }
    text = str(envelope.get("text") or "")
    preview = text[:64_000]
    if len(text) > len(preview):
        preview += "\n… remote result preview truncated; fetch operation-envelope.json"
    # Every bound here is DISCLOSED with an exact count beside the bounded list, the
    # way the export disclosure block does it. A bare `[:128]` told Home it had every
    # artifact row while silently dropping the 129th — and the complete envelope is
    # right there in the blob, so the omission is recoverable, but only if the reader
    # is told it happened.
    declared = [item for item in list(envelope.get("artifacts") or []) if isinstance(item, Mapping)]
    artifacts = [dict(item) for item in declared[:_MAX_ENVELOPE_ARTIFACT_ROWS]]
    listed = len(artifacts)
    artifacts.append(ref)
    diagnostic = envelope.get("diagnostic")
    if len(declared) > listed:
        preview += (
            f"\n⚠️ ARTIFACT_ROWS_TRUNCATED: {len(declared)} artifact rows were "
            f"produced, {listed} are listed here; the complete set is in "
            "operation-envelope.json."
        )
    return {
        "text": preview,
        "diagnostic": diagnostic if isinstance(diagnostic, Mapping) else None,
        "process": None,
        "artifacts": artifacts,
        "trace": {
            "completion": "complete",
            "externalized_result": ref,
            "artifact_rows_total": len(declared),
            "artifact_rows_listed": listed,
            "artifact_rows_truncated": len(declared) > listed,
        },
    }


def _missing_result(
    key: tuple[str, str],
    prepared_hash: str,
    *,
    unavailable: bool,
) -> dict[str, Any]:
    domain = "artifact" if unavailable else "protocol"
    code = "result_unavailable" if unavailable else "completion_unknown"
    message = (
        "Remote operation completed but its stored result is unavailable."
        if unavailable
        else "Remote operation started but no durable result is available."
    )
    phase = "import" if unavailable else "finalize"
    completion = "completed" if unavailable else "unknown"
    diagnostic = ExecutionDiagnostic(
        domain=domain,
        code=code,
        message=message,
        phase=phase,
        request_id=key[0],
        operation_id=key[1],
        completion=completion,
        retryable=False,
    )
    return {
        "completion": completion,
        "prepared_hash": prepared_hash,
        "envelope": _wire_json(ToolExecutionEnvelope(text=diagnostic.message, diagnostic=diagnostic)),
        "output_blobs": {},
    }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ouroboros-execd")
    parser.add_argument("--state-root")
    parser.add_argument("--workspace-root")
    parser.add_argument("--workspace-id", default="")
    parser.add_argument("--connection-id", default="")
    parser.add_argument("--project-id", default="")
    parser.add_argument("--server-generation", default="")
    parser.add_argument("--release-id", default="")
    parser.add_argument("--artifact-sha256", default="")
    parser.add_argument("--capability-manifest")
    parser.add_argument("--session-nonce", default="")
    parser.add_argument("--custodian")
    parser.add_argument("--custodian-id", default="")
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--print-host-id", action="store_true")
    parser.add_argument("--initialize-host-id", action="store_true")
    args = parser.parse_args(argv)
    if args.version:
        print(f"ouroboros-execd {EXECD_BUILD}")
        return 0
    if args.print_host_id and args.initialize_host_id:
        parser.error("--print-host-id and --initialize-host-id are mutually exclusive")
    if args.print_host_id:
        if not args.state_root:
            parser.error("--print-host-id requires --state-root")
        print(continuity_host_id(pathlib.Path(args.state_root)))
        return 0
    if args.initialize_host_id:
        if not args.state_root:
            parser.error("--initialize-host-id requires --state-root")
        print(initialize_continuity_host_id(pathlib.Path(args.state_root)))
        return 0
    if not args.server_generation:
        parser.error("missing required argument: --server-generation")
    if args.custodian:
        if not args.custodian_id:
            parser.error("--custodian requires --custodian-id")
        return _state.run_custodian(pathlib.Path(args.custodian), args.server_generation, args.custodian_id)
    required = {
        "--state-root": args.state_root,
        "--workspace-root": args.workspace_root,
        "--connection-id": args.connection_id,
        "--project-id": args.project_id,
        "--release-id": args.release_id,
        "--artifact-sha256": args.artifact_sha256,
        "--capability-manifest": args.capability_manifest,
        "--session-nonce": args.session_nonce,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")
    try:
        nonce = bytes.fromhex(args.session_nonce)
    except ValueError as exc:
        raise SystemExit("invalid --session-nonce") from exc
    capability_manifest = _state.read_json(
        pathlib.Path(args.capability_manifest),
        required=True,
    )
    assert capability_manifest is not None
    service = ExecdService(
        pathlib.Path(args.state_root),
        pathlib.Path(args.workspace_root),
        connection_id=args.connection_id,
        project_id=args.project_id,
        server_generation=args.server_generation,
        release_id=args.release_id,
        artifact_sha256=args.artifact_sha256,
        capability_manifest=capability_manifest,
        requested_workspace_id=args.workspace_id,
        start_custodian=True,
    )
    stdout = sys.stdout.buffer
    stdout.write(session_preamble(nonce))
    stdout.flush()
    ExecdProtocolServer(service, sys.stdin.buffer, stdout).serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
