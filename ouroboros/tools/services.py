"""Task-scoped long-running service manager.

``service_execution_facts`` is the only source of the sleep ``services`` selector:
the real Popen return code for host and local-executor services; a Docker service
reports no return code (its in-container leader observation is not the service's)
and an inconclusive probe reads ``unknown``. The park loop polls it about every
second as host work, never a model round.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shlex
import stat
import subprocess
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, BinaryIO, Dict, List

from ouroboros.owner_pause import OwnerPauseRefused
from ouroboros.launch_custody import (
    open_service_log, open_service_log_source, service_log_allows_standard_masking, service_log_archive_root,
    service_log_files, service_log_launch_id,
)
from ouroboros import service_outcomes, workspace_executor
from ouroboros.confined_files import native_name
from ouroboros.observability import read_blob_ref, redact_projection, write_blob
from ouroboros.secret_masking import redact_known_values
from ouroboros.platform_layer import (
    bootstrap_process_path,
    kill_process_group_id,
    kill_process_tree,
    process_group_id,
    request_process_tree_kill,
)
from ouroboros.process_interpreters import (
    active_node_resolution,
    apply_env_path_prepend,
    interpreter_path_overlay,
)
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.config import load_settings, selection_settings
from ouroboros.tools.tool_result import (
    ToolResult,
    _publish_tool_result,
    append_published_text,
    launch_refusal_result,
    safety_refusal,
)
from ouroboros.tool_access import (
    ResolvedResourceBinding,
    active_tool_profile,
    build_resolved_resource_binding,
    canonical_data_root,
    shell_cwd_block_message,
)
from ouroboros.tool_access_paths import PRIVATE_SERVICE_LOGS
from ouroboros.utils import append_jsonl, fsync_folder, utc_now_iso
from ouroboros.tools.kept_service_outputs import (
    KEPT_DECLARATION, KEPT_DIRECTORY, KeptCaptureBusy, KeptNamesake, capture_before_reuse, declare_kept,
    finish_own_outputs, pending_output_stop, release_after_stop, retry_named_outputs, retry_own_outputs,
    retire_kept_declaration, retire_kept_declarations, retract_unstarted, service_name_hold, writer_reservation,
)
from ouroboros.tools.shell_process import _executor_can_run_cwd
from ouroboros.workspace_executor import executor_ref_from_ctx
from ouroboros.workspace_executor import overlay_env, resolve_process_env, validate_process_env
from ouroboros.workspace_executor import kill_all_services as executor_kill_all_services
from ouroboros.workspace_executor import named_service as named_executor_service, replacement_state
from ouroboros.workspace_executor import service_execution_facts as executor_service_execution_facts
from ouroboros.workspace_executor import service_env as _service_env
from ouroboros.workspace_executor import service_key as _service_key
from ouroboros.workspace_executor import service_logs as executor_service_logs
from ouroboros.workspace_executor import service_status as executor_service_status
from ouroboros.workspace_executor import stop_service as executor_stop_service


@dataclass
class ServiceRecord:
    name: str
    service_id: str
    task_id: str
    cmd: List[str]
    cwd: str
    log_path: pathlib.Path
    proc: subprocess.Popen
    pgid: int = 0
    started_at: float = field(default_factory=time.time)
    readiness: Dict[str, Any] = field(default_factory=dict)
    ready: bool = False
    ready_observed_at: str = ""
    readiness_log_offset: int = 0
    readiness_log_carry: bytes = b""
    readiness_log_identity: tuple[int, int] | None = None
    outputs: List[str] = field(default_factory=list)
    cwd_root: str = ""
    cwd_base: str = ""
    cwd_source: str = ""
    skill_name: str = ""
    before_outputs: Dict[str, tuple[bool, int, str]] = field(default_factory=dict)
    keep_alive: bool = False
    env: Dict[str, str] = field(default_factory=dict, repr=False)
    secret_values: tuple[str, ...] = field(default=(), repr=False)
    leader_birth: str = ""  # measured: its task group row's, else read by the stop while the Popen holds the pid
    scope_marker: str = ""  # this launch's own scope marker: its stop's reach, never a task's
    drive_root: str = ""
    # An unconfirmed stop keeps this record: its typed facts and the identities still owed.
    known_members: List[Dict[str, Any]] = field(default_factory=list)
    cleanup: Dict[str, Any] = field(default_factory=dict)
    unwritten_outputs: Any = field(default=None, repr=False)
    stopped_outputs: Dict[str, Any] | None = field(default=None, repr=False)
    finalizing: bool = False  # a confirmed stop won this object: it holds the name until its log and outputs are done


_LOCK = threading.Lock()
_panic_requested = False
_SERVICES: Dict[str, ServiceRecord] = {}
_SERVICE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
_MAX_SERVICE_LOG_BLOB_BYTES = 5_000_000
_MAX_SERVICE_LOG_TAIL_CHARS = 80_000


def task_service_teardown(ctx: ToolContext) -> str:
    """Task-level service teardown policy: ``stop`` (default) or ``keep``.

    ``keep`` means services started by this task survive task finalization so
    an external party (benchmark verifier, the owner's own shell) can still
    reach them. They remain custody-ledgered and die with the server session.
    """
    meta = getattr(ctx, "task_metadata", None)
    if isinstance(meta, dict) and str(meta.get("service_teardown") or "").strip().lower() == "keep":
        return "keep"
    return "stop"


def _service_output_binding(
    ctx: ToolContext,
    *,
    cwd_root: str,
    cwd_base: str,
    cwd: str,
    cwd_source: str,
    skill_name: str,
) -> ResolvedResourceBinding | None:
    """Rehydrate the exact stored target identity without resolving it again."""

    if not str(cwd_base or "").strip() or not str(cwd or "").strip():
        return None
    return ResolvedResourceBinding(
        profile=active_tool_profile(ctx),
        root=str(cwd_root or "active_workspace"),
        operation="service",
        base_path=pathlib.Path(cwd_base).resolve(strict=False),
        target_path=pathlib.Path(cwd).resolve(strict=False),
        source=str(cwd_source or cwd_root or "active_workspace"),
        skill_name=str(skill_name or ""),
        state_drive_root=canonical_data_root(ctx),
    )


def _working_tree_changes(cwd: str) -> set[str]:
    """Git changes at stop, the fact run_command uses to accept a generated repo output."""
    from ouroboros.tools.shell import _resolve_git_root, _status_snapshot

    return set(_status_snapshot(_resolve_git_root(pathlib.Path(cwd))))


def _live_log_bytes(drive: pathlib.Path, path: pathlib.Path) -> bytes:
    with open_service_log_source(drive, path)[0] as handle:
        return handle.read(_MAX_SERVICE_LOG_BLOB_BYTES + 1)


def _executor_live_log_bytes(record: Any, docker_cli: Any) -> bytes:
    if record.executor.kind == "local":
        return _live_log_bytes(record.drive_root, pathlib.Path(record.backend_log_path))
    proc = docker_cli.run(
        ["docker", "exec", record.executor.container_name, "sh", "-lc",
         f"head -c {_MAX_SERVICE_LOG_BLOB_BYTES + 1} {shlex.quote(record.backend_log_path)}"],
        stdout=docker_cli.PIPE, stderr=docker_cli.PIPE, timeout=10,
    )
    if proc.returncode != 0 or not isinstance(proc.stdout, bytes):
        raise OSError("the backend log could not be read whole")
    return proc.stdout


def _read_local_service_marker(record: Any, path: pathlib.Path, marker: str) -> bool:
    """Incrementally scan the host log for readiness, never as display content."""
    needle = str(marker).encode("utf-8")
    if not needle:
        return True
    try:
        file_stat = path.stat()
        identity = (int(file_stat.st_dev), int(file_stat.st_ino))
        offset = int(getattr(record, "readiness_log_offset", 0) or 0)
        if getattr(record, "readiness_log_identity", None) != identity or file_stat.st_size < offset:
            record.readiness_log_identity = identity
            record.readiness_log_offset = 0
            record.readiness_log_carry = b""
        with path.open("rb") as fh:
            fh.seek(int(getattr(record, "readiness_log_offset", 0) or 0))
            carry = bytes(getattr(record, "readiness_log_carry", b"") or b"")
            carry_size = len(needle) - 1
            while chunk := fh.read(64 * 1024):
                record.readiness_log_offset += len(chunk)
                window = carry + chunk
                carry = window[-carry_size:] if carry_size else b""
                if needle in window:
                    record.readiness_log_carry = carry
                    return True
            record.readiness_log_carry = carry
    except OSError:
        return False
    return False


def _masked_live_log(record: Any, raw: bytes, chars: int) -> tuple[str, str, str]:
    """``(masked tail, masked full input, gap)``. A clipped raw input proves no masking context."""
    if len(raw) > _MAX_SERVICE_LOG_BLOB_BYTES:
        return "", "", f"log exceeds {_MAX_SERVICE_LOG_BLOB_BYTES} byte blob cap; tail masking unverified"
    limit = min(max(1, chars), _MAX_SERVICE_LOG_TAIL_CHARS)
    masked = _service_diagnostic(record, raw.decode("utf-8", errors="replace"))
    masked = masked.replace("\r\n", "\n").replace("\r", "\n")
    return masked[-limit:], masked, ""


def _sanitize_service_name(name: str) -> tuple[str, str]:
    service_name = str(name or "service").strip() or "service"
    if not _SERVICE_NAME_RE.fullmatch(service_name):
        return "", "⚠️ TOOL_ARG_ERROR (start_service): name must match [A-Za-z0-9_.-]{1,80}."
    return service_name, ""


def _readiness_timeout(readiness: Dict[str, Any] | None) -> tuple[float, str]:
    raw = (readiness or {}).get("timeout_sec", 5)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0, "⚠️ TOOL_ARG_ERROR (start_service): readiness.timeout_sec must be numeric."
    if value < 0:
        return 0.0, "⚠️ TOOL_ARG_ERROR (start_service): readiness.timeout_sec must be non-negative."
    return min(value, 25.0), ""


def _service_diagnostic(record: ServiceRecord, value: Any) -> Any:
    return redact_projection(redact_known_values(value, getattr(record, "secret_values", ()))).value


def _request_stop(record: ServiceRecord) -> None:
    """Emergency Stop's settle step: one native kill of the recorded group, no table read and no wait."""
    if record.finalizing or pending_output_stop(record) is not None:
        return  # an ended launch's historical group grants no signal authority
    if record.pgid:
        kill_process_group_id(record.pgid)
    elif record.proc.poll() is None:
        kill_process_tree(record.proc)


def _stop_host_record(record: ServiceRecord, *, graceful: bool = True,
                      ended: Dict[str, Any] | None = None) -> tuple[Dict[str, Any] | None, bool]:
    """Stop one host service record: ``(payload, confirmed)``, ``(None, False)`` when another stop retired it.

    The reach is the launch's own identity (``process_scope.end_held_processes``): its group, its
    descendants and every process that kept its launch marker. That pass reconciles with the launch's
    custody: only its confirmed answer lets the caller finalize the record's log and outputs, and it
    then frees the name (``_release_name``), before a new start can write its declared files; of stops that
    overlap, the first to confirm this object does that, and a late one touches nothing. An unconfirmed
    one keeps the record, its log and its output inputs, with the identities still owed for the next
    attempt (also on the launch's ledger row, the custody a later generation reads), and reports the
    leader's exit facts beside ``process_cleanup``. ``ended``: that answer from a combined pass."""
    from ouroboros.process_scope import end_held_processes, held_item

    with _LOCK:
        if _SERVICES.get(record.service_id) is not record or record.finalizing:
            return None, False
    answer = ended or end_held_processes([held_item(record, record.proc)], graceful=graceful)[record.service_id]
    facts, record.leader_birth, known = answer["facts"], answer["birth"], answer["known"]
    rc = record.proc.poll()
    payload = service_outcomes.process_facts(record, "exited" if rc is not None else "running", returncode=rc)
    payload["process_cleanup"] = facts
    if facts.get("exit_confirmed") and payload["state"] == "exited":
        with _LOCK:
            if _SERVICES.get(record.service_id) is not record or record.finalizing:
                return None, False  # a parallel stop won it first: that one finalizes its log and outputs
            record.finalizing = True
        record.known_members, record.cleanup = [], {}
        return payload, True
    record.known_members, record.cleanup = known, facts
    payload.update(stop_failed=True, cleanup_dispatched=False, stop_error=_stop_error(facts, payload))
    return payload, False


def _release_name(record: ServiceRecord) -> None:
    """The winning stop finalized this object: its name is free for a new start."""
    with _LOCK:
        if _SERVICES.get(record.service_id) is record:
            _SERVICES.pop(record.service_id)


def _stop_error(facts: Dict[str, Any], payload: Dict[str, Any]) -> str:
    if facts.get("census_failed"):
        return "the process table could not be read: no member of this service is confirmed ended"
    if facts.get("custody") == "owed":
        return "another stop of this launch still owes processes; its record and live log are kept until that is resolved"
    if facts.get("custody") == "unavailable":
        return "its custody could not be read or written: its record and live log are kept, nothing is claimed ended"
    parts = [f"{len(facts[key])} {label}" for key, label in (("survivors", "still alive"), ("unverified", "unverified"))
             if facts.get(key)]
    if payload.get("state") != "exited":
        parts.insert(0, "its leader is still running")
    return "service processes are not confirmed ended: " + (", ".join(parts) or "no exit was observed")


def _store_service_log(drive_root: pathlib.Path, masked: str) -> Dict[str, Any]:
    """Verify and sync the masked copy before releasing the only live log."""
    ref = write_blob(drive_root, masked, kind="txt")
    if read_blob_ref(drive_root, ref, expected_kind="txt", max_bytes=ref["size"]) != masked:
        raise OSError("service log archive did not verify")
    blob = pathlib.Path(ref["path"])
    with blob.open("r+b") as stored:
        os.fsync(stored.fileno())
    for directory in (blob.parent, blob.parent.parent, drive_root):
        fsync_folder(directory)  # the blobs/ and observability/ names may both be new
    return ref


def _full_log_text(handle: BinaryIO, size: int) -> str:
    handle.seek(0)
    raw = handle.read(min(size, _MAX_SERVICE_LOG_BLOB_BYTES) + 1)
    if len(raw) != size or size > _MAX_SERVICE_LOG_BLOB_BYTES:
        raise OSError("service log changed during capture")
    return raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


def _require_same_log(log_path: pathlib.Path, observed: os.stat_result) -> None:
    current = log_path.stat(follow_symlinks=False)
    if (not os.path.samestat(observed, current) or observed.st_size != current.st_size
            or observed.st_mtime_ns != current.st_mtime_ns):
        raise OSError("service log changed during capture")


def _record_service_log_archive(
    archive_root: pathlib.Path, log_path: pathlib.Path, observed: os.stat_result, text: str, *, event_type: str,
) -> Dict[str, Any]:
    """The canonical event binds the masked blob to this exact launch before either caller can unlink."""
    from ouroboros.task_custody import fence_publication

    launch_id = service_log_launch_id(log_path.name)
    if not launch_id:
        raise OSError("service log has no recorded launch identity")
    fence_publication()
    ref = _store_service_log(archive_root, text)
    _require_same_log(log_path, observed)
    fence_publication()
    if not append_jsonl(archive_root / "logs" / "events.jsonl", {
        "ts": utc_now_iso(), "type": event_type, "task_id": log_path.parent.name, "launch_id": launch_id,
        "name": log_path.stem.partition("@")[0], "full_log_ref": ref,
        "tail_chars": min(len(text), _MAX_SERVICE_LOG_TAIL_CHARS),
    }, ensure_record_boundary=True, require_lock=True, fsync=True):
        raise OSError("service log archive receipt unconfirmed")
    return ref


def _retain_masked_log(archive_root: pathlib.Path, task_id: str, result: service_outcomes.ServiceLogResult,
                       masked: str, *, archived: bool) -> None:
    """Publish in the task's canonical artifact store, even when no inline ledger will carry this ref."""
    from ouroboros.artifacts import persist_exact_text_source

    scope: service_outcomes.LogSourceScope = "full"
    if not archived and len(masked) > _MAX_SERVICE_LOG_TAIL_CHARS:
        masked = masked[-_MAX_SERVICE_LOG_TAIL_CHARS:]
        scope = "tail"
    digest = sha256(masked.encode("utf-8")).hexdigest()
    _, ref, failure = persist_exact_text_source(archive_root, task_id,
        source_id=service_outcomes.log_source_id(digest, scope=scope), text=masked)
    if failure:
        result["source_unavailable"] = "masked_source_not_retained"
    else:
        result.pop("source_unavailable")
        result.update(source_ref=ref, source_scope=scope)


def _finalize_service_log_for_drive(
    drive_root: pathlib.Path, record: ServiceRecord, *, log_path: pathlib.Path | None = None,
) -> service_outcomes.ServiceLogResult:
    """Settle one closed log before the launch forgets its masking context.

    A tail is cut from verified, already masked text, never from a bounded raw
    window. Keep an oversized source private rather than assert that its prefix
    is irrelevant. Only the existing durable canonical archive permits removal.
    """
    from ouroboros.process_census import LAUNCH_SCOPE_PREFIX
    from ouroboros.task_custody import fence_publication

    result: service_outcomes.ServiceLogResult = {"archive_outcome": "unconfirmed", "deleted_live_log": False,
                                                 "source_unavailable": "masked_source_not_verified"}
    archived, oversized, errors = False, False, 0
    log_path = log_path if log_path is not None else record.log_path
    task_id = log_path.parent.name
    try:
        marker = getattr(record, "scope_marker", "")
        if (getattr(record, "task_id", task_id) != task_id
                or marker and service_log_launch_id(log_path.name) != "launch-" + marker.removeprefix(LAUNCH_SCOPE_PREFIX)):
            raise OSError("service log does not belong to the stopped launch")
        try:
            handle, observed = open_service_log_source(drive_root, log_path, closed=True)
        except FileNotFoundError:
            return {"archive_outcome": "absent", "deleted_live_log": True, "source_unavailable": "log_not_found"}
        with handle:
            _require_same_log(log_path, observed)
            oversized = observed.st_size > _MAX_SERVICE_LOG_BLOB_BYTES
            if oversized:
                result["source_unavailable"] = "tail_masking_unverified"
            else:
                text = _service_diagnostic(record, _full_log_text(handle, observed.st_size))
                _require_same_log(log_path, observed)
                archive_root = None
                try:
                    archive_root = service_log_archive_root(drive_root, task_id)
                    if archive_root is None:
                        raise OSError("service log archive has no bound installation")
                    _record_service_log_archive(archive_root, log_path, observed, text, event_type="service_log_archived")
                    archived = True
                except Exception:
                    errors += 1
                    # A failed write can leave a safe diagnostic tail; a changed source cannot.
                    _require_same_log(log_path, observed)
                if archive_root is None:
                    result["source_unavailable"] = "masked_source_not_retained"
                else:
                    _retain_masked_log(archive_root, task_id, result, text, archived=archived)
                if archived:
                    _require_same_log(log_path, observed)
                    fence_publication()
                    log_path.unlink(missing_ok=True)
                    result["deleted_live_log"] = True
    except Exception:
        errors += 1
    if result["deleted_live_log"]:
        try:
            parent = log_path.parent
            if not any(parent.iterdir()):
                parent.rmdir()
        except Exception:
            errors += 1
    if not result["deleted_live_log"]:
        present = _live_log_present(log_path)
        if present:
            result["live_log"] = "retained"
            try:
                from ouroboros.maintenance_retirement import artifact_changed
                artifact_changed(log_path.parents[2], timestamp=log_path.stat(follow_symlinks=False).st_mtime)
            except Exception:
                errors += 1
        elif present is None:
            result["live_log"] = "unknown"
            errors += 1
        else:
            result["deleted_live_log"] = True
    if errors:
        result["error_count"] = errors
    if archived:
        result["archive_outcome"] = "saved"
    elif result["deleted_live_log"]:
        result["archive_outcome"] = "absent"
    elif result.get("live_log") == "retained":
        result["archive_outcome"] = "retained" if oversized else "not_saved"
    return result


def _without_log_tail(finalization: Dict[str, Any]) -> Dict[str, Any]:
    """The compact log shape: the display tail becomes its ``tail_chars`` size."""
    compact = dict(finalization)
    tail = compact.pop("tail", "")
    if tail:
        compact["tail_chars"] = len(str(tail))
    return compact


def retained_log_display(service: Dict[str, Any], *, drive_root: Any = None, task_id: str = "") -> Dict[str, Any]:
    """A service row whose tail gives way to a readable copy of its complete masked log.

    ``full_log_ref`` is gzip, which ``read_file`` does not open, so the verified
    blob text also becomes this task's exact text source (``source_ref``). Only
    then does the tail become ``tail_chars``. Without a readable source (no blob,
    ``full_log_omitted``, capture ``errors``, no task store) the tail stays; an
    unreadable blob keeps its digest marked ``unavailable``, with no path a review
    closure would have to fetch. A blob that cannot be located now keeps the row as
    it is, address included. Errors, gaps and sizes are never dropped.
    """
    from ouroboros.artifacts import persist_exact_text_source
    from ouroboros.evidence_packs import EvidenceUnavailable

    finalization = service.get("log_finalization")
    ref = finalization.get("full_log_ref") if isinstance(finalization, dict) else None
    if not (drive_root and task_id and isinstance(ref, dict) and ref.get("sha256")):
        return service
    try:
        text = read_blob_ref(pathlib.Path(drive_root), ref, expected_kind="txt", task_id=task_id)
    except EvidenceUnavailable:
        return service
    except Exception as exc:
        unavailable = {key: ref[key] for key in ("kind", "sha256", "size", "encoding") if key in ref}
        return {**service, "log_finalization": {**finalization, "full_log_ref": {
            **unavailable, "availability": "unavailable", "reason": type(exc).__name__}}}
    _exact, source, failure = persist_exact_text_source(
        drive_root, task_id, source_id=f"service-log-{ref['sha256'][:16]}", text=text)
    if failure:
        return service
    return {**service, "log_finalization": {**_without_log_tail(finalization), "source_ref": source}}


def _live_log_present(log_path: pathlib.Path) -> bool | None:
    """Whether the live log exists; ``None`` when that cannot be read."""
    try:
        log_path.stat(follow_symlinks=False)
        return True
    except FileNotFoundError:
        return False
    except Exception:  # an unread presence is unknown, never absence
        return None


def _finalize_service_log(drive_root: Any, record: Any, *,
                          log_path: pathlib.Path | None = None) -> service_outcomes.ServiceLogResult:
    """The log step of a winning stop on either backend and at shutdown: whatever happens to the log, what comes
    after it (the declared outputs, the name's release) still runs, and a collector that raised settled nothing."""
    try:
        return _finalize_service_log_for_drive(pathlib.Path(drive_root), record, log_path=log_path)
    except Exception:  # recorded beside the stop, its live log's state unknown
        return {"archive_outcome": "unconfirmed", "deleted_live_log": False, "live_log": "unknown", "error_count": 1,
                "source_unavailable": "masked_source_not_verified"}


def _archive_stale_service_log(
    drive_root: pathlib.Path, log_path: pathlib.Path, report: Dict[str, Any], *,
    archive_root: pathlib.Path, event_type: str,
) -> None:
    """Release the source only after a verified masked blob and durable receipt."""
    report["retained_files"] += 1
    from ouroboros.task_custody import fence_publication

    try:
        handle, observed = open_service_log_source(drive_root, log_path, closed=True)
        with handle:
            if observed.st_size > _MAX_SERVICE_LOG_BLOB_BYTES:
                return
            text = redact_projection(_full_log_text(handle, observed.st_size)).value
            _record_service_log_archive(archive_root, log_path, observed, text, event_type=event_type)
            report["archived_files"] += 1
            _require_same_log(log_path, observed)
            fence_publication()
            log_path.unlink()  # Windows: the held handle allows deletion, not a data writer
        report["deleted_files"] += 1
        report["retained_files"] -= 1
    except Exception as exc:
        report["errors"].append(f"{service_log_launch_id(log_path.name)}: archive: {type(exc).__name__}")


def archive_task_service_logs(
    drive_root: pathlib.Path,
    task_id: str,
    task: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Archive permitted terminal logs; raw files have already been placed privately by the launcher."""

    from ouroboros import body_candidate
    from ouroboros.confined_files import physical_path, root_spellings
    from ouroboros.source_retention import retained_child_root
    from ouroboros.task_custody import child_drive_candidates

    task_text = str(task_id or "").strip()
    report = {
        "task_id": task_text,
        "archived_files": 0,
        "deleted_files": 0,
        "deleted_dirs": 0,
        "retained_files": 0,
        "errors": [],
    }
    if not native_name(task_text):
        report["errors"].append("invalid task_id")
        return report
    spellings = root_spellings(drive_root)
    roots = [spellings[0]]
    archive_root = service_log_archive_root(spellings[0], task_text)
    if isinstance(task, dict):
        for candidate in child_drive_candidates(task):
            child = physical_path(spellings, candidate)
            if child not in roots and retained_child_root(drive_root, child, task_text):
                roots.append(child)
        report["roots"] = [str(root) for root in roots]
    live_writers = None
    sources = [(root, "services") for root in roots]
    if archive_root is not None:
        sources.append((archive_root, PRIVATE_SERVICE_LOGS))
    for root, store in sources:
        task_dir = root / store / task_text
        try:
            files = service_log_files(root, store, task_text)
            if files is None:
                continue
            for name in files:
                if archive_root is None or not service_log_allows_standard_masking(name):
                    report["retained_files"] += 1
                    continue
                if live_writers is None:
                    live_writers = bool(body_candidate.live_lineage_processes(archive_root, task_text))
                if live_writers:
                    report["retained_files"] += 1
                else:
                    _archive_stale_service_log(root, task_dir / name, report, archive_root=archive_root,
                                               event_type="service_log_archived")
            if not any(task_dir.iterdir()):
                from ouroboros.task_custody import fence_publication
                fence_publication()
                task_dir.rmdir()
                report["deleted_dirs"] += 1
        except FileNotFoundError:
            continue
        except OSError as exc:
            report["errors"].append(f"{task_dir}: {type(exc).__name__}")
    return report


def _refresh_ready(record: ServiceRecord) -> bool:
    with _LOCK:
        if record.proc.poll() is not None:
            record.ready = False
            return False
        if record.ready:
            return True
        readiness = record.readiness or {}
        contains = str(readiness.get("stdout_contains") or readiness.get("log_contains") or "").strip()
        if not contains:
            record.ready = True
            record.ready_observed_at = record.ready_observed_at or utc_now_iso()
            return True
        record.ready = _readiness_marker_observed(record, contains)
        if record.ready:
            record.ready_observed_at = record.ready_observed_at or utc_now_iso()
        return record.ready


def _readiness_marker_observed(record: ServiceRecord, marker: str) -> bool:
    """Scan only unseen service-log bytes without turning the display tail into truth."""

    return _read_local_service_marker(record, record.log_path, marker)


def _refused_before_start(ctx: ToolContext, refusal: str) -> str:
    """Producer fact: this start_service call refused during preparation, before
    any process, service record or executor submission existed. Errors after
    that point never carry it."""
    from ouroboros.tools.shell import _pre_spawn_refusal

    return _pre_spawn_refusal(ctx, refusal, tool="start_service")


def _new_process_refused(ctx, text, previous):
    """Admission may already have finalized a namesake. Only the new spawn is known not to have run."""
    meta = {"process_not_started": True}
    if previous is not None:
        meta["previous_service_retired"] = True
        text += "\n\nNo new process was started; this call finalized the previous same-named launch or its outputs."
        if previous:
            text += "\n" + json.dumps(previous, ensure_ascii=False, indent=2)
    return _publish_tool_result(ctx, ToolResult(status="error", code="LEGACY_TOOL_ERROR", text=text, meta=meta))


def _name_held(ctx: ToolContext, name: str, cause: str, facts: Any = None) -> Any:
    """The start's answer while a launch of ``name`` runs, may run or keeps its name: nothing new started. A retire
    before it may have signalled an ended launch's processes, so no operation outcome says the call changed nothing."""
    shown = json.dumps(service_outcomes.public(facts), ensure_ascii=False, indent=2) if facts else ""
    return _publish_tool_result(ctx, ToolResult(status="blocked", code="LEGACY_BLOCKED", meta={"process_not_started": True},
                                                text=f"⚠️ SERVICE_ALREADY_RUNNING: {name}: {cause}.\n\n" + shown))


def _check_retired(stopped, taken):
    if (stopped or {}).get("stop_failed"):
        raise KeptNamesake("unknown", stopped, "previous service termination is not confirmed: its processes are not confirmed ended; stop_service retries it")
    if stopped and stopped.get("artifact_output_owed"):
        raise KeptNamesake("owed", stopped, "service ended with uncaptured outputs")
    if taken:
        raise KeptNamesake("unknown", {}, "another call still holds this name")


def _admit_named_start(ctx: ToolContext, name: str, drive, admission: ExitStack, *, kept_writer: bool) -> tuple[dict | None, ExitStack]:
    """Settle a namesake, then reserve the writer until registration. None means no predecessor;
    an empty dict means one was finalized without gaps. A writer may transfer to an unwritten declaration."""
    task = str(getattr(ctx, "task_id", "") or "manual")
    admission.enter_context(service_name_hold(drive, task, name))
    key = _service_key(ctx, name)
    stopped = None
    try:
        with _LOCK:
            existing = _SERVICES.get(key)
        if existing is not None:
            try:
                running = pending_output_stop(existing) is None and existing.proc.poll() is None
            except OSError as exc:
                raise KeptNamesake("unknown", {}, "the previous service state could not be read") from exc
            if running:
                raise KeptNamesake("running", {}, f"pid={existing.proc.pid}")
            stopped, _ = _stop_selected(ctx, existing, "host", "replacement")
            with _LOCK:
                taken = key in _SERVICES
            _check_retired(stopped, taken)
        if (record := named_executor_service(ctx, name)) is not None:
            state = "exited" if pending_output_stop(record) else replacement_state(record)
            if state == "running":
                payload = {**service_outcomes.process_facts(record, state), "note": "already_running"}
                raise KeptNamesake(state, payload, "already running")
            if state != "exited":
                raise KeptNamesake("unknown", {}, "previous service termination is not confirmed; nothing was signalled")
            previous, _ = _stop_selected(ctx, record, "executor", "replacement")
            stopped = stopped or previous
            _check_retired(previous, named_executor_service(ctx, name) is not None)
        previous = capture_before_reuse(drive, task, name)
        if stopped is not None:
            previous = {**(previous or {}), **service_outcomes.restart_gaps(stopped)}
        writer = admission.enter_context(writer_reservation(drive, task, name, kept_writer))
        return previous, writer
    except (KeptNamesake, KeptCaptureBusy) as held:
        if stopped and not stopped.get("stop_failed"):  # this call confirmed that end: the refusal says so too
            held.retired = service_outcomes.restart_gaps(stopped)
        raise


def _start_service(
    ctx: ToolContext,
    cmd: List[str],
    name: str = "service",
    cwd: str = "",
    readiness: Dict[str, Any] | None = None,
    outputs: List[str] | None = None,
    keep_alive: bool = False,
    env: Dict[str, str] | None = None,
    env_from_settings: Dict[str, str] | None = None,
    _resolved_binding: ResolvedResourceBinding | None = None,
) -> str:
    if not isinstance(cmd, list) or not cmd or not all(str(x).strip() for x in cmd):
        return _refused_before_start(ctx, "⚠️ TOOL_ARG_ERROR (start_service): cmd must be a non-empty array of strings.")
    proposed_env = dict(env or {})
    try:
        refs = validate_process_env(env_from_settings)
        if refs:
            from ouroboros.tools.process_facts import settings_environment_allowed

            if not settings_environment_allowed(ctx):
                return _publish_tool_result(ctx, ToolResult(
                    status="blocked", code="ACCESS_BLOCKED",
                    text="⚠️ SERVICE_ENV_REFERENCE_BLOCKED: this task cannot select settings-backed service environment. A root task can start the service; existing literal environment and configured MCP access remain available.",
                    meta={"operation_outcome": "completed_no_effect"},
                ))
        env, secret_values = resolve_process_env(env, refs, settings=selection_settings(settings_reader=load_settings) if refs else None)
    except ValueError as exc:
        return _refused_before_start(ctx, f"⚠️ TOOL_ARG_ERROR (start_service): {exc}")
    service_name, name_error = _sanitize_service_name(name)
    if name_error:
        return _refused_before_start(ctx, name_error)
    readiness_timeout, readiness_error = _readiness_timeout(readiness)
    if readiness_error:
        return _refused_before_start(ctx, readiness_error)
    if _panic_requested:
        return _refused_before_start(ctx, "⚠️ SERVICE_START_ERROR: Emergency Stop has retired service admission")
    key = _service_key(ctx, service_name)
    with _LOCK:
        existing = _SERVICES.get(key)
    if existing and pending_output_stop(existing) is None:
        try:
            if existing.proc.poll() is None:
                return _refused_before_start(ctx, f"⚠️ SERVICE_ALREADY_RUNNING: {service_name} pid={existing.proc.pid}")
        except OSError:
            return _name_held(ctx, service_name, "the state of the previous service could not be read")
    try:
        binding = _resolved_binding or build_resolved_resource_binding(
            ctx,
            operation="service",
            process_cwd=cwd,
        )
        workdir = pathlib.Path(binding.target_path)
        cwd_root = binding.root
    except Exception as exc:
        # One failure class, one message (v6.54.3 SSOT): the canonical cwd block
        # names every allowed root as label=path instead of a bare rootless
        # ValueError echo; the SHELL_CWD_BLOCKED status is a typed policy denial.
        return _refused_before_start(ctx, shell_cwd_block_message(ctx, cwd, operation="service", error=exc))
    if _resolved_binding is None:
        # Registry dispatch has already checked this exact prepared binding.
        # A direct handler caller uses the same Supervisor before the first
        # process effect, rather than a second black-box text detector.
        from ouroboros.safety import check_safety

        allowed, advice = check_safety("start_service", {
            "cmd": cmd, "cwd": str(workdir), "name": service_name,
            "env": proposed_env, "env_from_settings": refs,
        }, messages=getattr(ctx, "messages", None), ctx=ctx, resolved_binding=binding)
        if not allowed:
            return _publish_tool_result(ctx, ToolResult(
                *safety_refusal(advice), text=advice, meta={"operation_outcome": "completed_no_effect"},
            ))
        if advice:
            ctx.emit_progress_fn(advice)
    declared_outputs = [str(item) for item in (outputs or []) if str(item or "").strip()]
    from ouroboros.tools.shell_outputs import _declared_output_baseline, _declared_output_refusal, _prepare_declared_output_context

    # A known refusal stops the start; only a baseline that could not be taken lets it proceed, disclosed.
    output_context, output_error = _prepare_declared_output_context(ctx, declared_outputs, workdir)
    if output_error is not None:
        return _publish_tool_result(ctx, output_error)
    output_refusal = _declared_output_refusal(ctx, declared_outputs, workdir, cwd_root=cwd_root, binding=binding,
                                             output_context=output_context)
    if output_refusal:
        return _refused_before_start(ctx, output_refusal)
    before_outputs, baseline_problem = _declared_output_baseline(
        ctx, declared_outputs, workdir, cwd_root=cwd_root, binding=binding, output_context=output_context)
    baseline_note = {"output_baseline_unavailable": baseline_problem} if baseline_problem else {}
    # Resolve the effective keep BEFORE choosing a backend: a task-level
    # service_teardown='keep' must reach the executor path too, else an
    # executor-backed service is recorded task-scoped and the custody reaper
    # kills it once the task ends — breaking the keep contract for a service a
    # verifier still needs (triad review r1, gpt-5.5 critical).
    keep_alive = bool(keep_alive) or task_service_teardown(ctx) == "keep"
    task_id = str(getattr(ctx, "task_id", "") or "manual")
    with ExitStack() as admission:
        try:
            previous, writer = _admit_named_start(ctx, service_name, ctx.drive_root, admission,
                                                   kept_writer=bool(declared_outputs))
        except KeptNamesake as kept:
            retired = service_outcomes.retired_note(getattr(kept, "retired", None))
            if kept.state == "running" and kept.launch.get("executor"):
                return json.dumps({**service_outcomes.public(kept.launch), "executor": {"type": kept.launch["executor"]["type"]},
                                   "note": "already_running"}, ensure_ascii=False, indent=2)
            if kept.state == "owed":
                return f"⚠️ SERVICE_START_ERROR: the previous service has uncaptured outputs; retry start_service or stop_service(name={service_name!r}) to capture them" + retired
            return append_published_text(ctx, _name_held(ctx, service_name, str(kept), kept.launch), retired)
        except KeptCaptureBusy as exc:
            return append_published_text(ctx, _name_held(ctx, service_name, str(exc)),
                                         service_outcomes.retired_note(getattr(exc, "retired", None)))
        from ouroboros import body_candidate

        if output_context is not None or _executor_can_run_cwd(ctx, workdir):
            try:
                payload = workspace_executor._launch_service(
                    ctx,
                    name=service_name,
                    cmd=[str(part) for part in cmd],
                    host_cwd=workdir,
                    cwd_root=cwd_root,
                    cwd_base=str(binding.base_path),
                    cwd_source=binding.source,
                    skill_name=binding.skill_name,
                    readiness=dict(readiness or {}),
                    outputs=declared_outputs,
                    before_outputs=before_outputs,
                    keep_alive=keep_alive,
                    env_overlay=interpreter_path_overlay(active_node_resolution(ctx)),
                    env=env,
                    secret_values=secret_values,
                    output_context=output_context,
                    admitted=admission.close, writer=writer,
                )
                return json.dumps({**payload, **baseline_note, **(previous or {})}, ensure_ascii=False, indent=2)
            except OwnerPauseRefused as exc:
                return append_published_text(ctx, _publish_tool_result(ctx, launch_refusal_result(exc)), service_outcomes.retired_note(previous))
            except body_candidate.CandidateRefused:
                raise
            except Exception as exc:
                text = redact_known_values(f"⚠️ SERVICE_START_ERROR: executor backend failed: {type(exc).__name__}: {exc}", secret_values)
                if getattr(exc, "process_not_started", False) is True:
                    return _new_process_refused(ctx, text, previous)
                return text + service_outcomes.retired_note(previous)

        from ouroboros.owner_pause import _member_coordinates
        from ouroboros.process_scope import record_task_group, scoped_child_env, task_launch_target
        from ouroboros.tool_custody import invocation_binding

        # A task-scoped service belongs to the task (task marker, group recorded); a keep /
        # session service is handed to the installation and sheds every task marker.
        scope_root, _, scope_task = _member_coordinates(ctx)
        scope_target = None if keep_alive else task_launch_target(
            invocation_binding(), drive_root=scope_root, task_id=scope_task)
        from ouroboros.process_census import LAUNCH_SCOPE_PREFIX, SCOPE_VALUE, mint_marker

        scope_marker = mint_marker(LAUNCH_SCOPE_PREFIX)  # this launch's own, kept or not: its stop's reach

        def publish_process(proc):
            record.proc = proc  # first: from here on a process of this launch exists, whatever fails next
            record.pgid = process_group_id(proc.pid)
            group = record_task_group(*scope_target, proc.pid) if scope_target is not None else None
            record.leader_birth = str((group or {}).get("leader_birth") or "")
            with _LOCK:
                _SERVICES[key] = record
            if _panic_requested:
                raise RuntimeError(f"Emergency Stop during service spawn: {request_process_tree_kill(proc)}")

        # Inside the bound candidate a failure to isolate refuses the start on either backend
        # (the registry types CandidateRefused); it never falls back to the serving environment.
        candidate_env = body_candidate.process_environment(ctx, workdir)
        service_base_env = candidate_env if candidate_env is not None else _service_env()

        try:
            log_path, log_fh = open_service_log(pathlib.Path(ctx.drive_root), task_id, service_name, scope_marker,
                                                needs_value_masking=bool(secret_values))
        except OSError as exc:
            text = f"⚠️ SERVICE_START_ERROR: private log unavailable ({type(exc).__name__})."
            return _refused_before_start(ctx, text) if previous is None else _new_process_refused(ctx, text, previous)
        record = ServiceRecord(
            name=service_name, service_id=key, task_id=task_id, cmd=[str(part) for part in cmd], cwd=str(workdir),
            log_path=log_path, proc=None, readiness=dict(readiness or {}), outputs=declared_outputs, cwd_root=cwd_root,
            cwd_base=str(binding.base_path), cwd_source=binding.source, skill_name=binding.skill_name,
            before_outputs=before_outputs, keep_alive=keep_alive, env=env, secret_values=secret_values,
            drive_root=str(ctx.drive_root), scope_marker=scope_marker)
        # Publish capture inputs under the launch's own id before it can write any output.
        kept_note = declare_kept(ctx, record, writer, prepared=True)
        writer.callback(retract_unstarted, ctx, record)  # as this admission ends: a start that spawned nothing takes them back
        if kept_note:
            log_fh.close()
            return _new_process_refused(ctx, "⚠️ SERVICE_START_ERROR: service declaration could not be saved.", previous)
        try:
            bootstrap_process_path()
            # Supervised spawn: the durable custody record means a SIGKILLed worker
            # can no longer orphan this service invisibly — the reaper finds it in
            # the ledger on the next server generation. keep_alive services use
            # session scope: they outlive their task on purpose, but a later server
            # generation still reaps them (no permanent orphans).
            from ouroboros.process_custody import spawn_supervised

            proc = spawn_supervised(
                [str(part) for part in cmd],
                drive_root=pathlib.Path(ctx.drive_root),
                purpose=f"service:{service_name}",
                scope="session" if keep_alive else "task",
                owner_task_id=task_id,
                on_spawn=publish_process,
                cwd=str(workdir),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                # The attested emergency bundled-node PATH prepend (post-gates
                # node resolver) applies on top of the allowlisted service env;
                # a healthy resolution leaves the env byte-identical.
                env={**scoped_child_env(
                    overlay_env(apply_env_path_prepend(service_base_env, active_node_resolution(ctx)), env), scope_target),
                    scope_marker: SCOPE_VALUE},
                scope_marker=scope_marker,
                service_log_name=log_path.name,
            )
            log_fh.close()
        except OwnerPauseRefused as exc:
            log_fh.close()
            return append_published_text(ctx, _publish_tool_result(ctx, launch_refusal_result(exc)), service_outcomes.retired_note(previous))
        except Exception as exc:
            log_fh.close()
            return redact_known_values(f"⚠️ SERVICE_START_ERROR: {type(exc).__name__}: {exc}", secret_values) + service_outcomes.retired_note(previous)
        if keep_alive or declared_outputs:  # the same declaration, now with a confirmed start
            kept_note = declare_kept(ctx, record)
        admission.close()
        try:
            system_root = pathlib.Path(
                getattr(ctx, "system_repo_dir", None) or getattr(ctx, "repo_dir")
            ).resolve(strict=False)
            mutates_system_repo = (
                binding.root == "system_repo"
                or pathlib.Path(binding.base_path).resolve(strict=False) == system_root
            )
            if mutates_system_repo:
                from ouroboros.tools.commit_gate import _invalidate_advisory

                _invalidate_advisory(
                    ctx,
                    changed_paths=[f"<service:{service_name}>"],
                    mutation_root=workdir,
                    source_tool="start_service",
                )
        except Exception:
            pass
        deadline = time.time() + readiness_timeout
        while time.time() < deadline:
            if _refresh_ready(record):
                break
            if proc.poll() is not None:
                break
            time.sleep(0.2)
        return json.dumps({**_status_payload(record), **baseline_note, **kept_note, **(previous or {})}, ensure_ascii=False, indent=2)


def _status_payload(record: ServiceRecord) -> Dict[str, Any]:
    _refresh_ready(record)
    rc = record.proc.poll()
    # The process can settle in the small window between the readiness poll
    # above and this state read.  Terminal state is never currently ready,
    # while the observation timestamp remains useful historical evidence.
    if rc is not None:
        record.ready = False
    state = "running" if rc is None else "exited"
    return {
        **service_outcomes.launch_facts(record),
        "pid": record.proc.pid,
        "pgid": record.pgid,
        "state": state,
        "ready": bool(record.ready),
        "ready_observed_at": record.ready_observed_at or None,
        "returncode": rc,
        "uptime_sec": round(max(0.0, time.time() - record.started_at), 3),
        "cwd": record.cwd,
        "cwd_root": record.cwd_root,
        "cwd_base": record.cwd_base,
        "cwd_source": record.cwd_source,
        "skill_name": record.skill_name,
        "cmd": _service_diagnostic(record, record.cmd),
        "outputs": list(record.outputs),
        "keep_alive": bool(record.keep_alive),
        "log_path": str(record.log_path),
        "ts": utc_now_iso(),
        **({"process_cleanup": dict(record.cleanup)} if record.cleanup else {}),  # an earlier stop left work owed
    }


def service_execution_facts(service_id: str) -> Dict[str, Any] | None:
    """One task service's start identity and execution state, for a sleep selector.

    Execution facts only: no readiness refresh, no log read. ``service_id`` is the
    ``task_id:name`` lookup key a stop/start reuses, so ``started_at`` with the
    process identity pins ONE start. ``returncode`` is the real exit status where
    this process holds the child, else ``None``. ``None`` = no record in this
    process: never started, stopped, or a registry a restart did not carry.
    """
    with _LOCK:
        record = _SERVICES.get(service_id)
    if record is None:
        return executor_service_execution_facts(service_id)
    rc = record.proc.poll()
    return {"service_id": service_id, "started_at": record.started_at, "pid": record.proc.pid,
            "pgid": record.pgid, "state": "running" if rc is None else "exited", "returncode": rc}


def _service_status(ctx: ToolContext, name: str = "service") -> str:
    service_name, name_error = _sanitize_service_name(name)
    if name_error:
        return name_error
    key = _service_key(ctx, service_name)
    with _LOCK:
        record = _SERVICES.get(key)
    if record:
        stopped = pending_output_stop(record)
        return json.dumps(service_outcomes.public(stopped) if stopped else _status_payload(record), ensure_ascii=False, indent=2)
    if executor_ref_from_ctx(ctx) is not None:
        payload = executor_service_status(ctx, service_name)
        if payload is None:
            return f"⚠️ SERVICE_NOT_FOUND: {name}"
        return json.dumps(payload, ensure_ascii=False, indent=2)
    return f"⚠️ SERVICE_NOT_FOUND: {name}"


def _service_logs(ctx: ToolContext, name: str = "service", tail: int = 8000) -> str:
    service_name, name_error = _sanitize_service_name(name)
    if name_error:
        return name_error
    try:
        tail_chars = int(tail or 8000)
    except (TypeError, ValueError):
        return "⚠️ TOOL_ARG_ERROR (service_logs): tail must be an integer."
    key = _service_key(ctx, service_name)
    with _LOCK:
        record = _SERVICES.get(key)
    if record and (stopped := pending_output_stop(record)) is not None:
        # Its stop already decided this log's fate: every reader gets that one saved answer; the file is not read again.
        return json.dumps(service_outcomes.display(stopped), ensure_ascii=False, indent=2)
    if record:
        ref = {}
        text = ""
        omitted_reason = ""
        try:
            drive = pathlib.Path(ctx.drive_root)
            text, full, omitted_reason = _masked_live_log(record, _live_log_bytes(drive, record.log_path), tail_chars)
            if not omitted_reason:
                ref = write_blob(drive, full, kind="txt")
        except FileNotFoundError:
            omitted_reason = "live log not found"
        except Exception as exc:
            omitted_reason = f"log capture unavailable ({type(exc).__name__})"
        return json.dumps({
            "service_id": record.service_id,
            "name": record.name,
            "tail": text,
            "full_log_ref": ref,
            "full_log_omitted": omitted_reason,
        }, ensure_ascii=False, indent=2)
    if executor_ref_from_ctx(ctx) is not None:
        payload = executor_service_logs(ctx, service_name, tail_chars)
        if payload is None:
            return _publish_tool_result(ctx, ToolResult(
                status="unavailable", code="LEGACY_UNAVAILABLE",
                text=f"⚠️ SERVICE_NOT_FOUND: {name}"))
        return json.dumps(payload, ensure_ascii=False, indent=2)
    # Status discovery and idempotent stop use this same marker as a warning.
    # A logs request has no requested payload to return; type only this branch.
    return _publish_tool_result(ctx, ToolResult(
        status="unavailable", code="LEGACY_UNAVAILABLE",
        text=f"⚠️ SERVICE_NOT_FOUND: {name}"))


def _stop_kept_service(ctx: ToolContext, service_id: str, reason: str) -> str:
    """A root task's stop of ONE kept service a FINISHED task left running, by its durable launch id (the owner's
    choice, quiz f4023bf9): the owner's Activity Stop and this land on ``launch_custody.stop_kept_service``, this one
    refusing while the starting task is live or not proven finished. A delegated or Presence-bounded task never
    reaches another task's service."""
    from ouroboros.launch_custody import REASON_MAX_CHARS, stop_kept_service
    from ouroboros.presence_authority import presence_ceiling_from_context
    from ouroboros.tool_access_types import _TOP_LEVEL_PRINCIPAL_PROFILES

    if str(active_tool_profile(ctx)) not in _TOP_LEVEL_PRINCIPAL_PROFILES or presence_ceiling_from_context(ctx) is not None:
        return _publish_tool_result(ctx, ToolResult(
            status="blocked", code="ACCESS_BLOCKED", meta={"operation_outcome": "completed_no_effect"},
            text="⚠️ SERVICE_STOP_BLOCKED: only a root task stops a service another task kept; stop_service(name=...) "
                 "still stops this task's own services."))
    receipt = stop_kept_service(canonical_data_root(ctx), service_id, reason=reason,
                                actor=f"agent:{getattr(ctx, 'task_id', '') or 'unknown'}", finished_producer=True)
    rendered = json.dumps(service_outcomes.public_retained_receipt(receipt), ensure_ascii=False, indent=2)
    outputs = receipt.get("outputs") or {}
    meta = {"artifact_registered": True} if outputs.get("registered") else {}
    if isinstance(outputs.get("observation"), dict) and (key := service_outcomes.output_launch(receipt)):
        meta["service_output_launch"] = dict(zip(("task_id", "launch_id"), key))
        if receipt.get("ok") and receipt.get("outcome") == "not_found":
            meta["service_output_recovered"] = True
    if receipt.get("ok"):  # ended, and its declared outputs are kept (a not_found one: by this call's capture)
        return _publish_tool_result(ctx, ToolResult(status="ok", code="OK", text=rendered, meta=meta))
    if receipt.get("outcome") == "stopped" or "outputs" in receipt and receipt.get("outcome") == "not_found":
        # ended, but its declared outputs are not all captured: typed, not done
        return _publish_tool_result(ctx, ToolResult(status="error", code="ARTIFACT_OUTPUT_ERROR", meta=meta, text=(
            "⚠️ ARTIFACT_OUTPUT_ERROR (stop_service): the service has ended; its declared outputs are "
            f"{receipt.get('outputs', {}).get('state', 'unknown')}.\n\n{rendered}")))
    if receipt.get("outcome") in ("refused", "not_found"):
        refused = receipt["outcome"] == "refused"  # each identifier spelled whole, so the producer harvest reads it
        detail = {"reason_required": "a reason is required", "reason_too_long": f"the reason is longer than {REASON_MAX_CHARS} characters",
                  "invalid_launch": "not a service launch id", "producer_unfinished": "the producer is not proven finished"}
        return _publish_tool_result(ctx, ToolResult(
            status="error" if refused else "unavailable",
            code="TOOL_ARG_ERROR" if refused else "LEGACY_UNAVAILABLE",
            meta={"operation_outcome": "completed_no_effect"},
            text=("⚠️ SERVICE_STOP_REFUSED" if refused else "⚠️ SERVICE_STOP_NOT_FOUND")
            + f": {detail.get(receipt.get('refusal'), 'no process was signalled')}.\n\n{rendered}"))
    return f"⚠️ SERVICE_STOP_ERROR (stop_service): termination was not confirmed.\n\n{rendered}"


def _stop_service(ctx: ToolContext, name: str = "service", service_id: str = "", reason: str = "") -> str:
    if str(service_id or "").strip():
        return _stop_kept_service(ctx, str(service_id), str(reason or ""))
    service_name, name_error = _sanitize_service_name(name)
    if name_error:
        return name_error
    answer = _stop_answer(name, _stop_by_name(ctx, service_name))
    return _publish_tool_result(ctx, answer) if isinstance(answer, ToolResult) else answer


def _stop_by_name(ctx: ToolContext, service_name: str) -> tuple[str, Dict[str, Any] | None, bool]:
    """Select once; the stop only acts on that record, never its future namesake."""
    with _LOCK:
        record = _SERVICES.get(_service_key(ctx, service_name))
    if record is not None:
        return "host", *_stop_selected(ctx, record, "host", "explicit")
    record = named_executor_service(ctx, service_name)
    if record is None:
        try:
            saved = retry_named_outputs(ctx, service_name)
        except (OSError, ValueError) as exc:
            # The local registries held no match. Failure to inspect outstanding files is not a stop observation.
            return "executor", {"name": service_name, "late_capture_unavailable": True, "detail": type(exc).__name__}, False
        return "executor", *service_outcomes.observe_stop(ctx, None, lambda: (saved, False), source="explicit")
    return "executor", *_stop_selected(ctx, record, "executor", "explicit")


def _stop_selected(ctx, record, backend, source):
    def stop():
        if backend == "host":
            return _stop_host_service(ctx, record)
        return executor_stop_service(ctx, record.name, record=record), False
    return service_outcomes.observe_stop(ctx, record, stop, source=source)


_STOP_WORDS = {  # backend -> (an unconfirmed end, declared outputs not finalized)
    "host": ("not every process of this service is confirmed ended; its record, log and declared outputs are kept, "
             "and stop_service retries the identities still owed", "declared service outputs were not finalized"),
    "executor": ("executor backend did not confirm service termination",
                 "declared executor service outputs were not finalized"),
}


def _stop_answer(name: str, outcome: tuple[str, Dict[str, Any] | None, bool]) -> str | ToolResult:
    """Use the stop's lifecycle and output facts; keep the established tool codes and texts."""
    backend, payload, registered = outcome
    if payload is None:
        return f"⚠️ SERVICE_NOT_FOUND: {name}"
    if payload.get("late_capture_unavailable"):
        shown = {key: payload[key] for key in ("name", "detail")}
        return ToolResult(status="unavailable", code="LEGACY_UNAVAILABLE", meta={"late_capture_unavailable": True},
                          text=("⚠️ SERVICE_CAPTURE_UNAVAILABLE (stop_service): no current record for this name; "
                                "could not check outstanding files. Nothing was stopped or signalled by this call.\n\n"
                                + json.dumps(shown, ensure_ascii=False, indent=2)))
    unconfirmed, outputs_failed = _STOP_WORDS[backend]
    shown = service_outcomes.display(payload)
    rendered = json.dumps(shown, ensure_ascii=False, indent=2)
    # Every answer below shows this receipt: the delivered result names it, so completion does not ask again.
    seen = service_outcomes.shown_stop(payload)
    if payload["lifecycle"] == "cleanup_pending":
        return ToolResult(status="error", code="LEGACY_TOOL_ERROR", meta=seen,
                          text=f"⚠️ SERVICE_STOP_ERROR (stop_service): {unconfirmed}.\n\n{rendered}")
    if payload.get("artifact_output_unknown"):
        outputs_failed = "the service ended; its declared output capture could not be confirmed"
    failed = bool(payload.get("artifact_output_failed"))
    # A sibling output registered on this stop remains an observed effect beside a capture failure.
    meta = {**seen, **({"artifact_registered": True} if registered else {})}
    retry = payload.get("output_retry") is True
    if (failed or retry) and payload.get("lifecycle") in ("stopped", "outputs_updated") and (key := service_outcomes.output_launch(payload)):
        meta["service_output_launch"] = dict(zip(("task_id", "launch_id"), key))
        if (retry and not failed and not payload.get("artifact_output_unknown")
                and (outputs := payload.get("output_outcomes"))
                and all(row.get("status") == "saved" for row in outputs)):
            meta["service_output_recovered"] = True
    if failed:
        return ToolResult(status="error", code="ARTIFACT_OUTPUT_ERROR", meta=meta,
                          text=f"⚠️ ARTIFACT_OUTPUT_ERROR (stop_service): {outputs_failed}.\n\n{rendered}")
    return ToolResult(status="ok", code="OK", text=rendered, meta=meta)


def _stop_host_service(ctx: ToolContext, record: ServiceRecord) -> tuple[Dict[str, Any] | None, bool]:
    """One host stop: ``(payload, registered)``, ``(None, False)`` when another stop retired this launch first.

    An unconfirmed stop keeps the record (``stop_failed``). A confirmed one finalizes the log, then the declared
    outputs. An unwritten declaration keeps the stopped record until its outputs are settled."""
    if pending_output_stop(record) is not None:
        return retry_own_outputs(ctx, record, _capture_record_outputs, _SERVICES, _LOCK)
    payload, confirmed = _stop_host_record(record)
    if payload is None or not confirmed:
        return payload, False
    registered = False
    try:
        payload["log_finalization"] = _finalize_service_log(ctx.drive_root, record)
    except Exception as exc:  # the log's fate alone is unknown: the declared files are still captured
        payload.update(log_outcome="unconfirmed", finalization_error=type(exc).__name__)
    try:
        registered = finish_own_outputs(ctx, record, payload, _capture_record_outputs)
    except Exception as exc:  # the exit is proven: what this stop already knows stands beside the fault
        service_outcomes.after_stop_error(payload, record.outputs, exc)
    finally:
        release_after_stop(record, payload, _SERVICES, _LOCK)
    return payload, registered


def _capture_record_outputs(ctx, record, payload):
    """Capture with the launch's own paths and executor, never fields from a diagnostic response."""
    executor = getattr(record, "executor", None)
    if not record.outputs and executor is None and record.cwd_root == "user_files":
        payload["artifact_audit_gap"] = True
    inputs = {key: getattr(record, key) for key in ("outputs", "cwd_root", "cwd_base", "cwd_source", "skill_name")}
    inputs.update(cwd=str(getattr(record, "host_cwd", None) or record.cwd),
                  output_outcomes=payload.get("output_outcomes"))
    try:
        return _capture_service_outputs(ctx, inputs, record.before_outputs, getattr(record, "output_context", None),
                                        executor=executor)
    finally:
        payload.update({key: inputs[key] for key in ("output_outcomes", "output_baseline_unavailable", "artifact_output_failed",
                        "artifact_output_owed", "artifact_output_unknown", "artifact_outputs") if key in inputs})


def _capture_service_outputs(ctx, payload, before_outputs, output_context=None, *, executor=...) -> bool:
    """One ended launch's declared outputs into ``payload``; True once one registered. ``artifact_output_owed``: a
    failed one may still hold bytes a later capture could take (``shell_outputs.outputs_owed``); absent, every failed
    one was proven absent and nothing more is owed."""
    outputs = payload.get("outputs")
    if not outputs:
        payload["artifact_output_failed"] = False
        return False
    from ouroboros.tools.shell import _register_process_outputs
    from ouroboros.tools.shell_outputs import outputs_owed

    settled = {row["index"]: {**row, **({"registered": False} if "registered" in row else {})}
               for row in payload.get("output_outcomes") or () if row["status"] in {"saved", "partial", "absent"}}
    pending = [index for index in range(1, len(outputs) + 1) if index not in settled]
    if not pending:  # only receipt/retirement may remain; no source read can improve these facts
        payload.update(output_outcomes=service_outcomes.declared_output_outcomes(outputs, settled.values()),
                       artifact_output_failed=any(row["status"] == "absent" for row in settled.values()))
        return False
    # A path this capture could not address (``DockerOutputContext.unresolved``) is alone unconfirmed, and owed.
    unresolved = getattr(output_context, "unresolved", None) or ()
    blocked = [index for index in pending if str(outputs[index - 1]).strip() in unresolved]
    pending = [index for index in pending if index not in blocked]
    failures, observed = [(None, "owed")] * bool(blocked), []
    note, failed, registered = "", False, False
    try:
        cwd = payload.get("host_cwd") or payload["cwd"]
        root = payload.get("cwd_root", "")
        binding = _service_output_binding(ctx, cwd_root=root, cwd_base=payload.get("cwd_base", ""), cwd=cwd,
                                          cwd_source=payload.get("cwd_source", ""), skill_name=payload.get("skill_name", ""))
        if pending:
            note, failed, registered = _register_process_outputs(
                ctx, [outputs[index - 1] for index in pending], pathlib.Path(cwd), cwd_root=root,
                changed_paths=_working_tree_changes(cwd), before_outputs=before_outputs, binding=binding,
                output_context=output_context, failures=failures, executor=executor, output_outcomes=observed)
    except Exception as exc:
        service_outcomes.after_stop_error(payload, outputs, exc)
        failed = True
        failures.append((None, "owed"))
    failed = bool(failed or blocked)
    settled.update({pending[row["index"] - 1]: {**row, "index": pending[row["index"] - 1]} for row in observed})
    payload["output_outcomes"] = service_outcomes.declared_output_outcomes(outputs, settled.values())
    payload["artifact_output_failed"] = bool(failed) or any(row["status"] == "absent" for row in settled.values())
    if outputs_owed(failed, failures):
        payload["artifact_output_owed"] = True
        payload["artifact_output_unknown"] = any(row["status"] == "unconfirmed" for row in payload["output_outcomes"])
    if not before_outputs:  # the start baseline holds every admitted output: none means it could not be taken (said at start)
        payload["output_baseline_unavailable"] = True
        note += "\n- output baseline was unavailable at start: a registered output may predate this service"
    if note:
        payload["artifact_outputs"] = note.strip()
    return registered or service_outcomes.output_registered(observed)


def stop_task_services(ctx: ToolContext) -> List[Dict[str, Any]]:
    """End this task's services on both backends; one outcome per service, tagged with ``lifecycle``.

    ``lifecycle`` is each stop's own answer, given here once: ``kept`` for a service marked ``keep_alive``
    (per-service flag or task-level ``service_teardown=keep``), NOT stopped but reported so the caller can
    surface pid/port metadata to whoever inherits responsibility for it (benchmark verifier, the owner);
    ``stopped`` once every process is confirmed ended, its log and declared outputs as finalized; else
    ``cleanup_pending``, the record kept. No service is left out: a stop that raised is reported with its error,
    ended only when its end was confirmed first. A launch another stop retired is that stop's to report.
    """
    from ouroboros.workspace_executor import _service_payload, _services_snapshot

    task_id = str(getattr(ctx, "task_id", "") or "manual")
    keep_all = task_service_teardown(ctx) == "keep"
    results: List[Dict[str, Any]] = []
    with _LOCK:
        hosts = [record for record in _SERVICES.values() if record.task_id == task_id]
    for record in hosts:
        if (keep_all or record.keep_alive) and pending_output_stop(record) is None:
            results.append(service_outcomes.kept_outcome(record, lambda: _status_payload(record)))
        elif (row := _stop_selected(ctx, record, "host", "task_end")[0]) is not None:
            results.append(row)
    for record in _services_snapshot():
        if str(getattr(record, "task_id", "")) != task_id:
            continue
        if (keep_all or record.keep_alive) and pending_output_stop(record) is None:
            results.append(service_outcomes.kept_outcome(record, lambda: _service_payload(record)))
        elif (row := _stop_selected(ctx, record, "executor", "task_end")[0]) is not None:
            results.append(row)
    return results


def kill_all_services(
    drive_root: pathlib.Path | None = None,
    *,
    wait: bool = True,
    include_keep_alive: bool = True,
    request_only: bool = False,
    durable: bool = True,
) -> List[Dict[str, Any]]:
    """Stop every tracked service for panic/shutdown paths.

    ``wait=True`` (shutdown, update) ends each service's processes by measured identity with
    KILL at once and keeps any record whose end is unconfirmed (``stop_failed``); ``wait=False``
    and ``request_only`` (Emergency Stop) only request the native kill, with no table read.
    ``include_keep_alive=False`` (graceful shutdown/restart) leaves keep_alive
    services running: they are session-scoped in the process custody ledger,
    so the next server generation's reaper still collects them. Panic and
    emergency cleanup keep the default and kill everything. ``durable=False``
    stops only this process's in-memory services: the exit stop
    (``owned_shutdown.stop_owned_work``) settles the durable records itself.
    """

    global _panic_requested
    if request_only:
        _panic_requested = True
        # A shallow builtin copy keeps the existing owner table readable even
        # when cleanup holds _LOCK; leave custody/log settlement to that owner.
        requested = [
            {"service_id": record.service_id, **request_process_tree_kill(record.proc)}
            for record in _SERVICES.copy().values()
            if not record.finalizing and pending_output_stop(record) is None and (include_keep_alive or not record.keep_alive)
        ]
        requested.extend(executor_kill_all_services(drive_root, request_only=True))
        return requested
    with _LOCK:
        records = [record for record in _SERVICES.values() if not record.finalizing and pending_output_stop(record) is None
                   and (include_keep_alive or not record.keep_alive)]
        if not wait:  # Emergency Stop's settle step owns them from here: one native request each, no wait
            for record in records:
                _SERVICES.pop(record.service_id, None)
    stopped: List[Dict[str, Any]] = []
    together: Dict[str, Any] = {}
    if wait and records:
        # The shutdown's own grace bounds this: KILL at once, no per-service TERM grace, and ONE
        # census and wait for every service, each answer attributed to its own launch.
        from ouroboros.process_scope import end_held_processes, held_item

        together = end_held_processes([held_item(record, record.proc) for record in records if not record.finalizing], graceful=False)
    for record in records:
        if not wait:
            _request_stop(record)
            stopped.append(_status_payload(record))
            continue
        payload, confirmed = _stop_host_record(record, graceful=False, ended=together.get(record.service_id))
        if payload is None:
            continue  # another stop retired it meanwhile
        if confirmed:
            try:
                if drive_root is not None:
                    payload["log_finalization"] = _finalize_service_log(drive_root, record)
            finally:
                _release_name(record)
        stopped.append(payload)
    try:
        stopped.extend(executor_kill_all_services(drive_root, wait=wait, durable=durable))
    except Exception:
        pass
    if wait and drive_root is not None and stopped:
        try:
            append_jsonl(pathlib.Path(drive_root) / "logs" / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "services_shutdown_cleanup",
                "services": [service_outcomes.display(payload) for payload in stopped],
            })
        except Exception:
            pass
    return stopped


def prune_service_logs(
    drive_root: pathlib.Path,
    retention_days: int | None = None,
    *,
    now: float | None = None,
) -> Dict[str, Any]:
    """Archive old logs, and retire a kept declaration naming no outputs (beside its log) with them, only after process
    custody releases their writers; an owed declaration waits for its capture (``retire_kept_declarations``)."""
    from ouroboros import body_candidate
    from ouroboros.retention import age_cutoff, get_gc_retention_days, keep_until

    # Explicit overrides are honored as-is (age_cutoff floors at 0 => explicit 0
    # prunes everything before `now`), uniform with the worktree/task prunes; only
    # the default (None) path reads the clamped owner knob.
    if retention_days is None:
        retention_days = get_gc_retention_days()
    cutoff = age_cutoff(retention_days, now)
    services_root = pathlib.Path(drive_root) / "services"
    report = {
        "enabled": True,
        "retention_days": retention_days,
        "deleted_dirs": 0,
        "deleted_files": 0,
        "archived_files": 0,
        "retained_files": 0,
        "errors": [],
    }
    task_dirs = []
    for root in (services_root, services_root.with_name(PRIVATE_SERVICE_LOGS)):
        try:
            if stat.S_ISDIR(root.lstat().st_mode):
                task_dirs.extend(root.iterdir())
        except FileNotFoundError:
            continue
        except OSError as exc:
            report["errors"].append(f"{root}: {type(exc).__name__}")
    retire_kept_declarations(pathlib.Path(drive_root), cutoff, report)
    for task_dir in task_dirs:
        try:
            task_stat = task_dir.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            report["errors"].append(f"{task_dir}: {type(exc).__name__}: {exc}")
            continue
        if not stat.S_ISDIR(task_stat.st_mode) or task_dir.name == KEPT_DIRECTORY:  # that directory: visited above
            continue
        try:
            live_writers = None
            archive_root = service_log_archive_root(pathlib.Path(drive_root), task_dir.name)
            if task_dir.parent == services_root:
                for declaration in task_dir.glob(f"*{KEPT_DECLARATION}"):
                    retire_kept_declaration(pathlib.Path(drive_root), task_dir.name, declaration, cutoff, report)
            for child in task_dir.glob("*.log"):
                try:
                    child_stat = child.lstat()
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    report["errors"].append(f"{child}: {type(exc).__name__}: {exc}")
                    continue
                if stat.S_ISREG(child_stat.st_mode) and child_stat.st_mtime >= cutoff:
                    keep_until(report, child_stat.st_mtime, retention_days)
                if stat.S_ISREG(child_stat.st_mode) and child_stat.st_mtime < cutoff:
                    if archive_root is None or not service_log_allows_standard_masking(child.name):
                        report["retained_files"] += 1
                        continue
                    stem = child.stem.partition("@")[0]
                    facts = service_execution_facts(f"{task_dir.name}:{stem}")
                    if facts is None and stem.endswith(".executor"):
                        facts = service_execution_facts(f"{task_dir.name}:{stem.removesuffix('.executor')}")
                    # A kept service can outlive its task or live in another worker's registry.
                    if live_writers is None:
                        live_writers = bool(body_candidate.live_lineage_processes(archive_root, task_dir.name))
                    if live_writers or (facts is not None and facts.get("state") != "exited"):
                        report["retained_files"] += 1
                        continue
                    _archive_stale_service_log(pathlib.Path(drive_root), child, report,
                                               archive_root=archive_root, event_type="service_log_pruned")
            if not any(task_dir.iterdir()):
                task_dir.rmdir()
                report["deleted_dirs"] += 1
        except Exception as exc:
            report["errors"].append(f"{task_dir}: {type(exc).__name__}: {exc}")
    return report


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("start_service", {
            "name": "start_service",
            "description": "Start a long-running process owned by this task (dev server, watcher) and return pid/readiness/state; it is stopped when the task ends unless kept alive (keep_alive). In runtime_mode=light a service whose cwd is the Ouroboros repository is refused (LIGHT_MODE_BLOCKED): pass an explicit external/task/artifact cwd.",
            "parameters": {"type": "object", "properties": {
                "cmd": {"type": "array", "items": {"type": "string"}},
                "cwd": {
                    "type": "string",
                    "default": "",
                    "description": (
                        "Omit for active_workspace; use system_repo[/subdir] for Ouroboros. "
                        "Docker backend aliases inside active_workspace are accepted; skill payload services are not supported."
                    ),
                },
                "name": {"type": "string", "default": "service"},
                "env": {"type": "object", "additionalProperties": {"type": "string"}, "description": "Ordinary literal environment overrides, passed unchanged on local and Docker executors over the minimal host baseline. Use env_from_settings for secrets."},
                "env_from_settings": {"type": "object", "additionalProperties": {"type": "string"}, "description": "Root tasks and Cyber acting tasks: environment name to saved setting key. References override literal env; Settings-classified secrets are masked in diagnostics. Explicit read-only tasks retain their existing limits."},
                "readiness": {"type": "object", "default": {}, "description": "Optional {log_contains|stdout_contains, timeout_sec} readiness probe."},
                "outputs": {"type": "array", "items": {"type": "string"}, "default": [], "description": "Paths captured into the task artifact store when the service stops. An eligible path is captured even if the service left it unchanged; outside Cyber Pro a system_repo file is eligible only while it has uncommitted changes. A declaration whose location is refused stops the start before the service launches. Only outputs accept a relative '../' spelling, judged by its resolved location like an absolute one; cwd and file-tool path rules are unchanged."},
                "keep_alive": {"type": "boolean", "default": False, "description": "Leave this service running after the task ends (e.g. a dev server the user or an external verifier still needs). It stays custody-ledgered and dies with the server session or panic."},
            }, "required": ["cmd"]},
        }, _start_service, is_code_tool=True, timeout_sec=30, mutates_worktree=True),
        ToolEntry("service_status", {
            "name": "service_status",
            "description": "Return pid/state/readiness/uptime for a task-scoped service.",
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string", "default": "service"},
            }, "required": []},
        }, _service_status),
        ToolEntry("service_logs", {
            "name": "service_logs",
            "description": "Read the masked log tail of a running service. The host may also return a reference to the full masked log; a backend that supplies only its tail has no full-log reference.",
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string", "default": "service"},
                "tail": {"type": "integer", "default": 8000},
            }, "required": []},
        }, _service_logs),
        ToolEntry("stop_service", {
            "name": "stop_service",
            "description": "Stop a service and the processes still traceable to it, then capture its declared outputs into the task artifact store.",
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string", "default": "service"},
                "service_id": {"type": "string", "default": "", "description": "Root tasks only: the launch_id in Runtime kept_services. Selects one launch a finished task kept running or whose declared files are still owed. Refused while the producer or its helpers may still run. A confirmed end needs only file capture, with no second process stop. Outputs go to the starting task; name is ignored."},
                "reason": {"type": "string", "default": "", "description": "Required with service_id: why it is stopped (at most 1000 characters), recorded with the outcome."},
            }, "required": []},
        }, _stop_service, is_code_tool=True, timeout_sec=30, mutates_worktree=True),
    ]
