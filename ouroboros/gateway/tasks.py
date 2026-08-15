"""Headless task gateway endpoints."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import pathlib
import shutil
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse

from ouroboros.gateway.task_placement import (
    _fold_request_executor_ref,
    _seal_placement_into_metadata,
    _place_attachments,
    _admit_task_placement,
)
from ouroboros.gateway._helpers import (
    coerce_int,
    json_error,
    json_exception,
    request_drive_root,
    request_json_or,
    request_repo_dir,
    wire_error_code,
)
# Re-exported SSE surface (split out by the 1600-line module gate): route
# wiring, the CLI, and long-standing monkeypatch pins address these names on
# gateway.tasks; task_events resolves its patched collaborators back through
# this namespace at call time (see task_events._tasks_namespace).
from ouroboros.gateway.task_events import (  # noqa: F401
    _TaskEventFollower,
    _read_live_jsonl_entries,
    api_task_events,
    iter_task_events,
)
from ouroboros.headless import (
    ARTIFACTS_DIR,
    ARTIFACT_STATUS_FAILED,
    ARTIFACT_STATUS_PENDING,
    HEADLESS_TASKS_DIR,
    prepare_task_drive,
    task_artifacts_dir,
)
from ouroboros.contracts.task_contract import (
    attach_task_contract,
    normalize_acceptance_claims,
    normalize_allowed_resources,
    normalize_answer_protocol,
    normalize_bool,
    normalize_disabled_tools,
    normalize_resource_policy,
)
from ouroboros.outcomes import public_task_result
from ouroboros.task_results import (
    STATUS_FAILED,
    STATUS_SCHEDULED,
    list_task_results,
    load_task_result,
    task_results_dir,
    validate_task_id,
    write_task_result,
)
from ouroboros.task_status import (
    _EventsTailIndex,
    effective_task_result,
    load_effective_task_result,
)
from ouroboros.utils import read_json_dict
from ouroboros.workspace_admission import (
    REMOTE_TRANSPORT_UNAVAILABLE,
    RemoteWorkspaceUnavailableError,
)
from ouroboros.workspace_ref import (
    SEALED_WORKSPACE_REF_KEY,
    workspace_display_root,
)


log = logging.getLogger(__name__)

_RESERVED_METADATA_KEYS = frozenset({
    "task_id",
    "parent_task_id",
    "root_task_id",
    "session_id",
    "actor_id",
    "delegation_role",
    "drive_root",
    "child_drive_root",
    "headless_child_drive_root",
    "budget_drive_root",
    "task_constraint",
    "task_contract",
    "allowed_resources",
    "deadline_at",
    "executor_ref",
    "workspace_executor",
    "project_id",
    # RWS v2: the placement is SEALED by admission. A client-supplied placement in
    # metadata would be a second placement authority that no fence can revalidate.
    # BOTH halves are listed, and neither is reachable through this handler any more:
    # `_admit_task_placement` refuses them in `metadata` with a typed 400 before the
    # strip runs. This stays as the belt — a stripped key is still better than a stored
    # one — but the REFUSAL is the rule, because a silent strip is what let a caller
    # believe they had chosen a placement (`connection_id` was missing here entirely, so
    # it was kept and ignored, which is worse than being dropped).
    "workspace_ref",
    "connection_id",
    SEALED_WORKSPACE_REF_KEY,
})


def _cleanup_api_admission_attempt(
    drive_root: pathlib.Path,
    task_id: str,
    admission_token: str,
    child_drive: Optional[pathlib.Path] = None,
) -> None:
    """Release one token and remove only its pre-admission task-local state."""
    from supervisor.queue import release_task_admission

    release_task_admission(task_id, admission_token)
    if child_drive is not None:
        try:
            from ouroboros.headless import remove_subagent_task_drive

            remove_subagent_task_drive(drive_root, task_id)
        except Exception:
            log.warning("Failed to clean child drive for rejected task %s", task_id, exc_info=True)
    try:
        shutil.rmtree(task_artifacts_dir(drive_root, task_id, create=False), ignore_errors=True)
    except Exception:
        log.warning("Failed to clean admission artifacts for task %s", task_id, exc_info=True)


def _external_subagent_label(body: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
    role_values = [
        body.get("delegation_role"),
        metadata.get("delegation_role"),
    ]
    return any(str(value or "").strip().lower() == "subagent" for value in role_values)


def _normalize_deadline_at(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("deadline_at must be an ISO-8601 datetime") from exc
    if parsed.tzinfo is None:
        raise ValueError("deadline_at must include a timezone offset or Z")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _fold_contract_policies(body: Dict[str, Any], raw_metadata: Dict[str, Any], metadata: Dict[str, Any]):
    """Normalize the declarative contract policies from the request body into task
    metadata (extracted from api_tasks_create for the function-size gate; pure).
    Returns (allowed_resources, resource_policy, disabled_tools, acceptance_claims,
    error) — error is non-empty for an invalid service_teardown."""
    allowed_resources = normalize_allowed_resources(body.get("allowed_resources") or raw_metadata.get("allowed_resources") or {})
    if allowed_resources:
        metadata["allowed_resources"] = allowed_resources
    resource_policy = normalize_resource_policy(body.get("resource_policy") or raw_metadata.get("resource_policy") or {})
    if resource_policy:
        metadata["resource_policy"] = resource_policy
    disabled_tools = normalize_disabled_tools(body.get("disabled_tools") or raw_metadata.get("disabled_tools") or [])
    if disabled_tools:
        metadata["disabled_tools"] = disabled_tools
    acceptance_claims = normalize_acceptance_claims(body.get("acceptance_claims") or raw_metadata.get("acceptance_claims") or [])
    if acceptance_claims:
        metadata["acceptance_claims"] = acceptance_claims
    # v6.60.0: adapter-declared answer protocol ("" | "final_answer_line") — flows into
    # the task contract (and to subagents via the normal contract inheritance).
    answer_protocol = normalize_answer_protocol(body.get("answer_protocol") or raw_metadata.get("answer_protocol"))
    if answer_protocol:
        metadata["answer_protocol"] = answer_protocol
    service_teardown = str(body.get("service_teardown") or raw_metadata.get("service_teardown") or "").strip().lower()
    if service_teardown:
        if service_teardown not in {"stop", "keep"}:
            return allowed_resources, resource_policy, disabled_tools, acceptance_claims, "service_teardown must be 'stop' or 'keep'"
        metadata["service_teardown"] = service_teardown
    return allowed_resources, resource_policy, disabled_tools, acceptance_claims, ""


def _admission_rejection_response(
    admitted: Any,
    *,
    drive_root: pathlib.Path,
    task_id: str,
    project_id: str,
    # Presence-only (drives artifact_status): the placement's display spelling, so a
    # REMOTE workspace task is recognized as a workspace task here too.
    workspace_root: Any,
    child_drive: Optional[pathlib.Path],
    status_code: int = 409,
    detail: str = "Task was not scheduled because its admission fence is closed.",
) -> Optional[JSONResponse]:
    """Terminalize a typed queue refusal so no scheduled phantom remains."""
    if not (isinstance(admitted, dict) and admitted.get("_admission_blocked")):
        return None
    reason_code = str(admitted.get("_admission_blocked") or "admission_fence")
    if reason_code in {"duplicate_task_id", "admission_reservation_lost"}:
        return JSONResponse(
            {
                "error": "Task id is already owned by another admission attempt.",
                "task_id": task_id,
                "status": "rejected",
                "admission": {"reason_code": reason_code},
            },
            status_code=409,
        )
    admission = {
        "reason_code": reason_code,
        "project_id": str(admitted.get("_project_id") or project_id),
        "project_lifecycle": str(admitted.get("_project_lifecycle") or ""),
        "acceptance_fence_token": str(admitted.get("_acceptance_fence_token") or ""),
        "acceptance_fence_status": str(admitted.get("_acceptance_fence_status") or ""),
    }
    write_task_result(
        drive_root,
        task_id,
        STATUS_FAILED,
        reason_code=reason_code,
        admission=admission,
        artifact_status=ARTIFACT_STATUS_FAILED if workspace_root else "",
        result=detail,
        cost_usd=0.0,
    )
    if child_drive is not None:
        from ouroboros.headless import remove_subagent_task_drive

        removed = remove_subagent_task_drive(drive_root, task_id)
        write_task_result(
            drive_root,
            task_id,
            STATUS_FAILED,
            admission_cleanup={"child_drive_removed": bool(removed)},
        )
    try:
        shutil.rmtree(task_artifacts_dir(drive_root, task_id, create=False), ignore_errors=True)
    except Exception:
        log.warning("Failed to clean rejected task artifacts for %s", task_id, exc_info=True)
    return JSONResponse(
        {
            "error": detail,
            "task_id": task_id,
            "status": STATUS_FAILED,
            "admission": admission,
        },
        status_code=status_code,
    )


def _enqueue_api_task_durably(
    task: Dict[str, Any],
    *,
    drive_root: pathlib.Path,
    task_id: str,
    admission_token: str,
    result_fields: Dict[str, Any],
) -> Dict[str, Any]:
    """Atomically enqueue, snapshot, and publish the scheduled task result."""
    from supervisor import queue

    with queue._queue_lock:
        admitted = queue.enqueue_task(task)
        if isinstance(admitted, dict) and admitted.get("_admission_blocked"):
            return admitted
        if queue.persist_queue_snapshot(reason="api_task_create") is not True:
            queue.PENDING[:] = [
                row for row in queue.PENDING
                if not (
                    isinstance(row, dict)
                    and str(row.get("id") or "") == task_id
                    and str(row.get("_admission_owner_token") or "") == admission_token
                )
            ]
            queue.persist_queue_snapshot(reason="api_task_create_rollback")
            return {
                **task,
                "_admission_blocked": "queue_snapshot_persist_failed",
                "_admission_status_code": 503,
            }
        write_task_result(drive_root, task_id, STATUS_SCHEDULED, **result_fields)
        queue.release_task_admission(task_id, admission_token)
        return admitted


def _complete_api_task_admission(
    task: Dict[str, Any],
    *,
    drive_root: pathlib.Path,
    task_id: str,
    admission_token: str,
    project_id: str,
    description: str,
    allowed_resources: Dict[str, Any],
    deadline_at: str,
    workspace_root: Optional[pathlib.Path],
    workspace_mode: str,
    memory_mode: str,
    child_drive: Optional[pathlib.Path],
    artifacts: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> JSONResponse:
    """Publish one API admission or roll back only its token-owned queue row.

    ``workspace_root`` is the LOCAL Home path of the sealed placement (``None`` for a
    remote one), so it gates the preflight-artifact status. The refusal path is told
    about the placement through the task's DISPLAY spelling instead, so a remote
    workspace task is terminalized as a workspace task rather than a bare one.

    That asymmetry is DELIBERATE here and the reason belongs beside it, because the
    local key reads like an oversight: `artifact_status` is a claim that a HOME-side
    workspace-patch artifact is coming, and the only producer of one
    (``headless.write_workspace_patch_artifacts``) needs a local Home path and is
    therefore not the producer on a remote placement — a remote task's patch is
    evidence gathered on the target through ``remote_patch_bridge``. So ``""`` is the
    true answer rather than a lost one: ``outcomes`` reads it as ``not_applicable``,
    the CLI's terminal-success and patch-extraction checks treat it as settled (they
    gate on ``pending``/``finalizing``/``failed``/``missing``), and a PENDING marker
    here would promise an artifact nothing on this path will ever finalize — which is
    the shape that leaves a task waiting on its own bookkeeping. Whether a remote
    placement should grow its own artifact lifecycle is a design question, not this
    line's bug."""
    result_fields = {
        "parent_task_id": task.get("parent_task_id"),
        "root_task_id": task.get("root_task_id"),
        "session_id": task.get("session_id"),
        "actor_id": task.get("actor_id"),
        "delegation_role": task.get("delegation_role"),
        "project_id": project_id,
        "description": description,
        "context": task.get("context"),
        "expected_output": task.get("expected_output"),
        "constraints": task.get("constraints"),
        "allowed_resources": allowed_resources,
        "deadline_at": deadline_at,
        "task_contract": task.get("task_contract"),
        "workspace_root": task.get("workspace_root"),
        "workspace_mode": workspace_mode,
        "memory_mode": memory_mode,
        "child_drive_root": str(child_drive or ""),
        "budget_drive_root": str(drive_root) if child_drive is not None else "",
        "artifacts": artifacts,
        "artifact_status": ARTIFACT_STATUS_PENDING if workspace_root else "",
        "metadata": metadata,
        "result": "Task accepted and durably scheduled.",
    }
    try:
        admitted = _enqueue_api_task_durably(
            task,
            drive_root=drive_root,
            task_id=task_id,
            admission_token=admission_token,
            result_fields=result_fields,
        )
        snapshot_failed = (
            str(admitted.get("_admission_blocked") or "")
            == "queue_snapshot_persist_failed"
        )
        rejection = _admission_rejection_response(
            admitted,
            drive_root=drive_root,
            task_id=task_id,
            project_id=project_id,
            workspace_root=task.get("workspace_root"),
            child_drive=child_drive,
            status_code=503 if snapshot_failed else 409,
            detail=(
                "Task was not scheduled because its durable queue snapshot could not be written."
                if snapshot_failed
                else "Task was not scheduled because its admission fence is closed."
            ),
        )
        if rejection is not None:
            return rejection
    except Exception as exc:
        try:
            from supervisor import queue as supervisor_queue

            with supervisor_queue._queue_lock:
                supervisor_queue.PENDING[:] = [
                    row for row in supervisor_queue.PENDING
                    if not (
                        isinstance(row, dict)
                        and str(row.get("id") or "") == task_id
                        and str(row.get("_admission_owner_token") or "")
                        == admission_token
                    )
                ]
            supervisor_queue.persist_queue_snapshot(
                reason="api_task_create_failed_rollback"
            )
        except Exception:
            log.warning(
                "Failed to roll back API task %s after admission error",
                task_id,
                exc_info=True,
            )
        write_task_result(
            drive_root,
            task_id,
            "failed",
            **{
                **result_fields,
                "artifact_status": ARTIFACT_STATUS_FAILED if workspace_root else "",
                "result": f"Failed to enqueue task: {exc}",
            },
        )
        _cleanup_api_admission_attempt(
            drive_root, task_id, admission_token, child_drive
        )
        return json_exception(exc, 503)
    return JSONResponse({"ok": True, "task_id": task_id, "status": STATUS_SCHEDULED})


async def api_tasks_create(request: Request) -> JSONResponse:
    """POST /api/tasks — enqueue a managed headless task."""

    body = await request_json_or(request, {})
    if not isinstance(body, dict):
        return json_error("request body must be a JSON object", 400)
    description = str(body.get("description") or "").strip()
    if not description:
        return json_error("description is required", 400)

    ready_error = _supervisor_ready_error(request)
    if ready_error:
        return ready_error

    drive_root = request_drive_root(request)
    repo_dir = request_repo_dir(request)
    try:
        task_id = validate_task_id(body.get("task_id") or uuid.uuid4().hex[:16])
    except ValueError as exc:
        return json_error(str(exc), 400)
    if load_task_result(drive_root, task_id):
        return json_error(f"task_id already exists: {task_id}", 409)
    if (drive_root / HEADLESS_TASKS_DIR / task_id).exists() or (drive_root / ARTIFACTS_DIR / task_id).exists():
        return json_error(f"task_id already has headless state: {task_id}", 409)
    try:
        workspace_ref = _admit_task_placement(
            body,
            system_repo_dir=repo_dir,
            drive_root=drive_root,
        )
    # ORDER MATTERS. `RemoteWorkspaceUnavailableError` subclasses ValueError on
    # purpose (so every legacy `except ValueError` admission site keeps refusing
    # loudly), which means the generic arm below silently swallowed it into a bare
    # 400 whose only clue was an UPPERCASE constant embedded in English prose. The
    # request was well formed and the target simply could not be consulted, and
    # `/api/projects` already answers exactly that with 503 + a typed code + a next
    # step — two answers to one condition, differing only by which door was used.
    except RemoteWorkspaceUnavailableError as exc:
        # The action rides on the refusal (`remote_refusal_actions`): this line used
        # to hardcode `bootstrap_connection` for every reason a target can be
        # unreachable, which is correct for a stale executor and a dead end for a
        # changed host identity, a stale bundle, or an absent broker.
        return json_error(
            str(exc), 503,
            error_code=wire_error_code(REMOTE_TRANSPORT_UNAVAILABLE),
            action=str(getattr(exc, "action", "") or "retry"),
        )
    except ValueError as exc:
        return json_error(str(exc), 400)
    # Home path of the ADMITTED placement: a real path for a local workspace, None for a
    # remote one. Nothing below may reconstruct a Home path for a remote placement — the
    # sealed ref is the only placement authority from here on.
    workspace_root = workspace_ref.home_path() if getattr(workspace_ref, "kind", "") == "local" else None
    workspace_mode = str(body.get("workspace_mode") or ("external" if workspace_ref else "")).strip()
    memory_mode = str(body.get("memory_mode") or ("forked" if workspace_ref else "shared")).strip().lower()
    if memory_mode not in {"forked", "empty", "shared"}:
        return json_error("memory_mode must be one of forked, empty, shared", 400)
    if workspace_ref and memory_mode == "shared":
        return json_error("memory_mode=shared is not allowed for external workspaces; use forked or empty", 400)
    raw_project_id = str(body.get("project_id") or "")
    if raw_project_id:
        from ouroboros.project_facts import explicit_project_id_ok

        # Validate the UNSTRIPPED value so leading/trailing whitespace (which would
        # collapse two inputs into one store) is rejected, not silently normalized.
        if not explicit_project_id_ok(raw_project_id):
            # Fail closed: an explicit project_id must already be filesystem-clean.
            # Reject (rather than silently normalize/empty -> canonical), so two
            # inputs never collapse to one store and isolation is never defeated.
            return json_error(
                "project_id must be filesystem-safe (alphanumeric/_/-/., no spaces or slashes)", 400)
    from ouroboros.project_facts import resolve_project_id as _resolve_pid

    _task_project_id = _resolve_pid({"project_id": raw_project_id, "workspace_root": str(workspace_root or "")})
    # D5 (Option A): keep the RECORDED memory_mode exactly as requested — shared/forked/
    # empty semantics are unchanged. Isolation for a project-scoped `shared` task comes
    # from MATERIALIZING an isolated child drive (data-root isolation), NOT from mutating
    # the recorded mode. The worker uses task['drive_root'] (the child), and a pure
    # --project-id task never shows the memory_mode line, so the recorded mode stays
    # purely informational while post-task writes still land on the isolated child.
    effective_drive_mode = "forked" if (_task_project_id and memory_mode == "shared") else memory_mode
    task_type = str(body.get("type") or "task")
    if task_type in {"evolution", "review", "deep_self_review"}:
        return json_error(
            f"task type {task_type!r} is internal-only and cannot be created via the task API "
            "(use /evolve or /review); evolution additionally requires advanced/pro runtime mode",
            400,
        )
    if workspace_ref and task_type != "task":
        return json_error("external workspace tasks must use type='task'", 400)
    try:
        chat_id = int(body.get("chat_id") if body.get("chat_id") is not None else 0)
        depth = int(body.get("depth") or 0)
    except (TypeError, ValueError):
        return json_error("chat_id and depth must be integers", 400)

    raw_metadata = dict(body.get("metadata") or {}) if isinstance(body.get("metadata"), dict) else {}
    if _external_subagent_label(body, raw_metadata):
        return json_error("delegation_role=subagent is only allowed through the internal schedule_subagent tool", 400)
    if str(body.get("parent_task_id") or "").strip() or str(body.get("root_task_id") or "").strip():
        return json_error("parent_task_id and root_task_id are internal lineage fields; external tasks must start as roots", 400)
    if "project_id" in raw_metadata:
        # project_id is a top-level field; silently dropping it from metadata would
        # let a caller believe isolation is active while the task runs unscoped.
        return json_error("project_id must be a top-level field, not metadata", 400)
    metadata = {str(k): v for k, v in raw_metadata.items() if str(k) not in _RESERVED_METADATA_KEYS}
    allowed_resources, resource_policy, disabled_tools, acceptance_claims, policy_error = (
        _fold_contract_policies(body, raw_metadata, metadata)
    )
    if policy_error:
        return json_error(policy_error, 400)
    if "executor_ref" in raw_metadata or "workspace_executor" in raw_metadata:
        return json_error("metadata.executor_ref/workspace_executor is reserved; pass executor_ref as a top-level task field", 400)
    if "executor_ref" in body:
        executor_error = _fold_request_executor_ref(
            body,
            metadata,
            workspace_ref=workspace_ref,
            workspace_root=workspace_root,
            repo_dir=repo_dir,
            drive_root=drive_root,
        )
        if executor_error:
            return json_error(executor_error, 400)
    try:
        deadline_at = _normalize_deadline_at(body.get("deadline_at") or raw_metadata.get("deadline_at") or "")
    except ValueError as exc:
        return json_error(str(exc), 400)
    timeout_sec = 0.0
    try:
        timeout_sec = float(body.get("timeout_sec") or body.get("timeout") or 0)
    except (TypeError, ValueError):
        timeout_sec = 0.0
    if not deadline_at and timeout_sec > 0:
        deadline_at = datetime.fromtimestamp(time.time() + timeout_sec, timezone.utc).isoformat().replace("+00:00", "Z")
    if deadline_at:
        metadata["deadline_at"] = deadline_at
    admission_token = uuid.uuid4().hex
    from supervisor.queue import reserve_task_admission

    reservation = reserve_task_admission(
        task_id,
        admission_token,
        require_worker_pool=True,
        drive_root=drive_root,
    )
    if reservation.get("status") != "reserved":
        reason = str(reservation.get("reason") or "admission_reservation_failed")
        status_code = 503 if reason.startswith("worker_pool_") else 409
        return json_error(
            f"task admission refused: {reason}",
            status_code,
            task_id=task_id,
            reason_code=reason,
            worker_pool_disabled_reason=str(
                reservation.get("worker_pool_disabled_reason") or ""
            ),
        )
    try:
        child_drive = prepare_task_drive(
            drive_root, task_id, effective_drive_mode, project_id=_task_project_id
        )
    except Exception as exc:
        _cleanup_api_admission_attempt(drive_root, task_id, admission_token)
        return json_exception(exc, 503)
    # Attachments land on the host that will DO the work: the task's own drive for a
    # local placement, the target's task-file cache for a remote one.
    effective_drive = child_drive or drive_root
    try:
        attachment_manifest, attachment_images, _attachment_note = _place_attachments(
            body.get("attachments"), workspace_ref,
            metadata=metadata, drive_root=effective_drive, task_id=task_id,
        )
    except Exception as exc:
        # Refusing the task is the honest answer: its prompt would otherwise
        # advertise attachment paths that do not exist on the host doing the work.
        _cleanup_api_admission_attempt(
            drive_root, task_id, admission_token, child_drive
        )
        return json_error(
            f"remote attachment staging failed: {exc}", 502,
            error_code="remote_attachment_stage_failed",
        )
    metadata.setdefault("session_id", str(body.get("session_id") or uuid.uuid4().hex))
    metadata.setdefault("actor_id", str(body.get("actor_id") or "cli"))
    metadata.setdefault("source", str(body.get("source") or "api_task"))
    metadata.setdefault("delegation_role", "root")
    parent_task_id = None
    root_task_id = task_id
    metadata.setdefault("task_id", task_id)
    metadata.setdefault("parent_task_id", parent_task_id or "")
    metadata.setdefault("root_task_id", root_task_id)
    display_root = workspace_display_root(workspace_ref)
    artifacts, workspace_preflight_summary = _seal_placement_into_metadata(
        metadata, workspace_ref, drive_root=drive_root, task_id=task_id
    )

    try:
        task_text = _compose_task_text(
            description,
            workspace_root=display_root or None,
            workspace_mode=workspace_mode,
            memory_mode=memory_mode,
            workspace_preflight=workspace_preflight_summary,
            attachments=attachment_manifest,
            attachment_note=_attachment_note,
        )
    except Exception as exc:
        _cleanup_api_admission_attempt(
            drive_root, task_id, admission_token, child_drive
        )
        return json_exception(exc, 503)
    task = {
        "id": task_id,
        "type": task_type,
        "chat_id": chat_id,
        "text": task_text,
        "description": description,
        "context": str(body.get("context") or ""),
        "expected_output": str(body.get("expected_output") or ""),
        "constraints": str(body.get("constraints") or ""),
        "context_requires_self_body_docs": normalize_bool(body.get("context_requires_self_body_docs")),
        "allowed_resources": allowed_resources,
        "resource_policy": resource_policy,
        "disabled_tools": disabled_tools,
        "acceptance_claims": acceptance_claims,
        "deadline_at": deadline_at,
        "depth": depth,
        "parent_task_id": parent_task_id,
        "root_task_id": root_task_id,
        "session_id": metadata["session_id"],
        "actor_id": metadata["actor_id"],
        "delegation_role": metadata["delegation_role"],
        # The DISPLAY spelling of the sealed placement (a Home path for local, the
        # target-native POSIX root for ssh). Presence still means "this is a workspace
        # task" for every existing string-plumbing consumer; the sealed ref in metadata
        # is what decides whether the spelling may become a `pathlib.Path`.
        "workspace_root": display_root,
        "workspace_mode": workspace_mode,
        "memory_mode": memory_mode,
        "project_id": _task_project_id,
        "metadata": metadata,
        # v6.52.0 (P1): the STAGED manifest (root/relpath/mime/is_image), not raw
        # host paths — relpaths resolve against task['drive_root'] at read time.
        "attachments": attachment_manifest,
        "attachment_images": attachment_images,
        # v6.52.0 (P1): record the effective drive (child when forked/empty, else the shared
        # drive) so build_user_content can resolve staged attachment IMAGES for EVERY task
        # shape — not just child-drive tasks. The child-drive block below re-affirms it.
        "drive_root": str(effective_drive),
        "_require_unique_task_id": True,
        "_require_worker_pool": True,
        "_admission_token": admission_token,
    }
    if getattr(workspace_ref, "kind", "") == "ssh":
        # RWSB2-05 for the HTTP surface: this handler resolves the project's placement
        # and only then enqueues, so the owner can rebind the project or retire the
        # connection inside that window. The fence records the routing facts that were
        # TRUE at resolution; `supervisor/queue.py` refuses a stale one under
        # `_queue_lock` rather than inserting work aimed at the previous target. Set for
        # a REMOTE placement only, so the local path keeps its exact refusal codes.
        from ouroboros.workspace_admission import PLACEMENT_FENCE_KEY, placement_fence_for

        task[PLACEMENT_FENCE_KEY] = placement_fence_for(
            drive_root, _task_project_id, workspace_ref
        )
    try:
        task = attach_task_contract(task)
    except Exception as exc:
        _cleanup_api_admission_attempt(
            drive_root, task_id, admission_token, child_drive
        )
        return json_exception(exc, 503)
    if child_drive is not None:
        task["drive_root"] = str(child_drive)
        task["child_drive_root"] = str(child_drive)
        task["budget_drive_root"] = str(drive_root)
        metadata["child_drive_root"] = str(child_drive)
        metadata["budget_drive_root"] = str(drive_root)
    return _complete_api_task_admission(
        task,
        drive_root=drive_root,
        task_id=task_id,
        admission_token=admission_token,
        project_id=_task_project_id,
        description=description,
        allowed_resources=allowed_resources,
        deadline_at=deadline_at,
        workspace_root=workspace_root,
        workspace_mode=workspace_mode,
        memory_mode=memory_mode,
        child_drive=child_drive,
        artifacts=artifacts,
        metadata=metadata,
    )


_TASKS_LIST_DEFAULT_LIMIT = 50
_TASKS_LIST_MAX_LIMIT = 500

# Bulk evidence fields omitted from LIST rows (v6.9x P2): they are the megabyte
# carriers of a task summary and have zero code consumers on the list surface
# (result_index and the UI detail views read them from GET /api/tasks/{id},
# which keeps the full envelope). `result` stays — pinned by test_headless_cli.
_LIST_ROW_OMITTED_FIELDS = frozenset({
    "loop_outcome",
    "trace_refs",
    "verification_ledger",
    "review_evidence",
    "subagent_envelope",
})

# Process-wide {(results_dir, filename) -> raw ts} memo for the unfiltered list
# path. The raw `ts` is CREATION-STABLE (write_task_result sets it on the first
# write; later updates touch only updated_at), so entries never need
# invalidation — only deletions are dropped and new names decoded. Keyed by the
# directory too, so multiple drive roots (tests, child drives) never collide.
# Concurrency note: worst case a race re-reads a file and stores the identical
# creation-stable value; no lock needed.
_RAW_TS_MEMO: Dict[tuple, str] = {}


def _compact_list_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Compact LIST projection: drop the five bulk evidence fields, keep the
    summary contract (task_id, status, ts/updated_at, result, description/
    objective/title/role, lineage, project_id, reason_code, artifact_status,
    workspace fields, the TASK_COST_META_FIELDS, outcome_axes — all preserved
    because the projection is subtractive, never a whitelist)."""
    return {key: value for key, value in row.items() if key not in _LIST_ROW_OMITTED_FIELDS}


def _raw_sorted_result_names(results_dir: pathlib.Path) -> List[str]:
    """Result filenames sorted newest-first by RAW creation ts (memoized).

    A row whose file lacks `ts` sorts as minus-infinity (oldest), tie-broken by
    filename for determinism. A file that fails to parse (torn concurrent
    write) is excluded from THIS request and left out of the memo, so the next
    request re-reads it — a torn write can never poison the memo."""
    try:
        with os.scandir(results_dir) as entries:
            names = [entry.name for entry in entries if entry.name.endswith(".json")]
    except OSError:
        return []
    dir_key = str(results_dir)
    present = set(names)
    for key in [k for k in list(_RAW_TS_MEMO) if k[0] == dir_key and k[1] not in present]:
        _RAW_TS_MEMO.pop(key, None)
    decorated: List[tuple] = []
    for name in names:
        key = (dir_key, name)
        raw_ts = _RAW_TS_MEMO.get(key)
        if raw_ts is None:
            data = read_json_dict(results_dir / name)
            if data is None:
                continue
            raw_ts = str(data.get("ts") or "")
            _RAW_TS_MEMO[key] = raw_ts
        decorated.append((raw_ts, name))
    decorated.sort(reverse=True)  # "" (no ts) sorts after every real timestamp
    return [name for _ts, name in decorated]


def _tasks_list_payload(
    drive_root: pathlib.Path,
    wanted: set,
    limit: Optional[int],
    queue_only: bool,
) -> Dict[str, Any]:
    """Assemble the /api/tasks response off the event loop.

    Unfiltered requests slice BEFORE projection (v6.9x P2): sort raw filenames
    by the creation-stable raw ts, decode/project only the top-`limit` files,
    then re-sort that slice by EFFECTIVE ts (a child-drive merge can replace ts
    with the child's). Residual, disclosed: top-N membership is decided on raw
    ts, so an old task freshly completed through its child can fall outside the
    slice until its raw file is rewritten. Status-filtered requests keep the
    full projection path — filtering needs every row's effective status (the
    child-drive promotion contract pinned by test_headless_cli)."""
    if queue_only:
        return {"tasks": [], "queue": _queue_snapshot(drive_root)}
    # One shared events-tail parse for every stale-running orphan check in this
    # request (lazy: zero reads when no running row consults it).
    events_index = _EventsTailIndex(drive_root)
    # List view is a status/cost projection: never materialize artifacts (no child
    # rebase copies, no artifact-dir scans, no disposition/sha claims) on a GET list.
    if wanted:
        rows = [
            _compact_list_row(public_task_result(effective_task_result(
                drive_root, row, materialize_artifacts=False, _events_index=events_index,
            )))
            for row in list_task_results(drive_root)
        ]
        rows = [row for row in rows if str(row.get("status") or "").lower() in wanted]
        rows.sort(key=lambda item: str(item.get("ts") or ""), reverse=True)
        if limit is not None:
            rows = rows[:limit]
        return {"tasks": rows, "queue": _queue_snapshot(drive_root)}
    results_dir = task_results_dir(drive_root, create=False)
    names = _raw_sorted_result_names(results_dir)
    if limit is not None:
        names = names[:limit]
    rows = []
    for name in names:
        raw = read_json_dict(results_dir / name)
        if raw is None:
            continue  # vanished/torn between the scandir and this read
        rows.append(_compact_list_row(public_task_result(effective_task_result(
            drive_root, raw, materialize_artifacts=False, _events_index=events_index,
        ))))
    # Re-sort the slice by effective ts: the child-drive merge may have replaced
    # ts, and the response order is the displayed order.
    rows.sort(key=lambda item: str(item.get("ts") or ""), reverse=True)
    return {"tasks": rows, "queue": _queue_snapshot(drive_root)}


async def api_tasks_list(request: Request) -> JSONResponse:
    """GET /api/tasks — compact list projection plus the queue snapshot.

    ``limit`` defaults to 50 and explicit positive values cap at 500 (both
    unchanged); ``limit=0`` returns ALL rows (new, v6.9x P2 — previously it
    coerced to 1). ``queue_only=1`` skips the task-results scan entirely and
    answers ``{tasks: [], queue}`` — the Activity dashboard consumes only the
    queue."""
    statuses = [
        item.strip()
        for item in str(request.query_params.get("status") or "").split(",")
        if item.strip()
    ]
    raw_limit = coerce_int(request.query_params.get("limit"), _TASKS_LIST_DEFAULT_LIMIT)
    limit = None if raw_limit == 0 else max(1, min(raw_limit, _TASKS_LIST_MAX_LIMIT))
    queue_only = str(request.query_params.get("queue_only") or "").strip().lower() in {"1", "true", "yes"}
    drive_root = request_drive_root(request)
    wanted = {status.lower() for status in statuses}
    payload = await asyncio.to_thread(_tasks_list_payload, drive_root, wanted, limit, queue_only)
    return JSONResponse(payload)


def _task_cost_breakdown_view(drive_root: pathlib.Path, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Read-side "where did the money go" projection for a ROOT task's detail.

    Computed from the physical-attempt ledger AT READ TIME and never persisted
    into the task result — the ledger stays the single monetary authority (P7);
    the stored envelope keeps only its existing own/subtree projections.
    ``children_usd`` is subtree − own − unattributed (the subtraction every
    reader had to do by hand); ``delegated`` is a filter over the execution
    axis (subscription sessions), not a third sum. Unavailable accounting
    returns None — the field is simply absent, never a confident $0. That
    covers BOTH an unreadable ledger and a readable one that holds no
    attributable row for this subtree (empty or legacy-only): ``_summary()``
    always returns a float for ``accounted_usd``, so "no accounting happened"
    is decided on the ROW COUNTS, never on the dollar sum being 0.0."""
    task_id = str(result.get("task_id") or "")
    root_id = str(result.get("root_task_id") or "") or task_id
    # Subtree math is ledger-attributable only at the root (child rows carry
    # the ROOT's id, not every ancestor's); non-root details omit the view.
    if not task_id or root_id != task_id:
        return None
    try:
        from ouroboros.usage_accounting import usage_breakdown

        breakdown = usage_breakdown(drive_root, root_task_id=root_id)
    except Exception:
        log.debug("cost breakdown view unavailable for %s", task_id, exc_info=True)
        return None
    subtree = breakdown.get("accounted_usd")
    counts = breakdown.get("attempt_counts")
    counts = counts if isinstance(counts, dict) else {}
    # `metadata_only` is a count of AMBIGUOUS legacy calls carrying no money, so
    # it can never make a $0 measured; only priced attempt rows or subscription
    # sessions can. With neither, nothing was accounted for this subtree and the
    # view is ABSENT — the empty/legacy-ledger case that a `0.0 == measured zero`
    # reading would have published as `own 0 / children 0 / cost_final true`.
    priced_rows = sum(int(value or 0) for key, value in counts.items() if key != "metadata_only")
    sessions = int(breakdown.get("subscription_sessions") or 0)
    if subtree is None or (priced_rows <= 0 and sessions <= 0):
        return None
    own_bucket = (breakdown.get("by_task") or {}).get(task_id)
    # No rows attributed to the root itself is a MEASURED zero (all spend was
    # children's), not an unknown — unknowns ride `unknown_unmetered` below.
    own = float(own_bucket.get("accounted_usd") or 0.0) if isinstance(own_bucket, dict) else 0.0
    # Money inside this subtree that no task id claims (legacy/blank-task rows)
    # is DISCLOSED on its own axis instead of being silently folded into the
    # children's share: own + children + unattributed == subtree.
    unattributed_bucket = (breakdown.get("unattributed") or {}).get("task")
    unattributed = (
        float(unattributed_bucket.get("accounted_usd") or 0.0)
        if isinstance(unattributed_bucket, dict) else 0.0
    )
    delegated = breakdown.get("delegated") if isinstance(breakdown.get("delegated"), dict) else {}
    return {
        "own_usd": round(own, 6),
        "children_usd": round(max(0.0, float(subtree) - own - unattributed), 6),
        "unattributed_usd": round(unattributed, 6),
        "delegated_disclosed_usd": round(float(delegated.get("settled_usd") or 0.0), 6),
        # C2: the explicit subtree total under its honest name — an accounted
        # UPPER BOUND (own + children + unattributed), not a settled receipt.
        "accounted_upper_bound_usd": round(float(subtree), 6),
        "subscription_sessions": sessions,
        "unknown_unmetered": breakdown.get("unknown_unmetered"),
        "non_final_rows": breakdown.get("non_final_rows"),
        "cost_final": bool(breakdown.get("cost_final")),
        "authority": "physical_attempt_ledger",
    }


async def api_task_get(request: Request) -> JSONResponse:
    try:
        task_id = validate_task_id(request.path_params.get("task_id"))
    except ValueError as exc:
        return json_error(str(exc), 400)
    drive_root = request_drive_root(request)
    data = load_effective_task_result(drive_root, task_id)
    if not data:
        return json_error("task not found", 404)
    payload = public_task_result(data)
    breakdown_view = _task_cost_breakdown_view(drive_root, data)
    if breakdown_view is not None:
        payload["cost_breakdown"] = breakdown_view
    return JSONResponse(payload)


async def api_task_artifact(request: Request):
    try:
        task_id = validate_task_id(request.path_params.get("task_id"))
    except ValueError as exc:
        return json_error(str(exc), 400)
    name = str(request.path_params.get("name") or "").strip()
    if not name or "/" in name or "\\" in name or name in {".", ".."} or ".." in pathlib.PurePosixPath(name).parts:
        return json_error("artifact name must be a simple filename", 400)
    drive_root = request_drive_root(request)
    result = load_effective_task_result(drive_root, task_id)
    if not result:
        return json_error("task not found", 404)
    artifact = _artifact_by_name(result, name)
    if artifact is None:
        return json_error("artifact not found", 404, task_id=task_id, artifact=name)
    base = task_artifacts_dir(drive_root, task_id).resolve(strict=False)
    path = pathlib.Path(str(artifact.get("path") or "")).resolve(strict=False)
    if path.name != name:
        return json_error("artifact metadata path does not match requested name", 500)
    try:
        path.relative_to(base)
    except ValueError:
        return json_error("artifact path is outside task artifact directory", 500)
    if not path.is_file():
        return json_error("artifact file is missing", 404, task_id=task_id, artifact=name)
    return FileResponse(path)


def _record_cascade_incident(task_id: str, kind: str, detail: str = "") -> None:
    """Durably record a cascade outcome the owner must be able to see.

    The client already holds ``ok:true`` and the card only resolves on a real
    ``task_done``, so a cascade that RAISED or that cancelled NOTHING is a silent
    lie unless it is recorded. Both land on the supervisor's own drive root (the
    same log every other cancel artifact uses) and are pushed to the live owner
    surfaces when a bridge exists.
    """
    row = {"type": kind, "task_id": task_id}
    if detail:
        row["error"] = detail
    try:
        from supervisor import queue as supervisor_queue
        from ouroboros.utils import append_jsonl, utc_now_iso

        append_jsonl(
            pathlib.Path(supervisor_queue.DRIVE_ROOT) / "logs" / "supervisor.jsonl",
            {"ts": utc_now_iso(), **row},
        )
    except Exception:
        log.debug("Failed to persist cascade cancel incident for %s", task_id, exc_info=True)
    try:
        from supervisor.message_bus import try_get_bridge

        bridge = try_get_bridge()
        if bridge is not None:
            from ouroboros.utils import utc_now_iso

            bridge.push_log({"ts": utc_now_iso(), **row})
    except Exception:
        log.debug("Failed to surface cascade cancel incident for %s", task_id, exc_info=True)


def _run_cascade_cancel(task_id: str) -> bool:
    """Subtree cancel for the HTTP cascade path, AWAITED by its caller.

    Returns True when the subtree is settled — cancelled, or already terminal
    (the benign completion-wins race) — and False when the teardown failed or
    refused while the tree is STILL live, which the endpoint reports rather than
    answering ok:true for a cancellation that did not happen. Failures stay
    durable and owner-visible as incidents either way.
    """
    try:
        from supervisor.queue import cancel_task_by_id

        if not cancel_task_by_id(task_id, cascade=True):
            # A cascade that cancelled nothing is only an incident when the subtree
            # is STILL live: the ordinary completion-wins race (the task reached
            # terminal between the pre-check and this call) is the benign case the
            # UI already handles, and reporting it would train the owner to ignore
            # the real "refused to cancel" signal.
            from supervisor.queue import task_subtree_is_live

            if task_subtree_is_live(task_id):
                _record_cascade_incident(task_id, "task_cancel_cascade_noop")
                return False
        return True
    except Exception as exc:
        log.warning("Cascade cancel failed for %s", task_id, exc_info=True)
        _record_cascade_incident(task_id, "task_cancel_cascade_error", repr(exc))
        return False


# Sentinel telling "the body did not parse" apart from a legitimate JSON null.
_NO_BODY = object()


async def api_task_cancel(request: Request) -> JSONResponse:
    try:
        task_id = validate_task_id(request.path_params.get("task_id"))
    except ValueError as exc:
        return json_error(str(exc), 400)
    # Optional JSON body {"cascade": true} (v6.82): cancel the task AND its
    # atomically-snapshotted live subtree, answering only once that teardown has
    # finished. An absent/empty body keeps today's single-task behavior
    # byte-identical for headless callers (the CLI posts {}).
    # An ABSENT body keeps the legacy single-task path; a body that is PRESENT but
    # unparseable (or not a JSON object) is a client error. Collapsing the two would
    # answer a malformed cascade request by quietly cancelling only the root and
    # leaving its descendants running.
    raw_body = (await request.body()) or b""
    if raw_body.strip():
        body = await request_json_or(request, _NO_BODY)
        if body is _NO_BODY or not isinstance(body, dict):
            return json_error("request body must be a JSON object", 400, task_id=task_id)
    else:
        body = {}
    # STRICT boolean (DEVELOPMENT.md): a string "false" must never select the
    # destructive subtree path, and a non-boolean value is a client error rather
    # than a silent single-task cancel.
    raw_cascade = body.get("cascade")
    if raw_cascade is not None and not isinstance(raw_cascade, bool):
        return json_error("cascade must be a boolean", 400, task_id=task_id)
    cascade = raw_cascade is True

    def _record_http_intent(
        source: str, *, cascade_scope: bool = False, allow_settled: bool = False,
    ) -> bool:
        """ALL cancel ingress goes through the durable intent (owner batch-4 1=A):
        the intent survives a lost event/crash mid-teardown and the supervisor
        watchdog re-feeds it into custody. FAIL-CLOSED (AR2-1, mirroring the
        agent tool lane): a cancel whose durable intent could not be recorded is
        REFUSED — teardown without the intent would recreate exactly the
        unfenced, unreplayable cancel the redesign removes.

        The cascade endpoint mints with ``scope=cascade`` AT THE INGRESS
        (GR2-1a): a crash before the supervisor's own scope stamp would
        otherwise leave a single-scope intent that a watchdog replay runs as a
        single cancel, settling the root while its descendants keep running.
        It also mints over an ALREADY-SETTLED root (GR2-1b): a settled root
        with live descendants still needs the durable cascade coordination
        intent — it is the watchdog's replay trigger for the descendants and
        settles only when the cascade's no-live postcondition passes.

        ``allow_settled`` (GR6-1) is the single lane's LIVE-OWNERSHIP fact: a
        settled RESULT with a live worker (post-task cognition still spending)
        must still mint, or the ingress no-ops while the worker burns —
        ``already_settled`` is terminal only when no live ownership remains.

        Returns "" on success, or a typed refusal kind: "projection_corrupt"
        (GR4-8 — the projection FILE is malformed; a retry cannot succeed
        until it is repaired) vs "write_failed" (transient — retry)."""
        try:
            from supervisor.queue import DRIVE_ROOT as _drive_root

            from ouroboros.cancel_intents import (
                CancelIntentProjectionCorrupt,
                SCOPE_CASCADE,
                request_cancel,
            )
        except Exception:
            log.warning("HTTP cancel-intent machinery unavailable for %s", task_id,
                        exc_info=True)
            return "write_failed"
        try:
            request_cancel(
                _drive_root, task_id, source=source,
                **({"scope": SCOPE_CASCADE} if cascade_scope else {}),
                allow_settled_target=bool(cascade_scope or allow_settled),
            )
            return ""
        except CancelIntentProjectionCorrupt:
            log.error("HTTP cancel refused for %s: intent projection corrupt", task_id)
            return "projection_corrupt"
        except Exception:
            log.warning("HTTP cancel-intent write failed for %s; cancel refused",
                        task_id, exc_info=True)
            return "write_failed"

    def _intent_write_refused(kind: str) -> JSONResponse:
        if kind == "projection_corrupt":
            # GR4-8: honest wording — "retry" cannot succeed while the file is
            # malformed. The corrupt state/cancel_intents.json was PRESERVED
            # (never overwritten) and a projection_corrupt_refused forensic row
            # was recorded in logs/supervisor.jsonl.
            return json_error(
                "the cancel-intent projection (state/cancel_intents.json) is corrupt; "
                "nothing was cancelled and retrying cannot succeed until the file is "
                "repaired — the malformed file was preserved (no overwrite) and a "
                "projection_corrupt_refused forensic row was recorded in "
                "logs/supervisor.jsonl",
                503, task_id=task_id, reason_code="cancel_intent_projection_corrupt",
            )
        return json_error(
            "durable cancel intent could not be recorded; nothing was cancelled — retry",
            503, task_id=task_id, reason_code="cancel_intent_write_failed",
        )

    if not cascade:
        try:
            from supervisor.queue import (
                CANCEL_CANCELLED, CANCEL_FAILED, cancel_task_custody,
                task_has_live_ownership as _live_ownership,
                task_subtree_is_live as _live_check,
            )

            # Intent only for a LIVE task: the plain path's legacy contract
            # answers 404 for an inactive id, and an intent minted for a dead id
            # would sit open until the watchdog settles it as not_found.
            # LIVE OWNERSHIP (GR6-1) widens the gate: a settled result whose
            # worker is still alive is not "inactive" — the intent is minted
            # with ``allow_settled`` so custody kills the spending worker.
            live_own = await asyncio.to_thread(_live_ownership, task_id)
            if live_own or await asyncio.to_thread(_live_check, task_id):
                refused = await asyncio.to_thread(functools.partial(
                    _record_http_intent, "http_single", allow_settled=live_own,
                ))
                if refused:
                    return _intent_write_refused(refused)
            # The TYPED outcome, not a boolean: a task whose worker refused to die
            # is neither cancelled nor absent, and answering 404 for it would tell
            # the caller the task is gone while it keeps running.
            outcome = await asyncio.to_thread(cancel_task_custody, task_id)
        except Exception as exc:
            return json_exception(exc, 503)
        if outcome == CANCEL_FAILED:
            return json_error(
                "cancellation did not settle; the task is still live",
                503, task_id=task_id,
            )
        if outcome != CANCEL_CANCELLED:
            # LEGACY CONTRACT preserved: the plain path has always answered 404 for
            # an INACTIVE task, and one that already settled on its own is exactly
            # that — the typed outcome must not silently widen the envelope.
            return json_error("task not found or not active", 404, task_id=task_id)
        return JSONResponse({"ok": True, "task_id": task_id})
    # Cascade path: ONE synchronous transaction. The caller is answered only once
    # the subtree is actually torn down, which is what makes the whole
    # split-transaction family (durable pre-acknowledgement latch, partial-latch
    # taxonomy, ownership handed to a background teardown, rollbacks that could
    # withdraw a concurrent cascade's fences) unnecessary rather than merely
    # guarded. The cost is honest and bounded: a large tree makes the caller wait
    # for the worker kills and joins it asked for. Off the event loop; process
    # kills and joins deliberately happen outside the supervisor queue lock. Repeats are
    # idempotent (the per-task cancel finalizes-on-miss) and a fully-cancelled tree
    # is no longer live, so it answers 404 like any other inactive task.
    try:
        from supervisor.queue import (
            task_has_live_ownership as _cascade_live_ownership,
            task_subtree_is_live,
        )

        # GR7-1b: the 404 pre-check consults the SAME live-ownership predicate
        # the single lane uses. `task_subtree_is_live` deliberately excludes a
        # settled-but-RUNNING root (so the cascade postcondition can converge
        # over a winding-down finalizer), which made this pre-check answer 404
        # for a settled root whose worker was still burning post-task
        # cognition — before any intent was minted. A settled-but-LIVE root
        # proceeds to mint + custody (the kill path preserves the stored
        # result); a genuinely settled-AND-dead tree keeps the 404 envelope.
        if not await asyncio.to_thread(
            _cascade_live_ownership, task_id,
        ) and not await asyncio.to_thread(task_subtree_is_live, task_id):
            return json_error("task not found or not active", 404, task_id=task_id)
        refused = await asyncio.to_thread(
            functools.partial(_record_http_intent, "http_cascade", cascade_scope=True),
        )
        if refused:
            return _intent_write_refused(refused)
        settled = await asyncio.to_thread(_run_cascade_cancel, task_id)
        if not settled:
            # The teardown refused or failed while the subtree is STILL live: an
            # ok:true here would report a cancellation that did not happen.
            return json_error(
                "subtree cancellation did not settle; the tree is still live",
                503, task_id=task_id,
            )
    except Exception as exc:
        return json_exception(exc, 503)
    return JSONResponse({"ok": True, "task_id": task_id, "cascade": True})


async def api_task_resume(request: Request) -> JSONResponse:
    """Resume only a replay-safe task paused before its first model dispatch."""
    try:
        task_id = validate_task_id(request.path_params.get("task_id"))
    except ValueError as exc:
        return json_error(str(exc), 400)
    try:
        from supervisor.queue import resume_budget_paused_task

        result = resume_budget_paused_task(task_id)
    except Exception as exc:
        return json_exception(exc, 503)
    if result.get("ok"):
        return JSONResponse(result)
    error = str(result.get("error") or "resume_refused")
    status = 409 if error in {
        "task_not_budget_paused", "replay_unsafe", "root_budget_fence_missing",
    } else 404
    return json_error(error, status, task_id=task_id, **({"action": result["action"]} if result.get("action") else {}))


# The task-event SSE endpoint and its follower live in gateway/task_events.py
# (split by the 1600-line module gate). Re-exported at the top of this module
# so route wiring, the CLI, and monkeypatch pins keep addressing gateway.tasks.














def _compose_task_text(
    description: str,
    *,
    # The placement's DISPLAY spelling (str for a remote root, Path for a local one) —
    # rendered into the prompt block, never resolved.
    workspace_root: Any,
    workspace_mode: str,
    memory_mode: str,
    workspace_preflight: Dict[str, Any],
    attachments: Any,
    attachment_note: str = "",
) -> str:
    parts = [description]
    if workspace_root is not None:
        from ouroboros.workspace_admission import compose_workspace_block

        # SSOT block (v6.58.0): the same [HEADLESS_WORKSPACE] guidance the promote
        # path embeds, so the two admission surfaces render identical context.
        workspace_lines = compose_workspace_block(
            workspace_root=workspace_root,
            workspace_mode=workspace_mode,
            memory_mode=memory_mode,
            workspace_preflight=workspace_preflight,
        )
        if "[HEADLESS_WORKSPACE]" in description and "[END_HEADLESS_WORKSPACE]" in description:
            marker = "[END_HEADLESS_WORKSPACE]"
            idx = description.rfind(marker)
            parts = [description[:idx].rstrip(), "\n", workspace_lines, description[idx:]]
        else:
            parts.append(f"\n\n[HEADLESS_WORKSPACE]\n{workspace_lines}[END_HEADLESS_WORKSPACE]")
    rendered = _render_attachment_lines(attachments)
    # An omission note without any surviving attachment still gets its block: the
    # owner attached files and is owed the sentence saying why none arrived, and an
    # empty [ATTACHMENTS] section is exactly the silence D7 forbids.
    if rendered or attachment_note:
        body = "\n".join(part for part in (rendered, attachment_note.strip()) if part)
        parts.append(f"\n\n[ATTACHMENTS]\n{body}\n[END_ATTACHMENTS]")
    return "".join(parts)




def _render_attachment_lines(attachments: Any) -> str:
    """Render READY attachment lines from a staged manifest.

    v6.52.0 (P1): each line is a ready-to-use read_file call against the canonical
    artifact_store root — NEVER a bare absolute host path. ``attachments`` is the
    manifest returned by ``stage_task_attachments`` (entries with root/relpath/mime/
    is_image)."""
    if not isinstance(attachments, list):
        return ""
    lines: List[str] = []
    for item in attachments:
        if not isinstance(item, dict):
            continue
        relpath = str(item.get("relpath") or "").strip()
        root = str(item.get("root") or "artifact_store").strip() or "artifact_store"
        label = str(item.get("label") or pathlib.Path(relpath).name).strip()
        if not relpath:
            continue
        kind = "image" if item.get("is_image") else (str(item.get("mime") or "").strip() or "file")
        # v6.54.3: also surface the REAL staged path for process tools — scripts
        # (openpyxl, audio, ffmpeg) open files by OS path, and omitting it made
        # models GUESS wrong absolute paths that tripped light-mode path guards.
        # The staged path lives inside this task's own artifact_store, so both
        # forms address the same file.
        abs_path = str(item.get("abs_path") or "").strip()
        script_hint = f" | script/process path: {abs_path}" if abs_path else ""
        lines.append(
            f"- {label} ({kind}): read_file(root='{root}', path='{relpath}'){script_hint}"
        )
    return "\n".join(lines)


def _artifact_by_name(result: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    for artifact in result.get("artifacts") or []:
        if not isinstance(artifact, dict):
            continue
        if str(artifact.get("name") or pathlib.Path(str(artifact.get("path") or "")).name) == name:
            return artifact
    return None


def _queue_snapshot(drive_root: pathlib.Path) -> Dict[str, Any]:
    path = pathlib.Path(drive_root) / "state" / "queue_snapshot.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _supervisor_ready_error(request: Request) -> Optional[JSONResponse]:
    state = getattr(request.app, "state", None)
    ready_event = getattr(state, "supervisor_ready_event", None) if state is not None else None
    if ready_event is not None and not ready_event.is_set():
        return json_error("supervisor is still starting", 503)
    try:
        from supervisor.workers import worker_pool_admission_state

        pool_state = worker_pool_admission_state()
        if ready_event is not None and not pool_state["available"]:
            return json_error(
                "supervisor worker pool is unavailable",
                503,
                reason_code="worker_pool_unavailable",
                worker_pool_disabled_reason=str(pool_state.get("disabled_reason") or ""),
            )
    except Exception as exc:
        if ready_event is not None:
            return json_error(
                "supervisor worker-pool state is unavailable",
                503,
                reason_code="worker_pool_state_unavailable",
                detail=f"{type(exc).__name__}: {exc}",
            )
    return None


__all__ = [
    "api_task_artifact",
    "api_task_cancel",
    "api_task_resume",
    "api_task_events",
    "api_task_get",
    "api_tasks_create",
    "api_tasks_list",
    "iter_task_events",
]
