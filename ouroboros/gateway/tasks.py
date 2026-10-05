"""Headless task gateway endpoints."""

from __future__ import annotations

import asyncio
import functools
import logging
import pathlib
import shutil
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.gateway._helpers import coerce_int, json_error, json_exception, request_drive_root, request_json_or, request_repo_dir, run_sync_to_completion, stage_initial_task_attachments
from ouroboros.gateway.cost_breakdown import _task_cost_breakdown_view  # noqa: F401
from ouroboros.gateway.contracts import TaskCreateRequest
from ouroboros.gateway.schema import validate_ingress
from ouroboros.depth_evidence import parse_task_depth
from ouroboros.project_naming import admission_names
from supervisor.log_addressing import ProjectThreadConflict, ingress_chat_id
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
# Re-exported hurry ingress (same module-size split as task_events): route
# wiring and tests address gateway.tasks.api_task_hurry.
from ouroboros.gateway.task_hurry import api_task_hurry  # noqa: F401
from ouroboros.gateway.task_pause import api_task_pause, owner_tree_control_routes  # noqa: F401 -- same split as hurry
from ouroboros.gateway.task_decision import api_decision_answer  # noqa: F401
from ouroboros.gateway.task_archive import (
    chat_media_identity, directory_archives, plain_segments, serve_directory_archive, serve_task_file,
    task_artifact_location, recorded_identity, serve_task_source,
)
from ouroboros.task_custody import task_artifact_stores
from ouroboros.headless import (
    ARTIFACTS_DIR,
    ARTIFACT_STATUS_FAILED,
    ARTIFACT_STATUS_PENDING,
    HEADLESS_TASKS_DIR,
    prepare_task_drive,
    task_artifacts_dir,
    write_workspace_preflight_artifact,
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
from ouroboros import artifacts as artifact_store
from ouroboros.task_result_schema import (
    emit_quarantine_event,
    quarantine_task_result,
    task_result_schema_refusal,
)
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
    observe_cancellation_target,
)
from ouroboros.utils import read_json_dict, utc_now_iso
from ouroboros.tool_access import path_is_relative_to, paths_overlap_casefold
from ouroboros.workspace_preflight import (
    collect_workspace_preflight,
    summarize_workspace_preflight,
)
from ouroboros.workspace_executor import normalize_executor_ref


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
    "input_sources",
    "allowed_resources",
    "deadline_at",
    "executor_ref",
    "workspace_executor",
    "project_id",
    # The owner door's stamp (read as ``run_origin.owner_ingress`` by the corpus
    # label and the routing issuer) belongs to owner routing, never to a caller.
    "origin_message_ref",
    "origin_suppressed",
})


def _cleanup_api_admission_attempt(
    drive_root: pathlib.Path,
    task_id: str,
    admission_token: str,
    child_drive: Optional[pathlib.Path] = None,
) -> None:
    """Clean a known pre-enqueue failure while its reservation excludes rivals."""
    from supervisor import queue

    with queue._queue_lock:
        if (queue.ADMISSION_RESERVATIONS.get(task_id) != admission_token
                or task_id in queue.RUNNING or any(row.get("id") == task_id for row in queue.PENDING)):
            return
        try:
            if load_task_result(drive_root, task_id, strict=True) is not None:
                return
        except Exception:
            return  # unreadable identity is not proof of our preparation ownership
    # Keep the reservation through cleanup, but not Q through drive preparation.
    # Even a raising prepare_task_drive may have created a partial owned drive.
    from ouroboros.headless import remove_subagent_task_drive
    try:
        remove_subagent_task_drive(drive_root, task_id, live=queue.task_settlement_liveness,
                                   guard=queue.task_settlement_interlock, admission_rollback=True)
    except Exception:
        log.warning("Failed to clean child drive for rejected task %s", task_id, exc_info=True)
    try:
        shutil.rmtree(task_artifacts_dir(drive_root, task_id, create=False), ignore_errors=True)
    except Exception:
        log.warning("Failed to clean admission artifacts for task %s", task_id, exc_info=True)
    queue.release_task_admission(task_id, admission_token)


def _external_subagent_label(body: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
    return any(
        str(source.get("delegation_role") or "").strip().lower() == "subagent"
        for source in (body, metadata)
    )


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
    workspace_root: Optional[pathlib.Path],
    child_drive: Optional[pathlib.Path],
    status_code: int = 409,
    detail: str = "Task was not scheduled because its admission fence is closed.",
) -> Optional[JSONResponse]:
    """Terminalize a typed queue refusal so no scheduled phantom remains."""
    if not (isinstance(admitted, dict) and admitted.get("_admission_blocked")):
        return None
    reason_code = str(admitted.get("_admission_blocked") or "admission_fence")
    if reason_code.startswith("project_routing_fence"):
        from ouroboros.project_dialogue import routing_refusal_cause

        detail = routing_refusal_cause("promote_chat_to_task", "failed", reason_code) + "."
    if reason_code == "invalid_task_depth":
        detail, status_code = str(admitted.get("_admission_detail") or "Task was not scheduled: depth must be a non-negative integer."), 400
    if reason_code == "task_id_lookup_failed":
        return JSONResponse(
            {
                "error": "Task identity authority is unreadable; no existing bytes were changed.",
                "task_id": task_id,
                "status": "rejected",
                "admission": {"reason_code": reason_code},
            },
            status_code=409,
        )
    if reason_code in {"duplicate_task_id", "admission_reservation_lost", "admission_reservation_owned"}:
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
        "detail": str(admitted.get("_admission_detail") or ""),
        "project_id": str(admitted.get("_project_id") or project_id),
        "project_lifecycle": str(admitted.get("_project_lifecycle") or ""),
        "acceptance_fence_token": str(admitted.get("_acceptance_fence_token") or ""),
        "acceptance_fence_status": str(admitted.get("_acceptance_fence_status") or ""),
    }
    from supervisor.task_admission import persist_never_admitted_refusal

    write_refusal = persist_never_admitted_refusal if admitted.get("_admission_never_admitted") else write_task_result
    write_refusal(
        drive_root,
        task_id,
        **({"admission_token": str(admitted.get("_admission_owner_token") or "")}
           if admitted.get("_admission_never_admitted") else {"status": STATUS_FAILED}),
        reason_code=reason_code,
        admission=admission,
        **({"admission_outcome": "never_admitted"} if admitted.get("_admission_never_admitted") else {}),
        artifact_status=ARTIFACT_STATUS_FAILED if workspace_root else "",
        result=detail,
        accounted_upper_bound_usd=0.0,
    )
    if child_drive is not None:
        from ouroboros.headless import remove_subagent_task_drive
        from supervisor.queue import task_settlement_interlock, task_settlement_liveness
        removed = remove_subagent_task_drive(drive_root, task_id, live=task_settlement_liveness,
                                             guard=task_settlement_interlock, admission_rollback=True)
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

    with queue.prepared_root_billing(task), queue._queue_lock:  # the ledger read happens before the lock
        admitted = queue.enqueue_task(task)
        if isinstance(admitted, dict) and admitted.get("_admission_blocked"):
            admitted.update(_admission_never_admitted=True, _admission_owner_token=admission_token)
            return admitted
        try:
            if queue.persist_queue_snapshot(reason="api_task_create") is not True:
                raise RuntimeError("Queue snapshot persistence was not confirmed")
            write_task_result(drive_root, task_id, STATUS_SCHEDULED, **result_fields)
            stored = load_task_result(drive_root, task_id, strict=True) or {}
            if stored.get("api_admission") != result_fields["api_admission"]:
                raise RuntimeError("The exact API admission receipt was not persisted")
        except Exception as exc:
            # An observer may raise AFTER atomic replacement. Read the exact
            # token; unknown persistence retains the queue row and its resources.
            try:
                stored = load_task_result(drive_root, task_id, strict=True) or {}
            except Exception:
                stored = {}
            if stored.get("api_admission") != result_fields["api_admission"]:
                return {**admitted, "_admission_uncertain": str(exc)}
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
    """Publish one API admission; retain custody when persistence is unknown."""
    result_fields = {
        **({"created_at": task["created_at"]} if task.get("created_at") else {}),
        **{key: task.get(key) for key in (
            "parent_task_id", "root_task_id", "session_id", "actor_id", "delegation_role",
            "chat_id", "title", "suggested_name", "context", "expected_output", "constraints",
            "task_contract", "workspace_root",
        )},
        "project_id": project_id,
        **{k: v for k, v in task.items() if k == "_project_admission"},
        "description": description,
        "allowed_resources": allowed_resources,
        "deadline_at": deadline_at,
        "workspace_mode": workspace_mode,
        "memory_mode": memory_mode,
        "child_drive_root": str(child_drive or ""),
        "budget_drive_root": str(drive_root) if child_drive is not None else "",
        "artifacts": artifacts,
        "artifact_status": ARTIFACT_STATUS_PENDING if workspace_root else "",
        "metadata": metadata,
        **{key: task[key] for key in ("attachment_manifest", "attachment_manifest_ref") if key in task},
        "api_admission": {"token": admission_token, "status": "accepted"},
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
        if admitted.get("_admission_uncertain"):
            return JSONResponse({"task_id": task_id, "status": "unconfirmed",
                                 "error": "Task admission persistence is unconfirmed; its resources were retained.",
                                 "detail": admitted["_admission_uncertain"]}, status_code=503)
        rejection = _admission_rejection_response(
            admitted,
            drive_root=drive_root,
            task_id=task_id,
            project_id=project_id,
            workspace_root=workspace_root,
            child_drive=child_drive,
        )
        if rejection is not None:
            return rejection
    except Exception as exc:
        # The queue may already have been observed or persisted. No broad
        # rollback can prove non-execution here, and unreadable is not refusal.
        log.warning("API task admission settlement is unconfirmed for %s", task_id, exc_info=True)
        return JSONResponse({"task_id": task_id, "status": "unconfirmed",
                             "error": f"Task admission settlement is unconfirmed: {exc}"}, status_code=503)
    _broadcast_task_named(task_id, str(task.get("suggested_name") or ""))
    return JSONResponse({
        "ok": True,
        "task_id": task_id,
        "status": STATUS_SCHEDULED,
        **{key: task[key] for key in ("attachment_manifest", "attachment_manifest_ref") if key in task},
    })


def _task_identity_occupied(drive_root: pathlib.Path, task_id: str) -> bool:
    """Whether a stored task-result row already owns *task_id*.

    ABI-2: identity collision is an AUTHORITY question, so the probe is the
    strict reader. The fail-soft default would QUARANTINE an inadmissible
    stored row as a side effect of this check and then report "no result",
    letting the endpoint reuse that row's task id; strict raises WITHOUT
    moving anything, and any stored row — admissible or not — keeps its
    identity occupied.
    """
    try:
        return load_task_result(drive_root, task_id, strict=True) is not None
    except ValueError:
        return True


def _broadcast_task_named(task_id: str, suggested_name: str) -> None:
    """Publish an admitted run's name so its card is never born nameless.

    The live card takes its title from ``suggested_name``; without this frame a
    project-homed run would paint as its status phrase until the first history
    replay. WS only — never a chat.jsonl row — and the client buffers a name that
    arrives before the card exists, so ordering does not matter. Fail-soft: a
    missing bridge (CLI/test process) simply means no live viewer.
    """
    if not suggested_name:
        return
    try:
        from supervisor.message_bus import try_get_bridge

        bridge = try_get_bridge()
        if bridge is not None:
            bridge.broadcast(
                {"type": "task_named", "task_id": task_id, "suggested_name": suggested_name}
            )
    except Exception:
        log.debug("task_named broadcast failed for %s", task_id, exc_info=True)


async def api_tasks_create(request: Request) -> JSONResponse:
    """POST /api/tasks — enqueue a managed headless task."""

    body = await request_json_or(request, {})
    return await run_sync_to_completion(_create_task_from_body, request, body)


def _api_executor_metadata(body: dict, raw_metadata: dict, workspace_root: Optional[pathlib.Path],
                           repo_dir: pathlib.Path, drive_root: pathlib.Path) -> dict:
    """Validate executor scope before reservation or preparation effects."""
    if "executor_ref" in raw_metadata or "workspace_executor" in raw_metadata:
        raise ValueError("metadata.executor_ref/workspace_executor is reserved; pass executor_ref as a top-level task field")
    if "executor_ref" not in body:
        return {}
    raw = body["executor_ref"]
    if not isinstance(raw, dict) or not raw:
        raise ValueError("executor_ref must be a JSON object")
    if workspace_root is None:
        raise ValueError("executor_ref requires an external workspace_root")
    executor = normalize_executor_ref(raw)
    if executor is None:
        return {}
    for mapping in executor.mappings:
        for protected_root, label in ((repo_dir, "Ouroboros system repo"), (drive_root, "Ouroboros data drive")):
            if paths_overlap_casefold(mapping.host_path, protected_root):
                raise ValueError(f"executor_ref mapping must not overlap the {label}")
    if not any(path_is_relative_to(workspace_root, mapping.host_path) for mapping in executor.mappings):
        raise ValueError("executor_ref mappings must cover workspace_root")
    return {"executor_ref": {
        "type": executor.kind, "id": executor.executor_id, "network": executor.network,
        "workspace_host_path": str(executor.mappings[0].host_path),
        "workspace_backend_path": executor.mappings[0].backend_path,
        "container_name": executor.container_name,
        "path_mappings": [{"host_path": str(mapping.host_path), "backend_path": mapping.backend_path}
                          for mapping in executor.mappings],
    }}


def _create_task_from_body(request: Request, body: Any) -> JSONResponse:
    """Settle the existing reservation, staging and durable admission as one unit."""
    if not isinstance(body, dict):
        return json_error("request body must be a JSON object", 400)
    if schema_errors := validate_ingress(body, TaskCreateRequest):  # executable gateway ABI (ABI-3, Q7=A): derived-schema ingress gate
        return json_error(f"invalid request body: {schema_errors[0]}", 400, schema_errors=schema_errors[:8])
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
        created_at = utc_now_iso() if not body.get("task_id") else ""
    except ValueError as exc:
        return json_error(str(exc), 400)
    if _task_identity_occupied(drive_root, task_id):
        return json_error(f"task_id already exists: {task_id}", 409)
    if (drive_root / HEADLESS_TASKS_DIR / task_id).exists() or (drive_root / ARTIFACTS_DIR / task_id).exists():
        return json_error(f"task_id already has headless state: {task_id}", 409)
    try:
        workspace_root = _resolve_workspace_root(
            body.get("workspace_root"),
            system_repo_dir=repo_dir,
            drive_root=drive_root,
        )
    except ValueError as exc:
        return json_error(str(exc), 400)
    workspace_mode = str(body.get("workspace_mode") or ("external" if workspace_root else "")).strip()
    memory_mode = str(body.get("memory_mode") or ("forked" if workspace_root else "shared")).strip().lower()
    if memory_mode not in {"forked", "empty", "shared"}:
        return json_error("memory_mode must be one of forked, empty, shared", 400)
    if workspace_root and memory_mode == "shared":
        return json_error("memory_mode=shared is not allowed for external workspaces; use forked or empty", 400)
    from ouroboros.project_facts import explicit_project_id_ok
    from ouroboros.projects_registry import project_scope_admission

    raw_project_id = str(body.get("project_id") or "")
    # Reject unsanitized ids: normalization would alias different memory stores.
    if raw_project_id and not explicit_project_id_ok(raw_project_id):
        return json_error("project_id must be filesystem-safe (alphanumeric/_/-/., no spaces or slashes)", 400)
    try:
        project_basis = project_scope_admission(
            drive_root, project_id=raw_project_id, workspace_root=str(workspace_root or ""))
    except (OSError, ValueError, RuntimeError) as exc:
        return json_error(f"Project state could not be checked: {exc}", 409,
                          reason_code="project_routing_fence_lookup_failed")
    _task_project_id = project_basis["project_id"]
    # Keep requested memory_mode; project-scoped shared memory runs on a forked
    # child data root, so post-task writes stay isolated even without a workspace.
    effective_drive_mode = "forked" if (_task_project_id and memory_mode == "shared") else memory_mode
    task_type = str(body.get("type") or "task")
    if task_type in {"evolution", "review", "deep_self_review"}:
        return json_error(
            f"task type {task_type!r} is internal-only and cannot be created via the task API "
            "(use /evolve or /review); evolution additionally requires advanced/pro/cyber_pro runtime mode",
            400,
        )
    if workspace_root and task_type != "task":
        return json_error("external workspace tasks must use type='task'", 400)
    try:
        chat_id = ingress_chat_id(body.get("chat_id"), drive_root, _task_project_id,
                                  source=body.get("source"), project_basis=project_basis)
        depth = parse_task_depth(body.get("depth"), default=0)
    except ProjectThreadConflict as exc:
        return json_error(str(exc), 400)
    except (TypeError, ValueError) as exc:
        return json_error(
            "depth must be a non-negative integer"
            if str(getattr(exc, "code", "")) == "negative_task_depth"
            else "chat_id and depth must be integers",
            400,
        )

    raw_metadata = dict(body.get("metadata") or {}) if isinstance(body.get("metadata"), dict) else {}
    if "input_sources" in raw_metadata:
        return json_error(
            "metadata.input_sources is reserved; source selection is only supported by schedule_subagent",
            400, reason_code="input_source_selection_unsupported")
    if _external_subagent_label(body, raw_metadata):
        return json_error("delegation_role=subagent is only allowed through the internal schedule_subagent tool", 400)
    if str(body.get("parent_task_id") or "").strip() or str(body.get("root_task_id") or "").strip():
        return json_error("parent_task_id and root_task_id are internal lineage fields; external tasks must start as roots", 400)
    for _top_level_only in ("project_id", "title"):
        # Top-level fields; silently dropping either from metadata would let a
        # caller believe isolation is active, or a name was accepted, when it was not.
        if _top_level_only in raw_metadata:
            return json_error(f"{_top_level_only} must be a top-level field, not metadata", 400)
    metadata = {str(k): v for k, v in raw_metadata.items() if str(k) not in _RESERVED_METADATA_KEYS}
    allowed_resources, resource_policy, disabled_tools, acceptance_claims, policy_error = (
        _fold_contract_policies(body, raw_metadata, metadata)
    )
    if policy_error:
        return json_error(policy_error, 400)
    try:
        metadata.update(_api_executor_metadata(body, raw_metadata, workspace_root, repo_dir, drive_root))
    except ValueError as exc:
        return json_error(str(exc), 400)
    try:
        deadline_at = _normalize_deadline_at(body.get("deadline_at") or raw_metadata.get("deadline_at") or "")
    except ValueError as exc:
        return json_error(str(exc), 400)
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
    # v6.52.0 (P1): stage attachments into the SAME drive the task will read from at
    # runtime — the child drive when forked/empty, else the shared drive (matches the
    # task['drive_root'] set at the end of this handler). The returned manifest renders
    # READY read_file(root='artifact_store', ...) lines and feeds native image blocks.
    effective_drive = child_drive or drive_root
    try:
        attachment_manifest, attachment_error = stage_initial_task_attachments(
            effective_drive, task_id, _normalize_attachments(body.get("attachments")),
            # Partial staging is the DEFAULT (В25c); explicit false = atomic admission.
            allow_partial=body.get("allow_partial_attachments") is not False,
        )
    except Exception as exc:
        _cleanup_api_admission_attempt(drive_root, task_id, admission_token, child_drive)
        return json_exception(exc, 503)
    if attachment_error is not None:
        _cleanup_api_admission_attempt(drive_root, task_id, admission_token, child_drive)
        return attachment_error
    from ouroboros.artifacts import attachment_manifest_projection
    metadata.setdefault("session_id", str(body.get("session_id") or uuid.uuid4().hex))
    metadata.setdefault("actor_id", str(body.get("actor_id") or "cli"))
    metadata.setdefault("source", str(body.get("source") or "api_task"))
    # Owner Surface Fact: assembled at its PRODUCER. An external admission
    # carries no browser observables, and a caller-built descriptor must not
    # smuggle past the closed-key web normalizer (a fake received_at would
    # impersonate a host stamp) — the caller-declared channel IS the fact.
    metadata["client_surface"] = {"channel": str(metadata.get("source") or "api_task")}
    metadata.setdefault("delegation_role", "root")
    metadata.setdefault("task_id", task_id)
    metadata.setdefault("parent_task_id", "")
    metadata.setdefault("root_task_id", task_id)
    metadata["resource_intent"] = ({"kind": "explicit_resource", "root": str(workspace_root)} if workspace_root  # #1315
                                   else {"kind": "explicit_none", "project_id": _task_project_id} if _task_project_id else {"kind": "system_repo"})
    artifacts: List[Dict[str, Any]] = []
    workspace_preflight_summary: Dict[str, Any] = {}
    if workspace_root:
        metadata["workspace_root"] = str(workspace_root)
        try:
            preflight = collect_workspace_preflight(workspace_root)
            workspace_preflight_summary = summarize_workspace_preflight(preflight)
            metadata["workspace_preflight"] = workspace_preflight_summary
            artifacts.append(write_workspace_preflight_artifact(drive_root, task_id, preflight))
        except Exception as exc:
            workspace_preflight_summary = {
                "schema_version": 1,
                "workspace_root": str(workspace_root),
                "error": f"{type(exc).__name__}: {exc}",
            }
            metadata["workspace_preflight"] = workspace_preflight_summary

    try:
        attachment_authority = attachment_manifest_projection(effective_drive, task_id, attachment_manifest)
        task_text = _compose_task_text(
            description,
            workspace_root=workspace_root,
            workspace_mode=workspace_mode,
            memory_mode=memory_mode,
            workspace_preflight=workspace_preflight_summary,
            attachments=attachment_authority,
        )
    except Exception as exc:
        _cleanup_api_admission_attempt(
            drive_root, task_id, admission_token, child_drive
        )
        return json_exception(exc, 503)
    _title, _suggested_name = admission_names(body, description)
    task = {
        "id": task_id,
        **({"created_at": created_at} if created_at else {}),
        "type": task_type,
        "chat_id": chat_id,
        "title": _title, "suggested_name": _suggested_name,
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
        "parent_task_id": None,
        "root_task_id": task_id,
        "session_id": metadata["session_id"],
        "actor_id": metadata["actor_id"],
        "delegation_role": metadata["delegation_role"],
        "workspace_root": str(workspace_root) if workspace_root else "",
        "workspace_mode": workspace_mode,
        "memory_mode": memory_mode,
        "project_id": _task_project_id,
        **({"_project_admission": project_basis} if _task_project_id else {}),
        "metadata": metadata,
        # v6.52.0 (P1): the STAGED manifest (root/relpath/mime/is_image), not raw
        # host paths — relpaths resolve against task['drive_root'] at read time.
        "attachments": attachment_authority["attachment_manifest"],
        **attachment_authority,
        "attachment_images": [m for m in attachment_manifest
                              if str(m.get("status") or "staged") == "staged" and m.get("is_image")],
        # v6.52.0 (P1): record the effective drive (child when forked/empty, else the shared
        # drive) so build_user_content can resolve staged attachment IMAGES for EVERY task
        # shape — not just child-drive tasks. The child-drive block below re-affirms it.
        "drive_root": str(effective_drive),
        "_require_unique_task_id": True,
        "_require_worker_pool": True,
        "_admission_token": admission_token,
    }
    try:
        task = attach_task_contract(task)
    except Exception as exc:
        _cleanup_api_admission_attempt(drive_root, task_id, admission_token, child_drive)
        return json_exception(exc, 503)
    if child_drive is not None:
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

# The raw creation-ts sort scan and the ABI-2 malformed-candidate admission
# live in ouroboros/task_result_facts.py (module-size split); imported
# here so this module keeps the endpoint wiring surface.
from ouroboros.task_result_facts import (  # noqa: E402
    _quarantine_malformed_candidates,
    _raw_sorted_result_names,
)


def _compact_list_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Compact LIST projection: drop the five bulk evidence fields, keep the
    summary contract (task_id, status, ts/updated_at, result, description/
    objective/title/role, lineage, project_id, reason_code, artifact_status,
    workspace fields, the TASK_COST_META_FIELDS, outcome_axes — all preserved
    because the projection is subtractive, never a whitelist)."""
    return {key: value for key, value in row.items() if key not in _LIST_ROW_OMITTED_FIELDS}


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
    child-drive promotion contract pinned by test_headless_cli).

    Admission at the slice boundary (ABI-2): a MALFORMED candidate discovered
    by the sort scan reaches the admission reader even when it lies beyond
    the slice window (the scan had to read its bytes anyway), so it is
    quarantined and counted in the same ONE batched scan event. Disclosed
    residual: a PARSEABLE but inadmissible row (unstamped/future stamp)
    beyond the window is not classified by this sliced request — its raw ts
    is all the sort reads — and is quarantined by the next scan that
    actually reaches it (a filtered request, ``limit=0``, or any
    list_task_results caller)."""
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
    names, malformed_names = _raw_sorted_result_names(results_dir)
    if limit is not None:
        names = names[:limit]
    rows = []
    quarantined: List[Dict[str, str]] = []
    for name in names:
        path = results_dir / name
        raw = read_json_dict(path)
        if raw is None and not path.is_file():
            continue  # vanished/torn between the scandir and this read
        # ABI-2: the sliced fast path is admission-aware like every other
        # reader — an inadmissible row is quarantined and never projected,
        # with ONE batched durable event for the whole scan (6.3=B), the
        # same semantics as the list_task_results fail-soft scan.
        refusal = task_result_schema_refusal(raw)
        if refusal:
            outcome = quarantine_task_result(path, refusal)
            if outcome == "kept_admissible":
                raw = read_json_dict(path)
                if raw is None or task_result_schema_refusal(raw):
                    continue
            else:
                if outcome == "moved":
                    quarantined.append({"task_id": path.stem, "reason": refusal})
                continue
        rows.append(_compact_list_row(public_task_result(effective_task_result(
            drive_root, raw, materialize_artifacts=False, _events_index=events_index,
        ))))
    # ABI-2: a candidate whose bytes failed to parse is NOT silently dropped —
    # it reaches the same admission reader (quarantine + the batched event)
    # even beyond the slice window (see task_result_facts).
    quarantined.extend(_quarantine_malformed_candidates(results_dir, malformed_names))
    emit_quarantine_event(drive_root, quarantined)
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


async def api_task_get(request: Request) -> JSONResponse:
    return await run_sync_to_completion(_task_get_response, request)


def _task_get_response(request: Request) -> JSONResponse:
    """Project the complete detail and ledger view off the HTTP loop (a pure read)."""
    try:
        task_id = validate_task_id(request.path_params.get("task_id"))
    except ValueError as exc:
        return json_error(str(exc), 400)
    drive_root = request_drive_root(request)
    data = load_effective_task_result(drive_root, task_id)
    if not data:
        try:
            (task_results_dir(drive_root, create=False) / f"{task_id}.json").stat()
        except FileNotFoundError:
            return json_error("task not found", 404)
        except OSError:
            pass
        return json_error("task result is unavailable", 503)
    payload = public_task_result(data)
    from ouroboros.owner_continue import continuation_offer  # Batch4: a Continue, or its accepted successor
    payload["continuation_offer"] = continuation_offer(data, task_id)
    if isinstance(payload.get("artifacts"), list):  # what ``?archive=<dir>`` would stream now, per top-level dir
        payload["artifact_archives"] = directory_archives(task_artifact_stores(drive_root, task_id),
                                                          payload["artifacts"], anchor=drive_root)
    breakdown_view = _task_cost_breakdown_view(drive_root, data)
    if breakdown_view is not None:
        payload["cost_breakdown"] = breakdown_view
    return JSONResponse(payload)


def api_task_artifact(request: Request):
    """Serve one task file read-only from its canonical or OWN child store (``task_archive``): a bare
    name selects only a top-level file (several nested matches: 409 ``artifact_name_ambiguous`` naming
    their ``relpaths``), ``?relpath=a/b/{name}`` is exact, ``?archive=<dir>`` with ``{name}`` =
    ``<basename>.zip`` streams a recorded directory; bytes leave only through ``serve_task_file``."""
    try:
        task_id = validate_task_id(request.path_params.get("task_id"))
    except ValueError as exc:
        return json_error(str(exc), 400)
    name = str(request.path_params.get("name") or "").strip()
    if not name or "/" in name or "\\" in name or name in {".", ".."} or ".." in pathlib.PurePosixPath(name).parts:
        return json_error("artifact name must be a simple filename", 400)
    source, relpath = request.query_params.get("source"), request.query_params.get("relpath")
    if relpath is not None and (source or plain_segments(relpath)[-1:] != [name]):
        return json_error("relpath must be store-relative plain segments ending in the name, without source",
                          400, reason_code="artifact_relpath_invalid", task_id=task_id, artifact=name)
    drive_root = request_drive_root(request)
    if (archive := request.query_params.get("archive")) is not None:
        return serve_directory_archive(drive_root, task_id, name, archive,
                                       other_selectors=source is not None or relpath is not None)
    stores = task_artifact_stores(drive_root, task_id)
    path = None if relpath is not None else artifact_store.resolve_chat_media_path(drive_root, task_id, name)
    if path is not None:  # content-addressed chat media: the canonical store, bytes that hash to the name
        media = task_artifact_location(stores[:1], path)
        return (serve_task_file(drive_root, stores[0], media[2], name, chat_media_identity(name), task_id=task_id)
                if media else json_error("artifact not found", 404, task_id=task_id, artifact=name))
    registered = artifact_store.registered_task_artifact(drive_root, task_id, name) if relpath in (None, name) else None
    # A registered immutable top-level file needs one identity check, not a result projection.
    fast = bool(not source and registered and registered.get("immutable")
                and (at := task_artifact_location(stores, registered.get("path"))) and at[2] == name)
    result = {} if fast else (load_effective_task_result(drive_root, task_id) or {})
    if not result and not registered:
        return json_error("task not found", 404)
    if source:
        return serve_task_source(drive_root, stores, result, task_id, name, source)
    rows = [] if fast else [row for row in result.get("artifacts") or [] if isinstance(row, dict) and (
        relpath is not None or str(row.get("name") or pathlib.Path(str(row.get("path") or "")).name) == name)]
    located = [(at, order, row) for order, row in enumerate(rows + ([registered] if registered else []))
               if (at := task_artifact_location(stores, row.get("path")))]
    matches = [item for item in located if item[0][2] == (relpath or name)]
    nested = sorted({item[0][2] for item in located if "/" in item[0][2]})
    if relpath is None and not matches and len(nested) > 1:
        return json_error("artifact name matches several nested files; select one with ?relpath=", 409,
                          reason_code="artifact_name_ambiguous", task_id=task_id, artifact=name, relpaths=nested)
    if not matches:
        if relpath is None and any(at[2].rsplit("/", 1)[-1] != name for at, _order, _row in located):
            return json_error("artifact metadata path does not match requested name", 500)
        if (rows or registered) and not located and relpath is None:
            return json_error("artifact path is outside task artifact directory", 500)
        return json_error("artifact not found", 404, task_id=task_id, artifact=name)
    (index, _path, relative), _order, artifact = min(matches, key=lambda item: (item[0][0], item[1]))
    return serve_task_file(drive_root, stores[index], relative, name, recorded_identity(artifact), task_id=task_id,
                           mutable=not artifact.get("immutable"))


def _record_cascade_incident(task_id: str, kind: str, detail: str = "") -> None:
    """Durably record a cascade outcome the owner must be able to see.

    The client already holds ``ok:true`` and the card only resolves on a real
    ``task_done``, so a cascade that RAISED or that cancelled NOTHING is a silent
    lie unless it is recorded. Both land on the supervisor's own drive root (the
    same log every other cancel artifact uses) and are pushed to the live owner
    surfaces when a bridge exists.
    """
    # ONE event object for the durable row and the live frame: a second
    # timestamp would defeat the Logs panel's backfill/live dedupe key.
    incident = {"type": kind, "task_id": task_id}
    if detail:
        incident["error"] = detail
    try:
        from ouroboros.utils import utc_now_iso

        incident = {"ts": utc_now_iso(), **incident}
        from supervisor import queue as supervisor_queue
        from supervisor.log_addressing import address_handler_push

        incident = address_handler_push(pathlib.Path(supervisor_queue.DRIVE_ROOT), incident)
    except Exception:
        log.debug("Failed to address cascade cancel incident for %s", task_id, exc_info=True)
    try:
        from supervisor import queue as supervisor_queue
        from ouroboros.utils import append_jsonl

        append_jsonl(
            pathlib.Path(supervisor_queue.DRIVE_ROOT) / "logs" / "supervisor.jsonl",
            incident,
        )
    except Exception:
        log.debug("Failed to persist cascade cancel incident for %s", task_id, exc_info=True)
    try:
        from supervisor.message_bus import try_get_bridge

        bridge = try_get_bridge()
        if bridge is not None:
            bridge.push_log(incident)
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
        from supervisor.queue import CANCEL_CANCELLED, drive_cancel_intent_scope

        if drive_cancel_intent_scope(task_id) != CANCEL_CANCELLED:
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


async def _graceful_stop_acknowledgement(task_id: str, *, cascade: bool, stop_action_id: str = "") -> JSONResponse:
    """S3 graceful ingress: durable finalize intent + IMMEDIATE pending ack.

    The socket is NOT held for the (up to) 120-second episode (§12.2 item 2):
    the durable intent is the whole owner will, one orchestration pass is
    kicked off a background thread (the ~20s intent sweep is the crash-safe
    watchdog replay), and the caller gets the typed pending acknowledgement.
    Stop-now stays available throughout and HARDENS the same intent.
    """
    import threading
    from supervisor.followup_policy import StopActionConflict

    from supervisor.queue import (
        DRIVE_ROOT as _drive_root,
        task_has_live_ownership as _live_ownership,
        task_subtree_is_live as _live_check,
    )

    from ouroboros.cancel_intents import (
        CancelIntentProjectionCorrupt,
        SCOPE_CASCADE,
        STOP_POLICY_FINALIZE,
        request_cancel,
        stop_policy,
    )

    live_own = await asyncio.to_thread(_live_ownership, task_id)
    if not live_own and not await asyncio.to_thread(_live_check, task_id):
        return json_error("task not found or not active", 404, task_id=task_id)
    observation = await asyncio.to_thread(observe_cancellation_target, _drive_root, task_id, request_origin={"kind": "http_client", "source": "http_graceful"})
    try:
        intent = await asyncio.to_thread(functools.partial(
            request_cancel, _drive_root, task_id,
            reason="owner requested finalize-then-stop",
            source="http_graceful", requested_by="owner",
            observation=observation,
            stop_action_id=stop_action_id,
            requested_stop_policy=STOP_POLICY_FINALIZE,
            allow_settled_target=bool(cascade or live_own),
            **({"scope": SCOPE_CASCADE} if cascade else {}),
        ))
    except StopActionConflict as exc:
        return json_error(str(exc), 409, task_id=task_id, reason_code="stop_action_conflict")
    except CancelIntentProjectionCorrupt:
        return json_error(
            "the cancel-intent projection is corrupt; nothing was requested",
            503, task_id=task_id, reason_code="cancel_intent_projection_corrupt",
        )
    except Exception:
        return json_error(
            "durable stop intent could not be recorded; nothing was requested — retry",
            503, task_id=task_id, reason_code="cancel_intent_write_failed",
        )
    if intent.get("already_settled"):
        return json_error("task not found or not active", 404, task_id=task_id)
    try:
        from supervisor.owner_stop import begin_graceful_stop

        physical_task_id = str(intent.get("task_id") or task_id)
        threading.Thread(
            target=begin_graceful_stop, args=(physical_task_id,),
            name=f"owner-stop-{physical_task_id[:8]}", daemon=True,
        ).start()
    except Exception:
        log.debug("owner-stop ingress thread failed for %s", task_id, exc_info=True)
    return JSONResponse(
        {
            "ok": True,
            "task_id": task_id,
            "cancel_state": "pending",
            # The EFFECTIVE policy: a graceful request over an already-hard
            # intent never softens it, and the answer says so.
            "stop_policy": stop_policy(intent),
            **({"cascade": True} if cascade else {}),
        },
        status_code=202,
    )


async def api_task_cancel(request: Request) -> JSONResponse:
    try:
        task_id = validate_task_id(request.path_params.get("task_id"))
    except ValueError as exc:
        return json_error(str(exc), 400)
    # Optional cascade cancels the snapshotted live subtree before hard reply.
    # Absent/empty body keeps legacy single-task behavior. Present malformed or
    # non-object JSON must refuse, never silently narrow a cascade to its root.
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
    # S3 (Q1): the OPTIONAL terminalization policy — an INDEPENDENT axis from
    # cascade (§13.1). Absent/empty stays today's immediate hard cancellation,
    # byte-identical for every existing caller (benchmarks post empty bodies).
    raw_policy = body.get("stop_policy")
    if raw_policy is not None and not isinstance(raw_policy, str):
        return json_error("stop_policy must be a string", 400, task_id=task_id)
    stop_policy_value = str(raw_policy or "").strip()
    if stop_policy_value not in {"", "immediate", "finalize_then_cancel"}:
        return json_error(
            "stop_policy must be 'immediate' or 'finalize_then_cancel'",
            400, task_id=task_id,
        )
    action_id = body.get("stop_action_id", "")
    if not isinstance(action_id, str) or len(action_id) > 200:
        return json_error("stop_action_id must be a string of at most 200 characters", 400, task_id=task_id)
    if stop_policy_value == "finalize_then_cancel":
        # Graceful ingress: immediate typed pending acknowledgement; the
        # synchronous teardown contract below stays hard/legacy-only.
        return await _graceful_stop_acknowledgement(task_id, cascade=cascade, stop_action_id=action_id)

    intent_target = {"task_id": task_id, "scope": ""}

    def _record_http_intent(
        source: str, *, cascade_scope: bool = False, allow_settled: bool = False,
    ) -> bool:
        """Persist the watchdog fence before teardown; failure refuses ingress.

        Cascade ingress records widen-only scope and allows settled roots with
        live descendants. Single ingress supplies allow_settled from live physical
        ownership, since stored completion can precede worker exit. Return an
        empty string on success or a typed write/corruption/action-conflict refusal.
        """
        try:
            from supervisor.followup_policy import StopActionConflict
            from supervisor.queue import DRIVE_ROOT as _drive_root

            from ouroboros.cancel_intents import (
                CancelIntentProjectionCorrupt,
                SCOPE_CASCADE,
                STOP_POLICY_IMMEDIATE,
                request_cancel,
            )
        except Exception:
            log.warning("HTTP cancel-intent machinery unavailable for %s", task_id,
                        exc_info=True)
            return "write_failed"
        try:
            intent = request_cancel(
                _drive_root, task_id, source=source,
                stop_action_id=action_id,
                observation=observe_cancellation_target(_drive_root, task_id, request_origin={"kind": "http_client", "source": source}),
                **({"scope": SCOPE_CASCADE} if cascade_scope else {}),
                allow_settled_target=bool(cascade_scope or allow_settled),
                # §13.1: an omitted/empty-body or explicit-immediate request IS
                # the immediate policy — it monotonically HARDENS a pending
                # graceful intent (Stop-now during the wait) and mints a
                # byte-identical legacy row when no intent exists.
                requested_stop_policy=STOP_POLICY_IMMEDIATE,
            )
            intent_target["task_id"] = str(intent.get("task_id") or task_id)
            intent_target["scope"] = str(intent.get("scope") or "")
            return ""
        except StopActionConflict:
            return "stop_action_conflict"
        except CancelIntentProjectionCorrupt:
            log.error("HTTP cancel refused for %s: intent projection corrupt", task_id)
            return "projection_corrupt"
        except Exception:
            log.warning("HTTP cancel-intent write failed for %s; cancel refused",
                        task_id, exc_info=True)
            return "write_failed"

    def _intent_write_refused(kind: str) -> JSONResponse:
        if kind == "stop_action_conflict":
            return json_error("stop_action_id reused with a different action", 409,
                              task_id=task_id, reason_code=kind)
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
            if intent_target["scope"] == "cascade":
                # Scope is widen-only durable authority.  Stop-now over an
                # existing graceful cascade must execute that cascade now even
                # when the new HTTP body omitted ``cascade``; replaying the raw
                # single shape can kill only the root or fail on a retry leaf.
                settled = await asyncio.to_thread(
                    _run_cascade_cancel, intent_target["task_id"],
                )
                if not settled:
                    return json_error(
                        "subtree cancellation did not settle; the tree is still live",
                        503,
                        task_id=task_id,
                    )
                return JSONResponse({
                    "ok": True,
                    "task_id": task_id,
                    "cascade": True,
                })
            # The TYPED outcome, not a boolean: a task whose worker refused to die
            # is neither cancelled nor absent, and answering 404 for it would tell
            # the caller the task is gone while it keeps running.
            outcome = await asyncio.to_thread(
                cancel_task_custody, intent_target["task_id"],
            )
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
    """Explicit owner Resume of a budget-paused task.

    A replay-safe zero-dispatch row is released; an exact mid-run continuation
    (#1196) receives ONE single-use grant and continues under the same task id.
    Every refusal is typed: money still exhausted, a live cancel intent, a
    passed deadline, an exhausted finite lifetime, a root that is itself still
    paused, or a missing/unreadable checkpoint all leave the task paused. A row
    HELD beside its pause (an unrestorable source at restart, an unwritten
    revocation, an acceptance fence at restore) is granted by the same call once
    its durable authority validates again; a fence-lifted zero-dispatch sibling
    is released by this same call as an explicit selection. A paused direct
    owner-chat turn is resumed here too, under its own task id.
    """
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
        # exact-continuation refusals (#1196): the task stays paused
        "budget_still_exhausted", "root_hard_cap_exhausted", "cancel_intent_active",
        "deadline_passed", "lifetime_exhausted", "root_still_paused", "resume_already_granted",
        "restart_no_resume", "pause_record_missing", "pause_source_unreadable",
        "pause_record_unreadable", "grant_not_recorded", "snapshot_not_persisted",
        "monetary_authority_unavailable", "cancellation_authority_unavailable", "task_terminal",
        # holds and root-grant refusals (#1196, owner Q9): the row stays paused/held
        "root_resume_grant_missing", "root_resume_generation_stale", "root_replay_unsafe",
        "root_accounting_unavailable", "root_accounting_degraded", "external_custody_unreadable",
        "accounting_unavailable", "resume_grant_revocation_unwritten",
        # fresh custody at grant (#1196, owner Q8): a delegated run not proven
        # terminal keeps the task paused; a marker/attempt drift is typed too
        "external_runs_unsettled", "pause_attempt_mismatch",
        "owner_pause_effects_unsettled", "owner_pause_custody_unreadable", "selection_authority_changed",
    } else 404
    return json_error(error, status, task_id=task_id, **({"action": result["action"]} if result.get("action") else {}))


# The task-event SSE endpoint and its follower live in gateway/task_events.py
# (split by the 1600-line module gate). Re-exported at the top of this module
# so route wiring, the CLI, and monkeypatch pins keep addressing gateway.tasks.


def _resolve_workspace_root(
    value: Any,
    *,
    system_repo_dir: pathlib.Path,
    drive_root: pathlib.Path,
) -> Optional[pathlib.Path]:
    """Delegates to the admission SSOT (v6.58.0): the gateway and the promote path
    validate a workspace root through ONE function (workspace_admission), so the two
    surfaces can never drift. WorkspaceRootError subclasses ValueError, so existing
    `except ValueError` call sites keep working unchanged."""
    from ouroboros.workspace_admission import validate_workspace_root

    return validate_workspace_root(value, system_repo_dir=system_repo_dir, drive_root=drive_root)


def _normalize_attachments(value: Any) -> List[Dict[str, str]]:
    if not value:
        return []
    if not isinstance(value, list):
        return []
    out: List[Dict[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            path = str(item.get("path") or "").strip()
            label = str(
                item.get("label") or item.get("display_name") or pathlib.Path(path).name
            ).strip()
        else:
            path = str(item or "").strip()
            label = pathlib.Path(path).name
        # Preserve one row per declaration, including an empty/invalid path.
        # ``stage_task_attachments`` owns the typed rejection reason.
        out.append({"path": path, "label": label})
    return out


def _compose_task_text(
    description: str,
    *,
    workspace_root: Optional[pathlib.Path],
    workspace_mode: str,
    memory_mode: str,
    workspace_preflight: Dict[str, Any],
    attachments: Any,
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
    if rendered:
        parts.append(f"\n\n[ATTACHMENTS]\n{rendered}\n[END_ATTACHMENTS]")
    return "".join(parts)


def _render_attachment_lines(attachments: Any) -> str:
    """Render the complete staged/rejected attachment report.

    v6.52.0 (P1): each line is a ready-to-use read_file call against the canonical
    artifact_store root — NEVER a bare absolute host path. ``attachments`` is the
    manifest returned by ``stage_task_attachments``.  Legacy staged-only rows
    remain readable; new rows carry ordinal/status/reason and rejected rows are
    rendered without source paths or secret contents."""
    ref = attachments.get("attachment_manifest_ref") if isinstance(attachments, dict) else None
    if isinstance(attachments, dict):
        attachments = attachments.get("attachment_manifest")
    if not isinstance(attachments, list):
        return ""
    lines: List[str] = []
    if ref:
        lines.append(f"Complete input manifest ({ref.get('count')} declarations): read_file(root='artifact_store', path='{ref.get('path')}'). Inline rows below are a preview.")
    for item in attachments:
        if not isinstance(item, dict):
            continue
        try:
            ordinal = int(item.get("ordinal"))
        except (TypeError, ValueError):
            ordinal = len(lines)
        status = str(item.get("status") or "staged")
        label = str(item.get("label") or f"attachment {ordinal + 1}").strip()
        if status == "rejected":
            reason = str(item.get("reason") or "staging_failed").strip()
            lines.append(f"- {label}: rejected (reason={reason}, ordinal={ordinal})" + (f" rule: {rule}" if (rule := str(item.get("rule") or "").strip()) else ""))
            continue
        relpath = str(item.get("relpath") or "").strip()
        root = str(item.get("root") or "artifact_store").strip() or "artifact_store"
        label = label or pathlib.Path(relpath).name
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
            f"- {label} ({kind}): read_file(root='{root}', path='{relpath}')"
            f"{script_hint} [status=staged, ordinal={ordinal}]"
        )
    return "\n".join(lines)


def _queue_snapshot(drive_root: pathlib.Path) -> Dict[str, Any]:
    path = pathlib.Path(drive_root) / "state" / "queue_snapshot.json"
    try:
        from ouroboros.project_admission import project_hold_fact
        snapshot = read_json_dict(path) or {}
        for row in [*snapshot.get("pending", []), *snapshot.get("running", [])]:
            task = row.get("task")
            if isinstance(task, dict):
                task["project_admission_hold"] = project_hold_fact(task)
        return snapshot
    except Exception:
        return {}


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
    "api_decision_answer",
    "api_task_hurry", "api_task_pause", "owner_tree_control_routes",
    "api_task_resume",
    "api_task_events",
    "api_task_get",
    "api_tasks_create",
    "api_tasks_list",
    "iter_task_events",
]
