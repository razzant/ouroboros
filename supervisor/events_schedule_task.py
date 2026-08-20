"""The schedule_task admission gates, the duplicate gate, and its refusals.

One owner for the facts the dispatch parent's schedule handler needs: the
chat-target gate, the semantic duplicate gate, the composed queue payload,
and every refusal path including worktree cleanup for a rejected subagent.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, Optional
from ouroboros.config import (
    MAX_ACTIVE_SUBAGENTS_HARD_CAP,
    get_max_active_subagents_per_root,
    get_max_subagent_depth,
)
from ouroboros.tool_capabilities import ACTING_SUBAGENT_MODE
from ouroboros.task_results import (
    STATUS_FAILED,
    STATUS_REJECTED_DUPLICATE,
    STATUS_SCHEDULED,
    write_task_result,
)
from ouroboros.subagents import intended_lane as intended_subagent_lane
from ouroboros.contracts.task_contract import build_task_contract, normalize_allowed_resources
from supervisor.events_chat_delivery import _bound_project_chat_id
from supervisor.events_subagent_admission import (
    _active_subagent_count,
    _compose_subagent_text,
    _record_delegation_constraint,
    _resolve_subagent_constraint,
    _send_subagent_rejection,
    _subagent_cap_blocks,
    _subagent_scheduled_meta,
)

log = logging.getLogger(__name__)


_PARENT_CONTEXT_MARKER = "[BEGIN_PARENT_CONTEXT"


_PARENT_CONTEXT_END = "[END_PARENT_CONTEXT]"


VALID_SUBAGENT_MEMORY_MODES = frozenset({"forked", "empty"})


def _build_scheduled_task_payload(fields: Dict[str, Any]) -> Dict[str, Any]:
    tid = str(fields.get("tid") or "")
    chat_id = int(fields.get("chat_id") or 0)
    text = str(fields.get("text") or "")
    desc = str(fields.get("desc") or "")
    expected_output = str(fields.get("expected_output") or "")
    constraints = str(fields.get("constraints") or "")
    role = str(fields.get("role") or "")
    task_context = str(fields.get("task_context") or "")
    depth = int(fields.get("depth") or 0)
    root_task_id = str(fields.get("root_task_id") or "")
    session_id = str(fields.get("session_id") or "")
    actor_id = str(fields.get("actor_id") or "")
    delegation_role = str(fields.get("delegation_role") or "")
    memory_mode = str(fields.get("memory_mode") or "")
    drive_root = str(fields.get("drive_root") or "")
    child_drive_root = str(fields.get("child_drive_root") or "")
    budget_drive_root = str(fields.get("budget_drive_root") or "")
    task_constraint = fields.get("task_constraint") if isinstance(fields.get("task_constraint"), dict) else None
    required_capabilities = fields.get("required_capabilities") if isinstance(fields.get("required_capabilities"), list) else []
    workspace_root = str(fields.get("workspace_root") or "")
    workspace_mode = str(fields.get("workspace_mode") or "")
    project_id = str(fields.get("project_id") or "")
    allowed_resources = fields.get("allowed_resources") if isinstance(fields.get("allowed_resources"), dict) else {}
    task_contract = fields.get("task_contract") if isinstance(fields.get("task_contract"), dict) else {}
    parent_id = fields.get("parent_id")
    # INTENT ONLY. `effective_model_lane`, `model`, `use_local_model`,
    # `effective_executor`, `reasoning_effort` and `capability_delta` are DERIVED at
    # dispatch and written by the worker onto the one record; carrying schedule-time
    # values for them through here is what made two records of the same child.
    requested_model_lane = str(fields.get("requested_model_lane") or fields.get("model_lane") or "auto")
    parent_model_lane = str(fields.get("parent_model_lane") or "")
    # An ADMISSION fact, not a derivation (F9): the lane an applicable
    # non-advisory `require_lane` constraint verified this child against.
    required_model_lane = str(fields.get("required_model_lane") or "")
    requested_executor = str(fields.get("requested_executor") or "").strip().lower() or "auto"
    task_group_id = str(fields.get("task_group_id") or "")
    task_group = fields.get("task_group") if isinstance(fields.get("task_group"), dict) else {}
    subagent_envelope = fields.get("subagent_envelope") if isinstance(fields.get("subagent_envelope"), dict) else {}
    task: Dict[str, Any] = {
        "id": tid,
        "type": "task",
        "chat_id": chat_id,
        "text": text,
        "description": desc,
        "objective": desc,
        "expected_output": expected_output,
        "constraints": constraints,
        "role": role,
        "context": task_context,
        "depth": depth,
        "root_task_id": root_task_id,
        "session_id": session_id,
        "actor_id": actor_id,
        "delegation_role": delegation_role,
        "memory_mode": memory_mode,
        "drive_root": drive_root,
        "child_drive_root": child_drive_root,
        "budget_drive_root": budget_drive_root,
        "task_constraint": task_constraint,
        "required_capabilities": required_capabilities,
        "workspace_root": workspace_root,
        "workspace_mode": workspace_mode,
        "project_id": project_id,
        "allowed_resources": allowed_resources,
        "task_contract": task_contract,
        "model_lane": requested_model_lane,
        "requested_model_lane": requested_model_lane,
        "parent_model_lane": parent_model_lane,
        "required_model_lane": required_model_lane,
        "requested_executor": requested_executor,
        "task_group_id": task_group_id,
        "task_group": task_group,
        "subagent_envelope": subagent_envelope,
        "metadata": {
            "parent_task_id": parent_id,
            "root_task_id": root_task_id,
            "session_id": session_id,
            "actor_id": actor_id,
            "delegation_role": delegation_role,
            "role": role,
            "memory_mode": memory_mode,
            "task_constraint": task_constraint,
            "required_capabilities": required_capabilities,
            "child_drive_root": child_drive_root,
            "workspace_root": workspace_root,
            "workspace_mode": workspace_mode,
            "allowed_resources": allowed_resources,
            "task_contract": task_contract,
            "model_lane": requested_model_lane,
            "requested_model_lane": requested_model_lane,
            "parent_model_lane": parent_model_lane,
            "requested_executor": requested_executor,
            "task_group_id": task_group_id,
            "task_group": task_group,
            "subagent_envelope": subagent_envelope,
        },
    }
    if not drive_root:
        task.pop("drive_root", None)
    if not budget_drive_root:
        task.pop("budget_drive_root", None)
    if task_constraint is None:
        task.pop("task_constraint", None)
        task["metadata"].pop("task_constraint", None)
    if not required_capabilities:
        task.pop("required_capabilities", None)
        task["metadata"].pop("required_capabilities", None)
    if parent_id:
        task["parent_task_id"] = parent_id
    return task


def _extract_task_description_and_context(task: Dict[str, Any]) -> tuple[str, str]:
    description = str(task.get("description") or "").strip()
    context = str(task.get("context") or "").strip()
    if description or context:
        return description, context

    text = str(task.get("text") or task.get("description") or "").strip()
    if not text:
        return "", ""
    if _PARENT_CONTEXT_MARKER not in text or _PARENT_CONTEXT_END not in text:
        return text, ""

    before_marker, after_marker = text.split(_PARENT_CONTEXT_MARKER, 1)
    description = before_marker.split("\n\n---\n", 1)[0].strip()
    if "]\n" in after_marker:
        after_marker = after_marker.split("]\n", 1)[1]
    context = after_marker.rsplit(_PARENT_CONTEXT_END, 1)[0].strip()
    return description, context


def _format_task_for_dedup(
    task_id: str,
    description: str,
    context: str,
    *,
    expected_output: str = "",
    constraints: str = "",
    role: str = "",
) -> str:
    sections = [
        f"Task ID: {task_id}\n"
        f"Description:\n{description or '(empty)'}\n\n"
        f"Context:\n{context or '(none)'}"
    ]
    if expected_output:
        sections.append(f"Expected output:\n{expected_output}")
    if constraints:
        sections.append(f"Constraints:\n{constraints}")
    if role:
        sections.append(f"Role:\n{role}")
    return "\n\n".join(sections)


def _find_duplicate_task(
    desc: str,
    task_context: str,
    pending: list,
    running: dict,
    *,
    expected_output: str = "",
    constraints: str = "",
    role: str = "",
    dedupe_identity: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Use a scoped light-model attempt to reject only true duplicate active tasks.

    Provider/parse failures remain fail-soft, but monetary-accounting rails propagate
    so an unavailable budget can never be mistaken for a semantic non-duplicate.
    """
    identity = dedupe_identity if isinstance(dedupe_identity, dict) else {}

    def _task_identifier(existing_task: Dict[str, Any]) -> str:
        return str(existing_task.get("id") or existing_task.get("task_id") or "").strip()

    def _is_subagent_ancestor_task(existing_task: Dict[str, Any]) -> bool:
        delegation_role = str(identity.get("delegation_role") or "")
        if delegation_role != "subagent":
            return False
        existing_id = _task_identifier(existing_task)
        parent = str(identity.get("parent_task_id") or "").strip()
        root = str(identity.get("root_task_id") or "").strip()
        if existing_id and existing_id in {parent, root}:
            return True
        existing_role = str(existing_task.get("delegation_role") or "")
        existing_root = str(existing_task.get("root_task_id") or "").strip()
        return bool(existing_role == "root" and root and existing_root == root)

    def _is_distinct_parallel_subagent(existing_task: Dict[str, Any]) -> bool:
        # Lineage/role are scheduler identity facts for parallel swarm slots;
        # semantic duplicate judgment still belongs to the LLM for remaining cases.
        delegation_role = str(identity.get("delegation_role") or "")
        if str(delegation_role or "") != "subagent":
            return False
        if str(existing_task.get("delegation_role") or "") != "subagent":
            return False
        root = str(identity.get("root_task_id") or "")
        if not root or str(existing_task.get("root_task_id") or "") != root:
            return False
        parent = str(identity.get("parent_task_id") or "")
        existing_parent = str(existing_task.get("parent_task_id") or "")
        if parent != existing_parent:
            return True
        new_role = str(role or "").strip()
        existing_role = str(existing_task.get("role") or "").strip()
        return bool(new_role and existing_role and new_role != existing_role)

    existing = []
    for task in pending:
        description, context = _extract_task_description_and_context(task)
        if (
            description.strip()
            and not _is_subagent_ancestor_task(task)
            and not _is_distinct_parallel_subagent(task)
        ):
            existing.append({
                "id": str(task.get("id", "?")),
                "description": description,
                "context": context,
                "expected_output": str(task.get("expected_output") or ""),
                "constraints": str(task.get("constraints") or ""),
                "role": str(task.get("role") or ""),
                "delegation_role": str(task.get("delegation_role") or ""),
                "parent_task_id": str(task.get("parent_task_id") or ""),
                "root_task_id": str(task.get("root_task_id") or ""),
            })
    for task_id, meta in running.items():
        task_data = meta.get("task") if isinstance(meta, dict) else None
        if not isinstance(task_data, dict):
            continue
        description, context = _extract_task_description_and_context(task_data)
        if (
            description.strip()
            and not _is_subagent_ancestor_task({"id": task_id, **task_data})
            and not _is_distinct_parallel_subagent(task_data)
        ):
            existing.append({
                "id": str(task_id),
                "description": description,
                "context": context,
                "expected_output": str(task_data.get("expected_output") or ""),
                "constraints": str(task_data.get("constraints") or ""),
                "role": str(task_data.get("role") or ""),
                "delegation_role": str(task_data.get("delegation_role") or ""),
                "parent_task_id": str(task_data.get("parent_task_id") or ""),
                "root_task_id": str(task_data.get("root_task_id") or ""),
            })

    if not existing:
        return None

    existing_lines = "\n\n".join(
        _format_task_for_dedup(
            e["id"],
            e["description"],
            e["context"],
            expected_output=e.get("expected_output", ""),
            constraints=e.get("constraints", ""),
            role=e.get("role", ""),
        )
        for e in existing
    )
    prompt = (
        "Determine whether the NEW task is a true duplicate of any EXISTING active task.\n"
        "Only return a task ID if the requested work is materially the same.\n"
        "Tasks that share a broad goal but differ in target model, creative focus, "
        "scope, parent context, or intended output are NOT duplicates.\n\n"
        "NEW TASK\n"
        f"{_format_task_for_dedup('NEW', desc, task_context, expected_output=expected_output, constraints=constraints, role=role)}\n\n"
        f"EXISTING ACTIVE TASKS\n{existing_lines}\n\n"
        "Reply ONLY with the task ID if duplicate, or NONE if not."
    )

    from dataclasses import replace

    from ouroboros.usage_accounting import (
        BudgetExceeded,
        UsageAccountingError,
        UsageScope,
        current_usage_scope,
        usage_scope,
    )

    base_scope = current_usage_scope()
    prospective_task_id = str(identity.get("task_id") or (base_scope.task_id if base_scope else ""))
    prospective_root_id = str(
        identity.get("root_task_id")
        or (base_scope.root_task_id if base_scope else "")
        or prospective_task_id
    )
    prospective_parent_id = str(
        identity.get("parent_task_id")
        or (base_scope.parent_task_id if base_scope else "")
    )
    prospective_budget_root: Any = identity.get("budget_drive_root") or (
        base_scope.drive_root if base_scope else None
    )
    if base_scope is not None:
        duplicate_scope = replace(
            base_scope,
            drive_root=prospective_budget_root,
            task_id=prospective_task_id,
            root_task_id=prospective_root_id,
            parent_task_id=prospective_parent_id,
            category="planning",
            source="task_duplicate_check",
        )
    else:
        try:
            global_limit = float(os.environ.get("TOTAL_BUDGET", "0") or 0)
        except (TypeError, ValueError):
            global_limit = 0.0
        try:
            root_limit = float(os.environ.get("OUROBOROS_PER_TASK_COST_USD", "0") or 0)
        except (TypeError, ValueError):
            root_limit = 0.0
        duplicate_scope = UsageScope(
            drive_root=prospective_budget_root,
            task_id=prospective_task_id,
            root_task_id=prospective_root_id,
            parent_task_id=prospective_parent_id,
            category="planning",
            source="task_duplicate_check",
            global_limit_usd=global_limit if global_limit > 0 else None,
            root_limit_usd=root_limit if root_limit > 0 else None,
        )

    try:
        from ouroboros.config import get_light_model
        from ouroboros.llm import LLMClient
        light_model = get_light_model()
        client = LLMClient()
        with usage_scope(duplicate_scope):
            resp_msg, _usage = client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=light_model,
                reasoning_effort="low",
                max_tokens=50,
            )
        answer = (resp_msg.get("content") or "NONE").strip()
        if answer.upper() == "NONE" or not answer:
            return None
        answer_lower = answer.lower()
        for e in existing:
            if e["id"].lower() in answer_lower:
                return e["id"]
        return None
    except (BudgetExceeded, UsageAccountingError):
        raise
    except Exception as exc:
        log.warning("LLM dedup unavailable, accepting task: %s", exc)
        return None


def _cleanup_rejected_worktree(tid: str, result_fields: Dict[str, Any]) -> None:
    """Tear down a write surface provisioned for an acting subagent that is then
    rejected by a later gate, so rejected schedules never leak a worktree or an
    empty genesis project."""
    tc = result_fields.get("task_constraint") if isinstance(result_fields, dict) else None
    if not (isinstance(tc, dict) and tc.get("mode") == ACTING_SUBAGENT_MODE):
        return
    surface = str(tc.get("surface") or "")
    write_root = str(tc.get("write_root") or "").strip()
    if not write_root:
        return
    try:
        from ouroboros import subagent_worktrees

        if surface == "self_worktree":
            subagent_worktrees.remove_worktree(task_id=str(tid))
        elif surface == "genesis":
            subagent_worktrees.remove_genesis_project(write_root)
    except Exception:
        log.debug("Failed to clean up rejected acting write surface for %s", tid, exc_info=True)


def _reject_schedule_task(
    ctx: Any,
    *,
    tid: str,
    chat_id: int,
    delegation_role: str,
    parent_id: Any,
    root_task_id: str,
    role: str,
    result_fields: Dict[str, Any],
    detail: str,
    status: str = STATUS_FAILED,
    fallback_message: str = "",
    reason_code: Optional[str] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist and notify a terminal schedule rejection."""
    _cleanup_rejected_worktree(tid, result_fields)
    log.warning("Rejecting scheduled task %s: %s", tid, detail)
    write_fields = {**result_fields, **(extra_fields or {})}
    if reason_code:
        write_fields["reason_code"] = reason_code
    try:
        write_task_result(
            ctx.DRIVE_ROOT,
            tid,
            status,
            **write_fields,
            result=detail,
            cost_usd=0.0,
        )
    except Exception:
        log.warning("Failed to persist schedule rejection for %s", tid, exc_info=True)
    # The terminal result is already durable above; never let a notification
    # failure (torn-down bus, etc.) propagate into the supervisor event loop.
    try:
        if chat_id:
            if delegation_role == "subagent":
                _send_subagent_rejection(
                    ctx,
                    chat_id,
                    tid=tid,
                    parent_id=parent_id,
                    root_task_id=root_task_id,
                    role=role,
                    status=status,
                    detail=detail,
                )
            elif fallback_message:
                ctx.send_with_budget(chat_id, fallback_message)
    except Exception:
        log.warning("Failed to notify schedule rejection for %s", tid, exc_info=True)


def _reject_if_no_chat_target(
    ctx: Any, *, desc: str, chat_id: int, delegation_role: str, tid: str, role: str,
    parent_id: Any, root_task_id: str, result_fields: Dict[str, Any],
) -> bool:
    """Chat-target gate. A non-subagent task needs a live chat to schedule to; a
    subagent returns its result to its PARENT, not a UI thread, so headless roots
    (created via /api/tasks with no chat_id and owner_chat_id=None — CLI/Terminal-
    Bench) schedule it without a chat target (the chat-only notification later is
    skipped when chat_id is 0). Returns True when rejected (caller must return)."""
    if not (desc and not chat_id):
        return False
    if delegation_role != "subagent":
        log.warning("Rejected scheduled task without chat target: task_id=%s desc=%s", tid, desc[:100])
        _reject_schedule_task(
            ctx, tid=tid, chat_id=chat_id, delegation_role=delegation_role,
            parent_id=parent_id, root_task_id=root_task_id, role=role,
            result_fields=result_fields,
            detail="Subagent rejected: no chat target is available for live scheduling.",
        )
        return True
    log.info("Scheduled headless subagent without live chat target: task_id=%s role=%s", tid, role)
    return False


def _handle_schedule_task(evt: Dict[str, Any], ctx: Any) -> None:
    st = ctx.load_state()
    owner_chat_id = st.get("owner_chat_id")
    try:
        event_chat_id = int(evt.get("chat_id") or 0)
    except (TypeError, ValueError):
        event_chat_id = 0
    try:
        owner_chat_int = int(owner_chat_id or 0)
    except (TypeError, ValueError):
        owner_chat_int = 0
    chat_id = event_chat_id or owner_chat_int
    tid = str(evt.get("task_id") or uuid.uuid4().hex[:8])
    desc = str(evt.get("objective") or evt.get("description") or "").strip()
    expected_output = str(evt.get("expected_output") or "").strip()
    constraints = str(evt.get("constraints") or "").strip()
    role = str(evt.get("role") or "researcher").strip() or "researcher"
    task_context = str(evt.get("context") or "").strip()
    depth = int(evt.get("depth", 0))
    parent_id = evt.get("parent_task_id")
    root_task_id = str(evt.get("root_task_id") or parent_id or tid)
    session_id = str(evt.get("session_id") or "")
    actor_id = str(evt.get("actor_id") or "ouroboros")
    delegation_role = str(evt.get("delegation_role") or "subagent")
    memory_mode = str(evt.get("memory_mode") or "").strip()
    drive_root = str(evt.get("drive_root") or "").strip()
    child_drive_root = str(evt.get("child_drive_root") or drive_root).strip()
    budget_drive_root = str(evt.get("budget_drive_root") or "").strip()
    # INTENT ONLY (see `_build_scheduled_task_payload`): the supervisor forwards what
    # the parent ASKED for. What the child gets is resolved once, at dispatch.
    requested_model_lane = str(evt.get("requested_model_lane") or evt.get("model_lane") or "auto").strip() or "auto"
    parent_model_lane = str(evt.get("parent_model_lane") or "").strip()
    requested_executor = str(evt.get("requested_executor") or "").strip().lower() or "auto"
    task_group_id = str(evt.get("task_group_id") or "").strip()
    task_group = evt.get("task_group") if isinstance(evt.get("task_group"), dict) else {}
    subagent_envelope = evt.get("subagent_envelope") if isinstance(evt.get("subagent_envelope"), dict) else {}
    task_constraint = evt.get("task_constraint") if isinstance(evt.get("task_constraint"), dict) else None
    required_capabilities = [
        str(item or "").strip().lower()
        for item in (evt.get("required_capabilities") if isinstance(evt.get("required_capabilities"), list) else [])
        if str(item or "").strip()
    ]
    workspace_root = str(evt.get("workspace_root") or "").strip()
    workspace_mode = str(evt.get("workspace_mode") or "").strip()
    project_id = str(evt.get("project_id") or "").strip()
    acting_reject_detail = ""
    if delegation_role == "subagent":
        task_constraint, workspace_root, workspace_mode, acting_reject_detail = _resolve_subagent_constraint(
            ctx, tid=tid, requested_constraint=task_constraint, workspace_root=workspace_root,
            workspace_mode=workspace_mode, base_sha=str(evt.get("base_sha") or ""), parent_task_id=str(parent_id or ""))
    allowed_resources = normalize_allowed_resources(evt.get("allowed_resources") or {})
    task_contract = evt.get("task_contract") if isinstance(evt.get("task_contract"), dict) else build_task_contract({
        "id": tid,
        "type": "task",
        "description": desc,
        "objective": desc,
        "expected_output": expected_output,
        "constraints": constraints,
        "workspace_root": workspace_root,
        "workspace_mode": workspace_mode,
        "allowed_resources": allowed_resources,
        "parent_task_id": parent_id,
        "root_task_id": root_task_id,
        "session_id": session_id,
        "delegation_role": delegation_role,
    })
    result_fields = {
        "parent_task_id": parent_id,
        "root_task_id": root_task_id,
        "session_id": session_id,
        "actor_id": actor_id,
        "delegation_role": delegation_role,
        "role": role,
        "description": desc,
        "objective": desc,
        "expected_output": expected_output,
        "constraints": constraints,
        "context": task_context,
        "workspace_root": workspace_root,
        "workspace_mode": workspace_mode, "project_id": project_id,
        "allowed_resources": allowed_resources,
        "task_contract": task_contract,
        "chat_id": chat_id or None,
        "memory_mode": memory_mode,
        "drive_root": drive_root,
        "child_drive_root": child_drive_root,
        "budget_drive_root": budget_drive_root,
        "task_constraint": task_constraint,
        "required_capabilities": required_capabilities,
        "model_lane": requested_model_lane,
        "requested_model_lane": requested_model_lane,
        "parent_model_lane": parent_model_lane,
        "requested_executor": requested_executor,
        "task_group_id": task_group_id,
        "task_group": task_group,
        "subagent_envelope": subagent_envelope,
    }
    if delegation_role == "subagent" and (not str(evt.get("objective") or "").strip() or not expected_output):
        detail = "Subagent rejected: schedule_subagent requires objective and expected_output."
        log.warning("Rejected subagent due to strict schedule_subagent schema violation: task_id=%s", tid)
        _reject_schedule_task(
            ctx, tid=tid, chat_id=chat_id, delegation_role=delegation_role,
            parent_id=parent_id, root_task_id=root_task_id, role=role,
            result_fields={**result_fields, "objective": str(evt.get("objective") or "").strip()},
            detail=detail,
        )
        return

    if delegation_role == "subagent" and acting_reject_detail:
        log.warning("Acting subagent request rejected: task_id=%s detail=%s", tid, acting_reject_detail[:160])
        _record_delegation_constraint(
            root_task_id,
            task_id=tid,
            role=role,
            directive="block_surface",
            scope={"surface": str((task_constraint or {}).get("surface") or evt.get("write_surface") or "")},
            rationale=acting_reject_detail,
            advisory=True,
        )
        _reject_schedule_task(
            ctx, tid=tid, chat_id=chat_id, delegation_role=delegation_role,
            parent_id=parent_id, root_task_id=root_task_id, role=role,
            result_fields=result_fields, detail=acting_reject_detail,
        )
        return

    if delegation_role == "subagent" and (memory_mode not in VALID_SUBAGENT_MEMORY_MODES or not child_drive_root):
        detail = (
            "Subagent rejected: internal schedule_subagent events must use memory_mode=forked or empty "
            "and include a child_drive_root."
        )
        log.warning("Rejected subagent due to invalid child-drive contract: task_id=%s memory_mode=%s child_drive_root=%s", tid, memory_mode, child_drive_root)
        _reject_schedule_task(
            ctx, tid=tid, chat_id=chat_id, delegation_role=delegation_role,
            parent_id=parent_id, root_task_id=root_task_id, role=role,
            result_fields=result_fields, detail=detail,
        )
        return

    # The lane an applicable, non-advisory require_lane constraint verified this
    # admission against (F9): stamped onto the child record so the dispatch-time
    # policy default cannot override the lane the gate just enforced.
    required_model_lane = ""
    if delegation_role == "subagent":
        try:
            from ouroboros.tool_access import subagent_profile_satisfies
            from ouroboros.tools.control_delegation import effective_delegation_budget
            from ouroboros.task_tree_ledger import open_delegation_constraints

            selected_profile = (
                "acting_subagent"
                if isinstance(task_constraint, dict)
                and task_constraint.get("mode") == ACTING_SUBAGENT_MODE
                and task_constraint.get("surface")
                else "local_readonly_subagent"
            )
            _ok, missing_caps = subagent_profile_satisfies(selected_profile, required_capabilities)
            constraints_for_tree = open_delegation_constraints(root_task_id)
            decision = effective_delegation_budget(
                task_contract.get("delegation_budget") if isinstance(task_contract, dict) else {},
                missing_capabilities=missing_caps,
                unresolved_constraints=constraints_for_tree,
                write_surface=str((task_constraint or {}).get("surface") or "") if isinstance(task_constraint, dict) else "",
                role=role,
                requested_lane=requested_model_lane,
                intended_lane=intended_subagent_lane(requested_model_lane, parent_model_lane),
                active_child_count=_active_subagent_count(root_task_id, getattr(ctx, "PENDING", []), getattr(ctx, "RUNNING", {})),
            )
            if not decision.ok:
                detail = f"Subagent rejected: {decision.reason_code}: {decision.detail}"
                _reject_schedule_task(
                    ctx, tid=tid, chat_id=chat_id, delegation_role=delegation_role,
                    parent_id=parent_id, root_task_id=root_task_id, role=role,
                    result_fields=result_fields, detail=detail,
                )
                return
            if isinstance(task_contract, dict) and decision.budget:
                task_contract = {**task_contract, "delegation_budget": decision.budget}
                result_fields["task_contract"] = task_contract
            required_model_lane = str(getattr(decision, "required_lane", "") or "")
        except Exception:
            log.debug("Delegation reconciliation failed open for %s", tid, exc_info=True)

    max_depth = get_max_subagent_depth()
    if depth > max_depth:
        detail = f"Subagent rejected: subtask depth limit ({max_depth}) exceeded."
        log.warning("Rejected task due to depth limit: depth=%d, desc=%s", depth, desc[:100])
        _reject_schedule_task(
            ctx, tid=tid, chat_id=chat_id, delegation_role=delegation_role,
            parent_id=parent_id, root_task_id=root_task_id, role=role,
            result_fields=result_fields,
            detail=detail,
            fallback_message=f"⚠️ Task rejected: subtask depth limit ({max_depth}) exceeded",
        )
        return

    if _reject_if_no_chat_target(
        ctx, desc=desc, chat_id=chat_id, delegation_role=delegation_role, tid=tid,
        role=role, parent_id=parent_id, root_task_id=root_task_id, result_fields=result_fields,
    ):
        return

    # Fail fast when the worker pool is disabled (e.g. after a crash storm put
    # the supervisor in direct-chat mode). Without this, the task is written as
    # 'scheduled' and enqueued but nothing can ever run it — a permanent "ghost"
    # the parent keeps polling. Give the parent a clear terminal signal instead
    # so it can do the work inline.
    if desc and not (getattr(ctx, "WORKERS", {}) or {}):
        _reject_schedule_task(
            ctx, tid=tid, chat_id=chat_id, delegation_role=delegation_role,
            parent_id=parent_id, root_task_id=root_task_id, role=role,
            result_fields=result_fields,
            detail=(
                "Subagent not scheduled: the worker pool is currently unavailable "
                "(workers_unavailable), likely disabled after repeated worker crashes "
                "(direct-chat mode). It was NOT left scheduled — do the work inline "
                "yourself, or retry after /restart."
            ),
            reason_code="workers_unavailable",
            fallback_message=f"⚠️ Task {tid} not scheduled: worker pool unavailable.",
        )
        return

    if desc:
        # Bible P5: duplicate judgment stays LLM-first, not hardcoded.
        from supervisor.queue import PENDING as QUEUE_PENDING, RUNNING as QUEUE_RUNNING
        pending_ref = getattr(ctx, "PENDING", QUEUE_PENDING)
        running_ref = getattr(ctx, "RUNNING", QUEUE_RUNNING)
        max_active = get_max_active_subagents_per_root()
        queued_behind_active_cap = False
        if delegation_role == "subagent" and _subagent_cap_blocks(root_task_id, parent_id, pending_ref, running_ref, max_active):
            active_count = _active_subagent_count(root_task_id, pending_ref, running_ref)
            if active_count >= MAX_ACTIVE_SUBAGENTS_HARD_CAP:
                log.warning("Rejected subagent due to hard active child cap: root=%s desc=%s", root_task_id, desc[:100])
                detail = (
                    "Subagent rejected: hard active child limit "
                    f"({MAX_ACTIVE_SUBAGENTS_HARD_CAP}) exceeded for root_task_id={root_task_id}."
                )
                _reject_schedule_task(
                    ctx, tid=tid, chat_id=chat_id, delegation_role=delegation_role,
                    parent_id=parent_id, root_task_id=root_task_id, role=role,
                    result_fields=result_fields, detail=detail,
                )
                return
            queued_behind_active_cap = True
            _record_delegation_constraint(
                root_task_id,
                task_id=tid,
                role=role,
                directive="cap_children",
                scope={"max_children": max_active},
                rationale=f"Queued behind active subagent cap {max_active}; wait for a slot before additional fan-out.",
                advisory=True,
            )
        dup_id = _find_duplicate_task(
            desc,
            task_context,
            pending_ref,
            running_ref,
            expected_output=expected_output,
            constraints=constraints,
            role=role,
            dedupe_identity={
                "delegation_role": delegation_role,
                "task_id": tid,
                "parent_task_id": str(parent_id or ""),
                "root_task_id": root_task_id,
                "budget_drive_root": budget_drive_root or str(ctx.DRIVE_ROOT),
            },
        )
        if dup_id:
            log.info("Rejected duplicate task: new='%s' duplicates='%s'", desc[:100], dup_id)
            detail = f"Task was rejected as semantically similar to already active task {dup_id}."
            _reject_schedule_task(
                ctx, tid=tid, chat_id=chat_id, delegation_role=delegation_role,
                parent_id=parent_id, root_task_id=root_task_id, role=role,
                result_fields=result_fields,
                detail=detail,
                status=STATUS_REJECTED_DUPLICATE,
                extra_fields={"duplicate_of": dup_id},
                fallback_message=f"⚠️ Task rejected: semantically similar to already active task {dup_id}",
            )
            return

        text = _compose_subagent_text(
            desc,
            role=role,
            expected_output=expected_output,
            constraints=constraints,
            context=task_context,
            task_constraint=task_constraint,
            delegation_budget=task_contract.get("delegation_budget") if isinstance(task_contract, dict) else None,
        ) if delegation_role == "subagent" else desc
        task = _build_scheduled_task_payload({
            "tid": tid,
            "chat_id": chat_id,
            "text": text,
            "desc": desc,
            "expected_output": expected_output,
            "constraints": constraints,
            "role": role,
            "task_context": task_context,
            "depth": depth,
            "root_task_id": root_task_id,
            "session_id": session_id,
            "actor_id": actor_id,
            "delegation_role": delegation_role,
            "memory_mode": memory_mode,
            "drive_root": drive_root,
            "child_drive_root": child_drive_root,
            "budget_drive_root": budget_drive_root,
            "task_constraint": task_constraint,
            "workspace_root": workspace_root,
            "workspace_mode": workspace_mode,
            "project_id": project_id,
            "allowed_resources": allowed_resources,
            "task_contract": task_contract,
            "required_capabilities": required_capabilities,
            "model_lane": requested_model_lane,
            "requested_model_lane": requested_model_lane,
            "parent_model_lane": parent_model_lane,
            "required_model_lane": required_model_lane,
            "requested_executor": requested_executor,
            "task_group_id": task_group_id,
            "task_group": task_group,
            "subagent_envelope": subagent_envelope,
            "parent_id": parent_id,
        })
        admitted = ctx.enqueue_task(task)
        if isinstance(admitted, dict) and admitted.get("_admission_blocked"):
            blocked_reason = str(admitted.get("_admission_blocked") or "admission_fence")
            if blocked_reason.startswith("project_routing_fence"):
                fence_status = str(admitted.get("_project_lifecycle") or "unavailable")
                detail = (
                    "Subagent not scheduled: the target Project has closed its routing/admission "
                    f"fence ({fence_status}) and cannot accept new work."
                )
                reason_code = blocked_reason
                extra = {
                    "project_id": str(admitted.get("_project_id") or project_id),
                    "project_lifecycle": fence_status,
                }
            elif blocked_reason == "root_cancelled":
                detail = (
                    "Subagent not scheduled: its root's subtree cancellation has "
                    "begun, so the tree accepts no new work."
                )
                reason_code = blocked_reason
                extra = {"root_task_id": str(root_task_id or "")}
            elif blocked_reason == "root_budget_fence":
                detail = (
                    "Subagent not scheduled: the root budget is paused and requires an "
                    "explicit replay-safe resume, cancellation, or a new run."
                )
                reason_code = blocked_reason
                extra = {
                    "root_task_id": str(admitted.get("_budget_root_task_id") or root_task_id),
                    "budget_fence_id": str(admitted.get("_budget_fence_id") or ""),
                }
            else:
                fence_status = str(admitted.get("_acceptance_fence_status") or "active")
                detail = (
                    "Subagent not scheduled: the root task is in its atomic task-acceptance "
                    f"phase ({fence_status}); admission is closed until an explicit revision round."
                )
                reason_code = "task_acceptance_fence"
                extra = {
                    "acceptance_fence_token": str(admitted.get("_acceptance_fence_token") or ""),
                    "acceptance_fence_status": fence_status,
                }
            _reject_schedule_task(
                ctx,
                tid=tid,
                chat_id=chat_id,
                delegation_role=delegation_role,
                parent_id=parent_id,
                root_task_id=root_task_id,
                role=role,
                result_fields=result_fields,
                detail=detail,
                reason_code=reason_code,
                extra_fields=extra,
            )
            return
        try:
            write_task_result(
                ctx.DRIVE_ROOT,
                tid,
                STATUS_SCHEDULED,
                **result_fields,
                result="Subagent accepted and scheduled." if delegation_role == "subagent" else "Task accepted and scheduled.",
            )
        except Exception:
            log.warning("Failed to persist scheduled task status for %s", tid, exc_info=True)
        progress_meta = {
            "root_task_id": root_task_id,
            "parent_task_id": parent_id,
            "delegation_role": delegation_role,
            "task_group_id": task_group_id,
            "required_capabilities": required_capabilities,
            "requested_model_lane": requested_model_lane,
            # v6.82 (P5): host-attested cancelability, ROOTS ONLY. Every task
            # admitted here is a supervisor-queue task the cancel endpoint can
            # reach, but the marker exists to gate the ROOT card's "Cancel run"
            # button — a subagent row must never carry it (its card is a child
            # card, and a lineage-less replay of a marked child row could mint a
            # root-shaped card with a live Cancel). Direct-chat turns never pass
            # through this path (or RUNNING).
            "cancelable": delegation_role != "subagent",
        }
        if delegation_role == "subagent":
            progress_meta.update(_subagent_scheduled_meta(
                tid=tid, role=role, task_constraint=task_constraint,
                task_group_id=task_group_id, requested_model_lane=requested_model_lane,
                active_subagent_count=_active_subagent_count(root_task_id, pending_ref, running_ref),
                max_active_subagents=max_active,
            ))
            if queued_behind_active_cap:
                progress_meta["queued_behind_active_cap"] = True
        else:
            progress_meta["task_event"] = "scheduled"
        workers = getattr(ctx, "WORKERS", {}) or {}
        if workers and not any(not getattr(worker, "busy_task_id", None) for worker in workers.values()):
            progress_meta["worker_saturation_warning"] = True
            suffix = " (all workers are currently busy; it will start when one is free)"
        else:
            suffix = ""
        if delegation_role == "subagent" and queued_behind_active_cap:
            suffix = (
                f" (queued behind active subagent cap {max_active}; it will start when a slot frees)"
            )
        # A subagent's scheduled notice routes to its root project thread by lineage (C4.4); else its own chat; a headless subagent (chat_id=0, no bound root) still skips.
        _notice_chat = (_bound_project_chat_id(ctx, tid, parent_id, root_task_id)
                        if delegation_role == "subagent" else 0) or chat_id
        if _notice_chat:
            ctx.send_with_budget(
                _notice_chat,
                f"🗓️ Scheduled subagent {tid} ({role}): {desc}{suffix}" if delegation_role == "subagent" else f"🗓️ Scheduled task {tid}: {desc}",
                is_progress=True, task_id=tid, progress_meta=progress_meta,
            )
        ctx.persist_queue_snapshot(reason="schedule_subagent_event")
