"""The schedule_task admission gates, the duplicate gate, and its refusals.

One owner for the facts the dispatch parent's schedule handler needs: the
chat-target gate, the semantic duplicate gate, the composed queue payload,
and every refusal path including worktree cleanup for a rejected subagent.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional
from ouroboros.tool_capabilities import ACTING_SUBAGENT_MODE
from ouroboros.task_results import (
    STATUS_FAILED,
    write_task_result,
)
from ouroboros.utils import utc_now_iso
from supervisor.events_subagent_admission import (
    _send_subagent_rejection,
)
from ouroboros.config import MAX_ACTIVE_SUBAGENTS_HARD_CAP
from ouroboros.config import get_max_active_subagents_per_root
from ouroboros.contracts.task_contract import build_task_contract
from ouroboros.contracts.task_contract import normalize_allowed_resources
from ouroboros.subagents import intended_lane as intended_subagent_lane
from ouroboros.task_results import STATUS_REJECTED_DUPLICATE
from ouroboros.task_results import STATUS_SCHEDULED
from ouroboros.tools.control_delegation import admitted_depth_cap
from ouroboros.tools.control_delegation import check_delegation_admission
from ouroboros.tools.control_delegation import durable_direct_child_count
from ouroboros.tools.control_delegation import stamp_depth_provenance
from supervisor.events_subagent_admission import _active_subagent_count
from supervisor.events_subagent_admission import _compose_subagent_text
from supervisor.events_subagent_admission import _record_delegation_constraint
from supervisor.events_subagent_admission import _resolve_subagent_constraint
from supervisor.events_subagent_admission import _subagent_cap_blocks
from supervisor.events_subagent_admission import _subagent_scheduled_meta
from supervisor.log_addressing import bound_project_chat_id as _bound_project_chat_id
from supervisor.message_bus import coerce_chat_identity, notification_chat_route
from ouroboros.contracts.chat_id_policy import HIDDEN_CHAT_ID
from supervisor.task_dispatch import build_scheduled_task_payload as _build_scheduled_task_payload
import uuid

log = logging.getLogger(__name__)


def _events():
    """The parent module, read at call time.

    The parent owns the rebindable module state and the members tests
    monkeypatch there; reading them through the module at each call keeps
    one binding, where a from-import would freeze the value this leaf saw
    at import time (the owner-approved D18/D33 mechanical exception).
    """
    from supervisor import events

    return events


_PARENT_CONTEXT_MARKER = "[BEGIN_PARENT_CONTEXT"


_PARENT_CONTEXT_END = "[END_PARENT_CONTEXT]"


VALID_SUBAGENT_MEMORY_MODES = frozenset({"forked", "empty"})


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
        from ouroboros.settings_setup_contract import resolve_total_budget_usd
        global_limit = resolve_total_budget_usd()
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
            global_limit_usd=global_limit,
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
    persist_result: bool = True,
) -> None:
    """Clean up and notify a rejection, preserving an existing result when asked."""
    _cleanup_rejected_worktree(tid, result_fields)
    log.warning("Rejecting scheduled task %s: %s", tid, detail)
    write_fields = {**result_fields, **(extra_fields or {})}
    if reason_code:
        write_fields["reason_code"] = reason_code
    if persist_result:
        try:
            write_task_result(
                ctx.DRIVE_ROOT,
                tid,
                status,
                **write_fields,
                result=detail,
                # ABI-3: a rejected schedule spent a confirmed zero — stamped
                # under the honest name (the retired alias is read-only).
                accounted_upper_bound_usd=0.0,
            )
        except Exception:
            log.warning("Failed to persist schedule rejection for %s", tid, exc_info=True)
    # A torn-down notification bus must not escape into the supervisor loop.
    try:
        if chat_id is not None:
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


def _handle_schedule_task(evt: Dict[str, Any], ctx: Any) -> None:
    st = ctx.load_state()
    owner_chat_id = st.get("owner_chat_id")
    try:
        owner_chat_int = int(owner_chat_id or 0)
    except (TypeError, ValueError):
        owner_chat_int = 0
    # Membership, not truthiness (C4): an explicit 0 is the hidden partition the
    # producer stamped; only a MISSING id is absence and falls to the owner chat.
    chat_id = coerce_chat_identity(evt.get("chat_id"), owner_chat_int)
    tid = str(evt.get("task_id") or uuid.uuid4().hex[:8])
    created_at = utc_now_iso() if not evt.get("task_id") else ""
    desc = str(evt.get("objective") or evt.get("description") or "").strip()
    expected_output = str(evt.get("expected_output") or "").strip()
    constraints = str(evt.get("constraints") or "").strip()
    role = str(evt.get("role") or "researcher").strip() or "researcher"
    task_context = str(evt.get("context") or "").strip()
    parent_id = evt.get("parent_task_id")
    root_task_id = str(evt.get("root_task_id") or parent_id or tid)
    session_id = str(evt.get("session_id") or "")
    actor_id = str(evt.get("actor_id") or "ouroboros")
    delegation_role = str(evt.get("delegation_role") or "subagent")
    # Idempotency/ownership is checked before parsing for every scheduling role
    # so a malformed replay cannot terminalize an already-owned task id. Fresh
    # events still reach the typed depth rejection below before provisioning or
    # enqueue; events without an explicit id use the normal fresh-id path.
    from supervisor.task_admission import subagent_schedule_preflight
    if subagent_schedule_preflight(
        ctx, evt, chat_id, delegation_role=delegation_role,
    ):
        return
    from supervisor.task_admission import parse_schedule_task_depth

    depth, depth_rejected = parse_schedule_task_depth(
        ctx,
        evt,
        tid=tid,
        chat_id=chat_id,
        delegation_role=delegation_role,
        parent_id=parent_id,
        root_task_id=root_task_id,
        role=role,
        desc=desc,
        expected_output=expected_output,
        constraints=constraints,
        task_context=task_context,
    )
    if depth_rejected:
        return
    memory_mode = str(evt.get("memory_mode") or "").strip()
    drive_root = str(evt.get("drive_root") or "").strip()
    child_drive_root = str(evt.get("child_drive_root") or drive_root).strip()
    budget_drive_root = str(evt.get("budget_drive_root") or "").strip()
    root_cost_ceiling_usd = evt.get("root_cost_ceiling_usd")
    # Forward parent-requested intent; dispatch resolves it once.
    requested_model_lane = str(evt.get("requested_model_lane") or evt.get("model_lane") or "auto").strip() or "auto"
    parent_model_lane = str(evt.get("parent_model_lane") or "").strip()
    requested_executor = str(evt.get("requested_executor") or "").strip().lower() or "auto"
    task_group_id = str(evt.get("task_group_id") or "").strip()
    task_group = evt.get("task_group") if isinstance(evt.get("task_group"), dict) else {}
    subagent_envelope = evt.get("subagent_envelope") if isinstance(evt.get("subagent_envelope"), dict) else {}
    configured_subagent = evt.get("configured_subagent") if isinstance(evt.get("configured_subagent"), dict) else {}
    parent_cognitive_route = evt.get("parent_cognitive_route") if isinstance(evt.get("parent_cognitive_route"), dict) else {}
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
    live_max_depth = _events().get_max_subagent_depth()
    max_depth = admitted_depth_cap(task_contract, live_max_depth)
    task_contract, depth_provenance = stamp_depth_provenance(
        task_contract,
        attempted_depth=depth,
        max_depth=max_depth,
    )
    result_fields = {
        **({"created_at": created_at} if created_at else {}),
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
        "depth_provenance": depth_provenance,
        "chat_id": chat_id,
        "memory_mode": memory_mode,
        "drive_root": drive_root,
        "child_drive_root": child_drive_root,
        "budget_drive_root": budget_drive_root,
        "root_cost_ceiling_usd": root_cost_ceiling_usd,
        "task_constraint": task_constraint,
        "required_capabilities": required_capabilities,
        "model_lane": requested_model_lane,
        "requested_model_lane": requested_model_lane,
        "parent_model_lane": parent_model_lane,
        "requested_executor": requested_executor,
        "task_group_id": task_group_id,
        "task_group": task_group,
        "subagent_envelope": subagent_envelope,
        "configured_subagent": configured_subagent,
        "parent_cognitive_route": parent_cognitive_route,
    }
    if delegation_role == "subagent":
        parent_budget = _events()._parent_delegation_budget(
            ctx, parent_id, budget_drive_root or getattr(ctx, "DRIVE_ROOT", "")
        )
        count_roots = [budget_drive_root or getattr(ctx, "DRIVE_ROOT", "")]
        canonical_root = getattr(ctx, "DRIVE_ROOT", "")
        if canonical_root and str(canonical_root) != str(count_roots[0]):
            count_roots.append(canonical_root)
        direct_child_counts = [
            durable_direct_child_count(root, parent_id, exclude_task_id=tid)
            for root in count_roots
        ]
        direct_child_count = (
            max(count for count in direct_child_counts if count is not None)
            if direct_child_counts and all(
                count is not None for count in direct_child_counts
            )
            else None
        )
        rights = check_delegation_admission(
            parent_budget,
            direct_child_count=direct_child_count,
        )
        if not rights.ok:
            detail = f"Subagent rejected: {rights.reason_code}: {rights.detail}"
            result_fields["delegation_admission"] = {
                "status": "rejected",
                "reason_code": rights.reason_code,
                "direct_child_count": rights.direct_child_count,
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
                reason_code=rights.reason_code,
            )
            return
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

        # Assignment, not admission, proves achieved depth.
        admitted_task_contract, admitted_depth_provenance = stamp_depth_provenance(
            task_contract,
            attempted_depth=depth,
            max_depth=max_depth,
            achieved_depth=None,
        )
        text = _compose_subagent_text(
            desc,
            role=role,
            expected_output=expected_output,
            constraints=constraints,
            context=task_context,
            task_constraint=task_constraint,
            delegation_budget=admitted_task_contract.get("delegation_budget") if isinstance(admitted_task_contract, dict) else None,
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
            "root_cost_ceiling_usd": root_cost_ceiling_usd,
            "task_constraint": task_constraint,
            "workspace_root": workspace_root,
            "workspace_mode": workspace_mode,
            "project_id": project_id,
            "allowed_resources": allowed_resources,
            "task_contract": admitted_task_contract,
            "depth_provenance": admitted_depth_provenance,
            "required_capabilities": required_capabilities,
            "model_lane": requested_model_lane,
            "requested_model_lane": requested_model_lane,
            "parent_model_lane": parent_model_lane,
            "required_model_lane": required_model_lane,
            "requested_executor": requested_executor,
            "task_group_id": task_group_id,
            "task_group": task_group,
            "subagent_envelope": subagent_envelope,
            "configured_subagent": configured_subagent,
            "parent_cognitive_route": parent_cognitive_route,
            "parent_id": parent_id,
        })
        scheduled_failure_reason = ""
        scheduled_failure_detail = ""
        persist_scheduled_failure = False
        if delegation_role == "subagent":
            from supervisor.task_admission import enqueue_subagent_with_scheduled_result

            (
                admitted,
                scheduled_failure_reason,
                scheduled_failure_detail,
                persist_scheduled_failure,
            ) = enqueue_subagent_with_scheduled_result(
                ctx,
                task,
                result_fields=result_fields,
                admitted_task_contract=admitted_task_contract,
                admitted_depth_provenance=admitted_depth_provenance,
                direct_child_count=direct_child_count,
                pending_ref=pending_ref,
            )
        else:
            admitted = ctx.enqueue_task(task)
        if isinstance(admitted, dict) and admitted.get("_admission_blocked"):
            from supervisor.task_admission import scheduled_admission_rejection

            _reject_schedule_task(
                ctx,
                tid=tid,
                chat_id=chat_id,
                delegation_role=delegation_role,
                parent_id=parent_id,
                root_task_id=root_task_id,
                role=role,
                result_fields=result_fields,
                **scheduled_admission_rejection(
                    admitted, project_id=project_id, root_task_id=root_task_id,
                ),
            )
            return
        if scheduled_failure_reason:
            if scheduled_failure_reason == "scheduled_event_replay":
                return
            result_fields["delegation_admission"] = {
                "status": "rejected",
                "reason_code": scheduled_failure_reason,
                "direct_child_count": direct_child_count,
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
                detail=scheduled_failure_detail,
                reason_code=scheduled_failure_reason,
                persist_result=persist_scheduled_failure,
            )
            ctx.persist_queue_snapshot(reason="schedule_subagent_receipt_rollback")
            return
        if delegation_role != "subagent":
            result_fields["task_contract"] = admitted_task_contract
            result_fields["depth_provenance"] = admitted_depth_provenance
            try:
                write_task_result(
                    ctx.DRIVE_ROOT,
                    tid,
                    STATUS_SCHEDULED,
                    **result_fields,
                    result="Task accepted and scheduled.",
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
        # A subagent's scheduled notice routes to its root project thread by lineage (C4.4); else its own chat, the hidden partition included (membership, not truthiness).
        _notice_chat = notification_chat_route(
            (_bound_project_chat_id(ctx, tid, parent_id, root_task_id) or None)
            if delegation_role == "subagent" else None,
            chat_id,
        )
        # A LIVE toast needs a chat a human reads. The hidden partition has none,
        # and a headless run's progress log is a benchmark trajectory input, so a
        # host toast there would be published as the agent's own narration. The
        # durable record still keeps the true address (row_chat_identity).
        if _notice_chat is not None and _notice_chat != HIDDEN_CHAT_ID:
            ctx.send_with_budget(
                _notice_chat,
                f"🗓️ Scheduled subagent {tid} ({role}): {desc}{suffix}" if delegation_role == "subagent" else f"🗓️ Scheduled task {tid}: {desc}",
                is_progress=True, task_id=tid, progress_meta=progress_meta,
            )
        ctx.persist_queue_snapshot(reason="schedule_subagent_event")
