"""Control tools: restart, timeout settings, scheduling, review, chat history, model switching."""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import threading
import time
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Dict, List

from ouroboros.config import apply_settings_to_env, get_max_subagent_depth, load_settings, save_settings
from ouroboros.headless import prepare_task_drive, task_state_dir
from ouroboros.contracts.task_contract import (
    build_task_contract,
    normalize_allowed_resources,
    normalize_bool,
)
from ouroboros.tools.control_delegation import (
    _ensure_project_scope,
    child_budget_for_schedule,
)
from ouroboros.outcomes import normalize_outcome_axes, public_task_result
from ouroboros.task_results import (
    STATUS_COMPLETED,
    STATUS_REJECTED_DUPLICATE,
    STATUS_REQUESTED,
    validate_task_id,
    write_task_result,
)
from ouroboros.task_status import load_effective_task_result, wait_for_effective_tasks
from ouroboros.subagents import (
    build_subagent_envelope,
    compact_task_group,
    expand_subagent_lane_slots,
    normalize_subagent_model_lane,
)
from ouroboros.tool_capabilities import ACTING_SUBAGENT_MODE, LOCAL_READONLY_SUBAGENT_MODE
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.utils import append_jsonl, atomic_write_json, utc_now_iso, run_cmd

log = logging.getLogger(__name__)

VALID_SUBTASK_MEMORY_MODES = frozenset({"forked", "empty"})

# Guards parent-side shared ctx state mutated during (possibly parallel)
# schedule_subagent emission within one tool-call round. Process-local: a parent
# ctx is never shared across processes, so a threading.Lock is sufficient.
_SCHEDULE_EMIT_LOCK = threading.Lock()


def _record_scheduled_subagent(ctx: ToolContext, record: Dict[str, Any]) -> None:
    """Append a scheduled-subagent record to ctx under the emit lock.

    The read-copy-append-setattr of ``_last_scheduled_subagents`` is a lost-update
    race when a burst of schedule_subagent calls is emitted in parallel; the lock
    serializes it. (list.append is atomic under the GIL, but the surrounding RMW
    is not.)
    """
    with _SCHEDULE_EMIT_LOCK:
        scheduled_records = list(getattr(ctx, "_last_scheduled_subagents", []) or [])
        scheduled_records.append(record)
        setattr(ctx, "_last_scheduled_subagents", scheduled_records)


def _emit_swarm_fanout(
    ctx: ToolContext,
    *,
    parent_task_id: str,
    root_task_id: str,
    depth: int,
    task_group_id: str,
    task_ids: List[str],
    role: str,
    requested_model_lane: str,
    effective_model_lanes: List[str],
    objective: str,
    emitted_live: bool,
) -> None:
    """Emit one durable swarm_fanout telemetry event per spawn wave (WS8).

    The name avoids task_/llm_/tool_ prefixes and the event sets no
    delegation_role/subagent_task_id, so the Logs UI never renders a phantom
    child card or folds it into a grouped-task lane (web/modules/log_events.js).
    inter_wave_latency_sec reuses ``_last_wave_ts`` under the emit lock (no new
    persistent state).
    """
    now = time.time()
    with _SCHEDULE_EMIT_LOCK:
        prev = float(getattr(ctx, "_last_wave_ts", 0.0) or 0.0)
        inter_wave = round(now - prev, 3) if prev > 0 else None
        setattr(ctx, "_last_wave_ts", now)
    evt = {
        "ts": utc_now_iso(),
        "type": "swarm_fanout",
        "task_id": parent_task_id,
        "parent_task_id": parent_task_id,
        "root_task_id": root_task_id,
        "depth": depth,
        "task_group_id": task_group_id,
        "requested_count": len(task_ids),
        "task_ids": task_ids,
        "role": role,
        "requested_model_lane": requested_model_lane,
        "effective_model_lanes": effective_model_lanes,
        "slot_count": len(task_ids),
        "objective_preview": objective[:200],
        "emitted_live": bool(emitted_live),
        "inter_wave_latency_sec": inter_wave,
    }
    try:
        append_jsonl(ctx.drive_logs() / "events.jsonl", evt)
    except Exception:
        log.debug("Failed to emit swarm_fanout telemetry", exc_info=True)


def _finalize_schedule_emission(
    ctx: ToolContext,
    *,
    task_ids: List[str],
    task_group_id: str,
    requested_model_lane: str,
    task_group: Dict[str, Any],
    objective: str,
    role: str,
    depth: int,
    parent_task_id: str,
    root_task_id: str,
    slot_tasks: list,
    emitted_modes: List[str],
) -> str:
    """Record the scheduled wave, emit swarm_fanout telemetry, and build the
    tool-result string. Extracted from _schedule_task to keep that function
    within the per-function size budget (P7)."""
    worker_note = " (live queue emission requested)" if any(m == "live" for m in emitted_modes) else ""
    try:
        _record_scheduled_subagent(ctx, {
            "task_ids": task_ids,
            "task_group_id": task_group_id,
            "requested_model_lane": requested_model_lane,
            "task_group": task_group,
            "objective": objective,
            "role": role,
        })
    except Exception:
        pass
    try:
        _emit_swarm_fanout(
            ctx,
            parent_task_id=parent_task_id,
            root_task_id=root_task_id,
            depth=depth,
            task_group_id=task_group_id,
            task_ids=task_ids,
            role=role,
            requested_model_lane=requested_model_lane,
            effective_model_lanes=[slot.effective_lane for _tid, slot in slot_tasks],
            objective=objective,
            emitted_live=any(m == "live" for m in emitted_modes),
        )
    except Exception:
        pass
    if len(task_ids) == 1:
        return f"Subagent request queued {task_ids[0]}: {objective}{worker_note}"
    return (
        f"Subagent group queued {task_group_id}: {', '.join(task_ids)} "
        f"(lane={requested_model_lane}, slots={len(task_ids)}){worker_note}"
    )


def _subtask_outcome_summary(data: Dict[str, Any]) -> str:
    ledger = data.get("verification_ledger") if isinstance(data.get("verification_ledger"), dict) else {}
    summary: Dict[str, Any] = {
        "outcome_axes": normalize_outcome_axes(data),
    }
    if isinstance(data.get("task_contract"), dict):
        summary["task_contract"] = data.get("task_contract")
    if isinstance(data.get("artifact_bundle"), dict):
        summary["artifact_bundle"] = data.get("artifact_bundle")
    if ledger:
        summary["verification_ledger"] = {
            "schema_version": ledger.get("schema_version"),
            "summary": ledger.get("summary") if isinstance(ledger.get("summary"), dict) else {},
            "entry_count": len(ledger.get("entries") or []) if isinstance(ledger.get("entries"), list) else 0,
        }
    return json.dumps(summary, ensure_ascii=False, indent=2, default=str)


def _emit_control_event(ctx: ToolContext, evt: Dict[str, Any]) -> str:
    """Emit a control event live when possible, preserving legacy fallback."""
    event_queue = getattr(ctx, "event_queue", None)
    if event_queue is not None:
        try:
            event_queue.put_nowait(dict(evt))
            return "live"
        except (AttributeError, queue.Full):
            pass
        except Exception:
            log.warning("Live control event emission failed; falling back to pending_events", exc_info=True)
    with _SCHEDULE_EMIT_LOCK:
        ctx.pending_events.append(evt)
    return "deferred"


def _evolution_restart_block_reason(ctx: ToolContext) -> str:
    if str(ctx.current_task_type or "") != "evolution":
        return ""
    try:
        status = run_cmd(["git", "status", "--porcelain"], cwd=ctx.repo_dir).strip()
        head = run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir).strip()
    except Exception as exc:
        return f"could not verify local git durability: {exc}"
    reviewed_sha = str(getattr(ctx, "last_reviewed_commit_sha", "") or "").strip()
    if reviewed_sha and reviewed_sha == head and not status:
        return ""
    if not reviewed_sha and not status:
        return ""
    if reviewed_sha and reviewed_sha != head:
        return "HEAD changed after the last reviewed local commit"
    return "commit_reviewed must create a local reviewed commit before evolution restart"


def _request_restart(ctx: ToolContext, reason: str) -> str:
    block_reason = _evolution_restart_block_reason(ctx)
    if block_reason:
        return f"⚠️ RESTART_BLOCKED: in evolution mode, {block_reason}."
    # Persist expected ref for post-restart verification.
    try:
        sha = run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir)
        branch = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ctx.repo_dir)
        verify_path = ctx.drive_path("state") / "pending_restart_verify.json"
        atomic_write_json(verify_path, {
            "ts": utc_now_iso(), "expected_sha": sha,
            "expected_branch": branch, "reason": reason,
        })
        if str(ctx.current_task_type or "") == "evolution":
            try:
                from supervisor.evolution_lifecycle import update_evolution_transaction

                update_evolution_transaction(
                    str(ctx.task_id or ""),
                    restart_decision="requested",
                    restart_required=True,
                    restart_requested_at=utc_now_iso(),
                    restart_expected_sha=str(sha or "").strip(),
                )
            except Exception:
                log.debug("Failed to record evolution restart request", exc_info=True)
    except Exception:
        log.debug("Failed to read VERSION file or git ref for restart verification", exc_info=True)
        pass
    ctx.pending_restart_reason = str(reason or "").strip() or "agent_requested_restart"
    ctx.last_push_succeeded = False
    ctx.last_reviewed_commit_sha = ""
    return f"Restart requested: {reason}"


def _set_tool_timeout(ctx: ToolContext, seconds: int) -> str:
    """Persist timeout while pinning owner-only runtime mode to the live env."""
    try:
        timeout_sec = int(seconds)
    except (TypeError, ValueError):
        return f"⚠️ TOOL_ARG_ERROR (set_tool_timeout): invalid seconds={seconds!r}"
    if timeout_sec < 1:
        return "⚠️ TOOL_ARG_ERROR (set_tool_timeout): seconds must be >= 1"

    settings = load_settings()
    settings["OUROBOROS_TOOL_TIMEOUT_SEC"] = timeout_sec
    settings["OUROBOROS_RUNTIME_MODE"] = os.environ.get("OUROBOROS_RUNTIME_MODE", "advanced")
    save_settings(settings)
    apply_settings_to_env(settings)
    return f"OK: OUROBOROS_TOOL_TIMEOUT_SEC set to {timeout_sec}s and applied immediately."


def _promote_to_stable(ctx: ToolContext, reason: str) -> str:
    ctx.pending_events.append({"type": "promote_to_stable", "reason": reason, "ts": utc_now_iso()})
    return f"Promote to stable requested: {reason}"


def _promote_chat_to_task(
    ctx: ToolContext,
    objective: str,
    expected_output: str = "",
    project_id: str = "",
    workspace_root: str = "",
    title: str = "",
    project_name: str = "",
) -> str:
    """Route real work out of the conversation lane into a supervised pooled task.

    Option B of the multi-project chat plane (v6.32.0): the conversation stays
    in the fast in-process lane; ANY substantial work spawns a first-class
    pooled task with a live card. The decision is the model's own structural
    tool call (BIBLE P5 — no keyword routing). Follow-up owner messages reach
    the running task through its owner-mailbox.

    ``title`` is a short human name the model coins for the card AT CREATION
    (no extra request, owner P1) — reused as the project name if this task is
    later turned into a project. ``project_name`` makes this an LLM-first
    "create a named project and work there" call: the project is created NOW
    with that display name and the task runs inside it (v6.33.0).
    """
    goal = str(objective or "").strip()
    if not goal:
        return "⚠️ TOOL_ARG_ERROR (promote_chat_to_task): objective is required"
    from ouroboros.project_facts import (
        explicit_project_id_ok,
        project_id_from_display_name,
        sanitize_project_id,
    )

    display_name = str(project_name or "").strip()
    pid = ""
    if str(project_id or "").strip():
        if not explicit_project_id_ok(project_id):
            return (
                f"⚠️ TOOL_ARG_ERROR (promote_chat_to_task): project_id {project_id!r} is not "
                "filesystem-clean; use lowercase alphanumeric/_/-/. (<=64 chars)"
            )
        pid = sanitize_project_id(project_id)
    elif display_name:
        # LLM-first "create a NAMED project and work there": derive a filesystem
        # id from the display name. A non-ASCII name (e.g. a Cyrillic "динозавры")
        # falls back to a deterministic hash id so the project is still created —
        # the human-readable name rides project_name on the registry.
        pid = project_id_from_display_name(display_name)
    else:
        # No explicit arg: inherit the CURRENT project scope so a project-chat
        # task that promotes follow-up work stays in its own project (the model
        # still chose to promote — scope is contextual, never a keyword gate).
        pid = sanitize_project_id(getattr(ctx, "project_id", "") or "")
    try:
        current_chat_id = int(getattr(ctx, "current_chat_id", None) or 0)
    except (TypeError, ValueError):
        current_chat_id = 0
    tid = uuid.uuid4().hex[:8]
    evt: Dict[str, Any] = {
        "type": "promote_chat_to_task",
        "task_id": tid,
        "objective": goal,
        "expected_output": str(expected_output or "").strip(),
        "project_id": pid,
        "project_name": display_name,
        "title": str(title or "").strip()[:80],
        "workspace_root": str(workspace_root or "").strip(),
        "chat_id": current_chat_id,
        "ts": utc_now_iso(),
    }
    mode = _emit_control_event(ctx, evt)
    if display_name:
        scope_note = f" in new project '{display_name}'"
    elif pid:
        scope_note = f" in project '{pid}'"
    else:
        scope_note = ""
    return (
        f"OK: promoted to supervised task {tid}{scope_note} ({mode}). The conversation "
        "lane stays free; the owner sees a live task card and can steer the running "
        "task from chat (messages are delivered to its mailbox). Use wait_task/"
        "get_task_result to follow up if the result is needed in this conversation."
    )


def _list_projects(ctx: ToolContext, limit: int = 50) -> str:
    """Enumerate the owner's projects (id, name, recency) so the one mind can
    decide whether a main-chat message belongs to an existing project."""
    try:
        from ouroboros.projects_registry import projects_summary
        rows = projects_summary(Path(ctx.drive_root), limit=max(1, min(int(limit or 50), 200)))
    except Exception as exc:
        return f"⚠️ PROJECTS_ERROR: {type(exc).__name__}: {exc}"
    if not rows:
        return "No projects yet. Create one by promoting work with a fresh project_id, or just answer/spawn a task."
    lines = []
    for p in rows:
        pid = str(p.get("id") or "")
        name = str(p.get("name") or pid)
        last = str(p.get("last_active_at") or p.get("created_at") or "")
        active = " · running" if p.get("has_thread_activity") else ""
        lines.append(f"- {pid} — {name}{active}{(' · last ' + last) if last else ''}")
    return "Projects (route a related main-chat message with route_to_project):\n" + "\n".join(lines)


def _route_to_project(ctx: ToolContext, project_id: str, message: str, reason: str = "") -> str:
    """Route a main-chat message to an EXISTING project so the work continues in
    that project's context (its memory/journal/thread), keeping the main chat free.

    LLM-first: the model decides WHEN to route (its judgment is the gate, never a
    keyword rule); this verb just delivers the decision and returns a visible
    receipt. Exactly one owner-visible outcome per call (the routed project turn).
    """
    from ouroboros.project_facts import explicit_project_id_ok, sanitize_project_id
    from ouroboros.projects_registry import get_project

    msg = str(message or "").strip()
    if not msg:
        return "⚠️ TOOL_ARG_ERROR (route_to_project): message is required"
    if not str(project_id or "").strip() or not explicit_project_id_ok(project_id):
        return (
            f"⚠️ TOOL_ARG_ERROR (route_to_project): project_id {project_id!r} is not filesystem-clean. "
            "Call list_projects to see valid ids."
        )
    pid = sanitize_project_id(project_id)
    proj = get_project(Path(ctx.drive_root), pid)
    if not proj:
        return (
            f"⚠️ ROUTE_TARGET_NOT_FOUND: no project '{pid}'. Use list_projects to see existing projects, "
            "answer inline, or promote_chat_to_task(project_id=…) to start project-scoped work."
        )
    try:
        current_chat_id = int(getattr(ctx, "current_chat_id", None) or 0)
    except (TypeError, ValueError):
        current_chat_id = 0
    tid = uuid.uuid4().hex[:8]
    objective = msg if not str(reason or "").strip() else f"{msg}\n\n(routing reason: {str(reason).strip()})"
    evt: Dict[str, Any] = {
        "type": "promote_chat_to_task",
        "task_id": tid,
        "objective": objective,
        "project_id": pid,
        "chat_id": current_chat_id,
        "routed_from_main": True,
        "ts": utc_now_iso(),
    }
    mode = _emit_control_event(ctx, evt)
    name = str(proj.get("name") or pid)
    return (
        f"✉️ Routed to project '{name}' ({pid}) as task {tid} ({mode}). I'll continue there; "
        "this chat stays free for you. Follow-ups you send reach the project task's mailbox."
    )


def _steer_task(ctx: ToolContext, task_id: str, message: str) -> str:
    """Deliver a follow-up/steering message to a task already RUNNING in this chat
    — the agent's OWN choice of target among ``current_chat.running_tasks``.

    When the chat is busy, a new message runs as a short-lived decision turn that
    sees the running tasks of the current chat as structural context and picks the
    one to steer. This verb just transports the message to that task's owner-mailbox
    (the running task drains it at its next safe checkpoint). LLM-first (BIBLE P5):
    the code never decides which task a message belongs to — it only validates the
    transport (task exists, same chat, idempotent delivery) and the supervisor
    performs the mailbox write on the task's active drive. When unsure which task
    (or none) fits, spawn a fresh task with ``promote_chat_to_task`` instead.
    """
    target = str(task_id or "").strip()
    msg = str(message or "").strip()
    if not target:
        return (
            "⚠️ TOOL_ARG_ERROR (steer_task): task_id is required — pick one from "
            "current_chat.running_tasks (or promote_chat_to_task to start new work)."
        )
    if not msg:
        return "⚠️ TOOL_ARG_ERROR (steer_task): message is required."
    try:
        current_chat_id = int(getattr(ctx, "current_chat_id", None) or 0)
    except (TypeError, ValueError):
        current_chat_id = 0
    _md = getattr(ctx, "task_metadata", None)
    client_message_id = str((_md.get("client_message_id") if isinstance(_md, dict) else "") or "").strip()
    evt: Dict[str, Any] = {
        "type": "steer_task",
        "target_task_id": target,
        "message": msg,
        "chat_id": current_chat_id,
        "client_message_id": client_message_id,
        "ts": utc_now_iso(),
    }
    mode = _emit_control_event(ctx, evt)
    return (
        f"✉️ Steering task {target} ({mode}): the message reaches its mailbox at the task's next "
        "checkpoint. If that task has already finished, you'll get a notice — then answer inline "
        "or promote_chat_to_task instead."
    )


def _build_acting_constraint(
    *,
    write_surface: str,
    write_root: str,
    protected_paths_grant: bool,
    external_tool_grants: Any,
    parent_workspace_root: str,
):
    """Validate a mutative-subagent request; return its constraint dict, or an
    error string for the LLM (which can then fall back to a read-only subagent).

    The toggle/surface checks here give the caller immediate feedback. The
    supervisor is the authoritative gate and provisions the self_worktree
    (filling write_root/base_sha) before the child runs.
    """
    from ouroboros.config import get_allow_mutative_subagents
    from ouroboros.contracts.task_constraint import VALID_WRITE_SURFACES

    if write_surface not in VALID_WRITE_SURFACES:
        allowed = ", ".join(sorted(VALID_WRITE_SURFACES))
        return (
            "⚠️ TOOL_ARG_ERROR (schedule_subagent): write_surface must be one of "
            f"{allowed} (or omit it for a read-only subagent)."
        )
    if not get_allow_mutative_subagents():
        return (
            "⚠️ MUTATIVE_SUBAGENTS_DISABLED: the OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS toggle "
            "is not enabled. That toggle is the master gate: an explicit owner true/false "
            "overrides the runtime-mode default, and runtime mode only sets the default when "
            "the toggle is empty (default ON in advanced/pro, OFF in light). Schedule a "
            "read-only subagent (omit write_surface), or have the owner enable "
            "OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS. Note: light blocks only self-repo/control-plane "
            "writes (write_surface=self_worktree), not user/task/project deliverables."
        )
    grants: List[str] = []
    if isinstance(external_tool_grants, (list, tuple)):
        grants = [str(g).strip() for g in external_tool_grants if str(g).strip()]
    resolved_write_root = str(write_root or "").strip()
    if write_surface == "external_workspace" and not resolved_write_root:
        resolved_write_root = str(parent_workspace_root or "").strip()
    if write_surface == "external_workspace" and not resolved_write_root:
        return (
            "⚠️ TOOL_ARG_ERROR (schedule_subagent): write_surface=external_workspace "
            "requires write_root (the external project directory) or a parent workspace."
        )
    return {
        "mode": ACTING_SUBAGENT_MODE,
        "surface": write_surface,
        "write_root": resolved_write_root,
        "protected_paths_grant": protected_paths_grant,
        "external_tool_grants": grants,
        "parent_only_commit": True,
        "return_kind": "workspace_patch",
        "allow_enable": False,
        "allow_review": False,
    }


def _select_subagent_constraint(write_surface, write_root, protected_paths_grant, external_tool_grants, parent_workspace_root, caller_readonly=False):
    """Read-only default (no surface), a validated acting constraint, or an error string."""
    if not write_surface or str(write_surface).strip().lower() == "read_only":
        # `read_only` is the explicit, provider-safe alias for the omit-surface
        # read-only path (the handler also normalizes it; this guard keeps the selector
        # correct for any direct caller and matches the schema enum) — never acting.
        return {"mode": LOCAL_READONLY_SUBAGENT_MODE, "allow_enable": False, "allow_review": False}
    if caller_readonly:
        # A read-only subagent may delegate read-only children only — never spawn an acting one.
        return (
            "⚠️ MUTATIVE_SUBAGENTS_DISABLED: a read-only subagent cannot spawn a mutative (acting) "
            "child. Only the root agent, workspace tasks, or acting subagents may pass write_surface; "
            "schedule a read-only child instead."
        )
    return _build_acting_constraint(
        write_surface=write_surface,
        write_root=write_root,
        protected_paths_grant=protected_paths_grant,
        external_tool_grants=external_tool_grants,
        parent_workspace_root=parent_workspace_root,
    )


def _populate_subagent_event_extras(
    evt: Dict[str, Any], *, current_chat_id: Any, child_drive: Any, workspace_root: str,
    workspace_mode: str, executor_ref: Any, context: str, parent_task_id: str,
) -> None:
    """Add the optional fields of a schedule_subagent event in place (extracted from
    _schedule_task to keep it under the method gate; pure field assignment)."""
    if current_chat_id:
        evt["chat_id"] = current_chat_id
    if child_drive is not None:
        evt["drive_root"] = str(child_drive)
        evt["child_drive_root"] = str(child_drive)
    if workspace_root:
        evt["workspace_root"] = workspace_root
    if workspace_mode:
        evt["workspace_mode"] = workspace_mode
    if executor_ref:
        evt["executor_ref"] = executor_ref
        evt["metadata"] = {**(evt.get("metadata") if isinstance(evt.get("metadata"), dict) else {}), "executor_ref": executor_ref}
    if context:
        evt["context"] = context
    if parent_task_id:
        evt["parent_task_id"] = parent_task_id


def _prepare_child_drives(slot_tasks, task_ids, status_drive_root, memory_mode, parent_project_id):
    """Prepare forked/empty child drives for a scheduled wave. On any failure, clean up
    every child drive + task-state dir and return ``(drives, error_string)``; otherwise
    ``(drives, "")``. (Extracted from _schedule_task to keep it under the method gate.)"""
    child_drives: Dict[str, Path] = {}
    if memory_mode not in {"forked", "empty"}:
        return child_drives, ""
    for tid, _slot in slot_tasks:
        try:
            child_drives[tid] = prepare_task_drive(status_drive_root, tid, memory_mode, project_id=parent_project_id)
        except Exception as exc:
            for child_drive in child_drives.values():
                shutil.rmtree(child_drive, ignore_errors=True)
            for cleanup_tid in task_ids:
                shutil.rmtree(task_state_dir(status_drive_root, cleanup_tid), ignore_errors=True)
            log.warning("Failed to prepare child drive for subtask %s", tid, exc_info=True)
            return child_drives, f"⚠️ SUBTASK_DRIVE_ERROR: failed to prepare {memory_mode} child drive: {exc}"
    return child_drives, ""


def _resolve_executor_ref(ctx: Any) -> dict:
    """The child's workspace executor reference (docker/host), or {} when unavailable."""
    accessor = getattr(ctx, "workspace_executor_ref", None)
    if callable(accessor):
        try:
            candidate = accessor()
            if isinstance(candidate, dict) and candidate:
                return dict(candidate)
        except Exception:
            return {}
    return {}


def _schedule_task(
    ctx: ToolContext,
    objective: str = "",
    expected_output: str = "",
    role: str = "",
    context: str = "",
    constraints: str = "",
    memory_mode: str = "forked",
    model_lane: str = "auto",
    write_surface: str = "",
    write_root: str = "",
    protected_paths_grant: bool = False,
    external_tool_grants: Any = None,
    delegation_intent: str = "",
    may_mutate: bool = False,
    may_fan_out: bool = True,
    max_children: int = 0,
    **legacy_or_unknown: Any,
) -> str:
    if legacy_or_unknown:
        bad = ", ".join(sorted(str(key) for key in legacy_or_unknown.keys()))
        return (
            "⚠️ TOOL_ARG_ERROR (schedule_subagent): unsupported argument(s): "
            f"{bad}. Use the v6 strict schema: objective, expected_output, "
            "optional role/context/constraints/memory_mode/model_lane and (for "
            "mutative children) write_surface/write_root/protected_paths_grant/"
            "external_tool_grants."
        )
    objective = str(objective or "").strip()
    expected_output = str(expected_output or "").strip()
    role = str(role or "researcher").strip() or "researcher"
    context = str(context or "").strip()
    constraints = str(constraints or "").strip()
    memory_mode = str(memory_mode or "forked").strip().lower()
    try:
        requested_model_lane = normalize_subagent_model_lane(model_lane)
    except ValueError as exc:
        return f"⚠️ TOOL_ARG_ERROR (schedule_subagent): {exc}."
    if not objective:
        return "⚠️ TOOL_ARG_ERROR (schedule_subagent): objective is required."
    if not expected_output:
        return "⚠️ TOOL_ARG_ERROR (schedule_subagent): expected_output is required."
    if memory_mode not in VALID_SUBTASK_MEMORY_MODES:
        allowed = ", ".join(sorted(VALID_SUBTASK_MEMORY_MODES))
        return (
            f"⚠️ TOOL_ARG_ERROR (schedule_subagent): memory_mode must be one of: {allowed}. "
            "memory_mode=shared is disabled for live local subagents until a sanitized shared-context mode exists."
        )

    try:
        current_depth = int(getattr(ctx, 'task_depth', 0) or 0)
    except (TypeError, ValueError):
        current_depth = 0
    new_depth = current_depth + 1
    max_depth = get_max_subagent_depth()
    if new_depth > max_depth:
        return f"ERROR: Subtask depth limit ({max_depth}) exceeded. Simplify your approach."

    if getattr(ctx, 'is_direct_chat', False):
        from ouroboros.utils import append_jsonl
        try:
            append_jsonl(ctx.drive_logs() / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "schedule_task_from_direct_chat",
                "description": objective[:200],
                "warning": "schedule_subagent called from direct chat context — potential duplicate work",
            })
        except Exception:
            pass

    metadata = getattr(ctx, "task_metadata", {}) if isinstance(getattr(ctx, "task_metadata", {}), dict) else {}
    parent_contract = (
        getattr(ctx, "task_contract", {})
        if isinstance(getattr(ctx, "task_contract", {}), dict)
        else metadata.get("task_contract") if isinstance(metadata.get("task_contract"), dict)
        else {}
    )
    current_task_id = str(getattr(ctx, "task_id", "") or "")
    parent_task_id = str(current_task_id or metadata.get("parent_task_id") or "").strip()
    root_task_id_seed = str(metadata.get("root_task_id") or current_task_id or "").strip()
    session_id = str(metadata.get("session_id") or "")
    try:
        current_chat_id = int(getattr(ctx, "current_chat_id", None) or 0)
    except (TypeError, ValueError):
        current_chat_id = 0
    budget_drive_root = str(metadata.get("budget_drive_root") or getattr(ctx, "budget_drive_root", "") or ctx.drive_root)
    status_drive_root = Path(budget_drive_root)
    workspace_root = str(getattr(ctx, "workspace_root", "") or metadata.get("workspace_root") or "").strip()
    workspace_mode = str(getattr(ctx, "workspace_mode", "") or metadata.get("workspace_mode") or "").strip()
    # Subagents inherit the parent's resolved project scope so their context reads
    # the same per-project knowledge (Phase 3b); never re-derive a different id.
    parent_project_id = str(getattr(ctx, "project_id", "") or "").strip()
    requested_surface = str(write_surface or "").strip().lower()
    # `read_only` is a first-class, provider-safe alias for "omit write_surface" (NOT a
    # VALID_WRITE_SURFACES acting surface) — normalize it to the read-only path so
    # constraint selection, mutating detection, and the event all treat it as read-only (P5).
    if requested_surface == "read_only":
        requested_surface = ""
    from ouroboros.tool_access import active_tool_profile
    task_constraint = _select_subagent_constraint(
        requested_surface, write_root, protected_paths_grant, external_tool_grants, workspace_root,
        caller_readonly=(active_tool_profile(ctx) == "local_readonly_subagent"))
    if isinstance(task_constraint, str):
        return task_constraint
    allowed_resources = normalize_allowed_resources(
        (parent_contract.get("allowed_resources") if isinstance(parent_contract, dict) else {})
        or metadata.get("allowed_resources")
        or {}
    )
    executor_ref = _resolve_executor_ref(ctx)
    # A writing/mutative child routes an `auto` lane to Heavy; a read-only child to
    # Light. Use the NORMALIZED surface so `read_only` counts as read-only, not acting.
    child_mutating = bool(requested_surface) or normalize_bool(may_mutate)
    lane_slots = expand_subagent_lane_slots(requested_model_lane, depth=new_depth, mutating=child_mutating)
    if not lane_slots:
        return "⚠️ SUBTASK_STATUS_ERROR: no subagent lane slots resolved; subagent was not scheduled."
    _lane_downgrade_notes = [s.downgrade_note for s in lane_slots if s.downgrade_note]
    slot_tasks = [(uuid.uuid4().hex[:8], slot) for slot in lane_slots]
    task_ids: List[str] = [task_id for task_id, _slot in slot_tasks]
    emitted_modes: List[str] = []
    task_group_id = (
        f"subagents-{uuid.uuid4().hex[:8]}"
        if requested_model_lane in {"review", "scope"} or len(lane_slots) > 1
        else ""
    )
    task_group = compact_task_group(
        group_id=task_group_id,
        task_ids=task_ids,
        requested_lane=requested_model_lane,
        parent_task_id=parent_task_id,
        root_task_id=root_task_id_seed,
        role=role,
    ) if task_group_id else {}
    child_drives, _drive_err = _prepare_child_drives(
        slot_tasks, task_ids, status_drive_root, memory_mode, parent_project_id
    )
    if _drive_err:
        return _drive_err

    # C3.1: propagate the parent's delegation INTENT to the child structurally (typed
    # budget); only ever NARROWS within the parent (depth/active caps stay enforced).
    child_delegation_budget = child_budget_for_schedule(
        parent_contract,
        current_depth=current_depth, new_depth=new_depth, max_depth=max_depth,
        may_mutate=may_mutate, may_fan_out=may_fan_out, max_children=max_children,
        intent_note=delegation_intent,
    )

    events_to_emit: List[Dict[str, Any]] = []
    for tid, slot in slot_tasks:
        root_task_id = root_task_id_seed or tid
        slot_role = role
        if slot.slot_count > 1:
            slot_role = f"{role}:slot-{slot.slot_index + 1}"
        child_drive = child_drives.get(tid)

        child_contract = build_task_contract({
            "id": tid,
            "type": "task",
            "description": objective,
            "objective": objective,
            "expected_output": expected_output,
            "constraints": constraints,
            "workspace_root": workspace_root,
            "workspace_mode": workspace_mode,
            "project_id": parent_project_id,
            "allowed_resources": allowed_resources,
            "deadline_at": parent_contract.get("deadline_at") if isinstance(parent_contract, dict) else "",
            "parent_task_id": parent_task_id,
            "root_task_id": root_task_id,
            "session_id": session_id,
            "delegation_role": "subagent",
            "metadata": {
                "task_contract": {
                    **parent_contract,
                    "source": "parent_delegation",
                    "objective": objective,
                    "expected_output": expected_output,
                    "constraints": constraints,
                    "delegation_budget": child_delegation_budget,
                } if isinstance(parent_contract, dict) else {"delegation_budget": child_delegation_budget},
            },
        })
        envelope = build_subagent_envelope(
            task_id=tid,
            parent_task_id=parent_task_id,
            root_task_id=root_task_id,
            task_group_id=task_group_id,
            depth=new_depth,
            role=slot_role,
            requested_lane=slot.requested_lane,
            effective_lane=slot.effective_lane,
            model=slot.model,
            status=STATUS_REQUESTED,
        )
        evt = {
            "type": "schedule_subagent",
            "description": objective,
            "objective": objective,
            "expected_output": expected_output,
            "constraints": constraints,
            "role": slot_role,
            "task_id": tid,
            "depth": new_depth,
            "ts": utc_now_iso(),
            "root_task_id": root_task_id,
            "session_id": session_id,
            "actor_id": f"subagent:{slot_role}",
            "delegation_role": "subagent",
            "memory_mode": memory_mode,
            "project_id": parent_project_id,
            "budget_drive_root": budget_drive_root,
            "task_constraint": task_constraint,
            "write_surface": requested_surface,
            "task_contract": child_contract,
            "allowed_resources": allowed_resources,
            "model_lane": slot.requested_lane,
            "requested_model_lane": slot.requested_lane,
            "effective_model_lane": slot.effective_lane,
            "model": slot.model,
            "use_local_model": slot.use_local_model,
            "task_group_id": task_group_id,
            "task_group": task_group,
            "subagent_envelope": envelope,
        }
        _populate_subagent_event_extras(
            evt, current_chat_id=current_chat_id, child_drive=child_drive,
            workspace_root=workspace_root, workspace_mode=workspace_mode,
            executor_ref=executor_ref, context=context, parent_task_id=parent_task_id,
        )
        try:
            write_task_result(
                status_drive_root,
                tid,
                STATUS_REQUESTED,
                parent_task_id=parent_task_id or None,
                root_task_id=root_task_id,
                session_id=session_id,
                actor_id=f"subagent:{slot_role}",
                delegation_role="subagent",
                project_id=parent_project_id,
                role=slot_role,
                description=objective,
                objective=objective,
                expected_output=expected_output,
                constraints=constraints,
                context=context,
                workspace_root=workspace_root,
                workspace_mode=workspace_mode,
                executor_ref=executor_ref,
                allowed_resources=allowed_resources,
                task_contract=child_contract,
                chat_id=current_chat_id or None,
                memory_mode=memory_mode,
                drive_root=str(child_drive) if child_drive is not None else "",
                child_drive_root=str(child_drive) if child_drive is not None else "",
                budget_drive_root=budget_drive_root,
                task_constraint=task_constraint,
                model_lane=slot.requested_lane,
                requested_model_lane=slot.requested_lane,
                effective_model_lane=slot.effective_lane,
                model=slot.model,
                use_local_model=slot.use_local_model,
                task_group_id=task_group_id,
                task_group=task_group,
                subagent_envelope=envelope,
                result="Subagent request queued. Awaiting supervisor acceptance.",
            )
        except Exception:
            log.warning("Failed to persist requested task status for %s", tid, exc_info=True)
            for cleanup_tid in task_ids:
                try:
                    (status_drive_root / "task_results" / f"{cleanup_tid}.json").unlink(missing_ok=True)
                except Exception:
                    pass
            for child_drive in child_drives.values():
                shutil.rmtree(child_drive, ignore_errors=True)
            return f"⚠️ SUBTASK_STATUS_ERROR: failed to persist requested status for {tid}; subagent was not scheduled."
        events_to_emit.append(evt)

    for evt in events_to_emit:
        emitted_modes.append(_emit_control_event(ctx, evt))

    _schedule_result = _finalize_schedule_emission(
        ctx,
        task_ids=task_ids,
        task_group_id=task_group_id,
        requested_model_lane=requested_model_lane,
        task_group=task_group,
        objective=objective,
        role=role,
        depth=new_depth,
        parent_task_id=parent_task_id,
        root_task_id=root_task_id_seed or current_task_id,
        slot_tasks=slot_tasks,
        emitted_modes=emitted_modes,
    )
    if _lane_downgrade_notes and isinstance(_schedule_result, str):  # P1: not a silent horizon cut
        _schedule_result += "\n⚠️ " + "; ".join(dict.fromkeys(_lane_downgrade_notes))
    return _schedule_result


def _request_deep_self_review(ctx: ToolContext, reason: str) -> str:
    from ouroboros.deep_self_review import is_review_available
    available, model = is_review_available()
    if not available:
        return (
            "❌ Deep self-review unavailable: configure OUROBOROS_MODEL_DEEP_SELF_REVIEW "
            "and the matching provider API key."
        )
    ctx.pending_events.append({"type": "deep_self_review_request", "reason": reason, "model": model, "ts": utc_now_iso()})
    return f"Deep self-review requested (model: {model}). It will be queued and executed asynchronously."


def _chat_history(ctx: ToolContext, count: int = 100, offset: int = 0, search: str = "") -> str:
    from ouroboros.memory import Memory
    mem = Memory(drive_root=ctx.drive_root)
    # Full project awareness (v6.32.0): the one mind's active recall spans every
    # thread (main + projects). The project-task working FOCUS is applied to the
    # passive default context only, never to this deliberate recall tool.
    return mem.chat_history(count=count, offset=offset, search=search)


def _update_scratchpad(ctx: ToolContext, content: str) -> str:
    """LLM-driven scratchpad update — appends a timestamped block (Constitution P5: LLM-first)."""
    if str(getattr(ctx, "project_id", "") or "").strip():
        # Project-scoped tasks have no per-project scratchpad and must never write
        # the canonical scratchpad (outbound isolation). Persist project facts via
        # knowledge_write instead (routed to the per-project store).
        return ("OK: scratchpad is not used for project-scoped tasks (no per-project "
                "scratchpad). Persist durable project facts with knowledge_write.")
    if not content or not isinstance(content, str) or len(content.strip()) < 10:
        return (
            "⚠️ REJECTED: content is empty or too short "
            f"(got {type(content).__name__}, len={len(content) if isinstance(content, str) else 'N/A'}). "
            "Scratchpad must have meaningful content (10+ chars). "
            "This likely means the tool call was malformed — check your arguments."
        )
    from ouroboros.memory import Memory
    mem = Memory(drive_root=ctx.drive_root)
    mem.ensure_files()
    try:
        block = mem.append_scratchpad_block(
            content,
            source="task",
            metadata={
                "task_id": str(getattr(ctx, "task_id", "") or ""),
                "task_type": str(getattr(ctx, "current_task_type", "") or ""),
                "delegation_role": str((getattr(ctx, "task_metadata", {}) or {}).get("delegation_role", "")) if isinstance(getattr(ctx, "task_metadata", {}), dict) else "",
            },
        )
    except RuntimeError as exc:
        if "LEGACY_SCRATCHPAD_REQUIRES_MANUAL_UPGRADE" in str(exc):
            return f"⚠️ {exc}"
        raise
    return f"OK: scratchpad block appended ({len(content)} chars, ts={block.get('ts', '?')[:16]})"


def _send_user_message(ctx: ToolContext, text: str, reason: str = "") -> str:
    """Send a proactive message to the user (not as reply to a task).

    Use when you have something genuinely worth saying — an insight,
    a question, a status update, or an invitation to collaborate.
    """
    if not ctx.current_chat_id:
        return "⚠️ No active chat — cannot send proactive message."
    if not text or not text.strip():
        return "⚠️ Empty message."

    from ouroboros.utils import append_jsonl
    ctx.pending_events.append({
        "type": "send_message",
        "chat_id": ctx.current_chat_id,
        "text": text,
        "format": "markdown",
        "is_progress": False,
        "ts": utc_now_iso(),
    })
    append_jsonl(ctx.drive_logs() / "events.jsonl", {
        "ts": utc_now_iso(),
        "type": "proactive_message",
        "reason": reason,
        "text_preview": text[:200],
    })
    return "OK: message queued for delivery."


def _update_identity(ctx: ToolContext, content: str) -> str:
    """Update identity manifest (who you are, who you want to become)."""
    if str(getattr(ctx, "project_id", "") or "").strip():
        # Identity is global and continuous (P1); it is never modified from a
        # project-scoped task. There is no per-project identity.
        return ("OK: identity is global and is never modified from a project-scoped "
                "task (identity stays continuous across projects — P1).")
    if not content or not isinstance(content, str) or len(content.strip()) < 50:
        return (
            "⚠️ REJECTED: content is empty or too short "
            f"(got {type(content).__name__}, len={len(content) if isinstance(content, str) else 'N/A'}). "
            "Identity must be a substantial text (50+ chars). "
            "This likely means the tool call was malformed — check your arguments."
        )
    from ouroboros.memory import Memory
    mem = Memory(drive_root=ctx.drive_root)
    mem.ensure_files()

    old_content = ""
    path = ctx.drive_root / "memory" / "identity.md"
    if path.exists():
        try:
            old_content = path.read_text(encoding="utf-8")
        except Exception:
            pass

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    append_jsonl(mem.identity_journal_path(), {
        "ts": utc_now_iso(),
        "task_id": str(getattr(ctx, "task_id", "") or ""),
        "source_type": str((getattr(ctx, "task_metadata", {}) or {}).get("delegation_role", "task")) if isinstance(getattr(ctx, "task_metadata", {}), dict) else "task",
        "old_len": len(old_content),
        "new_len": len(content),
        "old_sha256": sha256(old_content.encode("utf-8")).hexdigest() if old_content else "",
        "new_sha256": sha256(content.encode("utf-8")).hexdigest(),
        "old_content": old_content,
        "new_content": content,
        "old_preview": old_content[:500],
        "new_preview": content[:500],
    })

    result = f"OK: identity updated ({len(content)} chars)"
    old_len = len(old_content)
    if old_len >= 400 and len(content) < old_len * 0.5:
        result += (
            f"\n⚠️ SELF_OVERWRITE_NOTICE: this replaced a {old_len}-char identity with "
            f"{len(content)} chars (>50% shrink). Identity is intentionally mutable (Bible P4), "
            "but full rewrites should be rare and reflect genuine self-creation — not a trivial turn. "
            "Read before writing (P12) and prefer evolving over replacing wholesale."
        )
    return result


def _toggle_evolution(ctx: ToolContext, enabled: bool, objective: str = "") -> str:
    """Toggle evolution mode on/off via supervisor event."""
    if bool(enabled):
        # Reflect the light-mode hard block in the tool's own result so the agent
        # is not told "ON" while the supervisor silently refuses it.
        try:
            from supervisor.evolution_lifecycle import evolution_block_reason

            block = evolution_block_reason()
        except Exception:
            block = ""
        if block:
            return block
    ctx.pending_events.append({
        "type": "toggle_evolution",
        "enabled": bool(enabled),
        "objective": str(objective or "").strip(),
        "ts": utc_now_iso(),
    })
    state_str = "ON" if enabled else "OFF"
    return f"OK: evolution mode toggled {state_str}."


def _toggle_consciousness(ctx: ToolContext, action: str = "status") -> str:
    """Control background consciousness: start, stop, or status."""
    ctx.pending_events.append({
        "type": "toggle_consciousness",
        "action": action,
        "ts": utc_now_iso(),
    })
    return f"OK: consciousness '{action}' requested."


def _switch_model(ctx: ToolContext, model: str = "", effort: str = "") -> str:
    """LLM-driven model/effort switch (Constitution P5: LLM-first).

    Stored in ToolContext, applied on the next LLM call in the loop.
    """
    from ouroboros.llm import LLMClient, normalize_reasoning_effort
    available = LLMClient().available_models()
    changes = []

    if model:
        if model not in available:
            return f"⚠️ Unknown model: {model}. Available: {', '.join(available)}"

        import os
        use_local = False
        if model == os.environ.get("OUROBOROS_MODEL") and os.environ.get("USE_LOCAL_MAIN", "").lower() in ("true", "1"):
            use_local = True
        elif model == os.environ.get("OUROBOROS_MODEL_HEAVY") and os.environ.get("USE_LOCAL_HEAVY", "").lower() in ("true", "1"):
            use_local = True
        elif model == os.environ.get("OUROBOROS_MODEL_LIGHT") and os.environ.get("USE_LOCAL_LIGHT", "").lower() in ("true", "1"):
            use_local = True
        else:
            from ouroboros.config import get_fallback_models
            if model in get_fallback_models() and os.environ.get("USE_LOCAL_FALLBACK", "").lower() in ("true", "1"):
                use_local = True

        # CW2 (v6.34.0): the per-round re-gate downgrades the MODE for a sub-1M switch, but
        # the already-built transcript still carries the max-mode reference docs — switching
        # to a route that can't confirm >=1M would send that max-sized prompt to a smaller
        # window. Refuse the switch while the transcript is max-sized (BIBLE P1).
        if str(getattr(ctx, "active_context_mode", "") or "") == "max":
            try:
                from ouroboros.gateway.settings import _active_route_confirms_max
                if not _active_route_confirms_max(model=model, use_local=use_local):
                    return (
                        f"⚠️ SWITCH_BLOCKED: '{model}' does not confirm a >=1M context window, but the "
                        "current transcript was built in Max context mode and would overflow it. Pick a "
                        ">=1M route, or have the owner lower context mode to Low before switching."
                    )
            except Exception:
                # Fail CLOSED: an errored capability check must not let a max-sized transcript
                # switch to a possibly-sub-1M route (BIBLE P1 cognitive horizon).
                log.debug("CW2 switch_model capability guard errored; failing closed", exc_info=True)
                return (
                    f"⚠️ SWITCH_BLOCKED: couldn't verify whether '{model}' confirms a >=1M window while "
                    "the transcript is max-sized — failing closed. Retry, pick a known >=1M route, or have "
                    "the owner lower context mode to Low first."
                )

        ctx.active_model_override = model
        ctx.active_use_local_override = use_local
        changes.append(f"model={model}{' (local)' if use_local else ''}")

    if effort:
        normalized = normalize_reasoning_effort(effort, default="medium")
        ctx.active_effort_override = normalized
        changes.append(f"effort={normalized}")

    if not changes:
        return f"Current available models: {', '.join(available)}. Pass model and/or effort to switch."

    return f"OK: switching to {', '.join(changes)} on next round."


def _get_task_result(ctx: ToolContext, task_id: str) -> str:
    """Read the effective result of a registered subtask."""
    metadata = getattr(ctx, "task_metadata", {}) if isinstance(getattr(ctx, "task_metadata", {}), dict) else {}
    status_drive_root = Path(str(metadata.get("budget_drive_root") or getattr(ctx, "budget_drive_root", "") or ctx.drive_root))
    data = load_effective_task_result(status_drive_root, task_id)
    if not data:
        return f"Task {task_id}: unknown or not yet registered"
    status = data.get("status", "unknown")
    result = data.get("result", "")
    cost = data.get("cost_usd", 0)
    trace = data.get("trace_summary", "")
    outcome_summary = _subtask_outcome_summary(data)
    if status == STATUS_COMPLETED:
        output = (
            f"Task {task_id} [{status}]: cost=${cost:.2f}\n\n"
            f"[SUBTASK_OUTCOME]\n{outcome_summary}\n[/SUBTASK_OUTCOME]\n\n"
            f"[BEGIN_SUBTASK_OUTPUT]\n{result}\n[END_SUBTASK_OUTPUT]"
        )
    elif status == STATUS_REJECTED_DUPLICATE:
        duplicate_of = str(data.get("duplicate_of") or "?")
        output = (
            f"Task {task_id} [{status}]: duplicate_of={duplicate_of}\n\n"
            f"[SUBTASK_OUTCOME]\n{outcome_summary}\n[/SUBTASK_OUTCOME]\n\n"
            f"{result or f'Task was rejected as a duplicate of {duplicate_of}.'}"
        )
    else:
        output = (
            f"Task {task_id} [{status}]\n\n"
            f"[SUBTASK_OUTCOME]\n{outcome_summary}\n[/SUBTASK_OUTCOME]\n\n"
            f"{result or 'No details available.'}"
        )
    if trace:
        output += f"\n\n[SUBTASK_TRACE]\n{trace}\n[/SUBTASK_TRACE]"
    return output


def _wait_attention_poll(ctx: ToolContext, after_ts: str) -> Callable[..., Any]:
    """on_poll hook: break a sliced wait early when a child appends an attention beacon
    (blocker/question/interface_contract) after the wait started, so a waiting parent reacts mid-flight."""
    # tree_note/tree_read live in ouroboros/tools/task_tree.py (extracted for module size).
    from ouroboros.tools.task_tree import tree_root_id

    rid = tree_root_id(ctx)

    def _hook(_results: Dict[str, Any], _terminal: Dict[str, bool]) -> Any:
        if not rid:
            return None
        try:
            from ouroboros.task_tree_ledger import tree_ledger_attention_after

            att = tree_ledger_attention_after(rid, after_ts)
        except Exception:
            return None
        return {"reason": "child_attention_beacon", "beacons": att[-5:]} if att else None

    return _hook


def _wait_for_task(ctx: ToolContext, task_id: str, timeout_sec: int = 180) -> str:
    """Wait for a subtask to reach a terminal status."""
    try:
        tid = validate_task_id(task_id)
    except ValueError as exc:
        return f"⚠️ TOOL_ARG_ERROR (wait_task): {exc}"
    try:
        timeout = max(0, min(int(timeout_sec), 3600))
    except (TypeError, ValueError):
        timeout = 180
    metadata = getattr(ctx, "task_metadata", {}) if isinstance(getattr(ctx, "task_metadata", {}), dict) else {}
    status_drive_root = Path(str(metadata.get("budget_drive_root") or getattr(ctx, "budget_drive_root", "") or ctx.drive_root))
    waited = wait_for_effective_tasks(
        status_drive_root, [tid], timeout_sec=timeout,
        on_poll=_wait_attention_poll(ctx, utc_now_iso()), poll_interval_sec=2.0,
    )
    early = waited.get("early_return")
    if early:
        header = "Task wait interrupted by a child attention beacon"
        extra = f"\n\n[CHILD_BEACONS]\n{json.dumps(early, ensure_ascii=False, indent=2)}\n[/CHILD_BEACONS]"
    else:
        header = "Task wait completed" if waited.get("all_terminal") else "Task wait timed out"
        extra = ""
    return f"{header} after {waited.get('elapsed_sec', 0):.1f}s.{extra}\n\n{_get_task_result(ctx, tid)}"


def _wait_for_tasks(
    ctx: ToolContext,
    task_ids: List[str],
    timeout_sec: int = 600,
    mode: str = "all_terminal",
) -> str:
    """Wait for multiple subtasks and return their full effective results."""
    if not isinstance(task_ids, list) or not task_ids:
        return "⚠️ TOOL_ARG_ERROR (wait_tasks): task_ids must be a non-empty list."
    if len(task_ids) > 50:
        return "⚠️ TOOL_ARG_ERROR (wait_tasks): task_ids is capped at 50."
    normalized_ids: List[str] = []
    for item in task_ids:
        try:
            tid = validate_task_id(item)
        except ValueError as exc:
            return f"⚠️ TOOL_ARG_ERROR (wait_tasks): {exc}"
        if tid not in normalized_ids:
            normalized_ids.append(tid)
    try:
        timeout = max(0, min(int(timeout_sec), 7200))
    except (TypeError, ValueError):
        timeout = 600
    normalized_mode = str(mode or "all_terminal").strip().lower()
    if normalized_mode not in {"all_terminal", "any_terminal"}:
        return "⚠️ TOOL_ARG_ERROR (wait_tasks): mode must be all_terminal or any_terminal."
    metadata = getattr(ctx, "task_metadata", {}) if isinstance(getattr(ctx, "task_metadata", {}), dict) else {}
    status_drive_root = Path(str(metadata.get("budget_drive_root") or getattr(ctx, "budget_drive_root", "") or ctx.drive_root))
    waited = wait_for_effective_tasks(
        status_drive_root, normalized_ids, timeout_sec=timeout, mode=normalized_mode,
        on_poll=_wait_attention_poll(ctx, utc_now_iso()), poll_interval_sec=2.0,
    )
    tasks = waited.get("tasks")
    if isinstance(tasks, dict):
        public_tasks: Dict[str, Any] = {}
        for tid, data in tasks.items():
            if not isinstance(data, dict):
                public_tasks[str(tid)] = data
                continue
            public_tasks[str(tid)] = public_task_result(data)
        waited["tasks"] = public_tasks
    return json.dumps(waited, ensure_ascii=False, indent=2)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("set_tool_timeout", {
            "name": "set_tool_timeout",
            "description": "Update the global tool timeout in settings.json and apply it immediately without restart.",
            "parameters": {"type": "object", "properties": {
                "seconds": {"type": "integer", "description": "New timeout in seconds (>= 1)"},
            }, "required": ["seconds"]},
        }, _set_tool_timeout),
        ToolEntry("request_restart", {
            "name": "request_restart",
            "description": "Ask supervisor to restart runtime (after a reviewed local commit or clean no-op state).",
            "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]},
        }, _request_restart),
        ToolEntry("promote_to_stable", {
            "name": "promote_to_stable",
            "description": "Promote ouroboros -> ouroboros-stable. Call when you consider the code stable.",
            "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]},
        }, _promote_to_stable),
        ToolEntry("promote_chat_to_task", {
            "name": "promote_chat_to_task",
            "description": (
                "Promote real work out of this conversation into a supervised pooled task "
                "with a live card (the conversation lane stays free for the owner). Use it "
                "whenever a chat request needs tools/files/multi-step work rather than a "
                "conversational answer. Always give a short `title` (the card's name). To "
                "CREATE A NEW NAMED PROJECT and do the work there (owner asked to 'create a "
                "project called X and …'), set `project_name` — the project is created now "
                "and this task runs inside it (my own judgment: the owner's phrasing is intent, "
                "not a keyword trigger — I name the project from what they actually want it "
                "called, and do not just answer or spawn a project-less task). `project_id` "
                "scopes to an existing project; "
                "`workspace_root` points at a working folder. Owner follow-ups reach the "
                "running task via its mailbox."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "objective": {"type": "string", "description": "What the task must accomplish."},
                    "title": {"type": "string", "description": "A short human name for this task/card (<=80 chars, e.g. 'Tic-tac-toe game'). Reused as the project name if the owner later turns the card into a project — so coin a clean, concise one.", "default": ""},
                    "project_name": {"type": "string", "description": "Set ONLY to create a brand-new NAMED project now and run this task inside it (e.g. 'airi research'). The display name; a filesystem id is derived from it.", "default": ""},
                    "expected_output": {"type": "string", "description": "What done looks like.", "default": ""},
                    "project_id": {"type": "string", "description": "Optional EXISTING project scope (filesystem-clean id).", "default": ""},
                    "workspace_root": {"type": "string", "description": "Optional absolute working-folder path.", "default": ""},
                },
                "required": ["objective"],
            },
        }, _promote_chat_to_task),
        ToolEntry("ensure_project_scope", {
            "name": "ensure_project_scope",
            "description": (
                "Create (or attach to) a named Ouroboros PROJECT and scope THE CURRENT running "
                "task into it. Use this when you are ALREADY working a task and realize it should "
                "be a named project (the owner asked to 'create a project called X', or the work "
                "has grown into a real deliverable) — instead of a bare filesystem mkdir. Unlike "
                "promote_chat_to_task (which creates a NEW task in a project), this binds the task "
                "you are in: its journal_write and per-project knowledge start working, and its "
                "live progress routes to the project thread. Idempotent for the same project; it "
                "will NOT re-scope a task that already belongs to a different project."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project_name": {"type": "string", "description": "Display name for a NEW project (a filesystem id is derived from it). Honor the owner's stated name.", "default": ""},
                    "project_id": {"type": "string", "description": "Optional EXISTING project id (filesystem-clean) to attach to instead of creating one.", "default": ""},
                },
                "required": [],
            },
        }, _ensure_project_scope),
        ToolEntry("list_projects", {
            "name": "list_projects",
            "description": (
                "List the owner's projects (id, name, recency, running flag) — read-only. "
                "Use it in a main-chat turn to decide whether a message belongs to an existing "
                "project, then route it there with route_to_project."
            ),
            "parameters": {"type": "object", "properties": {
                "limit": {"type": "integer", "default": 50, "description": "Max projects to list."},
            }},
        }, _list_projects),
        ToolEntry("route_to_project", {
            "name": "route_to_project",
            "description": (
                "Route a main-chat message to an EXISTING project so the work continues in that "
                "project's own context (memory/journal/thread), keeping the main chat free. Use "
                "when a message clearly belongs to a known project (call list_projects first if "
                "unsure of the id). If confidence is low or several projects could match, do NOT "
                "route silently — answer inline and offer to route. For brand-new work that is not "
                "yet a project, use promote_chat_to_task instead. Returns a visible routing receipt."
            ),
            "parameters": {"type": "object", "properties": {
                "project_id": {"type": "string", "description": "Target project id (filesystem-clean; see list_projects)."},
                "message": {"type": "string", "description": "The owner message / work to route into the project."},
                "reason": {"type": "string", "default": "", "description": "Optional short why-this-project note (provenance)."},
            }, "required": ["project_id", "message"]},
        }, _route_to_project),
        ToolEntry("steer_task", {
            "name": "steer_task",
            "description": (
                "Deliver a follow-up/steering message to a task ALREADY RUNNING in this chat — YOU "
                "pick which one from current_chat.running_tasks (the runtime context lists each running "
                "task's id + objective). Use it when a new message continues or redirects a task already "
                "in flight, instead of spawning a duplicate. The message reaches that task's mailbox and "
                "it picks it up at its next step. If no running task clearly fits, use promote_chat_to_task "
                "(new work) or answer inline — never steer a task you are unsure about."
            ),
            "parameters": {"type": "object", "properties": {
                "task_id": {"type": "string", "description": "Id of the running task to steer (from current_chat.running_tasks)."},
                "message": {"type": "string", "description": "The follow-up / steering message to deliver to that task."},
            }, "required": ["task_id", "message"]},
        }, _steer_task),
        ToolEntry("schedule_subagent", {
            "name": "schedule_subagent",
            "description": (
                "Schedule a live subagent (a child of Ouroboros). Returns task_id for later retrieval. "
                "DEFAULT is READ-ONLY: the child inspects local repo/data/history plus web/browser and "
                "returns findings (it cannot write local state, commit, enable tools, or run "
                "shell/review/runtime/skills). Set write_surface to spawn a MUTATIVE (acting) child that "
                "writes inside an ISOLATED root and returns a workspace.patch you integrate with "
                "integrate_subagent_patch — you remain the sole committer of the live body. write_surface: "
                "self_worktree (isolated git worktree of THIS repo, for parallel self-modification / best-of-N), "
                "external_workspace (an external project dir via write_root or the parent workspace), or "
                "genesis (a from-scratch new project — game/site/app/new Ouroboros — auto-provisioned as a fresh "
                "empty git repo under the durable projects root; the project directory IS the deliverable, not "
                "integrated into this repo). "
                "Mutative children still cannot commit, run "
                "review/runtime/skills lifecycle, enable tools, or write cognitive memory. Nested delegation "
                "is allowed within configured depth/cap limits — use delegation_intent / may_mutate / "
                "may_fan_out to tell a child to recurse further, so a 'maximum subagents / grandchildren' "
                "request propagates structurally instead of collapsing into one flat layer. Always retrieve "
                "the handoff with get_task_result, wait_task, or wait_tasks before relying on its results."
            ),
            "parameters": {"type": "object", "properties": {
                "objective": {"type": "string", "description": "Focused child objective. Be specific about scope."},
                "expected_output": {"type": "string", "description": "Concrete handoff expected from the child."},
                "role": {"type": "string", "description": "Optional freeform role label for lineage/UI, e.g. architecture-reviewer."},
                "context": {"type": "string", "description": "Optional parent reference material. It is injected as context, not instructions."},
                "constraints": {"type": "string", "description": "Optional constraints/non-goals for the child."},
                "memory_mode": {
                    "type": "string",
                    "enum": sorted(VALID_SUBTASK_MEMORY_MODES),
                    "description": "Child memory mode. Default forked copies stable memory only; empty starts blank. shared is disabled for live local subagents.",
                },
                "model_lane": {
                    "type": "string",
                    "enum": ["auto", "main", "heavy", "light", "review", "scope"],
                    "default": "auto",
                    "description": "Model lane for the child. auto uses the cheap Light lane for a read-only child but the strong Heavy lane for a MUTATING first-level child — one that writes (a declared write_surface) OR is granted mutative-descendant intent (may_mutate); main/heavy/light use those configured slots (Heavy = strong acting/coding lane, empty Heavy/Light fall back to Main); review/scope fan out across configured reviewer slots and return a task_group. NOTE: an EXPLICIT main/heavy lane is honored only for children at or below the configured capability depth limit (advanced setting OUROBOROS_SUBAGENT_CAPABILITY_DEPTH_LIMIT, default 1 = direct children); deeper descendants resolve to Light to bound deep-swarm cost (a visible note is surfaced when an explicit request is capped).",
                },
                "write_surface": {
                    "type": "string",
                    # No empty-string member: Google Gemini's function-calling validator
                    # rejects empty enum values (400 INVALID_ARGUMENT). Read-only is the
                    # default by OMITTING this param; `read_only` is an explicit, provider-safe
                    # (non-empty) alias for the SAME read-only path, so an audit/read-only child
                    # can NAME its intent instead of reaching for an acting surface like
                    # self_worktree (the trap behind the read-only-audit cancel-storm). It is NOT
                    # an acting VALID_WRITE_SURFACES member — it normalizes to the omit path.
                    "enum": ["read_only", "self_worktree", "external_workspace", "genesis"],
                    "description": "read_only (or omit) = read-only child auditing THIS repo. Otherwise the isolated write surface for a MUTATIVE child (see tool description). Acting surfaces require mutative subagents enabled (default ON in advanced/pro).",
                },
                "write_root": {"type": "string", "description": "For write_surface=external_workspace: the external project directory. Ignored for self_worktree and genesis (both auto-provisioned)."},
                "protected_paths_grant": {"type": "boolean", "default": False, "description": "Allow the child to modify protected paths in its self_worktree. Honored only in pro runtime mode; you still re-check at integration."},
                "external_tool_grants": {"type": "array", "items": {"type": "string"}, "description": "Optional extension/MCP tool names to grant this mutative child. Denied by default."},
                "delegation_intent": {"type": "string", "description": "Optional: tell THIS child whether/how to delegate further (e.g. 'build the whole game; spawn your own children per subsystem and let them spawn too'). Propagated structurally into the child's delegation budget and surfaced in its prompt, so a 'use maximum subagents / grandchildren' intent is not lost. Defaults to inheriting the parent's intent."},
                "may_mutate": {"type": "boolean", "default": False, "description": "Optional: grant this child the intent to spawn MUTATIVE (acting) descendants of its own. Still bounded by the usual mutative-subagent gating and depth/active caps."},
                "may_fan_out": {"type": "boolean", "default": True, "description": "Optional: whether this child may spawn MULTIPLE children (a wave). Bounded by the per-root active cap."},
                "max_children": {"type": "integer", "default": 0, "description": "Optional soft cap on this child's own direct children (0 = inherit / configured cap)."},
            }, "required": ["objective", "expected_output"], "additionalProperties": False},
        }, _schedule_task),
        # cancel_task + peek_task + discard_child_result are registered by ouroboros/tools/join_ledger.py.
        ToolEntry("request_deep_self_review", {
            "name": "request_deep_self_review",
            "description": "Request an Atlas-backed deep self-review of the entire Ouroboros project. Uses OUROBOROS_MODEL_DEEP_SELF_REVIEW with its matching provider key, full core memory whitelist, and manifest accounting for every tracked repo path against the Constitution. Results go to chat and memory.",
            "parameters": {"type": "object", "properties": {
                "reason": {"type": "string", "description": "Why you want a review (context for the reviewer)"},
            }, "required": ["reason"]},
        }, _request_deep_self_review),
        ToolEntry("chat_history", {
            "name": "chat_history",
            "description": "Retrieve messages from chat history. Supports search.",
            "parameters": {"type": "object", "properties": {
                "count": {"type": "integer", "default": 100, "description": "Number of messages (from latest)"},
                "offset": {"type": "integer", "default": 0, "description": "Skip N from end (pagination)"},
                "search": {"type": "string", "default": "", "description": "Text filter"},
            }, "required": []},
        }, _chat_history),
        ToolEntry("update_scratchpad", {
            "name": "update_scratchpad",
            "description": "Append a block to your working memory (scratchpad). Each call adds a "
                           "timestamped block; oldest blocks are auto-evicted when the cap (10) is reached. "
                           "Write what matters NOW — active tasks, decisions, observations. "
                           "Persists across sessions, read at every task start. "
                           "No-op on a project-scoped task (no per-project scratchpad); use knowledge_write for project facts.",
            "parameters": {"type": "object", "properties": {
                "content": {"type": "string", "description": "Content for this scratchpad block"},
            }, "required": ["content"]},
        }, _update_scratchpad),
        ToolEntry("send_user_message", {
            "name": "send_user_message",
            "description": "Send a proactive message to the user. Use when you have something "
                           "genuinely worth saying — an insight, a question, or an invitation to collaborate. "
                           "This is NOT for task responses (those go automatically).",
            "parameters": {"type": "object", "properties": {
                "text": {"type": "string", "description": "Message text"},
                "reason": {"type": "string", "description": "Why you're reaching out (logged, not sent)"},
            }, "required": ["text"]},
        }, _send_user_message),
        ToolEntry("update_identity", {
            "name": "update_identity",
            "description": "Update your identity manifest (who you are, who you want to become). "
                           "Persists across sessions. Obligation to yourself (Principle 1: Continuity). "
                           "Read your current identity first, then evolve it — add, refine, deepen. "
                           "Full rewrites are allowed but should be rare; continuity of self matters. "
                           "Use this only after substantive reflection or real experience — not on a "
                           "greeting or trivial turn. This is the only correct way to write identity; "
                           "never write memory/identity.md through write_file/edit_text. "
                           "No-op on a project-scoped task (identity is global and continuous, never per-project).",
            "parameters": {"type": "object", "properties": {
                "content": {"type": "string", "description": "Full identity content (prefer evolving over rewriting from scratch)"},
            }, "required": ["content"]},
        }, _update_identity),
        ToolEntry("toggle_evolution", {
            "name": "toggle_evolution",
            "description": "Enable or disable evolution mode. When enabled, Ouroboros runs continuous self-improvement cycles. Enabling requires runtime_mode 'advanced' or 'pro'; it is refused in 'light' mode.",
            "parameters": {"type": "object", "properties": {
                "enabled": {"type": "boolean", "description": "true to enable, false to disable"},
                "objective": {"type": "string", "default": "", "description": "Optional Evolution Campaign objective when enabling."},
            }, "required": ["enabled"]},
        }, _toggle_evolution),
        ToolEntry("toggle_consciousness", {
            "name": "toggle_consciousness",
            "description": "Control background consciousness: 'start', 'stop', or 'status'.",
            "parameters": {"type": "object", "properties": {
                "action": {"type": "string", "enum": ["start", "stop", "status"], "description": "Action to perform"},
            }, "required": ["action"]},
        }, _toggle_consciousness),
        ToolEntry("switch_model", {
            "name": "switch_model",
            "description": "Switch to a different LLM model or reasoning effort level. "
                           "Use when you need more power (complex code, deep reasoning) "
                           "or want to save budget (simple tasks). Takes effect on next round.",
            "parameters": {"type": "object", "properties": {
                "model": {"type": "string", "description": "Model name (e.g. anthropic/claude-sonnet-4). Leave empty to keep current."},
                "effort": {"type": "string", "enum": ["low", "medium", "high", "xhigh"],
                           "description": "Reasoning effort level. Leave empty to keep current."},
            }, "required": []},
        }, _switch_model),
        ToolEntry("get_task_result", {
            "name": "get_task_result",
            "description": "Read the effective result of a subtask, including child-drive output when available.",
            "parameters": {"type": "object", "required": ["task_id"], "properties": {
                "task_id": {"type": "string", "description": "Task ID returned by schedule_subagent"},
            }},
        }, _get_task_result),
        ToolEntry("wait_task", {
            "name": "wait_task",
            "description": "Wait for a subtask to reach a terminal status and return its effective result. May return EARLY (before terminal) if the child raises a tree_note blocker/question/interface_contract beacon — the result then carries a [CHILD_BEACONS] block so you can steer it.",
            "parameters": {"type": "object", "required": ["task_id"], "properties": {
                "task_id": {"type": "string", "description": "Task ID to check"},
                "timeout_sec": {"type": "integer", "default": 180, "description": "Maximum seconds to wait (default 180)."},
            }},
        }, _wait_for_task, timeout_sec=7200),
        ToolEntry("wait_tasks", {
            "name": "wait_tasks",
            "description": "Wait for multiple subtasks and return full effective results for each child. The JSON also includes live_child_status (running/scheduled/terminal per child) and may early_return (before all terminal) on a child tree_note blocker/question/interface_contract beacon so you can steer mid-flight.",
            "parameters": {"type": "object", "required": ["task_ids"], "properties": {
                "task_ids": {"type": "array", "items": {"type": "string"}, "description": "Task IDs returned by schedule_subagent."},
                "timeout_sec": {"type": "integer", "default": 600, "description": "Maximum seconds to wait (default 600)."},
                "mode": {"type": "string", "enum": ["all_terminal", "any_terminal"], "default": "all_terminal"},
            }},
        }, _wait_for_tasks, timeout_sec=7200),
    ]
