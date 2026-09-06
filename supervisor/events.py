"""Dispatch worker EVENT_Q messages to supervisor handlers."""

from __future__ import annotations

import logging
import os  # noqa: F401
import subprocess  # noqa: F401
import threading  # noqa: F401
import time
from collections import deque  # noqa: F401
from typing import Any, Dict, Optional

from ouroboros.utils import append_jsonl, atomic_write_json, truncate_for_log, utc_now_iso  # noqa: F401
from ouroboros.config import get_max_subagent_depth  # noqa: F401 -- facade name tests monkeypatch
from ouroboros.tool_capabilities import ACTING_SUBAGENT_MODE, LOCAL_READONLY_SUBAGENT_MODE  # noqa: F401
from ouroboros.contracts.task_constraint import VALID_WRITE_SURFACES  # noqa: F401
from ouroboros.task_results import (
    STATUS_SCHEDULED,  # noqa: F401 -- facade name tests read
    load_task_result,
)
from ouroboros.cost_projection import carry_cost_meta, live_root_cost_projection, with_cost_aliases  # noqa: F401
from ouroboros.subagent_messages import subagent_message_meta  # noqa: F401
from supervisor.cognitive_operations import EVENT_HANDLERS as _CEH, _handle_cognitive_operation  # noqa: F401
from supervisor.task_dispatch import (  # noqa: F401 -- facade name tests import
    build_scheduled_task_payload as _build_scheduled_task_payload,
)
from supervisor.log_addressing import (  # re-export: one events surface
    address_ctx_event as _address_ctx,  # noqa: F401
    address_task_event as _address_task_event,  # noqa: F401  (tests pin it here)
    bound_project_chat_id as _bound_project_chat_id,  # noqa: F401  (facade name; owner leaf is events_chat_delivery)
    make_server_log_sink,  # noqa: F401  (server.py installs it)
)

log = logging.getLogger(__name__)


# A progress frame's ``task_id`` is a ROUTING address — it says which live card the
# line lands on, NOT who wrote the line. The supervisor narrates a task's terminal
# path (grace requested, grace withdrawn) onto that task's own card, so those frames
# carry the task's id while the task itself did nothing. Host-authored frames set
# this key; ``_handle_send_message`` refuses to count them as the task's work.
# Without it the supervisor's own voice answers its own question — the grace toast
# stamped last_progress_at, the next 0.5s tick read the task as resumed, and the
# episode it had just opened was withdrawn before the worker could ever drain it.


def _routing_attachments(value: Any) -> Optional[list]:
    return [dict(item) for item in value if isinstance(item, dict)] if isinstance(value, list) else None


def _parent_delegation_budget(ctx: Any, parent_task_id: Any, drive_root: Any) -> Dict[str, Any]:
    """Read the parent's canonical budget for supervisor-side admission."""
    parent = str(parent_task_id or "").strip()
    if not parent:
        return {}
    running = getattr(ctx, "RUNNING", {})
    if isinstance(running, dict):
        for meta in running.values():
            task = meta.get("task") if isinstance(meta, dict) else None
            if not isinstance(task, dict) or str(task.get("id") or "").strip() != parent:
                continue
            contract = task.get("task_contract")
            if isinstance(contract, dict) and isinstance(contract.get("delegation_budget"), dict):
                return contract["delegation_budget"]
    roots = [drive_root]
    canonical = getattr(ctx, "DRIVE_ROOT", "")
    if canonical and str(canonical) != str(drive_root):
        roots.append(canonical)
    for root in roots:
        try:
            row = load_task_result(root, parent)
        except Exception:
            row = None
        if isinstance(row, dict):
            contract = row.get("task_contract")
            if isinstance(contract, dict) and isinstance(contract.get("delegation_budget"), dict):
                return contract["delegation_budget"]
    return {}


# Durable terminal registry dedupes successful sends across restarts.


# In-flight latch for off-loop coop checkpoints: one commit run per root at a
# time. A re-trigger after completion is safe (the helper no-ops on a clean
# tree), so this is concurrency control, not a permanent phase marker. A
# trigger arriving WHILE a run is in flight cannot simply be dropped: the
# in-flight worker may have already sampled liveness and seen the (then-live)
# last child, so it will skip the commit — and the dropped trigger was the
# last one there is. Such triggers are remembered per root and replayed once
# after the latch clears; the replayed run revalidates liveness itself.


# Owner steering delivery (cancel-pending refusal + the steer_task handler)
# lives in supervisor/steering.py (module-size boundary for this pinned
# surface); imported back so the dispatch table and callers keep one name.
from supervisor.steering import (  # noqa: E402 -- intentional re-import
    _handle_steer_task,
    _refuse_steering_while_cancelling,  # noqa: F401 -- re-exported for callers/tests
)


def _handle_main_llm_call_state(evt: Dict[str, Any], ctx: Any) -> None:
    task_id = str(evt.get("task_id") or "")
    phase = str(evt.get("phase") or "").strip().lower()
    llm_call_id = str(evt.get("llm_call_id") or "")
    execution_id = str(evt.get("execution_id") or "")
    round_id = str(evt.get("round_id") or "")
    if (
        not task_id
        or phase not in {"started", "finished", "failed"}
        or not llm_call_id
        or not execution_id
        or not round_id
    ):
        return
    try:
        task_attempt = int(evt["task_attempt"])
        call_attempt = int(evt["call_attempt"])
    except (KeyError, TypeError, ValueError):
        return
    if task_attempt < 1 or call_attempt < 1:
        return
    running = getattr(ctx, "RUNNING", None)
    meta = running.get(task_id) if isinstance(running, dict) else None
    if not isinstance(meta, dict):
        return
    expected_attempt = meta.get("attempt")
    if expected_attempt is None:
        task = meta.get("task") if isinstance(meta.get("task"), dict) else {}
        expected_attempt = task.get("_attempt")
    try:
        if int(expected_attempt) != task_attempt:
            return
    except (TypeError, ValueError):
        return
    identity = {
        "task_attempt": task_attempt,
        "llm_call_id": llm_call_id,
        "execution_id": execution_id,
        "round_id": round_id,
        "call_attempt": call_attempt,
    }
    if phase == "started":
        meta["active_llm_call"] = {**identity, "started_at": time.time()}
        return
    active = meta.get("active_llm_call")
    if not isinstance(active, dict) or any(active.get(key) != value for key, value in identity.items()):
        return
    meta.pop("active_llm_call", None)


# v7next F1 (D08): moved spans live in their owner leaves; re-exported here
# so this facade stays the single import surface for callers and tests.
from supervisor.events_budget import (  # noqa: E402, F401 -- intentional public re-exports
    _handle_budget_pause,
    _handle_budget_root_fence,
    _handle_llm_usage,
    _set_root_budget_pause_locked,
)
from supervisor.telemetry_events import (  # noqa: E402
    TELEMETRY_EVENT_HANDLERS as _TELEMETRY_EVENT_HANDLERS,
)
from supervisor.events_chat_delivery import (  # noqa: E402, F401 -- intentional public re-exports
    EVENT_HANDLERS as _CDE,
    HOST_NARRATION,
    _DELIVERED_MESSAGE_IDS,
    _handle_send_document,
    _handle_send_message,
    _handle_send_photo,
    _handle_send_video,
    _handle_typing_start,
    _register_delivered,
)
from supervisor.events_coop_checkpoint import (  # noqa: E402, F401 -- intentional public re-exports
    _COOP_CHECKPOINT_DROPPED,
    _COOP_CHECKPOINT_INFLIGHT,
    _COOP_CHECKPOINT_LOCK,
    _checkpoint_coop_roots_on_root_done,
    _maybe_checkpoint_coop_on_tree_quiescence,
    _spawn_coop_checkpoint,
)
from supervisor.events_evolution_done import (  # noqa: E402, F401 -- intentional public re-exports
    _handle_evolution_task_done,
)
from supervisor.events_project_routing import (  # noqa: E402, F401 -- intentional public re-exports
    _emit_routing_receipt,
    _handle_ensure_project_scope,
    _handle_project_digest,
    _handle_promote_chat_to_task,
    _handle_routing_manual_target,
    _persist_promote_rejection,
    _prepare_promote_source_off_loop,
    _publish_routing_ack,
    _rollback_promoted_pending,
)
from supervisor.events_runtime_controls import (  # noqa: E402, F401 -- intentional public re-exports
    _handle_cancel_task,
    _handle_deep_self_review_request,
    _handle_owner_message_injected,
    _handle_promote_to_stable,
    _handle_toggle_consciousness,
    _handle_toggle_evolution,
)
from supervisor.events_schedule_task import (  # noqa: E402, F401 -- intentional public re-exports
    VALID_SUBAGENT_MEMORY_MODES,
    _PARENT_CONTEXT_END,
    _PARENT_CONTEXT_MARKER,
    _cleanup_rejected_worktree,
    _extract_task_description_and_context,
    _find_duplicate_task,
    _handle_schedule_task,
    _format_task_for_dedup,
    _reject_schedule_task,
)
from supervisor.events_subagent_admission import (  # noqa: E402, F401 -- intentional public re-exports
    _GIT_UNBORN_HEAD,
    _active_subagent_count,
    _compose_subagent_text,
    _depth_reservation_admits,
    _external_workspace_head,
    _is_active_subagent_task,
    _iter_tree_subagent_tasks,
    _record_delegation_constraint,
    _resolve_subagent_constraint,
    _send_subagent_rejection,
    _subagent_cap_blocks,
    _subagent_rejection_meta,
    _subagent_scheduled_meta,
    _task_own_id,
    _validate_external_workspace,
)
from supervisor.events_task_done import (  # noqa: E402, F401 -- intentional public re-exports
    _PROVIDER_DEATH_NOTIFIED,
    _authoritative_terminal_cost,
    _finish_task_done_dispatch,
    _handle_task_done,
    _maybe_notify_provider_death,
    _resolve_lifecycle_fault,
    _task_done_durable_fault,
    _task_done_review_projection,
)
from supervisor.events_worker_reports import (  # noqa: E402, F401 -- intentional public re-exports
    _handle_acceptance_fence,
    _handle_external_wait_lease,
    _handle_log_event,
    _handle_skill_lifecycle,
    _handle_task_dispatch_resolved,
    _handle_task_heartbeat,
    _handle_task_metrics,
)
from supervisor.queue_transitions import (  # noqa: E402, F401 -- intentional public re-exports
    _close_campaign_after_owner_stop,
)

EVENT_HANDLERS = {
    "llm_usage": _handle_llm_usage,
    "external_wait_lease": _handle_external_wait_lease,
    **_CEH,
    **_CDE,
    "main_llm_call_state": _handle_main_llm_call_state,
    "budget_pause": _handle_budget_pause,
    "budget_root_fence": _handle_budget_root_fence,
    "task_heartbeat": _handle_task_heartbeat,
    "task_dispatch_resolved": _handle_task_dispatch_resolved,
    "typing_start": _handle_typing_start,
    "send_message": _handle_send_message,
    "task_done": _handle_task_done,
    "task_metrics": _handle_task_metrics,
    "deep_self_review_request": _handle_deep_self_review_request,
    "promote_to_stable": _handle_promote_to_stable,
    # D06: the "schedule_task" key retired here had no producer anywhere in the
    # tree — it advertised a capability nothing could reach. The FUNCTION keeps
    # its name (its family and FUNCTION_DEBT key are pinned to it); only the
    # dead wire vocabulary is gone. supervisor/event_taxonomy.py declares the
    # pairing this closes.
    "schedule_subagent": _handle_schedule_task,
    "promote_chat_to_task": _handle_promote_chat_to_task,
    "ensure_project_scope": _handle_ensure_project_scope,
    "routing_manual_target": _handle_routing_manual_target,
    "steer_task": _handle_steer_task,
    "project_digest": _handle_project_digest,
    "cancel_task": _handle_cancel_task,
    "toggle_evolution": _handle_toggle_evolution,
    "toggle_consciousness": _handle_toggle_consciousness,
    "owner_message_injected": _handle_owner_message_injected,
    "log_event": _handle_log_event,
    **_TELEMETRY_EVENT_HANDLERS,
    "skill_exec_finished": _handle_skill_lifecycle,
    "skill_exec_failed": _handle_skill_lifecycle,
    "acceptance_fence": _handle_acceptance_fence,
}


def dispatch_event(evt: Dict[str, Any], ctx: Any) -> None:
    """Dispatch a single worker event to its handler."""
    if not isinstance(evt, dict):
        ctx.append_jsonl(
            ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "invalid_worker_event",
                "error": "event is not dict",
                "event_repr": repr(evt)[:1000],
            },
        )
        return

    event_type = str(evt.get("type") or "").strip()
    if not event_type:
        ctx.append_jsonl(
            ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "invalid_worker_event",
                "error": "missing event.type",
                "event_repr": repr(evt)[:1000],
            },
        )
        return

    handler = EVENT_HANDLERS.get(event_type)
    if handler is None:
        log.warning("No handler for worker event type %r — event dropped", event_type)
        ctx.append_jsonl(
            ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "unknown_worker_event",
                "event_type": event_type,
                "event_repr": repr(evt)[:1000],
            },
        )
        return

    try:
        handler(evt, ctx)
    except Exception as e:
        # Surface the failure with a full traceback. Previously this only wrote a
        # repr(e) to supervisor.jsonl, so a crashing handler (e.g. an ImportError
        # in a task_done/heartbeat handler) was invisible and left the UI stuck.
        log.warning("Worker event handler %r failed: %s", event_type, e, exc_info=True)
        ctx.append_jsonl(
            ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "worker_event_handler_error",
                "event_type": event_type,
                "error": repr(e),
            },
        )
