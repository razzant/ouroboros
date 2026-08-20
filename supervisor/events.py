"""Dispatch worker EVENT_Q messages to supervisor handlers."""

from __future__ import annotations

import logging
import os  # noqa: F401
import pathlib  # noqa: F401
import subprocess  # noqa: F401
import threading  # noqa: F401
import time  # noqa: F401
import uuid  # noqa: F401
from collections import deque  # noqa: F401
from typing import Any, Dict, Optional  # noqa: F401

from ouroboros.utils import append_jsonl, atomic_write_json, truncate_for_log, utc_now_iso  # noqa: F401
from ouroboros.config import (
    MAX_ACTIVE_SUBAGENTS_HARD_CAP,  # noqa: F401
    get_max_active_subagents_per_root,  # noqa: F401
    get_max_subagent_depth,  # noqa: F401
)
from ouroboros.tool_capabilities import ACTING_SUBAGENT_MODE, LOCAL_READONLY_SUBAGENT_MODE  # noqa: F401
from ouroboros.contracts.task_constraint import VALID_WRITE_SURFACES  # noqa: F401
from ouroboros.task_results import (
    STATUS_CANCELLED,  # noqa: F401
    STATUS_COMPLETED,  # noqa: F401
    STATUS_FAILED,  # noqa: F401
    STATUS_INTERRUPTED,  # noqa: F401
    STATUS_REJECTED_DUPLICATE,  # noqa: F401
    STATUS_SCHEDULED,  # noqa: F401
    load_task_result,  # noqa: F401
    write_task_result,  # noqa: F401
)
from ouroboros.cost_projection import carry_cost_meta, with_cost_aliases  # noqa: F401
from ouroboros.outcomes import infra_failed_axes, normalize_outcome_axes  # noqa: F401
from ouroboros.subagents import intended_lane as intended_subagent_lane  # noqa: F401
from ouroboros.contracts.task_contract import build_task_contract, normalize_allowed_resources  # noqa: F401
# The declared disposition of every event kind the runtime can produce. Data only:
# the dispatch table below stays the single execution authority, and the taxonomy
# answers the one question the table cannot — what a MISS means.
from supervisor.event_taxonomy import disposition_for

# Handler families owned by their own modules (module-size boundary). Each
# family is re-imported here so this module keeps ONE public surface for the
# dispatch table, its callers and its tests, and so the historical
# ``supervisor.events`` names keep resolving. The dependency is one-way: no
# owner below imports this module.
from supervisor.events_chat_delivery import (  # noqa: F401 -- supervisor/events.py facade re-exports
    HOST_NARRATION,
    _DELIVERED_MESSAGE_IDS,
    _bound_project_chat_id,
    _handle_send_document,
    _handle_send_message,
    _handle_send_photo,
    _handle_send_video,
    _handle_typing_start,
    _register_delivered,
)
from supervisor.events_subagent_admission import (  # noqa: F401 -- supervisor/events.py facade re-exports
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
from supervisor.events_schedule_task import (  # noqa: F401 -- supervisor/events.py facade re-exports
    VALID_SUBAGENT_MEMORY_MODES,
    _handle_schedule_task,
    _PARENT_CONTEXT_END,
    _PARENT_CONTEXT_MARKER,
    _build_scheduled_task_payload,
    _cleanup_rejected_worktree,
    _extract_task_description_and_context,
    _find_duplicate_task,
    _format_task_for_dedup,
    _reject_if_no_chat_target,
    _reject_schedule_task,
)
from supervisor.events_project_routing import (  # noqa: F401 -- supervisor/events.py facade re-exports
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
from supervisor.events_coop_checkpoint import (  # noqa: F401 -- supervisor/events.py facade re-exports
    _COOP_CHECKPOINT_DROPPED,
    _COOP_CHECKPOINT_INFLIGHT,
    _COOP_CHECKPOINT_LOCK,
    _checkpoint_coop_roots_on_root_done,
    _maybe_checkpoint_coop_on_tree_quiescence,
    _spawn_coop_checkpoint,
)
from supervisor.events_evolution_done import (  # noqa: F401 -- supervisor/events.py facade re-exports
    _handle_evolution_task_done,
)
# The owner-stop backstop is one honesty rule with ``stop_evolution_tasks`` and lives
# beside it; re-exported here because callers and tests reach it through this module.
from supervisor.queue_transitions import (  # noqa: F401 -- supervisor/events.py facade re-export
    _close_campaign_after_owner_stop,
)
from supervisor.events_task_done import (  # noqa: F401 -- supervisor/events.py facade re-exports
    _PROVIDER_DEATH_NOTIFIED,
    _authoritative_terminal_cost,
    _finish_task_done_dispatch,
    _handle_task_done,
    _maybe_notify_provider_death,
    _resolve_lifecycle_fault,
    _task_done_durable_fault,
    _task_done_review_projection,
)
from supervisor.events_budget import (  # noqa: F401 -- supervisor/events.py facade re-exports
    _handle_budget_pause,
    _handle_budget_root_fence,
    _handle_llm_usage,
    _handle_review_wave_budget_insufficient,
    _set_root_budget_pause_locked,
)
from supervisor.events_worker_reports import (  # noqa: F401 -- supervisor/events.py facade re-exports
    _handle_acceptance_fence,
    _handle_external_wait_lease,
    _handle_log_event,
    _handle_skill_lifecycle,
    _handle_task_dispatch_resolved,
    _handle_task_heartbeat,
    _handle_task_metrics,
)
from supervisor.events_runtime_controls import (  # noqa: F401 -- supervisor/events.py facade re-exports
    _handle_cancel_task,
    _handle_deep_self_review_request,
    _handle_owner_message_injected,
    _handle_promote_to_stable,
    _handle_toggle_consciousness,
    _handle_toggle_evolution,
)

log = logging.getLogger(__name__)


# Owner steering delivery (cancel-pending refusal + the steer_task handler)
# lives in supervisor/steering.py (module-size boundary for this pinned
# surface); imported back so the dispatch table and callers keep one name.
from supervisor.steering import (  # noqa: E402 -- intentional re-import
    _handle_steer_task,
    _refuse_steering_while_cancelling,  # noqa: F401 -- re-exported for callers/tests
)


EVENT_HANDLERS = {
    "llm_usage": _handle_llm_usage,
    "external_wait_lease": _handle_external_wait_lease,
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
    "schedule_subagent": _handle_schedule_task,
    "promote_chat_to_task": _handle_promote_chat_to_task,
    "ensure_project_scope": _handle_ensure_project_scope,
    "routing_manual_target": _handle_routing_manual_target,
    "steer_task": _handle_steer_task,
    "project_digest": _handle_project_digest,
    "cancel_task": _handle_cancel_task,
    "send_photo": _handle_send_photo,
    "send_video": _handle_send_video,
    "send_document": _handle_send_document,
    "toggle_evolution": _handle_toggle_evolution,
    "toggle_consciousness": _handle_toggle_consciousness,
    "owner_message_injected": _handle_owner_message_injected,
    "log_event": _handle_log_event,
    "review_wave_budget_insufficient": _handle_review_wave_budget_insufficient,
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
        disposition = disposition_for(event_type)
        if disposition is not None:
            # Declared, just not dispatched here: the server intercepts it, the log
            # envelope already answered it, or it is a fact for the ledger. Record
            # the fact under its declared tier instead of dropping it as unknown.
            log.debug(
                "Worker event %r has no dispatch handler by design (%s)",
                event_type, disposition.tier,
            )
            ctx.append_jsonl(
                ctx.DRIVE_ROOT / "logs" / "events.jsonl",
                {
                    "ts": utc_now_iso(),
                    **{key: value for key, value in evt.items() if key != "ts"},
                    "event_disposition": disposition.tier,
                },
            )
            return
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
