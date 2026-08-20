"""The declared disposition of every event a worker can put on the event queue.

An event without a disposition is dropped: the dispatcher logs "no handler" and
the fact the producer meant to report is gone. That is a silent hole, and it is
only visible if someone reads both ends. This table is the missing half — it
names, for every event the runtime produces, WHO answers it and HOW, so a
producer added without an answer, and an answer left behind by its last
producer, both become test failures instead of quiet losses.

Tiers, exhaustive:

``worker_handler``
    ``supervisor.events.EVENT_HANDLERS`` dispatches it to a supervisor handler.
    This is the ordinary case and the only tier the dispatch table may contain.

``server_intercept``
    The server's drain loop consumes it before dispatch, because the answer is
    a process-level action the supervisor thread cannot take.

``nested_log_event``
    It travels inside ``log_event.data`` and is answered by the nested branch of
    the log-event handler. A top-level copy is therefore already accounted for
    and needs no second answer.

``telemetry_only``
    The supervisor records it and takes no further action. The event is a FACT
    the owner may want in the ledger, not an instruction.

The table is data. Nothing here decides policy or dispatches anything; the
dispatch table stays the single execution authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

WORKER_HANDLER = "worker_handler"
SERVER_INTERCEPT = "server_intercept"
NESTED_LOG_EVENT = "nested_log_event"
TELEMETRY_ONLY = "telemetry_only"

TIERS: Tuple[str, ...] = (WORKER_HANDLER, SERVER_INTERCEPT, NESTED_LOG_EVENT, TELEMETRY_ONLY)


@dataclass(frozen=True)
class EventDisposition:
    """Who answers one event kind, and where its producers live."""

    tier: str
    answered_by: str
    producers: Tuple[str, ...]
    note: str = ""


def _handled(answered_by: str, *producers: str, note: str = "") -> EventDisposition:
    return EventDisposition(WORKER_HANDLER, answered_by, tuple(producers), note)


EVENT_DISPOSITIONS: Dict[str, EventDisposition] = {
    # --- worker_handler: the ordinary case, one dispatch-table entry each -----
    "acceptance_fence": _handled(
        "supervisor.events_worker_reports", "ouroboros/agent.py"),
    "budget_pause": _handled(
        "supervisor.events_budget", "ouroboros/agent.py"),
    "budget_root_fence": _handled(
        # v7 L-B split: the loop's fence emitter lives in the budget leaf.
        "supervisor.events_budget", "ouroboros/agent.py", "ouroboros/loop_budget.py"),
    "cancel_task": _handled(
        "supervisor.events_runtime_controls", "ouroboros/tools/join_ledger.py"),
    "deep_self_review_request": _handled(
        "supervisor.events_runtime_controls", "ouroboros/tools/control_runtime.py"),
    "ensure_project_scope": _handled(
        "supervisor.events_project_routing", "ouroboros/tools/control_delegation.py"),
    "external_wait_lease": _handled(
        "supervisor.events_worker_reports", "ouroboros/delegate_progress.py"),
    "llm_usage": _handled(
        "supervisor.events_budget", "ouroboros/agent.py", "ouroboros/consciousness.py",
        "ouroboros/pricing.py", "ouroboros/tools/search.py", "ouroboros/tools/vision.py",
        "ouroboros/tools/skill_publish.py", "ouroboros/tools/review_helpers.py"),
    "log_event": _handled(
        "supervisor.events_worker_reports", "ouroboros/utils.py",
        note="the envelope every worker log line rides; its nested payload types are their own rows"),
    "owner_message_injected": _handled(
        # v7 L-B split: the drain that emits the receipt lives in the round-limits leaf.
        "supervisor.events_runtime_controls", "ouroboros/loop_round_limits.py"),
    "project_digest": _handled(
        "supervisor.events_project_routing", "ouroboros/agent_task_pipeline.py"),
    "promote_chat_to_task": _handled(
        "supervisor.events_project_routing", "ouroboros/tools/control_routing.py"),
    "promote_to_stable": _handled(
        "supervisor.events_runtime_controls", "ouroboros/tools/control_runtime.py"),
    "review_wave_budget_insufficient": _handled(
        "supervisor.events_budget", "ouroboros/tools/review_helpers.py"),
    "routing_manual_target": _handled(
        "supervisor.events_project_routing", "ouroboros/tools/control_routing.py"),
    "schedule_subagent": _handled(
        "supervisor.events_schedule_task", "ouroboros/tools/control_scheduling.py",
        note="the only producer of the schedule handler; the retired schedule_task key had none"),
    "send_document": _handled(
        "supervisor.events_chat_delivery", "ouroboros/tools/core_artifacts.py"),
    "send_message": _handled(
        "supervisor.events_chat_delivery", "ouroboros/agent.py",
        "ouroboros/agent_task_pipeline.py", "ouroboros/consciousness.py",
        "ouroboros/tools/control_runtime.py", "supervisor/task_reaper.py",
        "supervisor/terminal_delivery.py"),
    "send_photo": _handled(
        "supervisor.events_chat_delivery", "ouroboros/tools/core_artifacts.py"),
    "send_video": _handled(
        "supervisor.events_chat_delivery", "ouroboros/tools/core_artifacts.py"),
    "skill_exec_failed": _handled(
        "supervisor.events_worker_reports", "ouroboros/tools/skill_exec.py"),
    "skill_exec_finished": _handled(
        "supervisor.events_worker_reports", "ouroboros/tools/skill_exec.py"),
    "steer_task": _handled(
        "supervisor.steering", "ouroboros/tools/control_routing.py"),
    "task_dispatch_resolved": _handled(
        "supervisor.events_worker_reports", "ouroboros/agent_dispatch.py"),
    "task_done": _handled(
        "supervisor.events_task_done", "ouroboros/agent_task_pipeline.py",
        "supervisor/queue.py", "supervisor/task_reaper.py", "supervisor/worker_health.py"),
    "task_heartbeat": _handled(
        "supervisor.events_worker_reports", "ouroboros/agent.py"),
    "task_metrics": _handled(
        "supervisor.events_worker_reports", "ouroboros/agent_task_pipeline.py"),
    "toggle_consciousness": _handled(
        "supervisor.events_runtime_controls", "ouroboros/tools/control_runtime.py"),
    "toggle_evolution": _handled(
        "supervisor.events_runtime_controls", "ouroboros/tools/control_runtime.py"),
    "typing_start": _handled(
        "supervisor.events_chat_delivery", "ouroboros/agent.py"),

    # --- server_intercept ----------------------------------------------------
    "restart_request": EventDisposition(
        SERVER_INTERCEPT, "server.py",
        ("ouroboros/agent_task_pipeline.py", "supervisor/evolution_lifecycle.py"),
        "restarting the process is not something the supervisor thread can do to "
        "itself, so the server's drain loop answers this one before dispatch",
    ),

    # --- nested_log_event ----------------------------------------------------
    "task_checkpoint": EventDisposition(
        NESTED_LOG_EVENT, "supervisor.events_worker_reports",
        ("ouroboros/agent.py",),
        "persisted from inside log_event.data; the worker log sink suppresses the "
        "duplicate copy, so a top-level arrival is already accounted for",
    ),

    # --- telemetry_only ------------------------------------------------------
    "plan_task_deadline_skip": EventDisposition(
        TELEMETRY_ONLY, "supervisor.events",
        ("ouroboros/tools/plan_review_runtime.py",),
        "the fact that a deadline left no useful planning window; recorded for the "
        "owner's ledger, no runtime action follows from it",
    ),
    "progress": EventDisposition(
        TELEMETRY_ONLY, "supervisor.events",
        ("ouroboros/tools/ci.py",),
        "a CI-wait progress line; recorded rather than dropped. Nothing renders it "
        "live today, which is a producer expectation this table makes visible",
    ),
}


def disposition_for(event_type: str) -> EventDisposition | None:
    """The declared disposition for one event kind, or None when undeclared."""
    return EVENT_DISPOSITIONS.get(str(event_type or "").strip())
