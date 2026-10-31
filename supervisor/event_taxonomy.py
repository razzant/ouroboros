"""The declared disposition of every event kind the runtime puts on EVENT_Q.

An event without a disposition is dropped: ``dispatch_event`` logs "no handler"
and the fact its producer meant to report is gone. That is a silent hole, and it
is only visible to someone reading BOTH ends. ``tests/test_worker_event_registry``
already reads one end — every emitted literal type has a handler. This table is
the other half: for every event the runtime answers, it names WHO answers and
WHERE the producers are, so a dispatch key that outlives its last producer stops
reading as live capability (``schedule_task`` did exactly that until D06).

Tiers, exhaustive:

``worker_handler``
    ``supervisor.events.EVENT_HANDLERS`` dispatches it to a supervisor handler
    that takes a runtime action. The ordinary case.

``telemetry_only``
    Also a dispatch-table entry, but its handler is the durable passthrough in
    ``supervisor.telemetry_events`` — the supervisor records the row and takes
    no further action. The event is a FACT the owner may want in the ledger,
    not an instruction. That registry IS the tier's membership list; this table
    only declares which side of the line each row sits on.

``server_intercept``
    The server's drain loop consumes it before dispatch, because the answer is a
    process-level action the supervisor thread cannot take on itself.

``nested_log_event``
    It travels inside ``log_event.data`` and is answered by the nested branch of
    the log-event handler, so it never needs a dispatch key of its own. Without
    a row here such an event reads as a producer nobody answers.

Lanes, exhaustive (``lane``): the one supervisor event bus carries two lanes and
its single consumer always serves ``control`` before ``telemetry``
(``supervisor/event_observation.py``), so a stream of worker telemetry can never
sit in front of a chat message or a start/stop/control event (owner decision,
14 Oct 2026). ``telemetry`` is the three high-volume streams whose handlers take
no runtime action and need no RUNNING row to be correct — heartbeats, usage
rows and the ``log_event`` envelope with every nested kind it carries. Everything
authority-changing stays ``control``: every ``send_message`` (progress narration
is a chat row), ``task_done``, scheduling, cancel, owner/model waits, worker
readiness and quiescence, the cognitive-operation lifecycle, fences, restart.
The rare ``telemetry_only`` tier rows stay in the control lane too: they are
single typed facts, not streams, and their readers (health checks, audits)
expect them in FIFO order with the task's own control events. A kind without a
row rides the control lane. The lane is declared here once; the bus reads it
and the drain loop never carries a second classification.

The table is data. Nothing here decides policy or dispatches anything; the
dispatch table stays the single execution authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

WORKER_HANDLER = "worker_handler"
TELEMETRY_ONLY = "telemetry_only"
SERVER_INTERCEPT = "server_intercept"
NESTED_LOG_EVENT = "nested_log_event"

TIERS: Tuple[str, ...] = (WORKER_HANDLER, TELEMETRY_ONLY, SERVER_INTERCEPT, NESTED_LOG_EVENT)

CONTROL_LANE = "control"
TELEMETRY_LANE = "telemetry"

LANES: Tuple[str, ...] = (CONTROL_LANE, TELEMETRY_LANE)


@dataclass(frozen=True)
class EventDisposition:
    """Who answers one event kind, where its producers live, and which lane it rides."""

    tier: str
    answered_by: str
    producers: Tuple[str, ...]
    note: str = ""
    idle_input: bool = False  # its handler can change an existing task-idle input
    lane: str = CONTROL_LANE  # the bus lane; a nested kind names the envelope's lane


def _handled(answered_by: str, *producers: str, note: str = "", idle_input: bool = False,
             lane: str = CONTROL_LANE) -> EventDisposition:
    return EventDisposition(WORKER_HANDLER, answered_by, tuple(producers), note, idle_input, lane)


def _telemetry(*producers: str, note: str = "") -> EventDisposition:
    return EventDisposition(TELEMETRY_ONLY, "supervisor.telemetry_events", tuple(producers), note)


EVENT_DISPOSITIONS: Dict[str, EventDisposition] = {
    # --- worker_handler: one dispatch-table entry each, each taking an action --
    "acceptance_fence": _handled(
        "supervisor.events_worker_reports", "ouroboros/agent.py"),
    "budget_pause": _handled(
        "supervisor.events_budget", "ouroboros/agent.py", idle_input=True),
    "mind_task_control": _handled(
        "supervisor.events_mind_controls", "ouroboros/tools/control_mind.py", idle_input=True),
    "budget_resume_child": _handled(
        # #1196 (owner Q9): the resumed root's model selects one paused child.
        "supervisor.events_budget", "ouroboros/tools/join_ledger.py", idle_input=True),
    "budget_root_fence": _handled(
        # v7 L-B split: the loop's fence emitter lives in the budget leaf.
        "supervisor.events_budget", "ouroboros/agent.py", "ouroboros/loop_budget.py"),
    "cancel_task": _handled(
        "supervisor.events_runtime_controls", "ouroboros/tools/join_ledger.py"),
    "cognitive_operation": _handled(
        "supervisor.cognitive_operations", "ouroboros/utils.py", idle_input=True),
    "deep_self_review_request": _handled(
        "supervisor.events_runtime_controls", "ouroboros/tools/control_runtime.py"),
    "ensure_project_scope": _handled(
        "supervisor.events_project_routing", "ouroboros/tools/control_delegation.py"),
    "external_wait_lease": _handled(
        "supervisor.events_worker_reports", "ouroboros/delegate_progress.py", idle_input=True),
    "llm_usage": _handled(
        "supervisor.events_budget", "ouroboros/agent.py",
        "ouroboros/pricing.py", "ouroboros/tools/search.py", "ouroboros/tools/vision.py",
        "ouroboros/tools/skill_publish.py", "ouroboros/tools/review_helpers.py", idle_input=True,
        lane=TELEMETRY_LANE,
        note="the producer's per-task endpoint stamps lineage and chat by value (a queue-less server "
             "review's row, put on the bus directly, carries its usage scope and its carrier's chat); "
             "the handler fills a missing field from RUNNING only as a fallback"),
    "log_event": _handled(
        "supervisor.events_worker_reports", "ouroboros/utils.py", lane=TELEMETRY_LANE,
        note="the envelope every worker log line rides; the payload types that "
             "need their own answer are the nested_log_event rows below"),
    "main_llm_call_state": _handled(
        "supervisor.events", "ouroboros/loop_llm_call.py",
        note="the last handler still resolving on the facade itself", idle_input=True),
    "owner_message_injected": _handled(
        # v7 L-B split: the drain that emits the receipt lives in the round-limits leaf.
        "supervisor.events_runtime_controls", "ouroboros/loop_round_limits.py"),
    "owner_wait": _handled("supervisor.worker_owner_wait", "ouroboros/owner_wait.py", idle_input=True),
    "worker_quiescent": _handled("supervisor.worker_pool_lifecycle", "supervisor/worker_process.py"),
    "project_digest": _handled(
        "supervisor.events_project_routing", "ouroboros/agent_task_pipeline.py"),
    "promote_chat_to_task": _handled(
        "supervisor.events_project_routing", "ouroboros/tools/control_routing.py"),
    "rollback_request": _handled("supervisor.events_runtime_controls", "ouroboros/tools/git_rollback.py"),
    "promote_to_stable": _handled(
        "supervisor.events_runtime_controls", "ouroboros/tools/control_runtime.py"),
    "review_late_result": _handled(
        "supervisor.cognitive_operations", "ouroboros/review_custody.py"),
    "schedule_subagent": _handled(
        "supervisor.events_schedule_task", "ouroboros/tools/control_scheduling.py",
        idle_input=True, note="the only producer of the schedule handler; the schedule_task key "
             "retired beside this row had none"),
    "send_document": _handled(
        "supervisor.events_chat_delivery", "ouroboros/tools/core_artifacts.py"),
    "send_links": _handled(
        "supervisor.events_chat_delivery", "ouroboros/tools/core_artifacts.py"),
    "send_card": _handled(
        "supervisor.events_chat_delivery", "ouroboros/tools/cards.py",
        note="the card is stored before the event; the row carries its identity, the body stays in the store"),
    "send_message": _handled(
        "supervisor.events_chat_delivery", "ouroboros/agent.py",
        "supervisor/task_reaper.py", "supervisor/terminal_delivery.py", idle_input=True),
    "send_photo": _handled(
        "supervisor.events_chat_delivery", "ouroboros/tools/core_artifacts.py"),
    "send_quiz": _handled(
        "supervisor.events_chat_delivery", "ouroboros/tools/core_artifacts.py"),
    "send_quiz_state": _handled(
        "supervisor.events_chat_delivery", "ouroboros/tools/core_artifacts.py",
        "ouroboros/gateway/task_decision.py",
        note="persists an explicit closure reason and discharges its delivery; "
             "the live quiz_state frame is a separate transport discriminator"),
    "send_video": _handled(
        "supervisor.events_chat_delivery", "ouroboros/tools/core_artifacts.py"),
    "skill_exec_failed": _handled(
        "supervisor.events_worker_reports", "ouroboros/tools/skill_exec.py"),
    "skill_exec_finished": _handled(
        "supervisor.events_worker_reports", "ouroboros/tools/skill_exec.py"),
    "steer_task": _handled(
        "supervisor.steering", "ouroboros/tools/control_relay.py",
        "ouroboros/gateway/routing_decision.py"),
    "task_dispatch_resolved": _handled(
        "supervisor.events_worker_reports", "ouroboros/agent_dispatch.py"),
    "task_focus_updated": _handled(
        "supervisor.events_worker_reports", "ouroboros/tools/project_journal.py"),
    "task_done": _handled(
        "supervisor.events_task_done", "ouroboros/agent_task_pipeline.py",
        "supervisor/queue.py", "supervisor/task_reaper.py", "supervisor/worker_health.py", idle_input=True),
    "task_heartbeat": _handled(
        "supervisor.events_worker_reports", "ouroboros/agent.py", lane=TELEMETRY_LANE,
        note="liveness only: a heartbeat of a task that already left RUNNING is a no-op"),
    "task_model_wait": _handled(
        "supervisor.task_model_wait", "ouroboros/model_wait.py", idle_input=True),
    "task_metrics": _handled(
        "supervisor.events_worker_reports", "ouroboros/agent_task_pipeline.py"),
    "toggle_consciousness": _handled(
        "supervisor.events_runtime_controls", "ouroboros/tools/control_runtime.py"),
    "toggle_evolution": _handled(
        "supervisor.events_runtime_controls", "ouroboros/tools/control_runtime.py"),
    "typing_start": _handled(
        "supervisor.events_chat_delivery", "ouroboros/agent.py", "ouroboros/model_sleep.py",
        note="a start, or the same attempt running again after a wait (``announce_resumed``)"),
    "ui_i18n_updated": _handled(
        "supervisor.events_worker_reports", "ouroboros/tools/control_ui_translation.py",
        note="the memory file is already written; the host broadcasts the same-named frame to its windows"),
    "update_progress_changed": _handled(
        "supervisor.events_worker_reports", "ouroboros/tools/control_update_description.py",
        note="the update description is already saved; the host broadcasts the same-named frame to its windows"),

    # --- telemetry_only: recorded in the ledger, no runtime action follows -----
    "plan_task_deadline_skip": _telemetry(
        "ouroboros/tools/plan_review_runtime.py",
        note="the fact that a deadline left no useful planning window"),
    "quiz_answer_injected": _telemetry(
        "ouroboros/owner_mailbox.py"),
    "review_density_probe": _telemetry(
        "ouroboros/tools/review_admission.py",
        note="one bounded exact-model send that calibrates tokenizer density before a size refusal"),
    "review_wave_admission_unavailable": _telemetry(
        "ouroboros/tools/parallel_review.py",
        note="the commit gate's money admission raised: the wave dispatched unadmitted (fail-open)"),
    "review_wave_budget_insufficient": _telemetry(
        "ouroboros/tools/review_helpers.py"),
    "review_wave_billing_authority_unavailable": _telemetry(
        "ouroboros/tools/review_helpers.py",
        note="not money: the precheck fails open and each seat refuses typed and unsent"),
    "review_wave_budget_partial_unknown": _telemetry(
        "ouroboros/tools/review_helpers.py",
        note="the v6.69.0-class omission: the fix registered one branch of "
             "review_helpers' if/else and missed this one"),
    "task_message_injected": _telemetry(
        "ouroboros/owner_mailbox.py",
        note="its own handler rather than the shared passthrough, because the "
             "row is projected to the fields duplicate-detection reads"),

    # --- server_intercept -----------------------------------------------------
    "restart_request": EventDisposition(
        SERVER_INTERCEPT, "ouroboros.server_liveness",
        ("ouroboros/tools/control_events.py", "ouroboros/agent_task_pipeline.py", "supervisor/evolution_lifecycle.py"),
        "restarting the process is not something the supervisor thread can do to "
        "itself, so the loop's bounded drain routes this one to the server's restart "
        "handler before dispatch",
    ),

    # --- nested_log_event (they ride the log_event envelope's telemetry lane) --
    "review_reference": EventDisposition(
        NESTED_LOG_EVENT, "supervisor.events_worker_reports",
        ("ouroboros/tools/plan_review_references.py",),
        "an opaque Plan-detail invalidation: the nested branch forwards it live "
        "and its producer appends the durable copy to progress.jsonl itself",
        lane=TELEMETRY_LANE,
    ),
    "task_checkpoint": EventDisposition(
        NESTED_LOG_EVENT, "supervisor.events_worker_reports",
        ("ouroboros/loop_messages.py",),
        "the one producer (the budget-pause checkpoint in agent.py and every loop "
        "checkpoint ride through it) appends the durable rows itself (events.jsonl, "
        "and progress.jsonl for a System row) before emitting the one live frame; "
        "the worker log sink suppresses the append's copy and the nested branch "
        "only addresses and pushes that frame",
        lane=TELEMETRY_LANE,
    ),
    "task_start_settings_reload_failed": EventDisposition(
        NESTED_LOG_EVENT, "supervisor.events_worker_reports",
        ("ouroboros/subagent_runtime.py",),
        "a durable owner disclosure (#285) persisted by the nested branch: "
        "without it the fact evaporates on the next page load",
        lane=TELEMETRY_LANE,
    ),
}


def disposition_for(event_type: str) -> EventDisposition | None:
    """The declared disposition for one event kind, or None when undeclared."""
    return EVENT_DISPOSITIONS.get(str(event_type or "").strip())
