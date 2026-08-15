"""WHERE a task will run: resolving the executor, recording it, refusing it.

Three moments of one decision. `resolve_dispatch_axes` settles the axes a dispatch is
about; `_record_executor_resolution` writes the answer down so the run is auditable
after the fact; `executor_blocked_outcome` and `_blocked_executor_terminal` are what
happens when there is no executor to resolve to — a task pointed at a backend that
cannot take it must END, typed, rather than quietly falling back to whatever is at
hand. `dispatch_executor_note` is the sentence the model reads about all of it.

They belong together because a resolution that is not recorded and a refusal that is
not terminal are the same bug wearing different clothes: the task runs somewhere
nobody chose. Extracted from `agent`, which owns the turn loop.
"""

from __future__ import annotations

import logging

import pathlib
from typing import Any, Dict, Optional, Tuple
from ouroboros.utils import (
    append_jsonl,
    utc_now_iso,
)
from ouroboros.task_results import STATUS_RUNNING
from ouroboros.subagents import (
    SubagentExecutorResolution,
    SubagentLaneResolution,
    SUBAGENT_RESOLUTION_FIELDS,
    SubagentDispatch,
    envelope_from_task,
    resolve_subagent_dispatch,
)

log = logging.getLogger(__name__)


def dispatch_executor_note(decision: Optional[SubagentExecutorResolution],
                           lane: Optional["SubagentLaneResolution"] = None) -> str:
    """The child's VISIBLE marker for a substrate decision it did not make ('' = silent).

    The rule table's `auto` rows are only honest if the child can see which way they
    went: a nanny must know to delegate, and a child that fell back to metered tokens
    must know its route was unavailable rather than discovering it by spending.

    ``lane`` is the same dispatch's lane resolution: a nanny that landed on the
    LIGHT lane by policy is told so, with the sanctioned escalation
    (``switch_model`` for real acceptance judgment) named beside it — a policy the
    child cannot see is a policy it will fight by accident.
    """
    if decision is None or decision.blocked:
        return ""
    if decision.executor == "harness":
        route = decision.route.route_id if decision.route else ""
        note = (
            f"EXECUTOR: your parent scheduled you on the delegated substrate ({route}). "
            "You are a NANNY. Decide your delegation plan FIRST — right after reading "
            "your objective and constraints, before any substantive work. Cost classes: "
            "a subscription-lane run has known-zero marginal cost when the route reports "
            "its settled spend as $0 (an estimated or undisclosed spend is estimated/unknown, "
            "not zero); every token YOU think on is metered API money. "
            "While the lane is healthy, delegate everything you can — even small tasks — "
            "with delegate_start / delegate_wait, and verify what comes back rather than "
            "believing it. After a delegated run SUCCEEDS, your job is to VERIFY and "
            "INTEGRATE its output — never to rebuild the same work yourself on metered "
            "tokens. Follow-up work (fixes, the next increment, a retry with a corrected "
            "prompt) is delegated too, with a new delegate_start; your own metered rounds "
            "are for judgment — acceptance, integration, honest settlement — not for "
            "co-building around a $0 run. If your run asks a question (delegate_wait "
            "returns waiting_on_user), answer it from the task context with "
            "delegate_answer; a question above your authority — money, scope, external "
            "actions — goes to your human via progress while you keep waiting (a timeout_at "
            "question benign-declines at the engine timeout; timeout_at=null waits until answered)."
        )
        if lane is not None and lane.provenance == "policy" and lane.effective_lane == "light":
            note += (
                " You run on the LIGHT model lane by dispatch policy: custody chores "
                "(starting runs, waiting, reading results, relaying) belong on this "
                "cheap lane. For a genuine acceptance or integration judgment you may "
                "raise your own power with switch_model and drop back after — that is "
                "the sanctioned escalation, not a workaround."
            )
        if decision.reset_at:
            note += (
                f" The route's plan window is currently spent and resets at "
                f"{decision.reset_at}. Decide explicitly: wait for the reset, deliver "
                "partial work, or say you fell back — do not drift into spending."
            )
        return note
    if decision.reason in {"requested_native", "harness_not_configured"}:
        return ""  # the ordinary case has nothing to announce
    if decision.reset_at:
        # D28's fallback, stated as the CAPABILITY DELTA it is: the parent asked for the
        # already-paid substrate to be used when available, every profile of it is spent,
        # and the work is proceeding on metered money instead. Destination 2 of 3 (the
        # child's own prompt); the durable event and the parent's envelope carry the same
        # two facts. The reset instant is named so the child can weigh waiting against
        # spending instead of guessing.
        return (
            "EXECUTOR CAPABILITY DELTA: every plan window of the configured delegated "
            f"substrate is spent (resets at {decision.reset_at}), so you FELL BACK to "
            "METERED API tokens. Your parent asked for 'auto', which permits this "
            "fallback rather than a wait — but it is real money that the subscription "
            "would have covered: keep the work proportionate, and say in your result "
            "that you ran below the substrate you were scheduled for and why."
        )
    return (
        f"EXECUTOR: the configured delegated substrate is unavailable "
        f"({decision.reason}), so you are running on METERED API tokens. Your parent "
        "asked for 'auto', which permits this — but say so in your result."
    )


def executor_blocked_outcome(decision: SubagentExecutorResolution) -> Tuple[str, Dict[str, Any]]:
    """The terminal ``(text, usage)`` of a child that was pinned and could not run.

    Deliberately NOT a fallback: the task ends unrun and typed, having spent nothing.
    """
    if decision.reason in ("delegate_tools_invisible", "delegate_visibility_unverified"):
        # Q1A preflight (2026-08-10 amendments): the route is healthy but the
        # child's MATERIALIZED toolset does not carry the delegate verbs — or
        # the toolset introspection itself failed, so visibility is UNKNOWN,
        # not disproven (distinct reason: the terminal states exactly what is
        # known). Either way the pin cannot be honored, and the fix is tool
        # policy/contract, not waiting for the route to recover.
        detail = (
            "the delegate tools (delegate_start/delegate_wait/delegate_cancel) "
            "are not visible in its materialized toolset"
            if decision.reason == "delegate_tools_invisible"
            else "the toolset introspection failed, so the delegate tools' "
            "(delegate_start/delegate_wait/delegate_cancel) visibility could "
            "not be verified"
        )
        text = (
            "⚠️ EXECUTOR_UNAVAILABLE: this subagent was pinned to the delegated "
            f"substrate (executor='harness'), but {detail}, so the pin cannot be "
            "honored. The task was NOT run on metered API tokens. Fix the tool "
            "policy / task contract that hides the delegate verbs, or schedule "
            "again with executor='auto' to accept metered spend."
        )
        # Literal codes (not `decision.reason`) so the provenance drift guard
        # keeps seeing every code the runtime can emit.
        if decision.reason == "delegate_visibility_unverified":
            return text, {"execution_status": "infra_failed", "reason_code": "delegate_visibility_unverified"}
        return text, {"execution_status": "infra_failed", "reason_code": "delegate_tools_invisible"}
    text = (
        "⚠️ EXECUTOR_UNAVAILABLE: this subagent was pinned to the delegated substrate "
        f"(executor='harness') and the route cannot run: {decision.reason}."
        + (f" It resets at {decision.reset_at}." if decision.reset_at else "")
        + " The task was NOT run on metered API tokens, because that spend is exactly "
        "what the pin exists to prevent. Reschedule once the route recovers, or "
        "schedule it again with executor='auto' to accept metered spend."
    )
    return text, {
        "execution_status": "infra_failed",
        "reason_code": "subagent_executor_unavailable",
    }


def _record_executor_resolution(
    drive_logs: Any, task: Dict[str, Any], dispatch: Optional[SubagentDispatch],
) -> None:
    """Durably record the typed substrate decision (re-homed from the retired
    `_announce_dispatch_executor`): who was asked for, who runs it, why, and —
    when every plan window is spent — the instant it heals."""
    if dispatch is None or dispatch.executor_resolution is None:
        return
    res = dispatch.executor_resolution
    row = {
        "ts": utc_now_iso(), "type": "subagent_executor_resolved",
        "task_id": str(task.get("id") or ""),
        "requested": res.requested,
        "executor": res.executor,
        "reason": res.reason,
        "reset_at": res.reset_at,
        "route": res.route.route_id if res.route else "",
    }
    append_jsonl(drive_logs / "events.jsonl", row)
    # ALSO the canonical events log: a delegated child's forked drive is pruned
    # with the task, so this used to be the ONLY copy of the substrate decision
    # (submarine forensics: zero subagent_executor_resolved rows in the canonical
    # events.jsonl). The accounting axis the task already carries names the
    # canonical root; the root agent's own drive IS canonical, so skip the dup.
    try:
        budget_root = str(task.get("budget_drive_root") or "").strip()
        if budget_root:
            canonical_logs = pathlib.Path(budget_root) / "logs"
            if canonical_logs.resolve(strict=False) != pathlib.Path(drive_logs).resolve(strict=False):
                append_jsonl(canonical_logs / "events.jsonl", row)
    except Exception:
        log.debug("Failed to mirror executor resolution to canonical events", exc_info=True)
    # D28 exhaustion beacon: surface the spent-window fact to the WAITING parent
    # NOW (via the task-tree attention channel the wait tools already poll),
    # not at absorption after the wait window burned.
    if res.reason == "subscription_window_exhausted" and str(task.get("parent_task_id") or "").strip():
        root_id = str(task.get("root_task_id") or "").strip()
        if root_id:
            try:
                from ouroboros.task_tree_ledger import record_subscription_window_exhausted

                record_subscription_window_exhausted(
                    root_id,
                    child_task_id=str(task.get("id") or ""),
                    reset_at=res.reset_at,
                    route=res.route.route_id if res.route else "",
                    executor=res.executor,
                )
            except Exception:
                log.debug("Failed to append subscription-window beacon", exc_info=True)


def _blocked_executor_terminal(cap_info: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """p34's typed terminal for a blocked executor pin, rebuilt from the facts
    cap_info carried across the (ctx, messages, cap_info) seam. The placeholder
    method p2 kept for exactly this synthesis is deleted; this is the one body."""
    text, usage = executor_blocked_outcome(SubagentExecutorResolution(
        requested=str(cap_info.get("executor_blocked_requested") or "harness"),
        executor="blocked",
        reason=str(cap_info.get("executor_blocked_reason") or ""),
        reset_at=str(cap_info.get("executor_blocked_reset_at") or ""),
    ))
    return text, usage, {"reasoning_notes": ["subagent_executor_unavailable"], "tool_calls": []}


def resolve_dispatch_axes(task: Dict[str, Any]) -> Optional[SubagentDispatch]:
    """Resolve WHAT THIS CHILD GETS, once, and stamp it onto the record it came from.

    ``None`` when the task is not a delegated child. This is the ONE place a child's
    model, effort, route, tool profile and effective executor are decided, and the
    one author of its ``capability_delta``. It writes back onto the live task dict so
    every downstream surface — the RUNNING task result, the task-metadata projection
    the loop reads, the completion write, the envelope — describes the SAME
    resolution instead of each re-deriving its own from whatever it happens to hold.
    """
    if str(task.get("delegation_role") or "").lower() != "subagent":
        return None
    dispatch = resolve_subagent_dispatch(task, task_type=str(task.get("type") or "task"))
    task.update(dispatch.record_fields())
    # The envelope is the child's public description, so it is rebuilt from the
    # record the resolution just wrote rather than left holding the requested-status
    # copy the scheduler made — through the ONE record->envelope mapping, so it
    # cannot describe a different child than the record does.
    task["subagent_envelope"] = envelope_from_task(task, status=STATUS_RUNNING)
    return dispatch


def emit_dispatch_resolution(
    event_queue: Any, task: Dict[str, Any], dispatch: Optional[SubagentDispatch],
) -> None:
    """Report the dispatch-time resolution back to the supervisor (XG-2R.1).

    ``resolve_dispatch_axes`` stamps the WORKER process's clone of the task; the
    supervisor's RUNNING copy — the one ``persist_queue_snapshot`` serializes — is a
    separate dict made at assignment, so without this report a restart restored the
    unresolved intent and lost `effective_model_lane`, `reasoning_effort`, the
    executor fields and `capability_delta`. The report rides the SAME worker event
    channel every other worker fact uses (no second channel);
    ``supervisor/events.py::_handle_task_dispatch_resolved`` merges exactly
    ``SUBAGENT_RESOLUTION_FIELDS`` into RUNNING under the queue lock. Best-effort by
    design: the durable task_result written moments before this remains the record
    of authority — the merge keeps the supervisor's live mirror and its snapshot
    telling the same story.
    """
    if dispatch is None or event_queue is None:
        return
    try:
        event_queue.put({
            "type": "task_dispatch_resolved",
            "task_id": str(task.get("id") or ""),
            "resolution": {
                key: task.get(key) for key in SUBAGENT_RESOLUTION_FIELDS if key in task
            },
            "ts": utc_now_iso(),
        })
    except Exception:
        log.debug("Failed to report dispatch resolution to the supervisor", exc_info=True)


__all__ = [
    "dispatch_executor_note",
    "executor_blocked_outcome",
    "_record_executor_resolution",
    "_blocked_executor_terminal",
    "resolve_dispatch_axes",
    "emit_dispatch_resolution",
]
