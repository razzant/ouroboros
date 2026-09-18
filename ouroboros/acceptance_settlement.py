"""What happens to a paid acceptance panel that outlives the answer it reviewed.

A reviewer panel is advice for its author, never a signature on bytes the
reviewers did not read (owner decision D4=A, 2026-09-16). The three moments
below share one fact — the panel keeps its custody and paid identity while
Main moves on — so they live together:

* the wave wakes Main at its own quorum and again when the last slot settles,
  and the wake carries each reviewer's own verdict
  (``announce_acceptance_settlement``);
* final delivery under that panel's feedback buys no second panel
  (``_deliver_under_running_panel``), whether feedback returned ready or pending.
  Main waits for a pending panel — the
  default, and the only option under blocking enforcement — or consciously
  finishes; a panel that PASSED the earlier revision accepts the task and the
  owner row says so (fork 1=B), while any other settled verdict hands the
  delivery to the ordinary acceptance path with the collected verdicts in its
  dialogue history;
* a panel that settles after its task ended is collected at $0, republished on
  the task's own review projection with the host's own settlement note, and
  announced once in the task's room as one row of that card's Reviews group
  (``attach_late_acceptance_settlement``); no model turn starts (fork 2=A).
"""
from __future__ import annotations

import logging
import pathlib
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# The host acceptance reason for "the paid panel approved the answer Main has
# since rewritten": the task is accepted on the reviewers' word about the
# earlier revision and the owner row says so. Not a member of
# ``outcomes._ACCEPTANCE_BLOCKED_TERMINAL_REASONS``: nothing was refused and no
# review cycle was spent.
REASON_PREVIOUS_REVISION_ACCEPTED = "previous_revision_accepted"

# The chat row a panel writes when it settles after its task already ended.
LATE_SETTLEMENT_SYSTEM_TYPE = "acceptance_late_settlement"

# The mailbox row a settling acceptance wave writes. It carries the reviewers'
# OWN verdicts: a wake that only said results existed, and asked Main to reply
# ``keep`` to read them, made the verdicts reachable only by not having moved
# on. The reducer's quorum verdict still lands through the ordinary collection
# at the next acceptance entry; these are the individual reviewers, unreduced.
ACCEPTANCE_SETTLEMENT_WAKE = (
    "Acceptance review {retry_key}: {settled} of {total} reviewer slot(s) have answered on "
    "the answer that was under review. These are their own verdicts — advice for you, not "
    "a signature on your current draft, and they do not stop you from finishing. A rewritten "
    "answer gets a verdict of its own only when you nominate it again."
)


def _reviewer_lines(wave: Dict[str, Any]) -> List[str]:
    """One line per roster slot: the reviewer's own verdict and note, or pending."""
    slots = wave.get("slots") or {}
    verdicts = wave.get("verdicts") or {}
    lines: List[str] = []
    for slot_id, status in slots.items():
        row = verdicts.get(slot_id) if isinstance(verdicts.get(slot_id), dict) else {}
        verdict = str(row.get("verdict") or "") or str(status or "pending")
        note = str(row.get("note") or "")
        lines.append(f"- {slot_id}: {verdict}" + (f" — {note}" if note else ""))
    return lines


def acceptance_settlement_message(request: Any, wave: Dict[str, Any]) -> str:
    """Render the settled wave as the reviewers' own lines, bounded per slot."""
    slots = wave.get("slots") or {}
    # Slots that answered before the release were collected by the drain: they
    # count as answered, and their verdicts sit in the collected panel itself.
    early = len(wave.get("answered_before_release_ids") or ())
    head = ACCEPTANCE_SETTLEMENT_WAKE.format(
        retry_key=str(getattr(request, "retry_key", "") or ""),
        settled=early + sum(1 for status in slots.values() if status),
        total=max(int(wave.get("total") or 0), len(slots) + early),
    )
    lines = _reviewer_lines(wave)
    if early:
        lines.append(f"- {early} slot(s) answered before the release; their verdicts are in the collected panel")
    return "\n".join([head, *lines])


def _result_root(usage_ctx: Any) -> pathlib.Path:
    """The root the task result lives under, resolved the way the acceptance
    projection writer resolves it (a forked drive keeps its budget root)."""
    meta = getattr(usage_ctx, "task_metadata", {})
    meta = meta if isinstance(meta, dict) else {}
    return pathlib.Path(meta.get("budget_drive_root") or getattr(usage_ctx, "budget_drive_root", None)
                        or usage_ctx.drive_root)


# The loop exit restores the context's ``_execution_trace`` to whatever it was
# before the loop ran (``loop_budget._cleanup_loop_resources``), so a wave that
# settles after the turn ended would find no trace to attach to. A panel that
# went pending registers the trace it belongs to here, keyed by the wave's
# retry key; the supplement reads it back and drops it once the wave settled.
_SETTLEMENT_TRACE_CAP = 8


def remember_settlement_trace(tools_ctx: Any, llm_trace: Dict[str, Any], run: Dict[str, Any]) -> None:
    """Keep the trace a pending panel belongs to reachable past the loop exit."""
    request = run.get("request") if isinstance(run, dict) else None
    retry_key = str((request or {}).get("retry_key") or "") if isinstance(request, dict) else ""
    if not retry_key:
        return
    traces = getattr(tools_ctx, "_acceptance_settlement_traces", None)
    if not isinstance(traces, dict):
        traces = {}
        tools_ctx._acceptance_settlement_traces = traces
    traces.pop(retry_key, None)
    traces[retry_key] = llm_trace
    while len(traces) > _SETTLEMENT_TRACE_CAP:
        traces.pop(next(iter(traces)))


def _settlement_trace(usage_ctx: Any, retry_key: str) -> Optional[Dict[str, Any]]:
    """The trace holding this wave's run: the live one while the loop runs, the
    remembered one after it exited."""
    def holds(trace: Any) -> bool:
        return isinstance(trace, dict) and any(
            isinstance(run, dict) and run.get("authority") == "host_root"
            and isinstance(run.get("request"), dict)
            and str(run["request"].get("retry_key") or "") == retry_key
            for run in (trace.get("review_runs") or []))

    live = getattr(usage_ctx, "_execution_trace", None)
    if holds(live):
        return live
    remembered = (getattr(usage_ctx, "_acceptance_settlement_traces", None) or {}).get(retry_key)
    return remembered if holds(remembered) else None


def announce_acceptance_settlement(usage_ctx: Any, request: Any, wave: Dict[str, Any]) -> None:
    """Deliver a settled acceptance wave to whoever can still act on it.

    While the turn is alive the reviewers' verdicts go to Main's existing
    mailbox, which is drained before every round and also ends an acceptance
    park. Once the task is terminal there is nobody to wake: the same verdicts
    are attached to the task result and announced once in the task's own room,
    so the owner sees them and the next turn reads them in chat history. Never a
    model turn, never a second paid dispatch. Runs on the settlement thread,
    outside custody locks.
    """
    if usage_ctx is None or not getattr(usage_ctx, "drive_root", None):
        return
    task_id = str(getattr(request, "task_id", "") or "")
    try:
        from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result

        row = load_task_result(_result_root(usage_ctx), task_id) or {}
        if str(row.get("status") or "") in _TRULY_TERMINAL_STATUSES:
            attach_late_acceptance_settlement(usage_ctx, request, wave, result=row)
            return
        from ouroboros.owner_mailbox import write_task_message
        trace = _settlement_trace(usage_ctx, str(getattr(request, "retry_key", "") or ""))
        source = next(({"task_id": task_id, "run_index": index,
                        "binding_hash": str(run.get("binding_hash") or "")}
                       for index, run in enumerate((trace or {}).get("review_runs") or [])
                       if run.get("authority") == "host_root"
                       and (run.get("request") or {}).get("retry_key") == request.retry_key), None)
        write_task_message(pathlib.Path(usage_ctx.drive_root), acceptance_settlement_message(request, wave),
                           task_id, source_task_id=task_id, provenance="system", review_feedback=source)
    except Exception:
        log.warning("Acceptance settlement delivery failed for %s", task_id, exc_info=True)


def panel_awaiting_this_turn(tools_ctx: Any, llm_trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Read the panel this turn released without changing delivery authority.

    The binding recorded when it went pending is the physical identity a
    re-authored answer cannot move; current owner-source validation belongs
    to delivery, never to the wait's settlement check.
    """
    binding = str(getattr(tools_ctx, "_task_acceptance_pending", "") or "")
    if not binding:
        return None
    return next((run for run in reversed(llm_trace.get("review_runs") or [])
                 if isinstance(run, dict) and run.get("authority") == "host_root"
                 and str(run.get("binding_hash") or "") == binding), None)


def awaited_panel_has_settled(tools_ctx: Any, llm_trace: Dict[str, Any]) -> bool:
    """Whether the panel this turn waits for has already settled (a $0 look).

    Settled means its verdicts already woke Main and sit in the transcript: the
    only thing left is the next model round — a control repair, for one — so
    the loop must run it instead of parking behind a settlement that will never
    arrive again (the keyless E2E lane hung that way until the task deadline).
    The run record is reconciled at $0 first, exactly as delivery does, because
    the trace learns of a settlement only through that collection.
    """
    from ouroboros.loop_acceptance_review import acceptance_run_pending
    from ouroboros.review_dispatch import reconcile_pending_acceptance_runs

    run = panel_awaiting_this_turn(tools_ctx, llm_trace)
    if run is None:
        return False
    if acceptance_run_pending(run):
        try:
            reconcile_pending_acceptance_runs(
                {"review_runs": [run]}, drive_root=pathlib.Path(tools_ctx.drive_root),
                usage_ctx=tools_ctx)
        except Exception:
            log.debug("awaited acceptance panel could not be reconciled", exc_info=True)
    return not acceptance_run_pending(run)


def acceptance_choice_offered() -> bool:
    """Whether the host can honour a wait/finish choice on this install.

    Blocking enforcement always waits; Cyber Pro never does (Main's final
    response is its decision there, unchanged). Only advisory enforcement on an
    ordinary runtime mode leaves the choice to Main, so only there is it offered.
    """
    from ouroboros.config import get_review_enforcement, get_runtime_mode
    from ouroboros.runtime_mode_policy import runtime_mode_at_least

    return (get_review_enforcement() != "blocking"
            and not runtime_mode_at_least(get_runtime_mode(), "cyber_pro"))


def acceptance_wait_chosen(tools_ctx: Any) -> bool:
    """Waiting is the default; only an explicit, current ``finish`` releases an
    answer over a panel that is still running where the install lets Main choose."""
    from ouroboros.config import get_review_enforcement, get_runtime_mode
    from ouroboros.runtime_mode_policy import runtime_mode_at_least

    if runtime_mode_at_least(get_runtime_mode(), "cyber_pro"):
        return False
    if get_review_enforcement() == "blocking":
        return True
    return str(getattr(tools_ctx, "_acceptance_pending_review_choice", "") or "") != "finish"


def _deliver_under_running_panel(ctx: Any, prior_run: Any) -> Optional[bool]:
    """A DELIVERY is not a nomination: it neither buys a panel nor is refused one.

    The paid panel keeps its identity whether it returned ready or pending.
    While it runs, Main waits (the default and the only option under blocking
    enforcement) or consciously finishes. Once it has settled on the earlier
    revision: a PASS accepts the task on the reviewers' word and the owner row
    says so (fork 1=B); any other verdict is not a verdict on this answer, so the
    ordinary path decides — a new panel while the review cap allows, otherwise
    its typed capacity refusal — with the collected verdicts already in its
    dialogue history. ``None`` means "not this case". While the panel is still
    running, a rewritten answer buys a NEW panel only when Main nominates it
    again (``_acceptance_review_only``).
    """
    from ouroboros import loop
    from ouroboros.loop_acceptance_review import (
        _end_acceptance_terminal, _finish_cyber_acceptance,
        _set_applied_host_acceptance_impact, acceptance_run_pending,
    )
    from ouroboros.loop_delivery import delivery_subject_hash
    from ouroboros.loop_messages import owner_source_sha256
    from ouroboros.outcomes import ACCEPTANCE_ACCEPTED

    tools_ctx = ctx.tools._ctx
    if prior_run is not None or getattr(tools_ctx, "_acceptance_review_only", False):
        return None
    run = panel_awaiting_this_turn(tools_ctx, ctx.llm_trace)
    if run is None:
        # Initial release may return a completed panel. Its delivered feedback
        # belongs to this turn without pretending the operation is still pending.
        run = next((row for row in reversed(ctx.llm_trace.get("review_runs") or [])
                    if isinstance(row, dict) and row.get("authority") == "host_root"), None)
        if (not run or not run.get("feedback_delivered")
                or run.get("superseded_reason") != "delivery_candidate_replaced"):
            return None
    reviewed_text = (run.get("request") or {}).get("subject", "")
    if run.get("subject_hash") != delivery_subject_hash(tools_ctx, ctx.llm_trace, reviewed_text):
        return None  # A held rewrite may have acquired new material after supersession.
    if run.get("superseded_by_revision") and run.get("superseded_reason") != "delivery_candidate_replaced":
        return None  # The existing subject/effect owner already invalidated this feedback.
    # Older owner premises cannot authorize this delivery (owner rule 4=A).
    # The panel keeps its physical custody and still arrives as advice.
    reviewed_source = str(run.get("owner_source_sha256") or "")
    if reviewed_source and reviewed_source != str(owner_source_sha256(tools_ctx) or ""):
        tools_ctx._task_acceptance_pending = ""
        return None
    if acceptance_run_pending(run):
        if acceptance_wait_chosen(tools_ctx):
            ctx.emit_progress("Task acceptance review is still running; holding the answer for its verdict.")
            return True
        return _finish_cyber_acceptance(ctx, SimpleNamespace(**run))
    tools_ctx._task_acceptance_pending = ""
    if str(run.get("aggregate_signal") or "").upper() != "PASS":
        return None
    result = SimpleNamespace(**run)
    _set_applied_host_acceptance_impact(run, result, requires_revision=False)
    ctx.llm_trace.setdefault("review_decision", {}).update({
        "panel_id": str(run.get("panel_id") or ""),
        "binding_hash": str(run.get("binding_hash") or ""),
    })
    _end_acceptance_terminal(ctx, "pass")
    loop._set_acceptance_decision(ctx.llm_trace, {
        "status": ACCEPTANCE_ACCEPTED,
        "reason": REASON_PREVIOUS_REVISION_ACCEPTED,
        "source": "task_acceptance_review",
        "reviewer_signal": "PASS",
        "reviewed_panel_id": str(run.get("panel_id") or ""),
        "reviewed_candidate_hash": str(run.get("candidate_hash") or ""),
    })
    ctx.emit_progress(
        "Task acceptance review: PASS on the earlier revision of this answer; accepted on the "
        "reviewers' word (the rewrite itself was not re-reviewed)."
    )
    return False


def _unsettled_head(run: Dict[str, Any]) -> str:
    """The head of a panel that settled without a PASS or FAIL.

    It states what the host holds (no settled verdict), not what the panel did:
    DEGRADED can be a reviewer's own deliberate answer. A reviewer whose PHYSICAL
    outcome the host does not know — custody lost, or a dispatch still owed —
    did not stay silent, so the row names it as unknown.
    """
    unknown = sum(1 for actor in (run.get("actors") or [])
                  if isinstance(actor, dict)
                  and (bool(actor.get("late_result_pending"))
                       or str(actor.get("operation_state") or "")
                       in ("custody_lost", "pending_dispatch")))
    tail = ("" if not unknown
            else " — 1 reviewer's outcome is still unknown" if unknown == 1
            else f" — {unknown} reviewers' outcomes are still unknown")
    return "Reviewers later returned no settled verdict on this answer" + tail + "."


def _late_settlement_text(run: Dict[str, Any], wave: Dict[str, Any]) -> str:
    """The owner row: which verdict, which revision, and the reviewers' own lines."""
    signal = str(run.get("aggregate_signal") or "").upper()
    head = ({"PASS": "Reviewers later passed this answer.",
             "FAIL": "Reviewers later rejected this answer."}.get(signal)
            or _unsettled_head(run))
    which = (" They reviewed the earlier version, which was rewritten before delivery."
             if run.get("superseded_by_revision") else " They reviewed the answer that was delivered.")
    return "\n".join([head + which, *_reviewer_lines(wave)])


def attach_late_acceptance_settlement(usage_ctx: Any, request: Any, wave: Dict[str, Any],
                                      *, result: Dict[str, Any]) -> bool:
    """A panel that settled after its task ended is a supplement, never a new turn.

    Its verdicts are collected at $0 over the recorded operation, republished on
    the task's own review projection through the existing locked writer, and
    announced ONCE in the task's room through the existing terminal-delivery
    outbox (durably owed, keyed by ``delivery_id``; a second settlement of the
    same wave finds nothing left to reconcile and announces nothing). The owner
    sees it; the next turn reads it in chat history. The acceptance twin of plan
    review's historical supplement (docs/architecture/06-agent-core.md).
    """
    retry_key = str(getattr(request, "retry_key", "") or "")
    task_id = str(getattr(request, "task_id", "") or "")
    trace = _settlement_trace(usage_ctx, retry_key) if retry_key else None
    if trace is None:
        log.debug("late acceptance settlement %s: no trace holds this wave (worker rebound?)", retry_key)
        return False  # the worker was rebound to another task; the record stays as published
    runs = [run for run in (trace.get("review_runs") or [])
            if isinstance(run, dict) and run.get("authority") == "host_root"
            and isinstance(run.get("request"), dict)
            and str(run["request"].get("retry_key") or "") == retry_key]
    from ouroboros.loop_acceptance_review import acceptance_run_pending
    from ouroboros.review_dispatch import reconcile_pending_acceptance_runs
    from ouroboros.review_projection import publish_acceptance_checkpoint
    from supervisor.terminal_delivery import enqueue_terminal_delivery

    root = pathlib.Path(usage_ctx.drive_root)
    # Only THIS wave's runs: reconciling every pending panel here would let one
    # settlement collect a sibling panel's verdicts and leave that panel's own
    # settlement with nothing to announce (the run objects are shared with the trace).
    advanced = reconcile_pending_acceptance_runs({"review_runs": runs}, drive_root=root, usage_ctx=usage_ctx)
    if not any(acceptance_run_pending(run) for run in runs):
        (getattr(usage_ctx, "_acceptance_settlement_traces", None) or {}).pop(retry_key, None)
    if not advanced:
        log.debug("late acceptance settlement %s: nothing reconciled (still pending or already collected)", retry_key)
        return False
    # The sentence has ONE author: it is stamped on the exact run this wave
    # reconciled, so the republished projection carries the same bytes the row
    # does and the card's Reviews group prints them verbatim.
    settled = runs[-1]
    note = _late_settlement_text(settled, wave)
    settled["late_settlement"] = {
        "note": note,
        "reviewed_revision": "earlier" if settled.get("superseded_by_revision") else "delivered",
        "settled_after_terminal": True,
    }
    publish_acceptance_checkpoint(usage_ctx, trace, task_id=task_id, drive_root=_result_root(usage_ctx),
                                  chat_id=result.get("chat_id"))
    return bool(enqueue_terminal_delivery(root, {
        "type": "send_message", "chat_id": int(result.get("chat_id") or 0), "task_id": task_id,
        "text": note,
        "role": "system", "system_type": LATE_SETTLEMENT_SYSTEM_TYPE,
        "delivery_id": f"acceptance-late:{retry_key}",
        # The verdict belongs inside the task's card, in its Reviews group, and
        # stays one row across live delivery, outbox replay and history.
        "progress_meta": {"card_row": "reviews", "card_row_id": f"acceptance-late:{retry_key}"},
    }, event_queue=getattr(usage_ctx, "event_queue", None)))


def expose_acceptance_feedback(trace: Dict[str, Any], messages: list, task_id: str) -> None:
    """Mark exact host feedback carried by a Main request that returned a response.

    Queuing or appending a message alone never validates an author response.
    """
    if not isinstance(trace, dict):
        return
    runs = trace.get("review_runs") or []
    for message in messages:
        if not isinstance(message, dict):
            continue
        for source in message.get("review_feedback") or []:
            if not isinstance(source, dict) or source.get("task_id") != task_id:
                continue
            outcome = trace.get("acceptance_review_outcome") or {}
            if source.get("outcome_binding_hash") and source["outcome_binding_hash"] == outcome.get("binding_hash"):
                outcome["feedback_delivered"] = True
                continue
            index = source.get("run_index")
            if type(index) is not int or not 0 <= index < len(runs):
                continue
            run = runs[index]
            if (isinstance(run, dict) and run.get("authority") == "host_root"
                    and str(run.get("binding_hash") or "") == source.get("binding_hash", "")):
                run["feedback_delivered"] = True
