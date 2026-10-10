"""What happens to a paid acceptance panel that outlives the answer it reviewed.

A reviewer panel is advice for its author, never a signature on bytes the
reviewers did not read (owner decision D4=A, 2026-09-16). The three moments
below share one fact — the panel keeps its custody and paid identity while
Main moves on — so they live together:

* the wave wakes Main at its own quorum and again when the last slot settles,
  and the wake carries each reviewer's own verdict
  (``announce_acceptance_settlement``);
* final delivery of a text-only rewrite under that panel's feedback buys no
  second panel (``_deliver_under_running_panel``), whether feedback returned
  ready or pending. Main waits for a pending panel — the
  default, and the only option under blocking enforcement — or consciously
  finishes; a panel that PASSED the earlier revision accepts the task and the
  owner row says so (fork 1=B), while a changed subject (criteria, material
  evidence, owner source) or any other settled verdict hands the
  delivery to the ordinary acceptance path with the collected verdicts in its
  dialogue history;
* a panel that settles after its task ended is collected at $0, republished on
  the task's own review projection with the host's own settlement note and a
  neutral late-evidence fact (which version the reviewers read, what the host
  proved it emitted, when it settled and the exact source), and announced once
  in the task's room as one row of that card's Reviews group
  (``attach_late_acceptance_settlement``). The fact is evidence for Ouroboros
  to judge, never a host decision: no model turn starts here (fork 2=A). A
  worker that no longer holds the trace falls back to the canonical published
  source, and a controller that died is collected by the existing maintenance
  pass through the same ``settle_acceptance_operation``;
* a forced rail that ends the turn while the panel is still out collects it at
  $0 before recording anything, so the rail's "never reviewed" reason is never
  stamped over a panel that ran (``forced_rail_panel_verdict``).
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import pathlib
import threading
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from ouroboros.plan_review_facts import learn_from_late_settlement
from ouroboros.utils import utc_now_iso
# Late settlement reads the canonical result, including a forked task's budget root.
from ouroboros.tool_access_paths import canonical_data_root as _result_root

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
    """One line per roster slot: the reviewer's own verdict and note, or pending.

    This is the MODEL mailbox's form (``acceptance_settlement_message``): slot
    ids, the verdict tokens and the bounded note with its disclosed omission
    marker. The owner's row is composed separately (``_owner_reviewer_lines``).
    """
    slots = wave.get("slots") or {}
    verdicts = wave.get("verdicts") or {}
    lines: List[str] = []
    for slot_id, status in slots.items():
        row = verdicts.get(slot_id) if isinstance(verdicts.get(slot_id), dict) else {}
        verdict = str(row.get("verdict") or "") or str(status or "pending")
        note = str(row.get("note") or "")
        lines.append(f"- {slot_id}: {verdict}" + (f" — {note}" if note else ""))
    return lines


# ---------------------------------------------------------------------------
# The owner's reading of a settled reviewer (issue #1369). The record keeps the
# tokens: ``reviewer_outputs`` carries slot id, operation and PASS/FAIL/DEGRADED,
# the mailbox line keeps the bounded note with its omission marker, the applied
# review JSON keeps every byte. The row a person reads names the reviewer by the
# model that answered, says the verdict in words and discloses a shortening in
# words (DESIGN §4: internal reason codes belong in details and diagnostics).
# ---------------------------------------------------------------------------

_OWNER_NOTE_LIMIT = 400
# The verdict a reviewer actually settled with, in the owner's words. DEGRADED is
# a reviewer's own deliberate non-verdict; it is neither a refusal nor an absence.
_OWNER_VERDICT_WORDS = {"PASS": "passed it", "FAIL": "rejected it", "DEGRADED": "inconclusive"}
# The slot states that carry no verdict: an answer still owed, a physical outcome
# the host does not know, a request never sent, and a settled failure.
_OWNER_AWAITED = "still awaited"
_OWNER_UNKNOWN = "outcome unknown"
_OWNER_NOT_SENT = "not sent"
_OWNER_UNAVAILABLE = "unavailable"
_OWNER_UNREADABLE = "answered, but no verdict could be read"
_OWNER_SHORTENED = "… (shortened; the complete text is in the review record)"


def _slot_row(rows: Any, slot_id: str) -> Dict[str, Any]:
    """One slot's row of a run's ``actors`` or its ``slot_roster``, or {}."""
    return next((row for row in rows or [] if isinstance(row, dict) and str(row.get("slot_id") or "") == slot_id), {})


def _reviewer_identity(actor: Dict[str, Any], roster_row: Dict[str, Any]) -> Dict[str, str]:
    """Which model this slot ran as: the positively observed one when the route
    reported it, else the requested one (labelled as requested), else unknown.

    Only an engine's final-attempt report (``usage.observed_attempt``) is an
    observation. ``usage.resolved_model`` is not one: a direct API route writes
    its own target there — what the host sent, not what answered."""
    usage = actor.get("usage") if isinstance(actor.get("usage"), dict) else {}
    attempt = usage.get("observed_attempt") if isinstance(usage.get("observed_attempt"), dict) else {}
    observed = str(attempt.get("model") or "").strip()
    requested = str(actor.get("model") or roster_row.get("model") or "").strip()
    if observed:
        label = observed
    elif requested:
        label = f"{requested} (requested)"
    else:
        label = "unknown reviewer"
    return {"label": label, "model": observed, "requested_model": requested}


def _owner_outcome(actor: Dict[str, Any], status: str, verdict: str) -> str:
    """The reviewer's own settled outcome in the owner's words; tokens stay in the record.

    Only an answer carries a verdict. PASS and FAIL are only ever parsed from one,
    but DEGRADED is also the parser's word for NO text: ``_settled_slot_verdict``
    gives it to a failed, refused or unknown slot, whose own state speaks instead.
    """
    parsed = actor.get("parsed")
    if verdict in {"PASS", "FAIL"} or (verdict == "DEGRADED" and parsed is not None):
        return _OWNER_VERDICT_WORDS[verdict]
    # Text outside the contract is an unreadable answer, whatever custody state the
    # row still carries: a row that holds an answer is judged by it.
    if parsed is not None or str(actor.get("raw_text") or "").strip():
        return _OWNER_UNREADABLE
    state = str(actor.get("operation_state") or "")
    if not status or state in {"pending_dispatch", "in_flight"}:
        return _OWNER_AWAITED
    if state == "custody_lost" or bool(actor.get("late_result_pending")):
        return _OWNER_UNKNOWN
    if status == "not_dispatched" or state == "not_dispatched":
        return _OWNER_NOT_SENT
    if status in {"ok", "empty"}:
        return _OWNER_UNREADABLE  # the call returned, with no text to read
    # The cause is the delegated engine's own ``failure.safeMessage`` (``run_failure_cause``),
    # never the provider's or the reviewer's: the row names whose words they are. A cut made
    # by the host's bound is said in words; its model-facing marker stays in the record.
    from ouroboros.gateways.claudexor import reported_cause_words

    words, shortened = reported_cause_words(actor.get("reported_cause"))
    cause = " ".join(words.split())
    quote = f'"{cause}…" (shortened)' if shortened else f'"{cause}"'
    return f"{_OWNER_UNAVAILABLE} — the engine reported: {quote}" if cause else _OWNER_UNAVAILABLE


def owner_bounded_text(text: str, limit: int, *, in_record: bool) -> str:
    """The SSOT display bound (``truncate_review_artifact``: same limit, same
    anti-waste floor) with its model-facing marker said in words for a person.
    The marker is replaced only where the SSOT itself appended it — no reviewer
    text is pattern-matched — and the words point at the review record that
    holds the complete text only when ``in_record`` says it holds it."""
    from ouroboros.utils import truncate_review_artifact

    full = str(text or "")
    bounded = truncate_review_artifact(full, limit=limit)
    marker = f"\n⚠️ OMISSION NOTE: truncated at {limit} chars; original length {len(full)}"
    if bounded == full or not bounded.endswith(marker):
        return bounded
    return bounded[: -len(marker)].rstrip() + (_OWNER_SHORTENED if in_record else "… (shortened)")


def _owner_shortened(text: str, limit: int = _OWNER_NOTE_LIMIT) -> str:
    """One reviewer's note on one line of the owner's row, bounded in words.

    The note is composed before the panel's review record is stored, and that
    store can fail, so the cut says only that it was shortened; the row's
    record link, offered only for a stored record, is the pointer."""
    return owner_bounded_text(" ".join(str(text or "").split()), limit, in_record=False)


def _owner_note(actor: Dict[str, Any], verdict_row: Dict[str, Any]) -> str:
    """The reviewer's own summary from its complete answer, shortened in words.

    The mailbox note (``verdict_row['note']``) is already bounded with the
    model-facing marker; a person's row re-reads the complete answer instead of
    editing that marker, and falls back to the mailbox note only for a slot that
    answered while the run holds no answer text, and only when that note carries
    no marker of its own."""
    from ouroboros.triad_review import parse_review_findings

    parsed = actor.get("parsed")
    findings: List[Any] = []
    raw = str(actor.get("raw_text") or "")
    if parsed is None and raw.strip():
        try:
            parsed, findings, _signal = parse_review_findings(raw)
        except Exception:
            log.debug("late acceptance note could not be parsed for the owner row", exc_info=True)
            parsed, findings = None, []
    elif isinstance(parsed, dict):
        rows = parsed.get("findings")
        findings = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    elif isinstance(parsed, list):
        findings = [row for row in parsed if isinstance(row, dict)]
    note = str(parsed.get("summary") or "") if isinstance(parsed, dict) else ""
    note = note or next((str(row.get("recommendation") or row.get("item") or "")
                         for row in findings if isinstance(row, dict)), "")
    # A failure code or parse diagnostic stays in the task detail and Logs; the
    # owner's line quotes only the reviewer's own words (or the engine's
    # reported cause, already in the outcome), never the exception text. The
    # mailbox note of a slot that never answered IS that text (its error, which
    # ``_settled_slot_verdict`` keeps whitespace-normalised), so it never falls back.
    answered = (str(actor.get("status") or "") in {"ok", "empty"}
                or str(verdict_row.get("verdict") or "").upper() in {"PASS", "FAIL"})
    if not note and answered and actor.get("parsed") is None and not raw.strip():
        mailbox = str(verdict_row.get("note") or "")
        note = "" if "⚠️ OMISSION NOTE" in mailbox else mailbox
    return _owner_shortened(note)


def _owner_reviewer_lines(run: Dict[str, Any], wave: Dict[str, Any]) -> List[str]:
    """One line per roster slot for the owner: the model that answered, its
    outcome in words and its own note. Two seats of the same model stay two
    lines, distinguished by seat number, never merged.

    Lines follow the run's recorded roster, as the review record lists it: a
    released wave collects its slots from a set, so the wave's own order is
    arbitrary and a seat number taken from it could name another reviewer."""
    slots = wave.get("slots") or {}
    verdicts = wave.get("verdicts") or {}
    rank: Dict[str, int] = {}
    for row in [*(run.get("slot_roster") or []), *(run.get("actors") or [])]:
        if isinstance(row, dict):
            rank.setdefault(str(row.get("slot_id") or ""), len(rank))
    labels: List[str] = []
    rows: List[tuple] = []
    for slot_id, status in sorted(slots.items(), key=lambda item: rank.get(str(item[0]), len(rank))):
        actor = _slot_row(run.get("actors"), str(slot_id))
        verdict_row = verdicts.get(slot_id) if isinstance(verdicts.get(slot_id), dict) else {}
        parsed = actor.get("parsed") if isinstance(actor.get("parsed"), dict) else {}
        verdict = (str(verdict_row.get("verdict") or "")
                   or str(actor.get("semantic_verdict") or actor.get("signal") or "")
                   or str(parsed.get("verdict") or parsed.get("status") or "")).upper()
        identity = _reviewer_identity(actor, _slot_row(run.get("slot_roster"), str(slot_id)))
        labels.append(identity["label"])
        rows.append((identity["label"], _owner_outcome(actor, str(status or ""), verdict),
                     _owner_note(actor, verdict_row) if actor or verdict_row else ""))
    lines: List[str] = []
    seen: Dict[str, int] = {}
    for label, outcome, note in rows:
        seen[label] = seen.get(label, 0) + 1
        seat = f" (seat {seen[label]})" if labels.count(label) > 1 else ""
        lines.append(f"- {label}{seat}: {outcome}" + (f" — {note}" if note else ""))
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
    for trace in (getattr(usage_ctx, "_execution_trace", None),
                  (getattr(usage_ctx, "_acceptance_settlement_traces", None) or {}).get(retry_key)):
        if isinstance(trace, dict) and any(
                isinstance(run, dict) and run.get("authority") == "host_root"
                and isinstance(run.get("request"), dict)
                and str(run["request"].get("retry_key") or "") == retry_key
                for run in trace.get("review_runs") or []):
            return trace
    return None


def acceptance_actor_ended(usage_ctx: Any, task_id: str, result: Dict[str, Any]) -> bool:
    """Canonical completion or a positively ended actor drain; unknown is not ended.

    File copyback may keep the canonical row running after the solve loop ended.
    Maintenance has no live context, so it uses that row's existing child binding.
    """
    from ouroboros.owner_mailbox import mailbox_drain_ended
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES
    from ouroboros.task_status import _child_drive_candidates

    if str(result.get("status") or "") in _TRULY_TERMINAL_STATUSES:
        return True
    actor = pathlib.Path(usage_ctx.drive_root)
    if actor.resolve() == _result_root(usage_ctx).resolve():
        actor = next(iter(_child_drive_candidates(result)), actor)
    return mailbox_drain_ended(actor, task_id)


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
        from ouroboros.task_results import load_task_result

        row = load_task_result(_result_root(usage_ctx), task_id) or {}
        ended = acceptance_actor_ended(usage_ctx, task_id, row)
        if ended and not all((wave.get("slots") or {}).values()):
            # A delayed quorum callback may collect the final actors but still
            # carry pending lines. Only the complete roster owns the terminal
            # notice; a stale wake must not reopen its publication duty either.
            return
        # A mailbox notice is not canonical consumption. Keep the operation's
        # duty through the author's terminal-write gap for maintenance.
        _unpublished(task_id, str(getattr(request, "retry_key", "") or ""), "unpublished")
        if ended:
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
        _unpublished(task_id, str(getattr(request, "retry_key", "") or ""), "unpublished")
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
    dialogue history. ``None`` means "not this case": a changed subject is one
    (the ordinary path reviews it), and while the panel is still running a
    text-only rewrite buys a NEW panel only when Main nominates it again
    (``_acceptance_review_only``).
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


def forced_rail_panel_verdict(tools_ctx: Any, llm_trace: Dict[str, Any], rail_reason: str) -> Dict[str, Any]:
    """What a forced rail may honestly record about the panel its turn owns.

    Every rail bypass reason says "the answer was never reviewed", so one may
    be stamped only when no panel ran. The turn's own panel is collected at $0
    first — the same free collection delivery performs — and then speaks for
    itself: a clean PASS on the SAME subject accepts the answer on the
    reviewers' word; reviewers who had not answered leave it unaccepted with
    ``review_pending``, because an answer that has not arrived is a gap and
    never a verdict; any other settled outcome leaves it unaccepted with no
    verdict established. Returns the decision fields the recorder merges — the
    existing acceptance vocabulary only, no reason is minted here.
    """
    from ouroboros.loop_acceptance_review import acceptance_run_pending
    from ouroboros.loop_delivery import delivery_subject_hash
    from ouroboros.loop_messages import owner_source_sha256
    from ouroboros.outcomes import ACCEPTANCE_ACCEPTED
    from ouroboros.review_dispatch import reconcile_pending_acceptance_runs
    from ouroboros.review_verdict import task_acceptance_is_clean

    run = next((row for row in reversed(llm_trace.get("review_runs") or [])
                if isinstance(row, dict) and row.get("authority") == "host_root"), None)
    unreviewed = {"reason": rail_reason}  # the one way this helper says "this answer was never reviewed"
    if run is None:
        return unreviewed
    if acceptance_run_pending(run):
        try:
            reconcile_pending_acceptance_runs({"review_runs": [run]}, usage_ctx=tools_ctx,
                                              drive_root=pathlib.Path(tools_ctx.drive_root))
        except Exception:
            log.debug("a forced rail could not collect its own acceptance panel", exc_info=True)
        if acceptance_run_pending(run):
            return {"reason": "review_degraded", "review_pending": True}
    reviewed_source = str(run.get("owner_source_sha256") or "")
    if run.get("superseded_by_revision") or (reviewed_source and reviewed_source != str(owner_source_sha256(tools_ctx) or "")):
        return unreviewed  # that panel judged an earlier revision or older owner premises
    reviewed = (run.get("request") or {}).get("subject", "")
    if (not task_acceptance_is_clean(SimpleNamespace(**run))
            or run.get("subject_hash") != delivery_subject_hash(tools_ctx, llm_trace, reviewed)):
        return {"reason": "review_degraded"}
    return {"status": ACCEPTANCE_ACCEPTED, "reason": "clean_pass", "reviewer_signal": "PASS",
            "reviewed_panel_id": str(run.get("panel_id") or "")}


def _unsettled_clause(run: Dict[str, Any]) -> str:
    """The verdict clause of a panel that settled without a PASS or FAIL.

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
    return "reviewers later returned no settled verdict" + tail + "."


# The version the reviewers read comes FIRST, then what they said about it: a
# categorical "this answer was rejected" followed by a qualifier misstates which
# bytes were judged (owner, Batch 2 §4). The version is proven only by exact
# emitted-byte receipts; everything else is "unknown".
_VERSION_CLAUSES = {
    "delivered": "On the delivered version of this answer",
    "different": "On a version of this answer other than the one delivered",
    "unknown": "On the reviewed version of this answer (whether it was the delivered one is unknown)",
}

# Late settlements this process could not publish: the operation's pointer then
# closes as ``unpublished`` so the existing maintenance pass retries it.
_LATE_UNPUBLISHED: set = set()
_LATE_LOCK = threading.Lock()
# Makes "is this notice already owed?" and its enqueue one step for every settler in this process.
_LATE_NOTICE_LOCK = threading.Lock()


def late_publication_owed(task_id: str, retry_key: str) -> bool:
    with _LATE_LOCK:
        return (str(task_id), str(retry_key)) in _LATE_UNPUBLISHED


def _late_settlement_text(run: Dict[str, Any], wave: Dict[str, Any],
                          fact: Optional[Dict[str, Any]] = None) -> str:
    """The owner row: which version, then the verdict, then the reviewers' own lines.

    The reviewer lines are the owner's form (model, outcome in words, note
    shortened in words); the model mailbox keeps its own ``_reviewer_lines``.
    """
    signal = str(run.get("aggregate_signal") or "").upper()
    verdict = ({"PASS": "reviewers later passed it.", "FAIL": "reviewers later rejected it."}.get(signal)
               or _unsettled_clause(run))
    version = str((fact or {}).get("reviewed_revision") or "unknown")
    return "\n".join([f"{_VERSION_CLAUSES.get(version, _VERSION_CLAUSES['unknown'])}, {verdict}",
                      *_owner_reviewer_lines(run, wave)])


def emitted_answer_fact(root: Any, task_id: str) -> Dict[str, Any]:
    """What the host PROVED it emitted as this task's answer: send receipts only.

    Exact text digest, routed chat and retained source, captured by the send
    handler after a real send (``terminal_delivery.terminal_answer_receipts``).
    An owed or unconfirmed send, a delivered id without a receipt, the task's
    current result or an author-disposition hash is never byte proof.
    """
    from supervisor.terminal_delivery import terminal_answer_receipts

    return terminal_answer_receipts(root, task_id)


def late_evidence_fact(run: Dict[str, Any], emitted: Dict[str, Any], *, settled_at: str) -> Dict[str, Any]:
    """Neutral late-review evidence: the exact subject read, what was emitted, and each reviewer's output."""
    request = run.get("request") if isinstance(run.get("request"), dict) else {}
    subject = str(request.get("subject") or "")
    digest, chars = hashlib.sha256(subject.encode("utf-8")).hexdigest(), len(subject)
    # Supersession only proves a later candidate existed; emitted-byte receipts
    # alone establish delivered/different. No receipt remains unknown.
    receipts = [row for row in emitted.get("delivered") or [] if isinstance(row, dict)]
    revision = ("delivered" if any(row.get("text_sha256") == digest for row in receipts)
                else "different" if receipts else "unknown")
    historical = (request.get("policy") or {}).get("historical_acceptance")
    if isinstance(historical, dict):
        from ouroboros.acceptance_history import historical_receipt_matches

        delivery = historical.get("confirmed_delivery") or historical.get("delivery") or {}
        valid = (historical.get("schema_version") == 1 and historical.get("task_id") == request.get("task_id")
                 and historical.get("task_attempt") == request.get("task_attempt")
                 and delivery.get("task_id") == request.get("task_id") and delivery.get("text_sha256") == digest)
        addressed = [row for row in receipts if row.get("basis") == "send_handler_returned"
                     and row.get("source_ref") and (not delivery.get("source_ref") or row["source_ref"] == delivery["source_ref"])
                     and all(row.get(key) == delivery.get(key)
                     for key in ("task_id", "delivery_id", "chat_id"))]
        revision = ("delivered" if valid and any(historical_receipt_matches(delivery, row) for row in addressed)
                    else "different" if valid and addressed else "unknown")
    return {
        "settled_after_terminal": True, "settled_at": settled_at, "reviewed_revision": revision,
        "reviewed_subject": {"retry_key": str(request.get("retry_key") or ""),
                             "panel_id": str(run.get("panel_id") or ""),
                             "binding_hash": str(run.get("binding_hash") or ""),
                             "candidate_hash": str(run.get("candidate_hash") or ""),
                             "subject_sha256": digest, "subject_chars": chars},
        **({"historical_delivery": copy.deepcopy(historical.get("confirmed_delivery"))} if historical else {}),
        "reviewed_superseded": bool(run.get("superseded_by_revision")),
        "emitted_answer": copy.deepcopy(emitted),
        "reviewed_is_emitted": {"delivered": True, "different": False}.get(revision),
        "reviewer_outputs": [
            {"slot_id": str(actor.get("slot_id") or ""), "operation_id": str(actor.get("operation_id") or ""),
             "operation_state": str(actor.get("operation_state") or ""),
             "verdict": str(actor.get("semantic_verdict") or actor.get("signal") or "").upper(),
             # The identity the owner row names (additive): the model the route
             # reported, and the one the roster asked for; "" where unreported.
             "model": _reviewer_identity(
                 actor, _slot_row(run.get("slot_roster"), str(actor.get("slot_id") or "")))["model"],
             "requested_model": _reviewer_identity(
                 actor, _slot_row(run.get("slot_roster"), str(actor.get("slot_id") or "")))["requested_model"],
             "response_ref": dict(actor.get("response_ref") or {})}
            for actor in (run.get("actors") or []) if isinstance(actor, dict)],
    }


def late_acceptance_facts(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every late acceptance fact of one task result, for a reader that wants the evidence.

    Each row names its panel, when it settled, the exact applied source and the
    neutral fact; nothing here ranks, summarizes or decides.
    """
    panels = ((result or {}).get("review_projection") or {}).get("panels") or []
    return [{"task_id": str(result.get("task_id") or ""), "panel_id": str(panel.get("panel_id") or ""),
             "settled_at": str(panel["late_settlement"].get("settled_at") or ""),
             "aggregate_signal": str(panel.get("aggregate_signal") or ""),
             "source_ref": panel.get("applied_source_ref"), "late_settlement": panel["late_settlement"]}
            for panel in panels
            if isinstance(panel, dict) and panel.get("surface") == "task_acceptance"
            and isinstance(panel.get("late_settlement"), dict)]


def canonical_acceptance_trace(root: Any, task_id: str, retry_key: str, *,
                               result: Optional[Dict[str, Any]] = None,
                               checkpoint: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """The wave's runs from the canonical PUBLISHED source when no live trace holds them.

    Each published acceptance panel carries its full applied host record; the
    one whose recorded operation matches this wave is loaded exactly. A panel
    never published falls back to the operation's pre-dispatch checkpoint, but
    never beside a published panel whose source cannot be read and that may be
    this operation: that returns ``source_status: unreadable`` and no runs.
    """
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.task_results import load_task_result

    row = result if isinstance(result, dict) and result.get("review_projection") else (load_task_result(root, task_id) or {})
    runs: List[Dict[str, Any]] = []
    unreadable: List[Dict[str, Any]] = []
    top = 0
    for panel in ((row.get("review_projection") or {}).get("panels") or []):
        ref = panel.get("applied_source_ref") if isinstance(panel, dict) else None
        if not isinstance(panel, dict) or panel.get("surface") != "task_acceptance":
            continue
        top = max(top, int(panel.get("publication_revision") or 0))
        try:
            run = json.loads(read_actor_source_bytes(root, task_id, ref))
        except (OSError, ValueError):
            log.debug("published acceptance source unreadable for %s", task_id, exc_info=True)
            unreadable.append(panel)
            continue
        if isinstance(run, dict) and (run.get("request") or {}).get("retry_key") == retry_key:
            runs.append({**run, "applied_source_ref": ref, "applied_source_status": "available",
                         "publication_revision": panel.get("publication_revision")})
    if runs:
        return {"review_runs": runs, "_acceptance_publication_revision": top}
    fallback = None
    if isinstance(checkpoint, dict):
        from ouroboros.review_operation import run_from_checkpoint

        try:
            fallback = run_from_checkpoint(root, task_id, checkpoint, str(checkpoint.get("owner_id") or ""), row)
        except (OSError, ValueError):
            fallback = None
    binding = str((fallback or {}).get("binding_hash") or "")
    if any(str(panel.get("binding_hash") or "") in {"", binding} for panel in unreadable):
        return {"review_runs": [], "source_status": "unreadable",
                "unreadable_panels": [str(panel.get("panel_id") or "") for panel in unreadable]}
    return {"review_runs": [fallback], "_acceptance_publication_revision": top} if fallback else None


def _collected_wave(run: Dict[str, Any]) -> Dict[str, Any]:
    """The reviewers' own lines for a wave whose release roster died with its worker."""
    from ouroboros.review_custody import _settled_slot_verdict

    slots, verdicts = {}, {}
    for actor in run.get("actors") or []:
        if not isinstance(actor, dict):
            continue
        slot_id = str(actor.get("slot_id") or "")
        slots[slot_id] = "" if actor.get("operation_state") in {"pending_dispatch", "in_flight"} else str(actor.get("status") or "settled")
        if slots[slot_id]:
            verdicts[slot_id] = _settled_slot_verdict(SimpleNamespace(**{"raw_text": "", "error": "", **actor}))
    return {"slots": slots, "verdicts": verdicts, "total": len(slots)}


def _stored_late_settlement(outcome: Dict[str, Any], run: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The late settlement the canonical record holds for this run's panel, read back.

    The first published settlement of a panel is the settlement (its bytes and
    ``settled_at``); None when the write failed, the panel conflicted or it is
    still pending in the stored record.
    """
    if outcome.get("status") != "published" or str(run.get("panel_id") or "") in outcome.get("rejected", []):
        return None
    from ouroboros.review_projection import _actor_pending

    for panel in (outcome.get("projection") or {}).get("panels") or []:
        if (isinstance(panel, dict) and panel.get("surface") == "task_acceptance"
                and panel.get("panel_id") == run.get("panel_id") and isinstance(panel.get("late_settlement"), dict)
                and not any(_actor_pending(actor) for actor in panel.get("actors") or [] if isinstance(actor, dict))):
            return panel
    return None


def attach_late_acceptance_settlement(usage_ctx: Any, request: Any, wave: Dict[str, Any],
                                      *, result: Dict[str, Any]) -> bool:
    """A panel that settled after its task ended is a supplement, never a new turn.

    Its verdicts are collected at $0 over the recorded operation, republished on
    the task's own review projection through the existing locked writer, and
    announced ONCE in the task's room through the existing terminal-delivery
    outbox (durably owed, keyed by ``delivery_id``; a second settlement of the
    same wave, concurrent or later, finds that notice already owed or delivered
    and queues no second live copy). The owner
    sees it; the next turn reads it in chat history, and one bounded reflection
    row carries the settled verdict to the learning log (plan_review_facts).
    """
    return settle_acceptance_operation(
        usage_ctx, retry_key=str(getattr(request, "retry_key", "") or ""),
        task_id=str(getattr(request, "task_id", "") or ""), result=result, wave=wave) == "announced"


def settle_acceptance_operation(usage_ctx: Any, *, retry_key: str, task_id: str, result: Dict[str, Any],
                                wave: Optional[Dict[str, Any]] = None, checkpoint: Optional[Dict[str, Any]] = None,
                                controller: Any = None) -> str:
    """Collect one wave of a terminal task purely, publish it and announce it once.

    A completed ``historical_acceptance`` run without a ``late_settlement`` is stamped and published as if advanced.
    Returns ``announced`` (publication read back, row enqueued), ``published`` (read back; the row
    was already owed or delivered), ``settled`` (nothing was pending), ``pending`` (still in
    flight), ``unpublished`` (the canonical record or durable notice custody did not take the
    settlement; retry duty remains), ``source_unreadable`` (a published source exists but cannot
    be read: never duplicated from the checkpoint) or ``unavailable`` (no trace and no canonical source).
    """
    from ouroboros.loop_acceptance_review import acceptance_run_pending
    from ouroboros.review_dispatch import reconcile_pending_acceptance_runs
    from ouroboros.review_projection import publish_acceptance_checkpoint
    root = _result_root(usage_ctx)
    trace = _settlement_trace(usage_ctx, retry_key) if retry_key else None
    partial = trace is None
    if trace is None and retry_key:
        trace = canonical_acceptance_trace(root, task_id, retry_key, result=result, checkpoint=checkpoint)
    if trace is None:
        log.debug("late acceptance settlement %s: neither a live trace nor a published source holds it", retry_key)
        return "unavailable"
    if trace.get("source_status") == "unreadable":
        return _unpublished(task_id, retry_key, "source_unreadable")
    runs = [run for run in (trace.get("review_runs") or [])
            if isinstance(run, dict) and run.get("authority") == "host_root"
            and isinstance(run.get("request"), dict)
            and str(run["request"].get("retry_key") or "") == retry_key]
    was_pending = any(acceptance_run_pending(run) for run in runs)
    # Only THIS wave's runs: reconciling every pending panel here would let one
    # settlement collect a sibling panel's verdicts and leave that panel's own
    # settlement with nothing to announce (the run objects are shared with the trace).
    advanced = reconcile_pending_acceptance_runs({"review_runs": runs}, drive_root=root, usage_ctx=usage_ctx,
                                                 **({"controller": controller} if controller is not None else {}))
    settled = runs[-1] if runs else {}
    historical = ((settled.get("request") or {}).get("policy") or {}).get("historical_acceptance")
    first_complete = (isinstance(historical, dict) and historical.get("schema_version") == 1
                      and bool(settled.get("actors")) and not acceptance_run_pending(settled)
                      and not isinstance(settled.get("late_settlement"), dict))
    # A settlement this process already stamped but could not publish is published again, never re-stamped.
    republish = (not advanced and isinstance(settled.get("late_settlement"), dict)
                 and not acceptance_run_pending(settled)
                 and (late_publication_owed(task_id, retry_key)
                      or (checkpoint or {}).get("state") in {"retained", "dispatched", "unpublished"}))
    if not advanced and not republish and not first_complete:
        log.debug("late acceptance settlement %s: nothing reconciled (still pending or already collected)", retry_key)
        if not was_pending:
            (getattr(usage_ctx, "_acceptance_settlement_traces", None) or {}).pop(retry_key, None)
            if partial:
                # This trace was reloaded from the canonical publication. A
                # settled local/mailbox trace alone cannot consume the duty.
                with _LATE_LOCK:
                    _LATE_UNPUBLISHED.discard((task_id, retry_key))
        return "pending" if was_pending else "settled"
    if advanced or first_complete:
        # The sentence has ONE author: it is stamped on the exact run this wave
        # reconciled, so the republished projection carries the same bytes the row
        # does and the card's Reviews group prints them verbatim.
        fact = late_evidence_fact(settled, emitted_answer_fact(root, task_id), settled_at=utc_now_iso())
        settled["late_settlement"] = {"note": _late_settlement_text(settled, wave or _collected_wave(settled), fact),
                                      **fact}
    chat_id = ((historical.get("confirmed_delivery") or historical.get("delivery") or {}).get("chat_id")
               if isinstance(historical, dict) else result.get("chat_id"))
    outcome = publish_acceptance_checkpoint(usage_ctx, trace, task_id=task_id, drive_root=root,
                                            chat_id=chat_id, partial_trace=partial or bool(historical))
    panel = _stored_late_settlement(outcome, settled)
    if panel is None:
        log.warning("late acceptance settlement %s was not published (%s); nothing announced", retry_key,
                    outcome.get("error") or outcome.get("status"))
        return _unpublished(task_id, retry_key, "unpublished")
    if not any(acceptance_run_pending(run) for run in runs):
        (getattr(usage_ctx, "_acceptance_settlement_traces", None) or {}).pop(retry_key, None)
    return enqueue_late_acceptance_settlement(usage_ctx, task_id, retry_key, result, panel)


def enqueue_late_acceptance_settlement(usage_ctx: Any, task_id: str, retry_key: str,
                                      result: Dict[str, Any], panel: Dict[str, Any]) -> str:
    """Owe the already-published fact, also after an old controller lost its outbox write."""
    from supervisor.terminal_delivery import (
        ENQUEUE_ALREADY_DELIVERED, ENQUEUE_QUEUED, enqueue_terminal_delivery_outcome, pending_deliveries,
    )

    root = _result_root(usage_ctx)
    late = panel["late_settlement"]
    # Recovery has only the retained panel. Its proven delivery owns the room,
    # even when today's task or caller is bound to a different chat.
    historical = late.get("historical_delivery")
    chat_id = historical.get("chat_id") if isinstance(historical, dict) else result.get("chat_id")
    # A compact, source-bound pointer rides the row itself (progress_meta survives
    # live delivery, replay and history); the full fact stays on the projection.
    evidence = {"task_id": task_id, "panel_id": str(panel.get("panel_id") or ""),
                "settled_at": str(late.get("settled_at") or ""), "reviewed_revision": late.get("reviewed_revision"),
                "reviewed_is_emitted": late.get("reviewed_is_emitted"),
                "source_ref": panel.get("applied_source_ref") or {}}
    if evidence["source_ref"].get("path"):
        from ouroboros.task_finalization import review_source_reader

        evidence["read"] = review_source_reader(task_id, evidence["source_ref"])
    delivery_id = f"acceptance-late:{retry_key}"
    with _LATE_NOTICE_LOCK:
        # Every settler of one wave ends here: the drain's own collection, the
        # last slot's callback (whose mailbox duty mark can arrive after the
        # drain already announced), maintenance, an owner rejoin. A notice the
        # outbox already owes has custody and replay, so no second live copy.
        owed = any(row.get("delivery_id") == delivery_id for row in pending_deliveries(root))
        # The operation outlives the execution drive; replay belongs to the same
        # canonical root as its publication and the supervisor's delivery registry.
        outcome = "" if owed else enqueue_terminal_delivery_outcome(root, {
            "type": "send_message", "chat_id": int(chat_id or 0), "task_id": task_id,
            "text": str(late.get("note") or ""),
            "role": "system", "system_type": LATE_SETTLEMENT_SYSTEM_TYPE,
            "delivery_id": delivery_id,
            # The verdict belongs inside the task's card, in its Reviews group, and
            # stays one row across live delivery, outbox replay and history. The
            # evidence pointer is neutral: it grants no action and starts no turn.
            "progress_meta": {"card_row": "reviews", "card_row_id": delivery_id,
                              "late_evidence": evidence},
        }, event_queue=getattr(usage_ctx, "event_queue", None))
    if not owed and outcome not in {ENQUEUE_QUEUED, ENQUEUE_ALREADY_DELIVERED}:
        # Publication survived, but a live queue alone is not durable custody.
        # The operation pointer carries this retry duty across controller exit.
        return _unpublished(task_id, retry_key, "unpublished")
    if historical:
        from ouroboros.review_operation import historical_publication_retained

        historical_publication_retained(root, task_id, retry_key)
    with _LATE_LOCK:
        _LATE_UNPUBLISHED.discard((str(task_id), str(retry_key)))
    if outcome == ENQUEUE_QUEUED:  # a NEW announcement, whichever path made it: one learning row, never on a replay
        learn_from_late_settlement(root, result, retry_key)
    return "announced" if outcome == ENQUEUE_QUEUED else "published"


def _unpublished(task_id: str, retry_key: str, status: str) -> str:
    with _LATE_LOCK:
        _LATE_UNPUBLISHED.add((str(task_id), str(retry_key)))
    return status


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
            # A local acceptance-preparation incident is identified by its own id
            # AND attempt, not by a review binding it never had (the pre-binding
            # hash is empty): a request that carried attempt 1 exposes nothing
            # about a later attempt of the same incident.
            incident_id = str(source.get("outcome_incident_id") or "")
            if incident_id:
                attempt = int(source.get("outcome_incident_attempt") or 0)
                if (incident_id == str(outcome.get("incident_id") or "")
                        and attempt == int(outcome.get("incident_attempt") or 0)):
                    outcome["feedback_delivered"] = True
                    record = trace.get("acceptance_preparation")
                    if (isinstance(record, dict) and str(record.get("incident_id") or "") == incident_id
                            and int(record.get("attempts") or 0) == attempt):
                        record["feedback_delivered"] = True
                        record["exposed_attempt"] = attempt
                continue
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
