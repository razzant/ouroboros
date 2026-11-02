"""Owner "Hurry up" control — typed, task-local, NO CHAT (HQ1, 2026-08-15).

The owner's hurry click is NOT dialogue: it submits one typed task-bound control
through the existing durable owner-mailbox rail (``owner_mailbox.KIND_HURRY``)
and never creates a chat message, ``steer_task`` text, outbox row, or bubble.
This module owns everything hurry-specific so the pinned surfaces
(``loop.py``, ``supervisor/events.py``) keep thin dispatch only:

- attempt identity (``task["_attempt"]`` — ``timeout_retry_from`` is NOT an
  attempt key: it stays equal to the original id across same-id retries);
- the attempt-local effect-idempotent latch the production drain arms;
- the DEDICATED locked task-result projection writer (mutates ONLY the
  ``owner_hurry``/``owner_hurry_history`` keys — never ``write_task_result``,
  whose required status argument and status-regression guard can silently drop
  the write);
- history rollover for same-``task_id`` retries plus the ONE shared retry-reset
  helper every same-id requeue producer calls (reaper timeout AND the
  ``supervisor/workers.py`` crash-requeue);
- the non-chat ``owner_hurry`` event family (phases requested | applied |
  effect | not_applied, always ``is_progress=False``);
- the task-local advisory plan-review projection.

Effects are HOST RAILS only: a task-local advisory reading of an open
force-plan gate. Commit triad/scope, advisory pre-review, deterministic P3
gates, and safety NEVER consult hurry (§19.7.2 item 10); no global setting is
read or written.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Callable, Dict, List, Optional

from ouroboros.review_projection import PLAN_REVIEW_OBJECTIONS_OPEN
from ouroboros.utils import utc_now_iso
from ouroboros.task_store import patch_document
from ouroboros.store import StoreError

log = logging.getLogger(__name__)

# The finite typed hurry reason (§19.7.2 item 8) every hurry block and event names —
# SSOT in ``outcomes.py``, where older task records also read it. Deliberately NOT a
# member of lifecycle vocabulary, truncation vocabulary, or BEST_EFFORT_REASON_CODES:
# hurry is an owner-approved detail fact, never a forced-rail or lifecycle reason.
from ouroboros.outcomes import (  # noqa: E402
    REASON_ACCEPTANCE_SKIPPED_OWNER_HURRY as REASON_OWNER_HURRY,
)

# Closed non-chat event phases (one event family, ``type=owner_hurry``).
PHASE_REQUESTED = "requested"
PHASE_APPLIED = "applied"
PHASE_EFFECT = "effect"
PHASE_NOT_APPLIED = "not_applied"

# Projection states for the ``owner_hurry`` task-result block.
STATE_REQUESTED = "requested"
STATE_APPLIED = "applied"
STATE_NOT_APPLIED_BEFORE_TERMINAL = "not_applied_before_terminal"

_HISTORY_CAP = 16


def attempt_key(task_or_attempt: Any) -> int:
    """The REAL attempt identity: ``task['_attempt']`` (default 1).

    §19.7.2 item 5: ``timeout_retry_from`` alone is NOT an attempt key (it stays
    equal to the original task id across same-id retries) and the plan-review
    ``current_attempt`` pointer is unrelated.
    """
    if isinstance(task_or_attempt, dict):
        raw = task_or_attempt.get("_attempt")
    else:
        raw = task_or_attempt
    try:
        value = int(raw) if raw is not None else 1
    except (TypeError, ValueError):
        value = 1
    return max(1, value)


def _mutate_projection(
    drive_root: Any, task_id: str,
    mutator: Callable[[Optional[Dict[str, Any]], List[Dict[str, Any]]], Optional[Dict[str, Any]]],
) -> Dict[str, Any]:
    """The DEDICATED locked hurry writer (§19.7.2 item 5).

    ``patch_document`` on the canonical task-result row, mutating ONLY the
    ``owner_hurry``/``owner_hurry_history`` keys. Concurrent terminal writers
    merge-write around these keys (``write_task_result`` spreads ``**existing``),
    so the projection survives a terminal race; this writer never routes through
    ``write_task_result`` because its required ``status`` argument and
    status-regression guard can silently drop the write.

    ``mutator(current_block, history) -> new_block | _KEEP`` where returning
    ``_KEEP`` aborts (file untouched); returning ``None`` clears the current
    block. Returns the post-write ``{"owner_hurry": ..., "owner_hurry_history":
    ...}`` view (or the pre-write view on abort).
    """
    from ouroboros.task_results import (
        require_writable_task_result_schema,
        stamp_task_result_schema,
    )

    view: Dict[str, Any] = {}

    def _mutate(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        view.clear()
        # ABI 7.0: never write over a row a newer schema owns; every write
        # stamps the row (stamp-on-write, Q8=B).
        require_writable_task_result_schema(current)
        block = current.get("owner_hurry")
        block = dict(block) if isinstance(block, dict) else None
        history_raw = current.get("owner_hurry_history")
        history = [dict(row) for row in history_raw if isinstance(row, dict)] if isinstance(history_raw, list) else []
        outcome = mutator(block, history)
        if outcome is _KEEP:
            view.update({"owner_hurry": block, "owner_hurry_history": history})
            return None
        updated = dict(current)
        if outcome is None:
            updated.pop("owner_hurry", None)
        else:
            updated["owner_hurry"] = outcome
        if history:
            updated["owner_hurry_history"] = history[-_HISTORY_CAP:]
        view.update({
            "owner_hurry": updated.get("owner_hurry"),
            "owner_hurry_history": updated.get("owner_hurry_history"),
        })
        return stamp_task_result_schema(updated)

    patch_document(pathlib.Path(drive_root), task_id, _mutate)
    return view


class _Keep:
    """Sentinel: the mutator decided the file must stay byte-identical."""


_KEEP = _Keep()


def record_requested(
    drive_root: Any, task_id: str, *,
    request_id: str,
    attempt: int,
    requested_at: str = "",
    requested_by: str = "owner",
) -> Dict[str, Any]:
    """Endpoint-side ``requested`` projection write — request-id idempotent.

    Returns ``{"state": ..., "duplicate": bool, "block": {...}}``. The SAME
    ``request_id`` on the live attempt returns the existing acknowledgement
    without a second write; a DIFFERENT id while a current block exists for the
    same attempt collapses onto the one executable effect latch (recorded as a
    duplicate acknowledgement too — one latch, not two rows).
    """
    stamp = str(requested_at or "") or utc_now_iso()
    outcome: Dict[str, Any] = {}

    def _mutator(block: Optional[Dict[str, Any]], history: List[Dict[str, Any]]) -> Any:
        if isinstance(block, dict) and int(block.get("attempt_key") or 0) == int(attempt):
            outcome.update({"state": str(block.get("state") or STATE_REQUESTED),
                            "duplicate": True, "block": dict(block)})
            return _KEEP
        if isinstance(block, dict):
            # A stale prior-attempt block that no retry-reset archived yet:
            # roll it into history now rather than overwriting it silently.
            history.append({**block, "archived_at": stamp, "archived_reason": "superseded_by_new_attempt"})
        fresh = {
            "attempt_key": int(attempt),
            "request_id": str(request_id or ""),
            "requested_by": str(requested_by or "owner"),
            "requested_at": stamp,
            "reason": REASON_OWNER_HURRY,
            "state": STATE_REQUESTED,
            "effects": {},
        }
        outcome.update({"state": STATE_REQUESTED, "duplicate": False, "block": dict(fresh)})
        return fresh

    _mutate_projection(drive_root, task_id, _mutator)
    return outcome


def record_applied(
    drive_root: Any, task_id: str, *,
    attempt: int,
    applied_at: str = "",
    effects: Optional[Dict[str, str]] = None,
) -> None:
    """Worker-side ``applied`` projection write — effect-idempotent, fail-soft."""
    stamp = str(applied_at or "") or utc_now_iso()

    def _mutator(block: Optional[Dict[str, Any]], history: List[Dict[str, Any]]) -> Any:
        if not isinstance(block, dict) or int(block.get("attempt_key") or 0) != int(attempt):
            return _KEEP
        updated = dict(block)
        if str(updated.get("state") or "") != STATE_APPLIED:
            updated["state"] = STATE_APPLIED
            updated["applied_at"] = stamp
        merged = dict(updated.get("effects") or {})
        merged.update({str(k): str(v) for k, v in (effects or {}).items()})
        updated["effects"] = merged
        return updated

    try:
        _mutate_projection(drive_root, task_id, _mutator)
    except Exception:
        log.debug("owner_hurry applied projection write failed for %s", task_id, exc_info=True)


def reconcile_terminal(drive_root: Any, task_id: str) -> bool:
    """task_done reconciliation: a still-``requested`` block lost the terminal
    race and is honestly ``not_applied_before_terminal`` — never a successful
    acceleration (§19.7.2 item 5). Returns whether a block was reconciled."""
    changed = {"value": False}

    def _mutator(block: Optional[Dict[str, Any]], history: List[Dict[str, Any]]) -> Any:
        if not isinstance(block, dict) or str(block.get("state") or "") != STATE_REQUESTED:
            return _KEEP
        updated = dict(block)
        updated["state"] = STATE_NOT_APPLIED_BEFORE_TERMINAL
        updated["reconciled_at"] = utc_now_iso()
        changed["value"] = True
        return updated

    try:
        _mutate_projection(drive_root, task_id, _mutator)
    except Exception:
        log.debug("owner_hurry terminal reconciliation failed for %s", task_id, exc_info=True)
    if changed["value"]:
        emit_event(
            None, drive_root, task_id,
            phase=PHASE_NOT_APPLIED,
            detail="terminal before the worker drained the control",
        )
    return changed["value"]


def archive_current(drive_root: Any, task_id: str, *, reason: str) -> bool:
    """Roll the current block into ``owner_hurry_history[]`` and clear it.

    The retry-reset half of §19.7.2 item 11: a same-``task_id`` retry starts
    WITHOUT an executable latch while the historical evidence survives.
    """
    changed = {"value": False}

    def _mutator(block: Optional[Dict[str, Any]], history: List[Dict[str, Any]]) -> Any:
        if not isinstance(block, dict):
            return _KEEP
        history.append({**block, "archived_at": utc_now_iso(), "archived_reason": str(reason or "")})
        changed["value"] = True
        return None

    try:
        _mutate_projection(drive_root, task_id, _mutator)
    except Exception:
        log.debug("owner_hurry archive failed for %s", task_id, exc_info=True)
    return changed["value"]


def retry_reset(task_drive: Any, canonical_drive_root: Any, task_id: str, *, reason: str) -> None:
    """The ONE shared same-id attempt reset (§19.7.2 item 11).

    Crash and planned-restart requeues (``worker_health.py``, ``workers.py``)
    and the hard-bound reaper, whose attempt ends without a retry, call this.
    Attempt-local controls are revoked; durable owner text and its ack ledger
    survive for canonical/replay continuity.  The current ``owner_hurry``
    block is archived into history. Fail-soft; never raises.
    """
    try:
        from ouroboros.owner_mailbox import reset_attempt_controls_for_retry

        reset_attempt_controls_for_retry(pathlib.Path(task_drive), task_id)
    except Exception:
        log.debug("owner_hurry retry-reset control reset failed for %s", task_id, exc_info=True)
    try:
        archive_current(pathlib.Path(canonical_drive_root), task_id, reason=reason)
    except Exception:
        log.debug("owner_hurry retry-reset archive failed for %s", task_id, exc_info=True)


# ---------------------------------------------------------------------------
# Attempt-local in-process latch (worker side)
# ---------------------------------------------------------------------------

def latched(ctx: Any) -> Optional[Dict[str, Any]]:
    """The armed attempt-local latch, or None."""
    latch = getattr(ctx, "_owner_hurry_latch", None)
    return latch if isinstance(latch, dict) else None


def latch_from_entry(ctx: Any, entry: Dict[str, Any]) -> bool:
    """Arm the attempt-local latch from one drained ``kind=hurry`` mailbox entry.

    Effect-idempotent: different mailbox ids collapse onto ONE latch. Never
    touches ``messages``/``_owner_directives`` — the caller (the production
    drain) already routed the entry structurally. Returns whether the latch was
    newly armed (first arm wins; restart re-drain re-arms identically).
    """
    if ctx is None:
        return False
    existing = latched(ctx)
    if existing is not None:
        return False
    ctx._owner_hurry_latch = {
        "msg_id": str(entry.get("msg_id") or ""),
        "requested_at": str(entry.get("ts") or ""),
        "reason": REASON_OWNER_HURRY,
    }
    return True


def apply_latch(ctx: Any, entry: Dict[str, Any], *, event_queue: Any = None) -> None:
    """Arm the latch and persist the ``applied`` projection ONCE per attempt.

    The one entry point the production drain calls for a ``kind=hurry`` mailbox
    entry. Fail-soft end to end: a projection/event failure never breaks the
    round loop, and a re-drain of the same (or another) hurry entry is a no-op.
    """
    if not latch_from_entry(ctx, entry):
        return
    task_id = str(getattr(ctx, "task_id", "") or "")
    root = _canonical_root(ctx)
    if not task_id or root is None:
        return
    # ``ctx.task_attempt`` is stamped by the agent from ``task["_attempt"]``;
    # a context without the stamp is attempt 1 (attempt_key(None) -> 1).
    attempt = attempt_key(getattr(ctx, "task_attempt", None))
    record_applied(
        root, task_id, attempt=attempt,
        effects={
            "plan_review": "task_local_advisory",
        },
    )
    emit_event(
        event_queue, root, task_id,
        phase=PHASE_APPLIED,
        msg_id=str(entry.get("msg_id") or ""),
        requested_at=str(entry.get("ts") or ""),
        attempt=attempt,
    )


def _canonical_root(ctx: Any) -> Optional[pathlib.Path]:
    """The canonical (budget) task-result root — same rule as force-plan reads."""
    raw = str(getattr(ctx, "budget_drive_root", "") or "") or str(getattr(ctx, "drive_root", "") or "")
    return pathlib.Path(raw) if raw else None


# ---------------------------------------------------------------------------
# Effects: plan-review projection
# ---------------------------------------------------------------------------

def _plan_facts(ctx: Any) -> tuple[Dict[str, Any], Dict[str, Any], bool]:
    """``(record, legacy, unreadable)``: the task's plan record, the read-only projection of
    a v1/v2 ``plan_review_state`` (an in-flight task at upgrade keeps its open blocking
    findings as objections; a finished one its selected claims), and whether either
    authority could not be read — a gate that cannot read its authority is engaged, not
    absent (I-17). Pending seats whose producer completed are settled at $0 first."""
    from ouroboros import plan_record

    root = _canonical_root(ctx)
    task_id = str(getattr(ctx, "task_id", "") or "").strip()
    if not task_id or root is None:
        return plan_record.empty_record(), {}, False
    unreadable = False
    try:
        record = plan_record.load_plan_record(root, task_id)
    except (plan_record.PlanRecordInvalid, OSError, TimeoutError, ValueError, StoreError):
        log.warning("Unable to read the durable plan record", exc_info=True)
        record, unreadable = plan_record.empty_record(), True
    legacy, legacy_error = plan_record.legacy_facts(root, task_id)
    if legacy_error.get("source") == "plan_review_state":  # an unreadable author plan is "no author plan" for the gate
        log.warning("Unable to read the durable legacy plan-review state: %s", legacy_error.get("error"))
        unreadable = True
    if not unreadable and plan_record.pending_asks(record):
        try:
            from ouroboros.tools.plan_ask import collect_pending

            record = collect_pending(ctx, record, state_root=root, task_id=task_id)
        except Exception:
            log.warning("pending plan asks could not be collected before the gate", exc_info=True)
    if not unreadable and legacy.get("pending_wave"):
        try:  # 03 D7: the earlier schema's owed wave, read-only through the same collector
            from ouroboros.tools.plan_review import collect_legacy_contributions

            legacy = collect_legacy_contributions(ctx, legacy, state_root=root, task_id=task_id)
        except Exception:
            log.warning("the earlier plan review's owed wave could not be collected before the gate", exc_info=True)
    return record, legacy, unreadable


def plan_gate_decision(
    ctx: Any, llm_trace: Dict[str, Any], *,
    hard_rail: str = "", enforcement: Optional[str] = None,
) -> Dict[str, Any]:
    """Project the planning gate at the task's normal final answer — the ONE seam.

    ``plan_record.gate_projection`` decides: held only under Blocking (outside Cyber
    Pro, no owner hurry) by an open objection an owner-marked reviewer raised in the
    ask it answered — for any task (owner batch 2, Q2=B); silence, refusals, pending
    asks and mind-added critics never hold. While the hurry latch is armed the
    projection is computed as locally advisory and says so (attribution for the task
    detail). ``enforcement`` is supplied by the loop wrapper from ITS namespace so the
    existing ``loop.get_review_enforcement`` test seam holds.
    """
    from ouroboros import plan_record
    from ouroboros.tools.review_helpers import review_enforcement_blocks

    record, legacy, unreadable = _plan_facts(ctx)
    if enforcement is None:
        from ouroboros.config import get_review_enforcement

        enforcement = get_review_enforcement()
    decision = plan_record.gate_projection(
        record, enforcement=str(enforcement or ""), cyber_pro=not review_enforcement_blocks("blocking"),
        hurry=latched(ctx) is not None,
        hard_rail=hard_rail, legacy=legacy, unreadable=unreadable)
    if decision.get("required") and decision.get("open_objections"):
        # The outcome class at delivery (``outcome_axes.execution.plan_review``).
        decision["plan_review_class"] = PLAN_REVIEW_OBJECTIONS_OPEN
    return decision


def _objection_lines(decision: Dict[str, Any], *, holding_only: bool) -> List[str]:
    from ouroboros.plan_record import ask_subject

    rows = [o for o in decision.get("open_objections") or [] if not holding_only or o.get("holds")]
    return [f"{o.get('seat_id')}/{o.get('id')} (ask {o.get('ask_id')}, {ask_subject(o.get('revision'))}"
            + ("; not adjudicated in its raiser's last reply" if o.get("not_adjudicated") else "")
            + ("; imported from the previous plan review" if o.get("legacy") else "") + f"): {o.get('text')}" for o in rows]


def plan_review_reminder(decision: Dict[str, Any]) -> str:
    """The user-turn reminder appended while the planning gate holds the normal final
    answer (``_enforce_plan_gate``)."""
    tag = "[PLAN_REVIEW_HOLD]"
    status = str(decision.get("status") or "")
    if status == "unreadable":
        return (f"{tag} This task's plan record cannot be read, so the gate holds until it can; "
                "plan_task() reads it, and finish_task(action='stop') ends honestly with unfinished work.")
    held = _objection_lines(decision, holding_only=True)
    return (f"{tag} Blocking enforcement: {len(held)} open objection{'' if len(held) == 1 else 's'} of owner-marked "
            "reviewers hold your normal final answer:\n- " + "\n- ".join(held) + "\n"
            "Only the participant who raised an objection withdraws it: revise the plan or answer the objection, "
            "then plan_task(ask=[<its seat id>], message=...) so it can withdraw or restate; argue in the message "
            "when you disagree. Other exits: finish_task(action='stop') with unfinished work, or the owner's hurry "
            "or a change of enforcement. Nobody votes and no verdict closes this.")


def plan_review_disclosure(decision: Dict[str, Any], forced_reason: str = "") -> str:
    """The loud host suffix when the planning gate did not hold the final answer but the
    council's talk was not over: open objections (released by a rail, Advisory, Cyber Pro
    or the owner's hurry), asks whose answers had not arrived, or an unreadable record
    released by a rail."""
    if not decision.get("required"):
        return ""
    objections = decision.get("open_objections") or []
    pending = decision.get("pending_asks") or []
    status = str(decision.get("status") or "")
    unreadable = bool(decision.get("unreadable"))
    if not objections and not pending and not unreadable:
        return ""
    held = [o for o in objections if o.get("holds")]
    count = (f"{len(objections)} open objection{'' if len(objections) == 1 else 's'}"
             + (f" ({len(held)} from owner-marked reviewers)" if held and len(held) != len(objections) else ""))
    listed = "; ".join(_objection_lines(decision, holding_only=False))
    subject = "Blocking plan review" if str(decision.get("enforcement") or "") == "blocking" else "Plan review"
    if status == "rail_degraded":
        rail_reason = str(forced_reason or decision.get("reason") or "")
        rail = f"the task-wide rail `{rail_reason}`" if rail_reason else "a task-wide rail"
        if objections:
            what = f"{count}: {listed}"
        elif pending:
            what = "answers still on their way for ask(s) " + ", ".join(pending)
        else:
            what = "the plan record could not be read"
        return f"\n\n⚠️ {subject} is open ({what}); {rail} required best-effort finalization."
    if objections:
        why = ("released by the owner's hurry request" if decision.get("owner_hurry_local_advisory") else
               "Cyber Pro: Ouroboros proceeded by its own judgment" if decision.get("decision_authority") == "cyber_pro" else
               "work proceeded under the owner-selected advisory enforcement")
        return f"\n\n⚠️ Plan review: {count} remain{'s' if len(objections) == 1 else ''} open — {listed}; {why}."
    if pending:
        return ("\n\n⚠️ Plan review: the answers of ask(s) " + ", ".join(pending)
                + " had not arrived when the task ended; silence holds nothing and is not agreement"
                + (f"; the dispatch outcome of ask(s) {', '.join(decision['unknown_outcome'])} is unknown — a new ask may pay twice"
                   if decision.get("unknown_outcome") else "") + ".")
    return ""


# ---------------------------------------------------------------------------
# Non-chat event family
# ---------------------------------------------------------------------------

def event_row(
    task_id: str, *,
    phase: str,
    request_id: str = "",
    msg_id: str = "",
    requested_at: str = "",
    attempt: Optional[int] = None,
    detail: str = "",
    effect: str = "",
    status: str = "",
) -> Dict[str, Any]:
    """One typed ``owner_hurry`` event row — ALWAYS ``is_progress=False``.

    Task-detail observability only (§19.7.2 item 4): rows may update local
    toast/card/task details; they must never enter ``send_message``,
    ``chat.jsonl``, the message bus/outbox, terminal delivery, a chat WebSocket
    frame, ``addMessage``, or assistant/system bubble rendering. The timeline
    summarizer returns ``visible=false`` for this family.
    """
    row: Dict[str, Any] = {
        "type": "owner_hurry",
        "task_id": str(task_id or ""),
        "phase": str(phase or ""),
        "requested_by": "owner",
        "reason": REASON_OWNER_HURRY,
        "is_progress": False,
    }
    if request_id:
        row["request_id"] = str(request_id)
    if msg_id:
        row["msg_id"] = str(msg_id)
    if requested_at:
        row["requested_at"] = str(requested_at)
    if attempt is not None:
        row["attempt_key"] = int(attempt)
    if detail:
        row["detail"] = str(detail)
    if effect:
        row["effect"] = str(effect)
    if status:
        row["status"] = str(status)
    return row


def emit_event(event_queue: Any, drive_root: Any, task_id: str, **kwargs: Any) -> None:
    """Emit one non-chat ``owner_hurry`` event — queue when live, else durable append."""
    row = event_row(task_id, **kwargs)
    try:
        if event_queue is not None:
            from ouroboros.loop_llm_call import _emit_live_log

            _emit_live_log(event_queue, row)
            return
        if drive_root is not None:
            from ouroboros.utils import append_jsonl

            append_jsonl(pathlib.Path(drive_root) / "logs" / "events.jsonl",
                         {"ts": utc_now_iso(), **row})
    except Exception:
        log.debug("owner_hurry event emission failed for %s", task_id, exc_info=True)
