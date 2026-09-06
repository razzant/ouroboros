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
- the effective acceptance-pacing overlay (remaining improvement passes -> 0)
  and the task-local advisory plan-review projection.

Effects are HOST RAILS only: skip the next otherwise-eligible acceptance
panel (typed reason ``owner_hurry``), zero remaining improvement passes, and a
task-local advisory reading of an open force-plan gate. Commit triad/scope,
advisory pre-review, deterministic P3 gates, and safety NEVER consult hurry
(§19.7.2 item 10); no global setting is read or written.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Callable, Dict, List, Optional

from ouroboros.utils import update_json_locked, utc_now_iso

log = logging.getLogger(__name__)

# The finite typed acceptance-skip reason (§19.7.2 item 8) — SSOT in
# ``outcomes.py`` beside the acceptance-decision vocabulary. Deliberately NOT a
# member of outcomes.ACCEPTANCE_BYPASS_REASON_BY_RAIL, lifecycle vocabulary,
# truncation vocabulary, or BEST_EFFORT_REASON_CODES: hurry is an owner-approved
# acceptance-skip/detail fact, never a forced-rail or lifecycle reason.
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


def _hurry_result_path(drive_root: Any, task_id: str) -> pathlib.Path:
    from ouroboros.task_results import task_result_path

    return task_result_path(pathlib.Path(drive_root), task_id)


def _mutate_projection(
    drive_root: Any, task_id: str,
    mutator: Callable[[Optional[Dict[str, Any]], List[Dict[str, Any]]], Optional[Dict[str, Any]]],
) -> Dict[str, Any]:
    """The DEDICATED locked hurry writer (§19.7.2 item 5).

    ``update_json_locked`` on the canonical task-result file, mutating ONLY the
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

    update_json_locked(_hurry_result_path(drive_root, task_id), _mutate)
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


def record_effect(drive_root: Any, task_id: str, *, attempt: int, effect: str, value: str) -> None:
    """Record one named effect status on the current block — fail-soft."""
    record_applied(drive_root, task_id, attempt=attempt, effects={effect: value})


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
    """The ONE shared same-id requeue reset (§19.7.2 item 11).

    EVERY same-id requeue producer — the reaper timeout path AND the
    ``supervisor/workers.py`` crash-requeue — calls this.  Attempt-local
    mailbox controls are revoked, while durable owner text and its ack ledger
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
            "improvement_passes": "zeroed_for_attempt",
            "task_acceptance": "skip_next_panel",
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
# Effects: acceptance pacing overlay, acceptance skip, plan-review projection
# ---------------------------------------------------------------------------

def effective_budget_profile(ctx: Any, profile: Any) -> Any:
    """The acceptance-pacing overlay (§19.7.2 item 7).

    While the latch is armed, EVERY production acceptance-pacing read resolves
    an effective profile with an explicit ``max_improvement_passes=0`` — the
    real ``improvement_pass_allowed`` call and the remaining-pass display/rails
    consume the same object. The immutable ``task_contract.budget_profile`` is
    never mutated; unlatched tasks get the input back unchanged.
    """
    if latched(ctx) is None:
        return profile
    if isinstance(profile, dict):
        # An explicit cap is authoritative under EVERY policy in
        # ``task_pacing.effective_max_improvement_passes`` — including
        # Required+Blocking, whose implicit local loop is otherwise bounded only
        # by the shared review-cycle cap (or unbounded when that is unlimited).
        return {**profile, "max_improvement_passes": 0}
    return profile


def acceptance_skip_decision(ctx: Any) -> Optional[Dict[str, Any]]:
    """The typed skip for the next otherwise-eligible acceptance panel.

    Returns None when no latch is armed; otherwise the exact
    ``_set_acceptance_decision`` payload (§19.7.2 item 8). The caller marks the
    host panel complete for this attempt and launches ZERO reviewer calls; a
    physically in-flight panel is neither cancelled nor relabeled.
    """
    if latched(ctx) is None:
        return None
    return {
        "status": "finalized_unaccepted",
        "reason": REASON_OWNER_HURRY,
        "source": REASON_OWNER_HURRY,
        "rationale": (
            "Owner requested acceleration (hurry): the next otherwise-eligible "
            "acceptance panel is skipped for this attempt with zero reviewer calls."
        ),
    }


def acceptance_skip_applied(
    ctx: Any,
    llm_trace: Dict[str, Any],
    *,
    task_id: str,
    drive_root: Any,
    set_decision: Callable[[Dict[str, Any], Dict[str, Any]], Any],
    emit_progress: Callable[[str], None],
) -> bool:
    """Consume an armed latch at the ONE acceptance launch site (loop.py).

    Extracted from ``loop._run_task_acceptance_review_once`` (byte offset for
    the pinned ``loop.py``). Marks the host panel complete for this attempt,
    stamps the typed trace decision through the caller-supplied
    ``set_decision`` seam, records the durable effect, and reports the skip via
    progress. Returns False untouched when no latch is armed.
    """
    skip = acceptance_skip_decision(ctx)
    if skip is None:
        return False
    ctx._task_acceptance_reviewed = True
    llm_trace["review_decision"].update(
        {"trigger": "owner_hurry", "skipped": "owner_hurry"},
    )
    set_decision(llm_trace, skip)
    root = getattr(ctx, "budget_drive_root", "") or drive_root
    if root:
        record_effect(
            root, task_id,
            attempt=attempt_key(getattr(ctx, "task_attempt", None)),
            effect="task_acceptance", value="skipped_owner_hurry",
        )
    emit_progress("Task acceptance review skipped: owner requested acceleration (hurry).")
    return True


def _plan_review_engaged(state: Any) -> bool:
    """Did this task CALL plan_task (any v2 attempt/wave, or a v1 record)? Plan §6:
    once a task entered the plan-review gate, the gate binds under blocking."""
    if not isinstance(state, dict):
        return False
    legacy = state.get("legacy_v1_projection") if isinstance(state.get("legacy_v1_projection"), dict) else {}
    return bool(state.get("waves") or state.get("current_attempt")
                or legacy.get("status") not in (None, "", "absent"))


def force_plan_decision(
    ctx: Any, llm_trace: Dict[str, Any], *,
    hard_rail: str = "", enforcement: Optional[str] = None,
) -> Dict[str, Any]:
    """Project the plan-review finalization gate — the ONE production consumer seam.

    Extracted from ``loop._force_plan_decision`` (byte-neutral extraction for the
    pinned ``loop.py``). The gate is REQUIRED for a Swarm-forced task
    (``task_metadata.force_plan``) AND for any task that itself CALLED plan_task
    (plan-review redesign 2026-08-15, plan §6: an open plan review holds under
    blocking enforcement whoever opened it). While the hurry latch is armed, the
    projection is computed under a TASK-LOCAL advisory enforcement (§19.7.2 item
    9): ``reviewed``/``open``/``unavailable`` states may proceed locally while
    ``absent``/``pending`` remain hold — durable review state, reviewer calls, and
    the configured global enforcement are untouched, and the attribution rides the
    decision for the task detail.

    ``enforcement`` is supplied by the loop wrapper from ITS module namespace so
    the existing ``loop.get_review_enforcement`` test/monkeypatch seam holds.
    """
    metadata = getattr(ctx, "task_metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    not_required = {"required": False, "allow": True, "status": "not_required"}
    if bool(getattr(ctx, "is_ephemeral_turn", False)):
        return not_required
    from ouroboros.task_results import (
        current_plan_review_wave, load_plan_review_state, plan_review_gate_projection,
    )

    try:
        root = _canonical_root(ctx)
        task_id = str(getattr(ctx, "task_id", "") or "").strip()
        state = load_plan_review_state(root, task_id) if (task_id and root) else None
        unreadable = False
    except (OSError, TimeoutError, ValueError):
        log.warning("Unable to read durable force-plan review state", exc_info=True)
        state, unreadable = None, True
    # I-17: an UNREADABLE state used to release a self-opened blocking plan (no state ->
    # "not required") while a Swarm-forced task failed closed on the same input. A gate that
    # cannot read its own authority is engaged, not absent.
    if not metadata.get("force_plan") and not unreadable and not _plan_review_engaged(state):
        return not_required
    if enforcement is None:
        from ouroboros.config import get_review_enforcement

        enforcement = get_review_enforcement()
    hurry_armed = latched(ctx) is not None
    effective = "advisory" if hurry_armed else enforcement
    decision = {
        "required": True,
        "self_opened": not bool(metadata.get("force_plan")),
        **plan_review_gate_projection(state, effective, hard_rail=hard_rail),
    }
    if decision.get("reviewer_slots_degraded"):
        # The reminder's replay promise is conditional on the recorded wave's
        # structural health epoch (empty epoch = a re-dispatch is PAID), so the
        # epoch fact rides the decision for plan_review_reminder.
        decision["degraded_health_epoch"] = (
            (current_plan_review_wave(state) or {}).get("health_epoch") or "")
    if hurry_armed and str(enforcement or "").lower() == "blocking":
        # Attribution only (task detail); the durable state and the configured
        # global enforcement are byte-identical before/after.
        decision["owner_hurry_local_advisory"] = True
        decision["configured_enforcement"] = "blocking"
    return decision


def plan_review_reminder(decision: Dict[str, Any]) -> str:
    """The user-turn reminder appended while a blocking plan review holds finalization
    (text moved out of the pinned ``loop.py``)."""
    outcome = str(decision.get("outcome") or "")
    status = str(decision.get("status") or "")
    tag = "[PLAN_REVIEW_HOLD]"
    if status == "legacy_open_requires_resubmission":
        return (
            f"{tag} An open plan review from a previous schema cannot be honored. Re-call "
            "plan_task with your goal, plan and spec to start a fresh review before finalizing."
        )
    if decision.get("reviewer_slots_degraded"):
        # B2: facts, never a retry coach (P5). The replay promise is CONDITIONAL —
        # wording SSOT: plan_render._degraded_replay_note (a free replay exists only
        # while a non-empty recorded epoch and the reviewer roster stand; an
        # empty-epoch wave re-dispatches a paid panel). Lazy import; plan_render
        # never imports owner_hurry, so no cycle.
        from ouroboros.tools.plan_render import _degraded_replay_note

        note = _degraded_replay_note({"health_epoch": decision.get("degraded_health_epoch")})
        return (
            f"{tag} Blocking plan review is OPEN with no parseable reviewer quorum (DEGRADED). "
            f"The recorded wave lists each reviewer slot's typed state and reset time; {note}; "
            "a changed spec starts the next paid cycle. Implementation stays held while the "
            "review is open."
        )
    if outcome == "REVIEW_REQUIRED":
        return (
            f"{tag} Blocking plan review remains REVIEW_REQUIRED. Re-call plan_task with a "
            "complete review_disposition as the only field, naming the latest fingerprint, "
            "then continue; do not rerun reviewers."
        )
    if outcome == "REVISE_PLAN":
        return (
            f"{tag} Blocking plan review requires a revised spec. Change the spec and call "
            "plan_task again (or reject the blocking findings with a rationale via "
            "review_disposition). Continue analysis and non-mutating preparation, but do not "
            "begin the work before the review closes or a real task-wide rail fires."
        )
    return (
        f"{tag} Call plan_task with a concrete goal, plan and spec. If review infrastructure "
        "is unavailable, continue analysis and non-mutating preparation, but do not begin the "
        "work before the review closes or a real task-wide rail fires."
    )


def plan_review_disclosure(decision: Dict[str, Any], forced_reason: str = "") -> str:
    """The loud host suffix for a plan review that did not close before finalization
    (text moved out of the pinned ``loop.py``): rail release, cap exhaustion under
    blocking, or the owner-selected advisory enforcement."""
    if not decision.get("required") or decision.get("status") == "closed":
        return ""
    outcome = str(decision.get("outcome") or "")
    if decision.get("reviewer_slots_degraded"):
        outcome = f"{outcome or 'open'}; no parseable reviewer quorum"
    subject = "Blocking plan review" if decision.get("enforcement") == "blocking" else "Plan review"
    if decision.get("status") == "rail_degraded":
        rail_reason = str(forced_reason or decision.get("reason") or "task_rail")
        detail = f" ({outcome})" if outcome else ""
        return (
            f"\n\n⚠️ {subject} remained open{detail} when the task-wide rail "
            f"`{rail_reason}` required best-effort finalization."
        )
    if decision.get("status") == "cycles_exhausted" and decision.get("enforcement") == "blocking":
        return (
            f"\n\n⚠️ Blocking plan review stayed open ({outcome or 'open'}) with the review-cycle "
            f"cap spent ({decision.get('cycles_paid')} paid cycle(s)); the task is finalized as "
            "blocked_with_evidence — the planned work must not be treated as done."
        )
    if decision.get("quorum_unreachable") and decision.get("enforcement") == "blocking":
        # B2b: the agent chose the honest blocked terminal while the reviewer quorum
        # was structurally unreachable; the review stays open and nothing was built.
        reset = str(decision.get("earliest_reset") or "")
        return (
            f"\n\n⚠️ Blocking plan review stayed open ({outcome or 'open'}) with its reviewer "
            "quorum structurally unreachable (typed window-exhausted reviewer lanes"
            + (f"; earliest recorded reset {reset}" if reset else "")
            + "); the task is finalized as blocked_with_evidence — the planned work must "
            "not be treated as done."
        )
    if decision.get("allow"):
        return (
            f"\n\n⚠️ Plan review remained {outcome or 'unavailable'}; work proceeded under the "
            "owner-selected advisory enforcement."
        )
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
