"""Durable custody of in-flight review invocations and attempt-history hygiene.

The checkpoint that records a pending review invocation before dispatch, the
paid fact merged onto a reserved attempt at its first physical send, what a new
call of a round owes its completed earlier wave (collect a usable verdict, or
retire the process-local replays of a wave without one), the
roster-row predicates that decide whether an attempt still holds active review
custody, the eviction rule that keeps settled attempts from starving the
bounded history, and the heavy-payload stripper applied on eviction. This is
post-cutoff upstream growth (the adaptive-timeout/custody train) cut out of
ouroboros/review_state.py by the v7next D06 lane as a NEW owner leaf — no
MIGRATION_v7 row names these symbols; review_state.py re-exports every name.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:  # annotation-only names; lazy under future annotations, never imported at runtime
    from ouroboros.review_state_model import AdvisoryReviewState
    from ouroboros.review_state_records import CommitAttemptRecord

log = logging.getLogger(__name__)


def _rs():
    """The parent review-state module, read at call time.

    The review-state members stay monkeypatch-addressable at their historical
    ``ouroboros.review_state`` bindings (tests rebind them there), so this leaf
    resolves every such cross-reference through the module at each call instead
    of freezing whatever object a from-import saw at import time.
    """
    from ouroboros import review_state

    return review_state


_ACTIVE_REVIEW_OPERATION_STATES = frozenset({"in_flight", "custody_lost"})
# A row that names its operation and whose state says that operation settled: a collector projects ``late_settled``
# only when no invocation is pending (``review_custody.finalize_review_actor``) and the start checkpoint never binds a
# settled row, so a start token such a row still carries is a lease nobody released, never custody (#2710). An
# outcome merged without an operation (``review_change_custody._merge_seat``) proves nothing: its row keeps the token.
_CLOSED_OPERATION_STATES = frozenset({"settled", "late_settled"})
# How long a seat launch waits for the shared review-state lock before binding its start token
# (``checkpoint_pending_review_invocation``). That binding is bookkeeping before a paid start, so
# other writers of the file are waited out instead of refusing the seat. A transport bound, not a
# judgment: longer than the lock's 90 s stale-holder horizon (``acquire_review_state_lock``), so a
# lock left by a dead writer is reclaimed inside the wait; only a live writer wedged mid-write
# outlasts it, and that refusal carries the lock facts.
SEAT_LAUNCH_LOCK_WAIT_SEC = 120.0


class ReservedReviewSlotRefused(ValueError):
    """The reserved commit-review row cannot take this start token: the attempt or row is missing,
    closed, or bound to another operation or token. The seat's own reason, never a busy lock."""


def _attempt_review_roster_rows(item: "CommitAttemptRecord") -> List[Dict[str, Any]]:
    """Return mutable row objects from both existing commit-review surfaces."""
    rows = [row for row in (item.triad_raw_results or []) if isinstance(row, dict)]
    scope = item.scope_raw_result
    if not isinstance(scope, dict) or not scope:
        return rows
    raw_results = scope.get("raw_results")
    if isinstance(raw_results, list):
        rows.extend(row for row in raw_results if isinstance(row, dict))
    elif "raw_results" not in scope:
        # Historical single-scope rows used the wrapper itself as the actor.
        rows.append(scope)
    return rows


def _start_token_open(row: Dict[str, Any]) -> bool:
    """Whether the row's start token still holds custody: it is there and the row's own named operation has not
    settled (``_CLOSED_OPERATION_STATES``). Every reading of a row's token goes through here."""
    return bool(str(row.get("pending_invocation_id") or "").strip()) and not (
        str(row.get("operation_id") or "").strip() and str(row.get("operation_state") or "") in _CLOSED_OPERATION_STATES)


def _review_roster_row_is_pending(row: Dict[str, Any]) -> bool:
    """A seat still owed: its start token holds (``_start_token_open``), its own late flag is up, or its operation is
    in flight or lost."""
    return bool(
        _start_token_open(row)
        or row.get("late_result_pending")
        or str(row.get("operation_state") or "") in _ACTIVE_REVIEW_OPERATION_STATES
    )


def _attempt_has_active_review_custody(item: "CommitAttemptRecord") -> bool:
    """One SSOT for overlap, startup, eviction, and payload compaction. Pure and in memory: it runs under the
    state lock on every save (the ledger's exact repair runs outside it, ``review_change_custody``)."""
    if str(getattr(item, "status", "") or "") == "reviewing" or bool(
        getattr(item, "late_result_pending", False)
    ):
        return True
    # A terminal lifecycle projection can still race a live delegated/API row;
    # preserve that exact roster. A synthetic malformed-container
    # ``custody_lost`` row on a historical terminal attempt is forensic damage,
    # not reopened physical work, so it does not make the old attempt active;
    # neither does a settled row's leftover start token.
    return any(
        _start_token_open(row)
        or str(row.get("operation_state") or "") == "in_flight"
        for row in _attempt_review_roster_rows(item)
    )


def counted_roster_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The seats whose answers count for the verdict: every row but an additional adviser's."""
    return [row for row in rows if not (row.get("additional") or (row.get("review_plan") or {}).get("additional"))]


def attempt_counted_seats_closed(item: "CommitAttemptRecord") -> bool:
    """Whether every counted seat of this attempt reached its outcome, whatever an additional seat still does: the
    row left ``reviewing`` and no counted seat is pending or holds a start token. An adviser's physical custody stays
    ``_attempt_has_active_review_custody``'s separate fact and never decides the counted outcome."""
    from ouroboros.review_custody import _row_is_pending
    from ouroboros.review_ledger import counted_pending

    rows = _attempt_review_roster_rows(item)
    counted = counted_roster_rows(rows)
    return bool(counted) and item.status != "reviewing" and not counted_pending(rows) and not any(
        _row_is_pending(row) or _start_token_open(row) for row in counted)


def _row_operation(row: Dict[str, Any]) -> tuple:
    return str(row.get("slot_id") or row.get("slot") or ""), str(row.get("operation_id") or "")


def _open(row: Dict[str, Any]) -> bool:
    return _review_roster_row_is_pending(row) or str(row.get("operation_state") or "") == "pending_dispatch"


def _non_send(row: Dict[str, Any]) -> bool:
    """A definite non-send (``not_dispatched``): no outcome, so a later send of that same operation may follow it."""
    return "not_dispatched" in (str(row.get("status") or ""), str(row.get("operation_state") or ""))


def late_write_outlived(existing: "CommitAttemptRecord | None", triad: Any, scope: Any,
                        settled_operations: Any = None) -> bool:
    """Whether a ``reviewing/late_wait`` write by a call still holding its pending snapshot adds nothing to its row,
    because a collector already closed that row's counted seats (#2587): every row it names is one the row holds
    under the same operation, settled or still running (an additional adviser). Written, it would reopen the
    settled phase or open a second row that never expires. A row compacted out of the recent history keeps no rows:
    the ``(slot, operation)`` pairs its bound settled record confirms closed stand for them (read outside the lock,
    ``review_change_custody.late_write_check``). A new operation, a row naming none, a definite non-send or a row not
    positively closed is written as before."""
    from types import SimpleNamespace

    if existing is not None and existing.raw_stripped and settled_operations is not None and existing.status != "reviewing":
        known = set(settled_operations)
    elif existing is not None and attempt_counted_seats_closed(existing):
        known = {_row_operation(row) for row in _attempt_review_roster_rows(existing) if not _non_send(row)}
    else:
        return False
    rows = _attempt_review_roster_rows(SimpleNamespace(triad_raw_results=triad or [], scope_raw_result=scope or {}))
    return bool(rows) and all(_row_operation(row)[1] and _row_operation(row) in known for row in rows)


def settled_row_outlives(old: Dict[str, Any], row: Dict[str, Any]) -> bool:
    """A seat's own outcome keeps its row against a late snapshot of the SAME operation still open (#2587): that
    writer read the seat before its collector settled it. A reopened seat names a new operation, and a definite
    non-send is no outcome, so a later send of its operation is written."""
    return (bool(_row_operation(row)[1]) and _row_operation(row) == _row_operation(old) and _open(row)
            and not _open(old) and not _non_send(old))


def checkpoint_pending_review_invocation(
    drive_root: pathlib.Path,
    *,
    repo_key: str,
    tool_name: str,
    task_id: str,
    attempt: int,
    review_retry_key: str,
    surface: str,
    slot_id: str,
    operation_id: str,
    invocation_id: str,
) -> None:
    """Bind one delegated start token to its pre-reserved commit-review row.

    The existing advisory-review state remains the only ledger.  This narrow
    locked patch is deliberately stricter than a whole-attempt merge: triad and
    scope start concurrently, so each may update only its exact reserved row
    and may never overwrite the other surface's token. A commit or review-only
    action may already have finished while its reserved reviewer is starting.
    Its logical status does not revoke that physical operation's custody; the
    paid attempt, retry key and exact in-flight slot below are the authority.
    This checkpoint records no verdict and never changes the commit status.
    A busy review-state lock is waited out (``SEAT_LAUNCH_LOCK_WAIT_SEC``): the binding is
    bookkeeping before a paid start. Only a reserved row that cannot take this token refuses,
    as ``ReservedReviewSlotRefused`` naming why.
    """
    expected = {
        "review_retry_key": str(review_retry_key or ""),
        "surface": str(surface or ""),
        "slot_id": str(slot_id or ""),
        "operation_id": str(operation_id or ""),
        "invocation_id": str(invocation_id or ""),
    }
    if not all(expected.values()):
        raise ReservedReviewSlotRefused("pending review invocation checkpoint is incomplete")
    if expected["surface"] not in {"multi_model_review", "scope_review"}:
        raise ReservedReviewSlotRefused(f"unsupported commit-review surface {surface!r}")

    def _mutate(state: AdvisoryReviewState) -> None:
        current = state.latest_attempt_for(
            repo_key=repo_key,
            tool_name=tool_name,
            task_id=task_id,
            attempt=int(attempt or 0),
        )
        if (
            current is None
            or not current.paid
            or current.review_retry_key != expected["review_retry_key"]
        ):
            raise ReservedReviewSlotRefused("reserved commit-review attempt is unavailable")
        if expected["surface"] == "multi_model_review":
            rows = current.triad_raw_results
        else:
            wrapper = current.scope_raw_result
            rows = wrapper.get("raw_results") if isinstance(wrapper, dict) else None
        if not isinstance(rows, list):
            raise ReservedReviewSlotRefused("reserved commit-review roster is unavailable")
        matches = [
            row for row in rows
            if isinstance(row, dict)
            and str(row.get("slot_id") or row.get("slot") or "") == expected["slot_id"]
        ]
        if len(matches) != 1:
            raise ReservedReviewSlotRefused("reserved commit-review slot identity is ambiguous")
        row = matches[0]
        if str(row.get("operation_id") or "") != expected["operation_id"]:
            raise ReservedReviewSlotRefused("reserved commit-review operation identity changed")
        row_state = str(row.get("operation_state") or "")
        if row_state == "pending_dispatch" and _left_before_launch_still_sends(expected["operation_id"]):
            # The author left only its own wait before this seat's first launch (its drain recorded it
            # pending_dispatch); it still sends, and this start makes it in flight. Read with the row,
            # under its lock: the drain may record the row while this start waits for that lock.
            row["operation_state"] = "in_flight"
        elif row_state != "in_flight":  # a Pause preparation never sends; a settled or lost row stays closed
            raise ReservedReviewSlotRefused("reserved commit-review operation is not in flight")
        prior = str(row.get("pending_invocation_id") or "")
        if prior and prior != expected["invocation_id"]:
            raise ReservedReviewSlotRefused("reserved commit-review invocation identity changed")
        row["pending_invocation_id"] = expected["invocation_id"]
        row["late_result_pending"] = True
        current.late_result_pending = True
        current.updated_ts = _rs()._utc_now()

    _rs().update_state(pathlib.Path(drive_root), _mutate, lock_timeout_sec=SEAT_LAUNCH_LOCK_WAIT_SEC)


def mark_review_attempt_paid(
    drive_root: pathlib.Path, *, repo_key: str, tool_name: str, task_id: str, attempt: int, review_retry_key: str,
) -> None:
    """Merge the write-ahead paid fact onto one reserved attempt row at its wave's first physical send.

    The reservation already wrote the roster and its owner before any worker started; this narrow locked
    patch adds only ``paid``, never a status, roster or late-result change, so a first send after its
    author left still stamps the wave's own row. A missing row or another round refuses the send.
    """
    from ouroboros.review_owner_custody import stamp_paid_review_owner

    def _mutate(state: AdvisoryReviewState) -> Any:
        current = state.latest_attempt_for(repo_key=repo_key, tool_name=tool_name, task_id=task_id,
                                           attempt=int(attempt or 0))
        if current is None or current.review_retry_key != str(review_retry_key or ""):
            raise ReservedReviewSlotRefused("reserved review attempt is unavailable for its paid dispatch")
        if current.paid:
            return _rs().ReviewStateMutation(None, changed=False)
        current.paid = True
        stamp_paid_review_owner(current, paid=True)
        current.updated_ts = _rs()._utc_now()

    _rs().update_state(pathlib.Path(drive_root), _mutate, lock_timeout_sec=SEAT_LAUNCH_LOCK_WAIT_SEC)


def settle_completed_round(
    usage_ctx: Any, drive_root: pathlib.Path, *, repo_key: str, tool_name: str, task_id: str, review_retry_key: str,
    exclude_attempt: int = 0,
) -> "CommitAttemptRecord | None":
    """What a new explicit call of a review round owes the newest earlier attempt of that round (#2410).

    Only an attempt whose every counted seat settled is decided, by ONE exact read of its own ledger record
    (``review_record_id``): a usable counted verdict (PASS/FAIL) is returned, for the caller to collect with
    that attempt's own roster instead of reserving a new wave, even while an additional seat still runs (its
    physical custody is its own; it never vetoes the counted outcome); a settled wave without one has its
    process-local replays (``review_custody``'s ``_review_settled_attempts``) retired, so the explicit call is a
    new physical attempt rather than a replay of old operations under a fresh roster. An open, lost or unknown
    counted seat, or an unreadable state or record, keeps every replay and returns nothing; ``_NO_RESEND`` and
    live workers are never touched. A row an earlier collector settled without its lifecycle projection gets its
    exact repair first (``review_change_custody.repair_settled_attempt``). ``exclude_attempt`` is the caller's
    own row. Custody's lock is taken after the reads, never inside."""
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK, _NO_RESEND, _row_is_pending
    from ouroboros.review_ledger import STATE_SETTLED, VERDICT_FAIL, VERDICT_PASS, counted_pending, ledger_root, load_record
    from ouroboros.tools.review_change_custody import repair_settled_attempt

    try:
        attempts = [item for item in _rs().load_state(pathlib.Path(drive_root)).filter_attempts(
                        repo_key=repo_key, tool_name=tool_name, task_id=task_id)
                    if review_retry_key and item.review_retry_key == review_retry_key
                    and int(item.attempt or 0) != int(exclude_attempt or 0)]
        attempt = max(attempts, key=lambda item: int(item.attempt or 0), default=None)
        if attempt is not None and _attempt_has_active_review_custody(attempt):
            attempt = repair_settled_attempt(drive_root, ledger_root(usage_ctx), attempt) or attempt
        rows = _attempt_review_roster_rows(attempt) if attempt is not None else []
        if attempt is None or not attempt_counted_seats_closed(attempt):
            return None
        record = load_record(ledger_root(usage_ctx), attempt.review_record_id) if attempt.review_record_id else None
    except Exception:
        log.warning("the round's earlier review attempt is unreadable; its settled replays stay", exc_info=True)
        return None
    if not record or counted_pending(record.get("rows") or []):
        return None
    if (record.get("verdict") or {}).get("aggregate") in (VERDICT_PASS, VERDICT_FAIL):
        return attempt
    if record.get("state") != STATE_SETTLED or any(_row_is_pending(row) or _start_token_open(row) for row in rows):
        return None  # retiring replays needs the whole wave closed, its additional seats included
    operations = {str(row.get("operation_id") or "") for row in rows} - {""}
    settled = getattr(usage_ctx, "_review_settled_attempts", None)
    with _ACTIVE_LOCK:
        if not isinstance(settled, dict) or operations & (
                {entry.operation_id for entry in _ACTIVE.values()} | set(_NO_RESEND.values())):
            return None
        for key in [key for key, actor in settled.items() if str(getattr(actor, "operation_id", "")) in operations]:
            settled.pop(key, None)
    return None


def _left_before_launch_still_sends(operation_id: str) -> bool:
    """This process's live seat for ``operation_id`` was left when only its author's wait ended, not by
    Pause (``review_pause.left_wait_only``, the rule its drain's pending row read): its preparation goes
    on and sends. The caller is that seat's own start checkpoint, under the review-state lock; custody's
    ``_ACTIVE_LOCK`` nests inside it, and no holder of that lock writes review state."""
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK
    from ouroboros.review_pause import left_wait_only

    with _ACTIVE_LOCK:
        entry = next((item for item in _ACTIVE.values() if item.operation_id == operation_id), None)
        return bool(entry is not None and left_wait_only(entry))


def _attempt_history_evictable(item: "CommitAttemptRecord") -> bool:
    """True only for rows the Max-Review-Cycles machinery derives NO authority
    from. Never evictable: paid rows (the per-root-task money count), rows of
    an attempt still in flight (their upcoming terminal/paid merge must find
    them), and review-VERDICT blocks (legacy rows classify by recorded reason
    via the commit gate's ``attempt_block_class``). Unclassifiable rows are kept — fail toward
    remembering."""
    if _attempt_has_active_review_custody(item):
        return False
    if bool(getattr(item, "paid", False)):
        return False
    try:
        from ouroboros.tools.commit_gate import BLOCK_CLASS_VERDICT, attempt_block_class

        return attempt_block_class(item) != BLOCK_CLASS_VERDICT
    except Exception:
        return False


_STRIPPED_DETAILS_LIMIT = 600


_STRIPPED_MESSAGE_LIMIT = 300


def _strip_attempt_heavy_payload(item: CommitAttemptRecord, *, force: bool = False) -> None:
    """Compact one preserved accounting row that fell outside the newest-50
    ledger window (F1 follow-up: paid-row immortality must not make the hot
    load-modify-save state file grow by full reviewer raw output forever).

    Dropped: the heavy forensic payloads NO gate reads from over-window rows —
    ``triad_raw_results`` and ``scope_raw_result`` (full reviewer raw_text) —
    plus the free-text fields bounded to what consumers render. Kept, exactly
    the facts the Max-Review-Cycles gates consume: ``paid``/``root_task_id``
    (the ceiling count), status/phase/``block_reason``/``block_class``/
    ``pre_review_fingerprint``/``review_contract_fingerprint``/
    ``rebuttal_sha256`` (the attempt's identity), and ``critical_findings`` plus a
    600-char ``block_details`` excerpt (what the attempt's history renders). A
    legacy scope verdict classifies THROUGH ``scope_raw_result``, so
    ``block_class`` is materialized before that evidence is dropped.
    ``raw_stripped=True`` marks the compaction for audits; full payloads live
    only in the newest-50 window. ``force=True`` re-compacts a row ALREADY
    marked stripped — the terminal-merge path uses it when a merge onto a
    stripped row would otherwise resurrect the heavy payloads it carries in."""
    if _attempt_has_active_review_custody(item):
        return
    if bool(getattr(item, "raw_stripped", False)) and not force:
        return
    if not item.block_class:
        try:
            from ouroboros.tools.commit_gate import attempt_block_class

            klass = attempt_block_class(item)
            if klass:
                item.block_class = klass
        except Exception:
            pass
    item.triad_raw_results = []
    item.scope_raw_result = {}
    item.block_details = _rs()._truncate_review_artifact(
        str(item.block_details or ""), limit=_STRIPPED_DETAILS_LIMIT
    )
    item.commit_message = _rs()._truncate_review_artifact(
        str(item.commit_message or ""), limit=_STRIPPED_MESSAGE_LIMIT
    )
    item.raw_stripped = True
