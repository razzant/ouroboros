"""Review custody: in-flight invocations, a Continue's explicit selection and attempt-history hygiene.

The older interface is post-cutoff upstream growth (the adaptive-timeout/custody train) cut out of
ouroboros/review_state.py by the v7next D06 lane as a NEW owner leaf (no MIGRATION_v7 row names its symbols); the facade
review_state.py re-exports only part of it. It covers the checkpoint that records a pending review invocation
before dispatch, the paid fact merged onto a reserved attempt at its first physical send, what a new call of a round
owes its completed earlier wave (collect a usable verdict, or retire the process-local replays of a wave without one),
the roster-row predicates that decide whether an attempt still holds active review custody, the eviction rule that
keeps settled attempts from starving the bounded history and the heavy-payload stripper applied on eviction.

The active continuation custody is owned here and imported from here: who still holds a continued wave's operations,
the owed duty a Continue's explicit selection records before its choice, the unsent choices closed once their holder
ended, and the loan that keeps another task's retained source drive from retirement (``source_lending_hold``; one read
for a planning pass over many drives: ``source_loan_probe``).
"""

from __future__ import annotations

import contextlib
import functools
import logging
import pathlib
from typing import Any, Callable, Dict, List, TYPE_CHECKING

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


def _known_wave_holders(origin: str, record: Any = None, extra: Any = ()) -> List[str]:
    """Exact task ids that may hold a ``review_operations`` pointer or continuation decision on ``record``."""
    from ouroboros.review_waiting import _wait_owner

    doc = record if isinstance(record, dict) else {}
    raw = [origin, str(doc.get("task_id") or ""), _wait_owner(doc) if doc else "", *(extra or ())]
    for seat in doc.get("rows") or ():
        if isinstance(seat, dict):
            for entry in seat.get("continuations") or ():
                if isinstance(entry, dict):
                    raw.extend(_choice_holders(entry))
    return list(dict.fromkeys(str(tid) for tid in raw if tid))


def _choice_holders(entry: Dict[str, Any]) -> tuple:
    """Every task that genuinely held one recorded choice: its author, each earlier claimant and its current holder."""
    return (entry.get("decided_by_task"), *(entry.get("earlier_holders") or ()), entry.get("holder_task_id"))


def _local_wave_live(retry_key: str) -> bool:
    from ouroboros import review_custody as rc
    from ouroboros import review_operation as ro

    if not retry_key:
        return False
    with rc._ACTIVE_LOCK:
        for entry in rc._ACTIVE.values():
            op = getattr(entry, "operation", None)
            if not entry.event.is_set() and (
                    str(getattr(entry, "wave_key", "") or "").endswith(f"|{retry_key}")
                    or getattr(op, "retry_key", "") == retry_key):
                return True
    with ro._LOCK:
        return any(not op.closed and op.retry_key == retry_key for op in ro._LIVE.values())


def _wave_member_live(root: Any, task_id: str, retry_key: str = "", *, row: Any = None) -> bool:
    """Whether ``task_id`` still owns an open review operation on ``retry_key`` (any wave when empty): in this
    process's live registry (a paused explicit continuation before its pointer write included) or on its canonical
    task row under a live or unverifiable controller; an unreadable row is never read as ended."""
    from ouroboros import review_operation as ro
    from ouroboros.task_results import load_task_result

    with ro._LOCK:
        for op in ro._LIVE.values():
            author = getattr(getattr(op, "continuation_author", None), "task_id", "")
            if not op.closed and task_id in (op.task_id, author) and (not retry_key or op.retry_key == retry_key):
                return True
    loader = (lambda tid: row if tid == task_id else (load_task_result(root, tid, strict=True) or {})) if isinstance(row, dict) else None
    try:
        for owner_id, _subject, entry in ro._task_operation_entries(root, task_id, result_loader=loader):
            if entry.get("state") not in ro._OPEN_STATES or (retry_key and entry.get("retry_key") != retry_key):
                continue
            with ro._LOCK:
                live = ro._LIVE.get(owner_id)
            if (live is not None and not live.closed) or ro.controller_state(entry.get("controller")) in {"alive", "unknown"}:
                return True
    except Exception:
        return True
    return False


def wave_has_live_review_operation(
    root: Any, retry_key: str, *, holders: Any = (), local_only: bool = False, rows: "Dict[str, Any] | None" = None,
) -> bool:
    """Whether any local worker/operation or an exact known holder's ``review_operations`` pointer runs ``retry_key``."""
    from ouroboros.store import probe
    from ouroboros.task_results import load_task_result

    if not retry_key:
        return False
    if _local_wave_live(retry_key):
        return True
    if local_only or not holders:
        return False
    try:
        if probe(pathlib.Path(root))[0] != "present":
            return False
        for tid in dict.fromkeys(str(h) for h in holders if h):
            if rows is not None and tid in rows:
                status, row = rows[tid]
                if status != "ok":
                    return True
            else:
                row = load_task_result(root, tid, strict=True)
                if rows is not None:
                    rows[tid] = ("ok", row)
            if _wave_member_live(root, tid, retry_key, row=row if isinstance(row, dict) else {}):
                return True
        return False
    except Exception:
        return True


def _verified_lineage(root: Any, origin: str, target: str, *, rows: "Dict[str, Any] | None" = None) -> str:
    """``'ok'`` when ``target`` is ``origin`` or a verified Continue successor of ``origin``; ``'store_busy'`` or
    ``'task_result_unreadable'`` when a row along the chain cannot be read; ``''`` when the chain ends elsewhere."""
    from ouroboros.owner_continue import continued_by_binding
    from ouroboros.store import StoreBusy, StoreError
    from ouroboros.task_results import load_task_result

    if not target or target == origin:
        return "ok" if target else ""
    if not origin:
        return ""
    current, seen = origin, set()
    while current and current not in seen:
        seen.add(current)
        if rows is not None and current in rows:
            status, row = rows[current]
            if status != "ok":
                return status
        else:
            try:
                row = load_task_result(root, current, strict=True)
            except StoreBusy:
                return "store_busy"
            except (OSError, ValueError, StoreError, TypeError, KeyError):
                return "task_result_unreadable"
            if rows is not None:
                rows[current] = ("ok", row)
        if not isinstance(row, dict) or not row:
            return "task_result_unreadable"
        nxt = str(continued_by_binding(row).get("successor_task_id") or "")
        if nxt == target:
            return "ok"
        current = nxt
    return ""


def _holder_ended(
    root: Any, origin: str, holder: str, retry_key: str = "", *, ended_owner: str = "", rows: "Dict[str, Any] | None" = None,
) -> bool:
    """Whether ``holder`` is a verified Continue member from ``origin`` whose task and review operations have ended."""
    from ouroboros import review_operation as ro
    from ouroboros.task_results import load_task_result

    if not holder or _verified_lineage(root, origin, holder, rows=rows) != "ok":
        return False
    if rows is not None and holder in rows:
        status, row = rows[holder]
        if status != "ok":
            return False
    else:
        try:
            row = load_task_result(root, holder, strict=True)
        except Exception:
            return False
        if rows is not None:
            rows[holder] = ("ok", row)
    if not isinstance(row, dict) or not row:
        return bool(holder == ended_owner == origin and not _wave_member_live(root, holder, retry_key, row={}))
    if _wave_member_live(root, holder, retry_key, row=row):
        return False
    try:
        return bool(ro._actor_ended(pathlib.Path(root), holder, row))
    except Exception:
        return False


def continuation_decision_state(root: Any, record: Dict[str, Any], seats: Any = (), *, ended_owner: str = "") -> tuple:
    """For ``record``: ``(live_operations, closable_seats)`` over ``current_decisions``. An operation stays live while
    its exact holder (``_decision_holder``) is not proven ended on the verified Continue chain from ``task_id``, or
    while a live review operation or local worker still runs on this wave."""
    from ouroboros import review_custody as rc
    from ouroboros.review_continuation import _decision_holder, current_decisions

    decisions = current_decisions(record)
    if not decisions:
        return set(), set()
    origin = str(record.get("task_id") or ended_owner or "")
    retry_key = str(record.get("review_wave_id") or "")
    task_rows: Dict[str, Any] = {}
    holders = _known_wave_holders(origin, record, (ended_owner,))
    wave_live = wave_has_live_review_operation(root, retry_key, holders=holders, rows=task_rows)
    with rc._ACTIVE_LOCK:
        local_ops = {str(e.operation_id) for e in rc._ACTIVE.values()
                     if not e.event.is_set() and getattr(e, "operation_id", "")}
    ended_cache: Dict[str, bool] = {}
    live_ops, closable = set(), set()
    for sid, decision in decisions.items():
        op_id = str(decision.get("operation_id") or "")
        if not op_id:
            continue
        holder = _decision_holder(record, decision)
        if holder not in ended_cache:
            ended_cache[holder] = _holder_ended(root, origin, holder, retry_key, ended_owner=ended_owner, rows=task_rows)
        if wave_live or op_id in local_ops or not ended_cache[holder]:
            live_ops.add(op_id)
        else:
            closable.add(sid)
    return live_ops, closable


def claim_rejoined_holder(payload: Dict[str, Any], seat: "Dict[str, Any] | None", choice: Any) -> bool:
    """Stamp ``holder_task_id`` on an open rejoined choice when a new actor selects it. ``decided_by_task`` and each
    replaced holder (``earlier_holders``) stay history: their genuine choices keep their owed after-end duties."""
    from ouroboros.review_continuation import _decision_holder, _last_decision
    from ouroboros.review_ledger import seat_open

    dec = _last_decision(seat or {})
    target = str((choice.rejoin or choice.entry).get("operation_id") or "")
    holder = str(choice.entry.get("holder_task_id") or choice.entry.get("decided_by_task") or "")
    if (seat is None or not seat_open(seat) or str(dec.get("operation_id") or "") != target
            or not holder or _decision_holder(payload, dec) == holder):
        return False
    conts = [dict(item) for item in seat.get("continuations") or ()]
    if conts[-1].get("holder_task_id"):  # the holder this claim replaces stays named beside the choice's author
        conts[-1]["earlier_holders"] = [*(conts[-1].get("earlier_holders") or ()), conts[-1]["holder_task_id"]]
    seat["continuations"] = [*conts[:-1], {**conts[-1], "holder_task_id": holder}]
    return True


def selection_duty_unrecorded(root: Any, duty: Dict[str, Any]) -> bool:
    """True only when a pre-choice selection duty's ``selected_operations`` never landed for ``duty['task_id']`` on a
    readable ledger record (a crash or refusal between persisting the duty and recording the choice); unreadable raises."""
    from ouroboros import review_ledger as ledger

    selected = [str(op) for op in (duty.get("selected_operations") or ()) if op]
    owner, record_id = str(duty.get("task_id") or ""), str(duty.get("record_id") or "")
    if not selected or not owner or not record_id:
        return False
    doc = ledger.load_record(root, record_id)
    return doc is not None and not any(
        str(c.get("operation_id") or "") in selected and owner in _choice_holders(c)
        for s in doc.get("rows") or () if isinstance(s, dict)
        for c in s.get("continuations") or () if isinstance(c, dict)
    )


def record_explicit_selection(
    ctx: Any, drive: Any, record: Dict[str, Any], roster: Any, choices: list, writer: Any, *,
    attempt: int = 0, attempt_root: Any = None,
) -> Dict[str, Any]:
    """Persist the deciding actor's owed after-end duty BEFORE recording the choice or holder claim on ``record``,
    so a crash after ``writer`` never strands an unwaited choice, and retire a newly created pre-choice duty if
    ``writer`` refuses or fails before recording it. A commit attempt on ANOTHER task's retained drive is lent first
    (``lend_source_drive``): under that drive owner's custody lock, which the drive's retirement takes before its move,
    a retired drive refuses at $0 before anything is written, sent or recreated, and a present one is held by this
    duty until it is discharged (``source_lending_hold``)."""
    from ouroboros import review_pause, store
    from ouroboros.late_review_marks import OWED_SET
    from ouroboros.review_continuation import ContinuationRefused, _decision_holder, continuable_seats
    from ouroboros.review_ledger import seat_open
    from ouroboros.task_custody import task_custody_lock
    from ouroboros.tool_access_paths import canonical_data_root

    rows_by_seat = {str(r.get("seat_id") or ""): r for r in record.get("rows") or () if isinstance(r, dict)}
    active = [
        c for c in choices
        if c.rejoin is None or (
            seat_open(rows_by_seat.get(c.seat_id) or {})
            and str((rows_by_seat.get(c.seat_id) or {}).get("operation_id") or "") == str(c.rejoin.get("operation_id") or "")
        )
    ]
    if not active:
        return record
    raw_surface = str(record.get("surface") or "")
    surface = ("" if raw_surface == "intention" else "commit" if raw_surface == "commit_gate" or attempt_root is not None
               else raw_surface if raw_surface in {"result", "skill"} else "change")
    actor = str(getattr(ctx, "task_id", "") or "")
    record_id, retry_key = str(record.get("record_id") or ""), str(roster.retry_key or "")
    created_duty_id = ""
    selected_ops = sorted({str((c.rejoin or c.entry).get("operation_id") or "") for c in active} - {""})
    canonical = canonical_data_root(ctx) if getattr(ctx, "drive_root", None) else None
    dutiable = bool(surface and canonical is not None and actor and record_id and retry_key)
    key, owner = source_drive_key(canonical, attempt_root) if dutiable and attempt_root is not None else ("", "")
    lent = bool(key) and owner != actor  # the actor's own drive is held by the actor's own liveness
    with task_custody_lock(canonical, owner) if lent else contextlib.nullcontext(True) as locked:
        if not locked or (lent and not (pathlib.Path(attempt_root) / _rs()._STATE_RELPATH).is_file()):
            raise ContinuationRefused("roster_unavailable", "the predecessor's retained drive is being settled; choose "
                                      "again" if not locked else "the predecessor's retained drive was retired",
                                      continuable_seats(record))
        if dutiable:
            new_ops = {c.seat_id: str(c.entry["operation_id"]) for c in active if c.rejoin is None}
            prospective = sorted(
                new_ops[sid] if sid in new_ops else str(row.get("operation_id") or "")
                for sid, row in rows_by_seat.items()
                if sid in new_ops or row.get("superseded_operation_ids")
            )
            base_spec = {"surface": surface, "task_id": actor, "record_id": record_id, "retry_key": retry_key,
                         "attempt": int(attempt or 0)}
            base_id = review_pause._left_mark_id(base_spec)
            target_id = review_pause._left_mark_id({**base_spec, "continued": prospective})
            try:
                with store.open_store(canonical, write=False) as txn:
                    already_owed = txn.get_row(OWED_SET, base_id) is not None or txn.get_row(OWED_SET, target_id) is not None
                if lent:  # the loan names the duty before it exists: a duty never stands without its loan
                    lend_source_drive(canonical, key, (base_id, target_id))
            except Exception as exc:
                raise ContinuationRefused("duty_unwritable", f"the continuation duty store is unreadable: {exc}",
                                          continuable_seats(record)) from exc
            if not already_owed:
                if not review_pause.owe_unwaited_review(
                    ctx, surface=surface, retry_key=retry_key, record_id=record_id, attempt=int(attempt or 0),
                    attempt_root=attempt_root, attempt_task_id=str(roster.task_id or ""),
                    continued=prospective, selected_operations=selected_ops,
                ):
                    raise ContinuationRefused("duty_unwritable", "the continuation custody duty could not be recorded",
                                              continuable_seats(record))
                created_duty_id = target_id
    needs_write = any(c.rejoin is None or (actor and _decision_holder(record, c.rejoin) != actor) for c in active)
    if not needs_write:
        return record
    try:
        return writer(drive, record, active)
    except Exception:
        if created_duty_id and canonical is not None:
            try:
                probe_duty = {"task_id": actor, "record_id": record_id, "selected_operations": selected_ops}
                if selection_duty_unrecorded(canonical, probe_duty):
                    with store.open_store(canonical, write=True) as txn:
                        cur_row = txn.get_row(OWED_SET, created_duty_id) or {}
                        if (cur_row.get("left_review") or {}).get("selected_operations") == selected_ops:
                            txn.delete_row(OWED_SET, created_duty_id)
            except Exception:
                log.debug("pre-choice continuation duty %s left for after-end check", created_duty_id, exc_info=True)
        raise


def _matched_attempt(state: Any, attempt: Any) -> Any:
    cur = state.latest_attempt_for(repo_key=attempt.repo_key, tool_name=attempt.tool_name,
                                   task_id=attempt.task_id, attempt=int(attempt.attempt or 0))
    if cur is not None and attempt.review_record_id and cur.review_record_id != attempt.review_record_id:
        return None
    return cur


def _handoff_unsent_invocation(inv: "Dict[str, Any] | None", task_id: str, sid: str, op_id: str) -> bool:
    from ouroboros.delegate_invocation import HANDOFF_UNSENT

    return bool(inv and inv.get("state") == "pending" and inv.get("handoff") == HANDOFF_UNSENT
                and str(inv.get("operation_id") or "") == op_id and str(inv.get("slot_id") or "") == sid
                and str(inv.get("task_id") or "") == task_id)


def plan_unsent_continuation_rows(
    drive: Any, attempt_root: Any, attempt: Any, decisions: Dict[str, Dict[str, Any]],
    closable: Any, carried: Any, live_ops: set,
) -> Dict[tuple, Dict[str, Any]]:
    """Read-only check of ``closable`` continuation choices before acquiring the ledger lock: plans ``not_dispatched``
    previews without mutating the attempt row, and keeps ``live_ops`` fail-closed on unreadable state or live controllers."""
    from ouroboros import delegate_invocation
    from ouroboros import review_operation as ro
    from ouroboros.review_continuation import START_UNSENT, unsent_row

    if not closable or attempt is None or attempt_root is None:
        return {}
    try:
        current = _matched_attempt(_rs()._load_state_unlocked(pathlib.Path(attempt_root), strict_attempt_authority=True), attempt)
    except (OSError, ValueError, TypeError):
        for sid in closable:
            if decisions.get(sid, {}).get("operation_id"):
                live_ops.add(str(decisions[sid]["operation_id"]))
        return {}
    if current is None:
        return {}
    seats = {str(r.get("slot_id") or r.get("slot") or ""): r for r in _attempt_review_roster_rows(current)}
    planned = {}
    for sid in sorted(closable):
        dec = decisions.get(sid) or {}
        op_id, sup, tok = (str(dec.get(k) or "") for k in ("operation_id", "supersedes", "invocation_id"))
        cur = seats.get(sid)
        if not op_id or not sup or cur is None:
            continue
        cur_op, cur_state = str(cur.get("operation_id") or ""), str(cur.get("operation_state") or "")
        try:
            inv = delegate_invocation.invocation_record(drive, tok) if tok else None
        except Exception:
            live_ops.add(op_id)
            continue
        carried_unbound = cur_state == "in_flight" or (
            cur_state == "custody_lost" and (cur.get("usage") or {}).get("custody_state") == ro.START_UNBOUND)
        if cur_op == op_id and carried_unbound:
            ctrl = (cur.get("usage") or {}).get("review_controller")
            if isinstance(ctrl, dict) and ctrl and ro.controller_state(ctrl) in {"alive", "unknown"}:
                live_ops.add(op_id)
            elif _handoff_unsent_invocation(inv, str(attempt.task_id), sid, op_id):
                planned[(sid, op_id)] = unsent_row(cur, dec, cur.get("recovery_binding") or {}, START_UNSENT)
        elif cur_op == op_id and cur_state == "not_dispatched":
            planned[(sid, op_id)] = dict(cur)
        elif cur_op == sup and (sid, op_id) not in carried:
            if inv is None or inv.get("state") == "failed_definite" or _handoff_unsent_invocation(inv, str(attempt.task_id), sid, op_id):
                planned[(sid, op_id)] = unsent_row(cur, dec, {}, START_UNSENT)
    return planned


def confirm_unsent_continuation_row(
    drive: Any, attempt_root: Any, attempt: Any, sid: str, dec: Dict[str, Any],
) -> "Dict[str, Any] | None":
    """Run the locked attempt-row CAS and START retirement for one ended unsent choice (called inside ``_ledger_lock``)."""
    from ouroboros import delegate_invocation
    from ouroboros import review_operation as ro
    from ouroboros.review_continuation import START_UNSENT, _retire_start, unsent_row
    from ouroboros.tools.review_collection import attempt_roster

    if attempt is None or attempt_root is None:
        return None
    op_id, sup, tok = (str(dec.get(k) or "") for k in ("operation_id", "supersedes", "invocation_id"))
    try:
        inv = delegate_invocation.invocation_record(drive, tok) if tok else None
    except Exception:
        return None
    if inv is not None and inv.get("state") != "failed_definite" and not _handoff_unsent_invocation(inv, str(attempt.task_id), sid, op_id):
        return None
    try:
        roster = attempt_roster(attempt_root, attempt)
        surface = (roster.rows.get(sid) or ("multi_model_review", {}))[0]
        cur = roster.carried(sid)
        if not cur:
            return None
        cur_op, cur_state = str(cur.get("operation_id") or ""), str(cur.get("operation_state") or "")
        if cur_op == sup:
            now = roster.transition(sid, dec, lambda row: {k: v for k, v in unsent_row(row, dec, {}, START_UNSENT).items()
                                                           if k != "pending_invocation_id"})
            after = {"surface": surface, **now} if now.get("operation_id") == op_id and now.get("operation_state") == "not_dispatched" else None
        elif cur_op == op_id and cur_state == "not_dispatched":
            after = {"surface": surface, **cur}
        else:
            rs, box = _rs(), {}

            def mutate(state: Any) -> Any:
                current = _matched_attempt(state, attempt)
                row = next((r for r in (_attempt_review_roster_rows(current) if current is not None else ())
                            if str(r.get("slot_id") or r.get("slot") or "") == sid), None)
                if row is None:
                    return rs.ReviewStateMutation(None, changed=False)
                r_op, r_state = str(row.get("operation_id") or ""), str(row.get("operation_state") or "")
                if r_op == op_id and r_state == "not_dispatched":
                    box["after"] = {"surface": surface, **row}
                    return rs.ReviewStateMutation(None, changed=False)
                ctrl = (row.get("usage") or {}).get("review_controller")
                unbound = r_state == "in_flight" or (
                    r_state == "custody_lost" and (row.get("usage") or {}).get("custody_state") == ro.START_UNBOUND)
                if (r_op != op_id or not unbound
                        or (isinstance(ctrl, dict) and ctrl and ro.controller_state(ctrl) in {"alive", "unknown"})
                        or not _handoff_unsent_invocation(inv, str(attempt.task_id), sid, op_id)):
                    return rs.ReviewStateMutation(None, changed=False)
                built = {**unsent_row(row, dec, row.get("recovery_binding") or {}, START_UNSENT),
                         "superseded_operation_ids": list(row.get("superseded_operation_ids") or [])}
                built.pop("pending_invocation_id", None)
                row.clear()
                row.update(built)
                current.updated_ts = rs._utc_now()
                box["after"] = {"surface": surface, **built}
                return rs.ReviewStateMutation(None, changed=True)

            rs.update_state(pathlib.Path(attempt_root), mutate)
            after = box.get("after")
    except Exception:
        return None
    if after is not None and inv is not None and inv.get("state") == "pending":
        _retire_start(drive, str(attempt.task_id), sid, dec, START_UNSENT)
    return after


SOURCE_LOANS = "review_source_loans"  # canonical obligation set: drive relpath -> {"duties": [owed mark ids]}


def source_drive_key(canonical_root: Any, root: Any) -> tuple:
    """``(relpath, owner)`` when ``root`` is a task's own child layout under ``canonical_root``
    (``task_custody.child_layout_owner``), else ``("", "")``: the canonical root and foreign paths are never lent."""
    from ouroboros.task_custody import child_layout_owner

    try:
        parts = pathlib.Path(root).resolve(strict=False).relative_to(pathlib.Path(canonical_root).resolve(strict=False)).parts
    except (TypeError, ValueError):
        return "", ""
    owner = child_layout_owner(parts)
    return ("/".join(parts), owner) if owner else ("", "")


@contextlib.contextmanager
def existing_attempt_write(canonical: Any, root: Any):
    """Serialize a foreign attempt-row write with retirement, never creating its source drive."""
    from ouroboros.confined_files import real_directory
    from ouroboros.task_custody import task_custody_lock

    key, owner = source_drive_key(canonical, root)
    with task_custody_lock(canonical, owner, timeout_sec=0) if key else contextlib.nullcontext(True) as locked:
        if not locked:
            yield "source_custody_busy"
        else:
            yield "" if real_directory(root) and (pathlib.Path(root) / _rs()._STATE_RELPATH).is_file() else "source_attempt_unavailable"


def lend_source_drive(canonical_root: Any, key: str, duties: Any) -> None:
    """Record that the after-end duties ``duties`` (owed mark ids, existing or about to be written) use the drive
    ``key``. The caller holds the drive owner's custody lock; ids whose duty was discharged are dropped here."""
    from ouroboros import store
    from ouroboros.late_review_marks import OWED_SET

    with store.open_store(canonical_root, write=True) as txn:
        kept = [d for d in (txn.get_row(SOURCE_LOANS, key) or {}).get("duties") or () if txn.get_row(OWED_SET, d) is not None]
        txn.put_row(SOURCE_LOANS, key, {"duties": list(dict.fromkeys([*kept, *(str(d) for d in duties if d)]))})


def _loan_owed(txn: Any, loan: "Dict[str, Any] | None") -> bool:
    """Whether ``loan`` still names an owed after-end duty (``late_review_marks.OWED_SET``): a spent loan holds nothing."""
    from ouroboros.late_review_marks import OWED_SET

    return loan is not None and any(txn.get_row(OWED_SET, d) is not None for d in loan.get("duties") or ())


def source_loan_probe(canonical_root: Any) -> Callable[[Any], str]:
    """``source_lending_hold`` without ``forget``, for a pass that only plans over many drives (the Storage preview): the
    loan set, one row per lent drive, is read once, in one short read-only transaction, by the first drive that has a key,
    however many ask after it. ``probe(drive)`` gives the hold's reason, a busy store as ``store_busy`` where the hold
    raises ``StoreBusy``: a plan reports, it never retires. A retirement rechecks with its own reads."""
    from ouroboros import store

    @functools.cache
    def read() -> tuple:
        try:
            with store.open_store(canonical_root, write=False) as txn:
                return frozenset(key for key, loan in txn.set_rows(SOURCE_LOANS).items() if _loan_owed(txn, loan)), ""
        except store.StoreBusy:
            return frozenset(), "store_busy"
        except store.StoreError:
            return frozenset(), "review_state_unreadable"

    def probe(drive: Any) -> str:
        key = source_drive_key(canonical_root, drive)[0]
        owed, unreadable = read() if key else (frozenset(), "")
        return unreadable or ("review_custody_active" if key in owed else "")

    return probe


def source_lending_hold(canonical_root: Any, drive: Any, *, forget: bool = False) -> str:
    """``review_custody_active`` while a verified Continue's explicit selection lent ``drive`` and one of the after-end
    duties it named is still owed (``record_explicit_selection``), else ``""``. Each check of a drive nobody selected
    from costs one indexed lookup (two for a retirement that moves it: its first gate and Phase B), the canonical root
    none. Under the owner's custody lock (retirement's Phase B) ``forget`` deletes a loan whose duties were all
    discharged before the move may proceed; without that lock a loan is only read. Phase B's read and deletion run
    under the queue lock and never wait for the store: a busy store, for either, raises ``StoreBusy``; another refusal
    holds as ``review_state_unreadable``: the drive stays for the next pass, which reads and deletes again."""
    from ouroboros import store

    key = source_drive_key(canonical_root, drive)[0]
    if not key:
        return ""
    nonblocking = store.acquisition_budget(0) if forget else contextlib.nullcontext()
    try:
        with nonblocking, store.open_store(canonical_root, write=False) as txn:
            loan = txn.get_row(SOURCE_LOANS, key)
            if _loan_owed(txn, loan):
                return "review_custody_active"
    except store.StoreBusy:
        raise
    except store.StoreError:
        return "review_state_unreadable"
    if loan is not None and forget:  # a spent loan leaves before its drive, never after it
        try:
            with store.acquisition_budget(0), store.open_store(canonical_root, write=True) as txn:
                txn.delete_row(SOURCE_LOANS, key)
        except store.StoreBusy:
            raise
        except store.StoreError:
            return "review_state_unreadable"
    return ""
