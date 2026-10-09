"""The custody side of one ``review_change`` wave.

``ouroboros/tools/review_change.py`` owns the request, the wave and its ledger
record; what the shared review state (``review_state``) knows about the wave lives
here: the write-ahead paid attempt row the per-task cycle ceiling counts, the
delegated start token checkpointed onto its exact reserved seat, the settled row
carrying the wave's final roster — and the rejoin. Before a rerun pays again, the
open operation of the SAME logical round on the SAME task (identity c,
``review_subject.review_retry_key``) is collected through the commit gate's
reconcile-only path (``review_custody``: settled seats replay, the exact pending
delegated invocation is rejoined, nothing new is sent). A rejoin reads the retained
checkout at the deterministic path of this task's round
(``review_subject.checkout_token``), so its custody attempt key and every
operation's recovery binding are those of the operation it rejoins, in this process
or after a restart; another task's wave of the round has its own path.
"""

from __future__ import annotations

import copy
import logging
import pathlib
from typing import Any, Dict, Optional

from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

TOOL_NAME = "review_change"


def attempt_record(wave: Any, ctx: Any, **fields: Any) -> Any:
    """One attempt row of this wave (the gate's ``CommitAttemptRecord``): a review only.
    A paid row is bound to the process that pays it (``stamp_paid_review_owner``, as the
    gate's rows are): should that process die mid-wave, the next server generation's
    startup reconciliation proves the owner dead and closes the tokenless row as an
    infra failure instead of leaving an open operation nobody can collect; a row whose
    seats hold durable delegated tokens stays recoverable by their exact rejoin."""
    from ouroboros.review_owner_custody import stamp_paid_review_owner
    from ouroboros.review_state import CommitAttemptRecord, make_repo_key

    attempt = CommitAttemptRecord(
        ts=utc_now_iso(), commit_message=wave.label, task_id=str(getattr(ctx, "task_id", "") or ""),
        root_task_id=wave.root_task_id, repo_key=make_repo_key(wave.root), tool_name=TOOL_NAME,
        pre_review_fingerprint=str(wave.frozen.diff_sha), review_retry_key=wave.retry_key,
        rebuttal_sha256=wave.rebuttal_sha, review_contract_fingerprint=wave.contract_fp,
        review_record_id=wave.record_id, **fields)
    stamp_paid_review_owner(attempt, paid=bool(getattr(attempt, "paid", False)))
    return attempt


def pending_round_attempt(ctx: Any, *, root: pathlib.Path, retry_key: str) -> Optional[Any]:
    """The open operation of THIS round on THIS task, or ``None``: the newest attempt
    row of (root, ``review_change``, task) under the wave's retry key whose custody is
    still active — a seat late, in flight or lost (``review_state_custody``). Found,
    the rerun collects it instead of paying for a second physical review; only a
    NEW paid wave meets the per-task cycle ceiling. An unreadable review state
    raises: it is not evidence that nothing is owed."""
    from ouroboros.review_state import _utc_now, make_repo_key, update_state

    repo_key, task_id = make_repo_key(root), str(getattr(ctx, "task_id", "") or "")

    def _active(state: Any) -> list:
        state.expire_stale_attempts(now_ts=_utc_now())
        return [item for item in state.get_active_attempts(repo_key=repo_key)
                if item.tool_name == TOOL_NAME and item.task_id == task_id and item.review_retry_key == retry_key]

    rows = update_state(pathlib.Path(ctx.drive_root), _active)
    return rows[-1] if rows else None


def arm_rejoin(ctx: Any, wave: Any) -> None:
    """Put the wave on the commit gate's reconcile-only path for the open operation it
    collects: ``parallel_review`` freezes the attempt's roster
    (``review_custody.prepare_frozen_review_reconciliation``), the custody layer
    replays each settled seat and rejoins the exact pending delegated invocation, and
    ``git_review_cycle._reconcile_and_clear_review_roster`` merges the roster back.
    Read inside the wave context, after its reset. A no-op for a new wave."""
    rejoin = getattr(wave, "rejoin", None)
    if rejoin is None:
        return
    ctx._review_reconcile_only = True
    ctx._pending_review_attempt = rejoin
    ctx._current_review_attempt_number = int(rejoin.attempt or 0)


def install_paid_stamp(ctx: Any, wave: Any) -> Dict[str, int]:
    """The write-ahead paid fact at the wave's first physical dispatch, keyed by the
    REVIEWED root so the shared per-task-tree ceiling of (this root, ``review_change``)
    counts it; and the delegated start token's checkpoint onto its reserved seat
    (``review_state_custody.checkpoint_pending_review_invocation``), so a restart
    finds the exact invocation to rejoin. A rejoin writes onto the attempt row it
    collects, never a second paid row; its empty reservation leaves that row's
    roster as recorded."""
    from ouroboros.review_dispatch import ReviewPaidStamp
    from ouroboros.review_state import checkpoint_pending_review_invocation, make_repo_key, update_state

    holder = {"attempt": int(getattr(getattr(wave, "rejoin", None), "attempt", 0) or 0)}
    drive, repo_key = pathlib.Path(ctx.drive_root), make_repo_key(wave.root)
    task_id = str(getattr(ctx, "task_id", "") or "")

    def _write() -> None:
        reserved = getattr(ctx, "_review_reserved_roster", None)
        reserved = reserved if isinstance(reserved, dict) else {}
        triad = copy.deepcopy(list(reserved.get("multi_model_review") or []))

        def _mutate(state: Any) -> None:
            number = holder["attempt"] or state.next_attempt_number(repo_key, TOOL_NAME, task_id)
            state.record_attempt(attempt_record(
                wave, ctx, status="reviewing", phase="review", paid=True, attempt=number,
                triad_raw_results=triad))
            holder["attempt"] = number

        update_state(drive, _mutate)

    def _checkpoint(*, surface: str, slot_id: str, operation_id: str, invocation_id: str) -> None:
        checkpoint_pending_review_invocation(
            drive, repo_key=repo_key, tool_name=TOOL_NAME, task_id=task_id, attempt=holder["attempt"],
            review_retry_key=wave.retry_key, surface=surface, slot_id=slot_id, operation_id=operation_id,
            invocation_id=invocation_id)

    ctx._review_paid_stamp = ReviewPaidStamp(_write, fail_closed=True)
    ctx._review_pending_invocation_checkpoint = _checkpoint
    return holder


def settle_attempt(ctx: Any, wave: Any, outcome: Dict[str, Any], payload: Dict[str, Any],
                   forensic: Dict[str, Any]) -> None:
    """Close the attempt row this wave opened or rejoined (a review only; never a
    commit) with the wave's final roster: a seat still owed stays
    ``late_result_pending`` with its exact operation and start token, so a later
    rerun of the round — in this process or after a restart — finds what to collect."""
    number = (int((outcome.get("attempt") or {}).get("attempt") or 0)
              or int(getattr(getattr(wave, "rejoin", None), "attempt", 0) or 0))
    if number <= 0:
        return
    from ouroboros.review_state import update_state

    verdict = dict(payload.get("verdict") or {})
    try:
        update_state(pathlib.Path(ctx.drive_root), lambda state: state.record_attempt(attempt_record(
            wave, ctx, status="reviewed", phase="review", paid=True, attempt=number,
            late_result_pending=payload.get("state") == "pending",
            block_reason=str(outcome.get("block_reason") or ""),
            triad_raw_results=copy.deepcopy([row for row in forensic.get("triad_raw") or [] if isinstance(row, dict)]),
            degraded_reasons=[str(item) for item in verdict.get("degraded_reasons") or []])))
    except Exception:
        log.warning("review_change attempt row was not settled", exc_info=True)
