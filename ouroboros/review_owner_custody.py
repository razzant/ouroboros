"""Causal owner-death reconciliation for process-local review workers."""

from __future__ import annotations

import logging
import os
import pathlib
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Set


log = logging.getLogger(__name__)


def stamp_paid_review_owner(attempt: Any, *, paid: bool) -> None:
    """Bind an attempt whose reviewer workers this process owns (a reserved roster, paid or not yet)
    to the current process custody owner; ``paid`` False with no reservation binds nothing."""
    if not paid or int(getattr(attempt, "review_owner_pid", 0) or 0) > 0:
        return
    from ouroboros.process_custody import current_custody_session_id

    attempt.review_owner_session_id = str(current_custody_session_id() or "")
    attempt.review_owner_pid = os.getpid()


def _recoverable_review_invocations(drive_root: pathlib.Path) -> Dict[str, str]:
    """Unique durable delegated tokens keyed by their reserved operation."""
    from ouroboros import delegate_custody
    from ouroboros.delegate_custody_current import snapshot

    # ONE store read: the pending invocations and the held runs, whose rows carry the
    # START_REQUESTED event that minted them (no chain fold per worker death).
    view = snapshot(drive_root, closed=False)
    candidates: Dict[str, Set[str]] = {}
    records = [(record, str(record.get("invocation_id") or "")) for record in view["pending"]]
    for run in delegate_custody.open_runs(drive_root, state=view["state"]):
        token = str(getattr(run, "invocation_id", "") or "")
        if token and getattr(run, "request_event", None):
            records.append((delegate_custody._request_facts(drive_root, run.request_event), token))
    for record, token in records:
        operation_id = str(record.get("operation_id") or "")
        if (
            operation_id
            and token
            and str(record.get("surface") or "")
            in {"multi_model_review", "scope_review"}
        ):
            candidates.setdefault(operation_id, set()).add(token)
    return {
        operation_id: next(iter(tokens))
        for operation_id, tokens in candidates.items()
        if len(tokens) == 1
    }


@contextmanager
def _locked_review_state(root: pathlib.Path) -> Iterator[Any]:
    from ouroboros import review_state

    lock_fd = review_state.acquire_review_state_lock(root)
    if lock_fd is None:
        raise TimeoutError(f"Could not acquire review state lock for {root}")
    try:
        yield review_state._load_state_unlocked(root, strict_attempt_authority=True)
    finally:
        review_state.release_review_state_lock(root, lock_fd)


def _reconcile_review_owners(
    root: pathlib.Path, owner_is_dead: Callable[[Any], bool], *, expire_unpaid: bool = False,
) -> Dict[str, Any]:
    """Discover once per death batch, then mutate only the current locked state."""
    from ouroboros import review_state

    def eligible_pids(state: Any) -> set[int]:
        return {
            int(item.review_owner_pid) for item in state.attempts
            if int(getattr(item, "review_owner_pid", 0) or 0) > 0
            and review_state._attempt_has_active_review_custody(item)
            and owner_is_dead(item)
        }

    with _locked_review_state(root) as state:
        dead_pids = eligible_pids(state)
        if not dead_pids:
            expired = state.expire_stale_attempts() if expire_unpaid else []
            if expired:
                review_state._save_state_unlocked(root, state)
            return {"reconciled": [], "expired": expired}

    # Archive I/O must not hold the review-state lock. A concurrent checkpoint
    # or late result owns its current row; never write this earlier view back.
    recoverable = _recoverable_review_invocations(root)
    with _locked_review_state(root) as state:
        stamp = review_state._utc_now()
        reconciled = state.reconcile_process_local_review_custody_after_owner_loss(
            now_ts=stamp, recoverable_invocations=recoverable,
            confirmed_dead_owner_pids=dead_pids & eligible_pids(state),
        )
        expired = state.expire_stale_attempts(now_ts=stamp) if expire_unpaid else []
        if reconciled or expired:
            review_state._save_state_unlocked(root, state)
        return {"reconciled": reconciled, "expired": expired}


def reconcile_review_custody_on_process_start(drive_root: pathlib.Path) -> Dict[str, Any]:
    """Settle only review owners proven dead from an older server generation."""
    from ouroboros.platform_layer import pid_is_alive
    from ouroboros.process_custody import current_custody_session_id

    current_session = str(current_custody_session_id() or "")
    return _reconcile_review_owners(
        pathlib.Path(drive_root),
        lambda item: bool(item.review_owner_session_id)
        and item.review_owner_session_id != current_session
        and not pid_is_alive(int(item.review_owner_pid)),
        expire_unpaid=True,
    )


def reconcile_review_custody_after_confirmed_process_death(
    drive_root: pathlib.Path,
    owner_pid: int,
) -> Dict[str, Any]:
    """Settle tokenless review rows owned by one process confirmed dead."""
    return reconcile_review_custody_after_confirmed_process_deaths(drive_root, {int(owner_pid or 0)})


def reconcile_review_custody_after_confirmed_process_deaths(
    drive_root: pathlib.Path, owner_pids: set[int],
) -> Dict[str, Any]:
    """Settle one confirmed-dead batch without replaying history per worker."""
    pids = {int(pid) for pid in owner_pids if int(pid) > 0}
    if not pids:
        return {"reconciled": [], "expired": []}
    return _reconcile_review_owners(
        pathlib.Path(drive_root), lambda item: int(item.review_owner_pid) in pids,
    )


def reconcile_confirmed_dead_review_owner(
    drive_root: pathlib.Path,
    owner_pid: int,
) -> None:
    """Fail softly after a caller has proved this exact process dead."""
    reconcile_confirmed_dead_review_owners(drive_root, {int(owner_pid or 0)})


def reconcile_confirmed_dead_review_owners(drive_root: pathlib.Path, owner_pids: set[int]) -> None:
    """Fail softly without discarding unresolved review custody."""
    try:
        reconcile_review_custody_after_confirmed_process_deaths(drive_root, owner_pids)
    except Exception:
        # State admission remains fail-closed if this write cannot complete; a
        # later retry sees the active paid attempt instead of buying a duplicate.
        log.warning(
            "Failed to reconcile review custody for dead worker pids %s",
            sorted(owner_pids),
            exc_info=True,
        )
