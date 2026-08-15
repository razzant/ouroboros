"""CUSTODY of a cancel intent: who is settling this, right now, and by what right.

A cancel intent is a claim before it is an outcome. Exactly one process may be acting
on it at a time, so every mutation here is fenced by a claim GENERATION: a custody
attempt that lost its claim mid-flight must not settle or release what it no longer
owns, and it re-verifies immediately before the one durable write the kill/join window
could poison.

"Abandoned" is the delicate word and it is defined narrowly on purpose: a claimant
whose pid probes ALIVE is never stolen from, however old the claim. Only a provably
dead claimant, or one stale with liveness UNKNOWN, is taken over — and taking over
also recovers the worker slot its owner left marked `reaping`, because a claim that
outlived its process would otherwise strand the slot forever.

Split from `task_lifecycle`, which owns the cancellation PATHS. This owns the ledger
those paths write to.
"""

from __future__ import annotations

import logging

from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


def _queue_module():
    from supervisor import queue

    return queue


def _active_intent(q: Any, task_id: str) -> Dict[str, Any]:
    """The durable intent row for this task, or ``{}`` (fail-soft)."""
    try:
        from ouroboros.cancel_intents import active_intent

        return active_intent(q.DRIVE_ROOT, task_id) or {}
    except Exception:
        log.debug("cancel-intent read failed for %s", task_id, exc_info=True)
        return {}


def _reaping_owner_abandoned(intent: Dict[str, Any]) -> bool:
    """Whether a ``reaping`` slot's custody owner is provably gone.

    The ONLY takeover signal. A slot marked by the REAPER carries no claim, and a
    live custody's claim is fresh — neither is taken. An abandoned CLAIM (dead
    process or aged past ``CLAIM_STALE_SEC``) names a custody attempt that will
    never come back, and leaving its marker in place skips that worker slot for
    the rest of the process's life while the watchdog re-feeds the same intent
    into a permanent ``failed``.
    """
    try:
        from ouroboros.cancel_intents import claim_is_abandoned

        return bool(intent) and claim_is_abandoned(intent)
    except Exception:
        log.debug("cancel-intent abandonment check failed", exc_info=True)
        return False


def _recover_stranded_reaping_slot(q: Any, task_id: str, intent: Dict[str, Any]) -> bool:
    """Clear (and respawn) a worker slot a DEAD custody attempt left ``reaping``.

    Mirrors the reaper's own self-heal: assignment, ``ensure_workers_healthy``
    and the crash detector all skip a ``reaping`` slot, so a marker whose owner
    crashed removes a worker from the pool permanently. Gated on the same
    abandoned-claim proof as the takeover — the reaper's own markers are never
    touched.
    """
    if not _reaping_owner_abandoned(intent):
        return False
    from supervisor import workers

    target = None
    with q._queue_lock:
        for worker in list(workers.WORKERS.values()):
            if worker.busy_task_id == task_id and getattr(worker, "reaping", False):
                target = worker
                break
        if target is None:
            return False
        try:
            alive = bool(target.proc.is_alive())
        except Exception:
            alive = False
        if alive:
            # The process outlived its custody: releasing the marker alone would
            # hand a live process back to assignment, so leave the slot owned and
            # let the next custody attempt kill it.
            return False
    log.warning(
        "Recovering worker slot %s stranded at reaping by an abandoned cancellation custody (task %s)",
        getattr(target, "wid", "?"), task_id,
    )
    try:
        workers.respawn_worker(target.wid)
    except Exception:
        log.warning("Respawn of stranded slot for %s failed; clearing the marker", task_id, exc_info=True)
        with q._queue_lock:
            slot = workers.WORKERS.get(target.wid)
            if slot is not None:
                slot.reaping = False
    return True


def _claim_intent(q: Any, task_id: str) -> Dict[str, Any]:
    """Claim the durable intent for this custody attempt.

    Called BEFORE any custody mutation (GR2-2 claim-first): a refused claim
    (another LIVE custody owns the teardown) comes back with
    ``claim_refused: True`` and the caller exits ``failed`` having touched
    nothing — the interleaving where a capture-miss loser settled in parallel
    with the capture winner is structurally impossible once the claim is the
    first move.

    The two remaining shapes are deliberately DISTINCT (AR2-2):

    - ``{}`` means NO ACTIVE INTENT exists — the legacy/no-intent path. Custody
      may proceed: capture under the queue lock is the mutual exclusion for a
      task nobody minted an intent for (pre-migration legacy latches, direct
      custody callers), and the later ``_settle_intent`` no-ops harmlessly.
    - A claim attempt that RAISED cannot tell whether a live owner exists, so
      it is treated as refused: proceeding would settle without the exclusivity
      the fence exists to prove.
    """
    try:
        from ouroboros.cancel_intents import claim_intent

        return claim_intent(q.DRIVE_ROOT, task_id, owner="cancel_task_custody") or {}
    except Exception:
        log.warning("cancel-intent claim failed for %s; refusing custody", task_id, exc_info=True)
        return {"claim_refused": True, "claim_error": "claim_read_failed"}


def _settle_intent(
    q: Any, task_id: str, *, outcome: str, detail: str = "",
    intent: Optional[Dict[str, Any]] = None,
) -> None:
    """Settle (remove) the durable intent with its terminal outcome (fail-soft).

    ``intent`` is the row this custody CLAIMED: its generation fences the write,
    so a custody attempt that was taken over cannot delete an intent the new
    owner is still working.

    CASCADE OWNERSHIP (GR3-1, superseding the GR2-1e live-descendants gate): a
    ``scope=cascade`` intent is the WHOLE TREE's watchdog replay trigger AND
    the postcondition's summary obligation — per-task custody NEVER settles
    it, even when every descendant is already dead. A per-task settle over a
    dead-descendants cascade root would skip the tree's one owed summary (the
    incident's replay-to-silence shape). The refusal is enforced ATOMICALLY
    inside ``cancel_intents.settle_intent`` against the CURRENT durable scope
    — so a stale claim snapshot of an intent widened to cascade mid-flight
    cannot settle it either — and this caller's fenced claim is released in
    the same write, keeping the intent watchdog-replayable.
    """
    try:
        from ouroboros.cancel_intents import settle_intent

        settle_intent(
            q.DRIVE_ROOT, task_id, outcome=outcome, detail=detail,
            expected_generation=(intent or {}).get("generation"),
            request_id=str((intent or {}).get("request_id") or ""),
        )
    except Exception:
        log.debug("cancel-intent settle failed for %s", task_id, exc_info=True)


def _release_intent_claim(
    q: Any, task_id: str, *, error: str,
    expected_generation: Optional[int] = None, request_id: str = "",
    intent: Optional[Dict[str, Any]] = None,
) -> None:
    """Return a claimed intent to ``requested`` so the watchdog retries (fail-soft)."""
    if intent is not None:
        expected_generation = intent.get("generation")
        request_id = str(intent.get("request_id") or "")
    try:
        from ouroboros.cancel_intents import release_claim

        release_claim(
            q.DRIVE_ROOT, task_id, error=error,
            expected_generation=expected_generation, request_id=request_id,
        )
    except Exception:
        log.debug("cancel-intent claim release failed for %s", task_id, exc_info=True)


def _intent_outcome_fields(intent: Dict[str, Any]) -> Dict[str, Any]:
    """``parent_decision`` written only at OUTCOME (phase A): a parent-requested
    cancel stamps its decision on the SETTLED cancelled result, never at intent
    time — so a child that finished first keeps a decision-free completed record."""
    if not isinstance(intent, dict) or not intent.get("requested_by"):
        return {}
    fields: Dict[str, Any] = {"parent_decision": "cancelled"}
    if intent.get("reason"):
        fields["parent_decision_reason"] = str(intent.get("reason") or "")
    return fields


def _restore_custody(
    task_id: str, *, pending: Any = None, worker: Any = None,
    worker_reaping: bool = False,
) -> None:
    """Release custody after a failed cancellation.

    A captured PENDING task is put back in the queue. A RUNNING task needs no
    re-insert — capture never removed its row, so there is no ghost state to
    reconstruct; releasing the slot marker is the whole restore (a stranded
    ``reaping`` slot is skipped by assign and the health check forever).

    ``worker_reaping`` is the marker value to restore. The default False is
    right for the OWNING custody (it set the marker itself and its claim is
    released for the watchdog); a LOSER whose claim was refused passes the
    as-found value instead, because a True it found belongs to the concurrent
    winner still mid-kill (AR2-11).
    """
    q = _queue_module()
    with q._queue_lock:
        if pending is not None and all(str(t.get("id")) != task_id for t in q.PENDING):
            q.PENDING.append(pending)
        if worker is not None:
            worker.reaping = worker_reaping


__all__ = [
    "_queue_module",
    "_active_intent",
    "_reaping_owner_abandoned",
    "_recover_stranded_reaping_slot",
    "_claim_intent",
    "_settle_intent",
    "_release_intent_claim",
    "_intent_outcome_fields",
    "_restore_custody",
]
