"""Cancellation CUSTODY: the one settle owner of a durable cancel intent.

Claim the intent exclusively before any custody mutation, capture the task,
confirm the worker's death, re-check the child's real settled result (natural
completion wins), reconcile delegated runs, capture artifacts, write the settled
result with reconstructed-or-unknown cost, register the owner's terminal answer
as OWED, only then settle the intent, and only then publish task_done. Every
mutation is fenced by the claim generation, so a taken-over attempt can neither
settle nor release what it no longer owns.

The cascade protocol - fences, tokens, and the subtree sweep - stays with
``supervisor.task_lifecycle``: it is one protocol over module-local state, and
this boundary deliberately does not cut through it.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, Optional
from supervisor.cancel_publication import (
    CANCEL_ALREADY_SETTLED,
    CANCEL_CANCELLED,
    CANCEL_FAILED,
    CANCEL_NOT_FOUND,
    _cancel_result_fields,
    _deliver_on_miss,
    _load_result_row,
    _publish_cancelled_task,
    _reconcile_delegated_runs_on_kill,
    _reconstructed_cost_fields,
    _register_owed_terminal_delivery,
    _salvage_cancelled_output,
    _settle_or_reopen_intent,
)

log = logging.getLogger(__name__)


def _queue_module():
    from supervisor import queue

    return queue


def _durable_settled_status(q: Any, task_id: str) -> str:
    """The task's own already-settled outcome, or "" — read once, off the hot path."""
    try:
        from ouroboros.task_results import load_task_result
        from ouroboros.task_status import SETTLED_STATUSES

        status = str((load_task_result(q.DRIVE_ROOT, task_id) or {}).get("status") or "")
        return status if status in SETTLED_STATUSES else ""
    except Exception:
        log.debug("Could not read durable status for %s", task_id, exc_info=True)
        return ""


def cancel_task_custody(task_id: str, *, deliver: bool = True) -> str:
    """Cancel one task and return a TYPED outcome, never a bare boolean.

    The ONE settle owner for cancellation (phase A): every ingress records a
    durable cancel intent first (``ouroboros.cancel_intents``); this custody
    CLAIMS that intent before teardown and SETTLES it with the terminal outcome.
    The supervisor watchdog only re-feeds open intents back here — it never
    settles on its own.

    CUSTODY model, in strictly ordered phases:

    0. CLAIM FIRST (GR2-2). The durable intent is claimed BEFORE any custody
       mutation. Two custody attempts racing the same task used to interleave —
       the loser entered the capture-miss lane before the winner claimed, saw
       no live claim, and double-settled (two ``cancelled`` writes, two
       ``task_done`` events). A refused claim is now ``failed`` with ZERO
       mutation; ``{}`` (no intent at all) keeps the legacy path, where the
       capture under the queue lock is the mutual exclusion.
    1. UNDER THE QUEUE LOCK — capture. A pending task leaves q.PENDING; a running
       task keeps its authoritative q.RUNNING row and its worker slot is marked
       ``reaping`` so no other actor can dispatch, reap, or respawn it.
       A task that already reached its OWN settled result is not captured at
       all: natural completion wins, keeps its result AND its own event.
    2. OUTSIDE THE LOCK — kill and JOIN the worker. Process teardown must never
       hold the global queue lock (it blocks every admission and dispatch for
       the duration), and the death must be CONFIRMED, not assumed.
    3. Only after confirmed death AND a successful durable write does the task
       become publicly cancelled: terminal result, `task_done`, worker respawn,
       drive cleanup, snapshot. If either step fails, custody is RESTORED (the
       task goes back where it came from), the intent claim is released for the
       watchdog to retry, and the outcome is ``failed`` — the caller must not
       report a cancellation that did not happen.

    ``deliver=False`` suppresses the per-task salvage chat delivery (cascade
    sweeps deliver ONE root message with a children digest instead).
    """
    q = _queue_module()
    from supervisor import workers

    task_id = str(task_id or "").strip()
    if not task_id:
        return CANCEL_NOT_FOUND

    # Read the durable intent BEFORE claiming it. The pre-claim row is what the
    # reaping-takeover gate below judges: a slot already marked ``reaping`` is
    # normally owned (reaper or a live custody) and must not be taken — but a
    # custody attempt that DIED mid-teardown leaves that marker behind forever
    # (assignment, the health check and the crash detector all skip a reaping
    # slot), so the watchdog would re-feed the intent into a permanent
    # CANCEL_FAILED loop. An ABANDONED claim is the proof the previous owner is
    # gone, and the only condition under which its slot is taken over.
    intent_before = _active_intent(q, task_id)

    # ---- phase 0: claim the intent BEFORE any mutation (GR2-2) -------------
    # Exclusivity comes from the claim, not from capture order: whichever
    # custody claims first owns the settle; the loser exits with ``failed``
    # having touched nothing, so it can never re-insert a captured row or
    # double-settle through the miss lane.
    intent = _claim_intent(q, task_id)
    if intent.get("claim_refused"):
        return CANCEL_FAILED
    generation = intent.get("generation")
    request_id = str(intent.get("request_id") or "")
    # Takeover authority (AR2-11, re-based on claim-first): our claim proves a
    # takeover ONLY if the pre-claim row was an ABANDONED custody claim on the
    # SAME intent. A live claimant would have refused us; a reaper-marked slot
    # carries no claim at all (the reaper owns that kill, and our trivially-
    # successful claim of a ``requested`` row grants no right to its slot).
    # The old under-lock re-read is superseded: a concurrent custody that
    # re-claimed after our pre-read would have made OUR claim the refused one.
    took_over_abandoned_claim = bool(
        intent
        and isinstance(intent_before, dict)
        and _reaping_owner_abandoned(intent_before)
        and str(intent_before.get("request_id") or "") == request_id
    )

    # ---- phase 1: capture under the lock -----------------------------------
    captured_was_reaping = False
    captured_pending = None
    captured_worker = None
    captured_meta = None
    with q._queue_lock:
        settled = _durable_settled_status(q, task_id)
        if settled:
            # Natural completion (or an earlier cancel) already decided this task.
            # A QUEUED row for a task with a terminal result is a ghost and is
            # dropped. A live WORKER is a different fact (GR6-1: the pipeline
            # persists the terminal result BEFORE post-task cognition ends), so
            # a settled RESULT does not mean a dead PROCESS — a busy worker is
            # captured below exactly like the unsettled path and driven through
            # kill/join. Completion wins on the write (the monotonic guard
            # keeps the stored terminal result) and the intent settles
            # ``already_settled`` only after the confirmed death.
            for index, item in enumerate(list(q.PENDING)):
                if str(item.get("id")) == task_id:
                    q.PENDING.pop(index)
                    break
        else:
            for index, item in enumerate(list(q.PENDING)):
                if str(item.get("id")) == task_id:
                    captured_pending = q.PENDING.pop(index)
                    break
        if captured_pending is None:
            for worker in workers.WORKERS.values():
                if worker.busy_task_id == task_id:
                    if settled and not _worker_possibly_alive(worker):
                        # Settled result AND provably dead process: no live
                        # ownership remains — the fast path below settles and
                        # recovers a stranded ``reaping`` marker. Only a
                        # possibly-ALIVE worker (post-task cognition still
                        # spending) is worth the capture/kill path.
                        break
                    captured_was_reaping = bool(getattr(worker, "reaping", False))
                    if captured_was_reaping and not took_over_abandoned_claim:
                        # The slot is ALREADY owned — by the reaper or
                        # another in-flight custody. Exactly one owner
                        # kills, publishes and respawns; a second taker
                        # would double-kill and double-respawn the slot.
                        # `failed` is honest here: the task is not settled
                        # yet, the caller's sweep retries, and the
                        # postcondition keeps refusing success until the
                        # real owner confirms death and persists the
                        # outcome. Our claim is released so the watchdog
                        # (or the real owner) is not blocked by a claim
                        # whose holder deliberately backed off.
                        break
                    captured_worker = worker
                    # ONE ownership state, shared with the reaper: the slot is
                    # marked `reaping` (assign_tasks, ensure_workers_healthy and
                    # the crash detector all skip it), and the task REMAINS in
                    # RUNNING — authoritatively visible, lineage intact — until
                    # its death is confirmed and its terminal result persisted.
                    # Popping the row here would blind task_subtree_is_live for
                    # the whole off-lock kill window, letting a concurrent
                    # cascade report a settled tree over a still-live process.
                    captured_meta = dict(q.RUNNING.get(task_id) or {})
                    captured_worker.reaping = True
                    break

    if settled and captured_worker is None and not captured_was_reaping:
        # A slot stranded at ``reaping`` by a custody attempt that crashed is
        # recovered HERE too: the task settled on its own afterwards, so nothing
        # else will ever revisit that worker.
        _recover_stranded_reaping_slot(q, task_id, intent_before)
        # GR5-3: the task is dead but its delegated runs may not be — the fast
        # already-settled path audits custody exactly like the kill path and
        # threads the disclosure into the miss-lane delivery.
        unreconciled = _reconcile_delegated_runs_on_kill(q, task_id)
        owed_ok = True
        if intent and deliver:
            # GR2-4 (fast already-settled re-entry): the settled answer is
            # delivered idempotently BEFORE the fenced settle removes the
            # intent — a crash between the two replays through the watchdog
            # and the durable-outbox dedupe suppresses any double. GR4-1: an
            # unowed answer reopens the intent instead of being settled over.
            owed_ok = _deliver_on_miss(
                q, task_id,
                _load_result_row(q, task_id), settled,
                unreconciled_runs=unreconciled,
            )
        _settle_or_reopen_intent(q, task_id, owed_ok=owed_ok, intent=intent,
                                 outcome=SETTLED_ALREADY, detail=settled)
        return CANCEL_ALREADY_SETTLED
    if captured_was_reaping and captured_worker is None:
        # The reaping-refusal branch above: nothing was mutated; give the claim
        # back so the real owner or the watchdog can finish.
        if intent:
            _release_intent_claim(
                q, task_id, error="slot owned by reaper or live custody",
                expected_generation=generation, request_id=request_id,
            )
        return CANCEL_FAILED
    try:
        if captured_pending is not None:
            return _finish_captured_pending(task_id, captured_pending, intent=intent)
        if captured_worker is not None:
            # ``settled_status`` (GR6-1b): a settled RESULT with a live WORKER
            # goes through the SAME kill/join path — the stored terminal truth
            # is preserved and the intent settles only after confirmed death.
            return _finish_captured_running(
                task_id, captured_worker, captured_meta or {},
                intent=intent, deliver=deliver, settled_status=settled,
            )
        return _finalize_cancel_intent_on_miss(task_id, intent=intent)
    except Exception:
        # A crash BETWEEN the capture and the respawn is what strands a slot at
        # ``reaping`` forever (the reaper's step-5 self-heal has the same
        # shape). Give the custody back and reopen the intent so the watchdog
        # retries instead of skipping the slot for the rest of the process life.
        log.error("Cancellation custody for %s raised; releasing custody", task_id, exc_info=True)
        _restore_custody(task_id, pending=captured_pending, worker=captured_worker)
        _release_intent_claim(
            q, task_id, error="custody raised mid-teardown",
            expected_generation=generation, request_id=request_id,
        )
        return CANCEL_FAILED


# Forensic settle outcome for "the task had already settled on its own".
SETTLED_ALREADY = "already_settled"


def _worker_possibly_alive(worker: Any) -> bool:
    """Whether a captured slot's process may still be running — fail-CLOSED.

    Used only by the settled-capture gate (GR6-1b): a probe that raises must
    answer "possibly alive" so custody proceeds through the kill path and
    CONFIRMS the death, never assumes it.
    """
    try:
        return bool(worker.proc.is_alive())
    except Exception:
        return True


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


def _finish_captured_pending(
    task_id: str, task: Dict[str, Any], *, intent: Optional[Dict[str, Any]] = None,
) -> str:
    """A queued task has no process: persist first, publish second."""
    q = _queue_module()
    from ouroboros.task_results import STATUS_CANCELLED, load_task_result, write_task_result

    cost_fields = _reconstructed_cost_fields(q, task_id, task)
    try:
        existing = load_task_result(q.DRIVE_ROOT, task_id) or {}
        stored = write_task_result(
            q.DRIVE_ROOT, task_id, STATUS_CANCELLED,
            **_cancel_result_fields(
                task, existing=existing, result="Task cancelled by user/agent request.",
                **cost_fields, **_intent_outcome_fields(intent or {}),
            ),
        )
    except Exception:
        log.warning("Cancel persistence failed for pending task %s", task_id, exc_info=True)
        _restore_custody(task_id, pending=task)
        _release_intent_claim(q, task_id, error="pending cancel persistence failed", intent=intent)
        return CANCEL_FAILED
    if str((stored or {}).get("status") or "") != STATUS_CANCELLED:
        # The writer's monotonic guard refused it: the task settled on its own
        # between capture and write. Its outcome and event stand.
        _settle_intent(q, task_id, outcome=SETTLED_ALREADY,
                       detail=str((stored or {}).get("status") or ""), intent=intent)
        return CANCEL_ALREADY_SETTLED
    _settle_intent(q, task_id, outcome="cancelled", detail="cancelled while pending", intent=intent)
    q._emit_cancel_task_done(task, task_id, cost_fields=cost_fields)
    q.persist_queue_snapshot(reason="cancel_pending")
    return CANCEL_CANCELLED


def _finish_captured_running(
    task_id: str, worker: Any, meta: Dict[str, Any], *,
    intent: Optional[Dict[str, Any]] = None, deliver: bool = True,
    settled_status: str = "",
) -> str:
    """A running task: CONFIRM the process is dead, persist, then publish.

    A4 ordering: confirmed death → natural child-result copy (completion WINS) →
    workspace artifact capture from the REAL tree → settled durable result →
    delivery + ``task_done`` → drive cleanup.

    ``settled_status`` (GR6-1b) names a task whose durable result settled
    BEFORE custody captured its still-live worker (post-task cognition burning
    past the terminal write). The kill/join above the durable boundary is
    identical; afterwards nothing is rewritten — the stored terminal truth is
    the answer (no salvage, no artifact re-capture over a result that already
    carries its own), it is registered as owed and delivered idempotently, and
    the intent settles ``already_settled`` after the confirmed death.
    """
    q = _queue_module()
    from ouroboros.platform_layer import kill_pid_tree
    from ouroboros.task_results import STATUS_CANCELLED, load_task_result, write_task_result

    task = meta.get("task") if isinstance(meta.get("task"), dict) else {}

    # ---- phase 2: kill and join OUTSIDE the lock ---------------------------
    # EVERY exit from this phase restores custody: an exception from the platform
    # kill, the service-pid lookup or a join would otherwise strand a possibly-live
    # worker outside RUNNING, where `task_subtree_is_live` cannot see it and the
    # cascade would report a settled tree.
    try:
        keep = q._kept_service_pids()
        if worker.proc.pid:
            kill_pid_tree(worker.proc.pid, exclude_pids=keep)
        elif worker.proc.is_alive():
            worker.proc.terminate()
        worker.proc.join(timeout=5)
        if worker.proc.is_alive() and worker.proc.pid:
            kill_pid_tree(worker.proc.pid, exclude_pids=keep)
            worker.proc.join(timeout=2)
    except Exception:
        log.error("Worker teardown for %s raised; cancellation refused", task_id, exc_info=True)
        _restore_custody(task_id, worker=worker)
        _release_intent_claim(q, task_id, error="worker teardown raised", intent=intent)
        return CANCEL_FAILED
    if worker.proc.is_alive():
        # A stubborn process is NOT a cancelled task: restoring custody keeps the
        # tree honest (still live, still owned by this worker) so the caller can
        # report a refusal instead of an imaginary success.
        log.error("Worker for %s survived kill escalation; cancellation refused", task_id)
        _restore_custody(task_id, worker=worker)
        _release_intent_claim(q, task_id, error="worker survived kill escalation", intent=intent)
        return CANCEL_FAILED

    unreconciled = _reconcile_delegated_runs_on_kill(q, task_id)

    if settled_status:
        # GR6-1b short-circuit, hoisted ABOVE every mutating step (GR7-2): the
        # result settled before the capture, the worker is now confirmed dead —
        # the kill is about the PROCESS, never the result, so the stored row
        # must survive BYTE-IDENTICAL. The old order ran child copy-back /
        # artifact finalize / memory export first, which mutated the settled
        # row (``headless_child_drive_root`` + a ``memory_export.json``
        # artifact on a shared drive; a split-drive copy-back REPLACING the
        # canonical settled answer — completion-wins violations). Deliver +
        # settle exactly like the natural-completion branch.
        from ouroboros.task_results import TASK_COST_META_FIELDS

        stored = load_task_result(q.DRIVE_ROOT, task_id) or {}
        stored_cost = {
            key: stored[key] for key in TASK_COST_META_FIELDS if key in stored
        } or {"cost_accounting_status": "unavailable", "cost_final": False,
              "cost_usd": None}
        owed_ok = _register_owed_terminal_delivery(
            q, task, task_id, stored, deliver=deliver,
            unreconciled_runs=unreconciled,
        )
        if not owed_ok and intent and intent.get("request_id"):
            _release_intent_claim(
                q, task_id,
                error="owed terminal-delivery registration failed", intent=intent,
            )
        else:
            _settle_intent(q, task_id, outcome=SETTLED_ALREADY,
                           detail=str(stored.get("status") or settled_status),
                           intent=intent)
        return _publish_cancelled_task(
            q, task_id, task, worker, stored, stored_cost,
            deliver=deliver, unreconciled_runs=unreconciled,
        )

    # POST-KILL natural-completion re-check (the incident's root cause, fixed):
    # forked/workspace/subagent tasks self-finalize on the CHILD drive and are
    # copied back only on task_done. The child's REAL result decides — SETTLED
    # statuses only (the old FINAL_STATUSES check read the cancel latch back as
    # "terminal" and published intent as an outcome). Natural completion WINS
    # (owner 4=A): a child that finished before the kill keeps its completed
    # result and artifacts; the cancel settles as "already settled".
    try:
        from ouroboros.headless import (
            copy_child_task_result, finalize_task_artifacts, task_is_readonly_subagent,
        )
        from ouroboros.task_results import TASK_COST_META_FIELDS
        from ouroboros.task_status import SETTLED_STATUSES

        child_result = copy_child_task_result(pathlib.Path(q.DRIVE_ROOT), task)
        if child_result and str(child_result.get("status") or "") in SETTLED_STATUSES:
            # A4 ordering: artifact capture/finalize BEFORE publication, so the
            # kept natural result carries its real artifacts.
            try:
                if not task_is_readonly_subagent(task):
                    finalize_task_artifacts(pathlib.Path(q.DRIVE_ROOT), task)
            except Exception:
                log.debug("Artifact finalize failed for naturally-settled %s", task_id, exc_info=True)
            child_cost = {
                key: child_result[key]
                for key in TASK_COST_META_FIELDS
                if key in child_result
            } or {"cost_accounting_status": "unavailable", "cost_final": False,
                  "cost_usd": None}
            kept_row = load_task_result(q.DRIVE_ROOT, task_id) or child_result
            # GR2-4: the kept answer is registered as OWED before the intent
            # settles — a crash between the two must not lose both the
            # watchdog trigger and the delivery. GR3-4: a registration that
            # could NOT be made durable leaves the intent OPEN (claim released
            # for the watchdog) instead of settling over an unowed answer —
            # the retry finds the settled result and re-delivers on the miss
            # lane.
            owed_ok = _register_owed_terminal_delivery(
                q, task, task_id, kept_row, deliver=deliver,
                unreconciled_runs=unreconciled,
            )
            if not owed_ok and intent and intent.get("request_id"):
                _release_intent_claim(
                    q, task_id,
                    error="owed terminal-delivery registration failed", intent=intent,
                )
            else:
                _settle_intent(q, task_id, outcome=SETTLED_ALREADY,
                               detail=str(child_result.get("status") or ""), intent=intent)
            return _publish_cancelled_task(
                q, task_id, task, worker, kept_row,
                child_cost, deliver=deliver, unreconciled_runs=unreconciled,
            )
    except Exception:
        log.debug("Child-drive terminal re-check failed for %s", task_id, exc_info=True)

    # Cost reconstruction is EVIDENCE, not custody: a ledger read that fails must
    # degrade to unknown fields rather than strand a task whose worker is already
    # dead (supervisor/events.py::_authoritative_terminal_cost treats unavailable
    # accounting the same way).
    cost_fields = _reconstructed_cost_fields(q, task_id, task)
    # Rescue the partial result BEFORE the durable write — symmetrically with the
    # timeout kill (task_reaper), and for a stronger reason: publication below
    # DELETES a subagent's drive, so the observability blobs this reads are the
    # only copy of the work the cancelled task had already done (BIBLE P1). An
    # owner who cancels a task should not lose strictly more than a supervisor
    # timeout would.
    salvage_note, salvage_text, salvage_path = _salvage_cancelled_output(q, task, task_id)
    # A4: capture the REAL workspace tree BEFORE the settled write — the patch
    # artifacts come from git facts (commits/dirtiness), never a blanket
    # "missing" stamp (owner batch-1 9=A). WORKSPACE tasks only: for a plain
    # task there is no tree to capture, and ``finalize_task_artifacts`` on a
    # task without a durable result would default-stamp a fabricated
    # ``completed`` status. A capture that fails persists ``failed`` with its
    # error; ``_cancel_result_fields`` below preserves any terminal artifact
    # status this call recorded.
    # A4/F5 — the honesty fence on the capture. ``finalize_task_artifacts``
    # DEFAULTS a task with no durable result to ``completed``: a task killed
    # inside the spawn→RUNNING-write window has no result file yet, so the
    # capture used to write a FABRICATED completion, which the monotonic guard
    # then defended against the real ``cancelled`` write — and the invented
    # ``completed`` was published AND delivered to the owner. So the capture runs
    # only when a durable row already exists to carry its own honest status; a
    # task that never got one has nothing captured and says so (``missing``,
    # "cancelled before workspace patch finalization"), instead of claiming a
    # completion that never happened.
    captured = "never_started"
    try:
        from ouroboros.headless import (
            _workspace_root_from_task, finalize_task_artifacts, task_is_readonly_subagent,
        )

        if _workspace_root_from_task(task) is not None and not task_is_readonly_subagent(task):
            if load_task_result(q.DRIVE_ROOT, task_id):
                captured = "attempted"
                finalize_task_artifacts(pathlib.Path(q.DRIVE_ROOT), task)
            else:
                # A4 (§8: провал capture = failed, не missing). The capture was
                # OWED — a RUNNING workspace task was killed — but cannot run,
                # because with no durable row ``finalize_task_artifacts`` would
                # fabricate a ``completed`` status (the F5 class). That is a
                # capture FAILURE, not an honest "nothing was ever due".
                captured = "owed_no_result"
    except Exception:
        log.debug("Cancel-path artifact capture failed for %s", task_id, exc_info=True)
    # GR3-2 minimal write-fence: the kill/join window above is where a stale
    # takeover could have re-claimed the intent. Re-verify OUR claim (pid +
    # generation) immediately before the durable terminal write; a lost claim
    # aborts the publication — the new owner (or the watchdog) writes the
    # terminal. Deliberately NOT a renewable-lease subsystem: one re-read at
    # the one write that matters. The release below is fenced, so it no-ops
    # when the claim really moved and only reopens OUR claim when the re-read
    # merely failed (fail-closed toward the watchdog, never a wedged claim).
    if intent and intent.get("request_id"):
        try:
            from ouroboros.cancel_intents import claim_still_owned

            still_ours = claim_still_owned(q.DRIVE_ROOT, task_id, intent)
        except Exception:
            still_ours = False
        if not still_ours:
            log.error(
                "Cancellation custody for %s lost its intent claim before the "
                "terminal write; aborting publication", task_id,
            )
            _restore_custody(task_id, worker=worker)
            _release_intent_claim(
                q, task_id, error="claim lost before terminal write", intent=intent,
            )
            return CANCEL_FAILED
    try:
        existing = load_task_result(q.DRIVE_ROOT, task_id) or {}
        stored = write_task_result(
            q.DRIVE_ROOT, task_id, STATUS_CANCELLED,
            **_cancel_result_fields(
                task, existing=existing, artifact_capture=captured, **cost_fields,
                **_intent_outcome_fields(intent or {}),
                **({"delegated_runs_unreconciled": unreconciled} if unreconciled else {}),
                result="Running task cancelled and worker terminated." + salvage_note,
            ),
        )
    except Exception:
        log.warning("Cancel persistence failed for running task %s", task_id, exc_info=True)
        _restore_custody(task_id, worker=worker)
        _release_intent_claim(q, task_id, error="cancel persistence failed", intent=intent)
        return CANCEL_FAILED

    # ---- DURABLE BOUNDARY CROSSED -----------------------------------------
    # The task's terminal truth is on disk. Everything past this line is
    # publication and slot hygiene: it is FAIL-SOFT and idempotent, because
    # answering 503 now would report a cancellation that demonstrably happened,
    # and a raising respawn must never leave the slot stranded at `reaping`.
    stored_status = str((stored or {}).get("status") or STATUS_CANCELLED)
    # GR2-4 (owed-before-settle): the owner's terminal answer is durably
    # registered as OWED before the intent settles. A crash between the settle
    # and the send used to lose BOTH the watchdog trigger (intent gone) and the
    # answer (nothing owed); now the boot/tick outbox replay delivers it, and
    # the publish below enqueues the same event idempotently by delivery_id.
    # GR3-4: a registration that could NOT be made durable leaves the intent
    # OPEN (claim released for the watchdog) instead of settling over an
    # unowed answer — the retry finds the settled result and re-delivers on
    # the miss lane.
    owed_ok = _register_owed_terminal_delivery(
        q, task, task_id, stored, deliver=deliver,
        salvage_text=salvage_text, salvage_path=salvage_path,
        unreconciled_runs=unreconciled,
    )
    if not owed_ok and intent and intent.get("request_id"):
        _release_intent_claim(
            q, task_id, error="owed terminal-delivery registration failed",
            intent=intent,
        )
    elif stored_status == STATUS_CANCELLED:
        _settle_intent(q, task_id, outcome="cancelled", detail="worker terminated",
                       intent=intent)
    else:
        # Completion wins (owner 4=A): the worker persisted its own terminal
        # result and the monotonic guard refused ours. Stamping a forensic
        # ``cancelled`` outcome over a task that COMPLETED would put the lie back
        # into the ledger the redesign exists to clean.
        _settle_intent(q, task_id, outcome=SETTLED_ALREADY, detail=stored_status,
                       intent=intent)
    return _publish_cancelled_task(
        q, task_id, task, worker, stored, cost_fields,
        deliver=deliver, salvage_text=salvage_text, salvage_path=salvage_path,
        unreconciled_runs=unreconciled,
    )


def _finalize_cancel_intent_on_miss(
    task_id: str, *, intent: Optional[Dict[str, Any]] = None,
) -> str:
    """Neither queued nor running: settle an open cancel intent (or a legacy
    ``cancel_requested`` latch file) as cancelled with reconstructed cost.

    Two things this lane must NOT do. It must not invent a task: an intent for an
    id that has no durable result at all names a task that never existed, and
    fabricating a ``cancelled`` row with $0 for it would put a phantom task in the
    ledger — it settles as ``not_found`` instead. And it must not bury a child
    that finished: when the row names a child drive, the child's own result is
    copied back BEFORE the cancelled write, so a crash of the split-drive
    copy-back window cannot cost a completed answer.
    """
    q = _queue_module()
    from ouroboros.task_results import (
        STATUS_CANCEL_REQUESTED, STATUS_CANCELLED, load_task_result, write_task_result,
    )

    try:
        active = dict(intent or {})
        if not active:
            active = _active_intent(q, task_id)
        existing = load_task_result(q.DRIVE_ROOT, task_id) or {}
        legacy_latch = str(existing.get("status") or "") == STATUS_CANCEL_REQUESTED
        if not active and not legacy_latch:
            return CANCEL_NOT_FOUND
        if not existing:
            # No durable row ANYWHERE for this id: nothing was ever scheduled
            # under it (a mistyped/stale id reaching the cancel ingress). Settle
            # the intent honestly rather than minting a cancelled task.
            _settle_intent(q, task_id, outcome="not_found",
                           detail="no durable task result for this id", intent=intent)
            return CANCEL_NOT_FOUND
        # A concurrent custody attempt may have captured this task between our
        # own capture miss and here (the pending double-settle probe). If the
        # live claim is no longer ours, it owns the settle — refuse and let it,
        # or the watchdog, finish.
        current = _active_intent(q, task_id)
        if (
            intent
            and current
            and str(current.get("request_id") or "") == str(intent.get("request_id") or "")
            and int(current.get("generation") or 0) != int(intent.get("generation") or 0)
        ):
            log.warning(
                "Cancel finalize-on-miss for %s yielded to a newer custody claim", task_id,
            )
            return CANCEL_FAILED
        # A4/completion-wins on the split-drive lane: promote the child's own
        # terminal result first when the row names a child drive.
        try:
            from ouroboros.headless import copy_child_task_result

            if str(existing.get("child_drive_root") or "").strip():
                copy_child_task_result(pathlib.Path(q.DRIVE_ROOT), {
                    "id": task_id,
                    "drive_root": str(existing.get("child_drive_root") or ""),
                    "child_drive_root": str(existing.get("child_drive_root") or ""),
                    "delegation_role": str(existing.get("delegation_role") or ""),
                })
        except Exception:
            log.debug("Finalize-on-miss child copy-back failed for %s", task_id, exc_info=True)
        # GR5-3: neither queued nor running — the worker is gone, but its
        # delegated runs may still be live; audit custody like the kill path
        # and thread the disclosure into every miss-lane delivery below.
        unreconciled = _reconcile_delegated_runs_on_kill(q, task_id)
        settled = _durable_settled_status(q, task_id)
        if settled:
            _recover_stranded_reaping_slot(q, task_id, active)
            owed_ok = _deliver_on_miss(
                q, task_id, load_task_result(q.DRIVE_ROOT, task_id) or existing, settled,
                unreconciled_runs=unreconciled,
            )
            _settle_or_reopen_intent(q, task_id, owed_ok=owed_ok, intent=intent,
                                     outcome=SETTLED_ALREADY, detail=settled)
            return CANCEL_ALREADY_SETTLED
        existing = load_task_result(q.DRIVE_ROOT, task_id) or existing
        cost_fields = _reconstructed_cost_fields(q, task_id, existing)
        stored = write_task_result(
            q.DRIVE_ROOT, task_id, STATUS_CANCELLED,
            **_cancel_result_fields(
                existing, existing=existing, **cost_fields,
                **_intent_outcome_fields(active),
                result="Task cancelled (was neither queued nor running at supervisor teardown).",
            ),
        )
        stored_status = str((stored or {}).get("status") or "")
        if stored_status != STATUS_CANCELLED:
            # The monotonic guard refused: something settled it while we worked.
            owed_ok = _deliver_on_miss(q, task_id, stored or existing, stored_status,
                                       unreconciled_runs=unreconciled)
            _settle_or_reopen_intent(q, task_id, owed_ok=owed_ok, intent=intent,
                                     outcome=SETTLED_ALREADY, detail=stored_status)
            return CANCEL_ALREADY_SETTLED
        # GR2-4 ordering: the delivery seam registers the answer as OWED before
        # the intent settles — a crash between the two replays instead of losing
        # both the watchdog trigger and the answer. GR4-1: an unowed answer
        # reopens the intent; the publication below still proceeds — the
        # terminal truth is on disk.
        owed_ok = _deliver_on_miss(q, task_id, stored or existing, STATUS_CANCELLED,
                                   unreconciled_runs=unreconciled)
        _settle_or_reopen_intent(q, task_id, owed_ok=owed_ok, intent=intent,
                                 outcome="cancelled", detail="finalized on miss")
        q._emit_cancel_task_done(existing, task_id, cost_fields=cost_fields)
        q.persist_queue_snapshot(reason="cancel_finalize")
        return CANCEL_CANCELLED
    except Exception:
        log.debug("Cancel finalize-on-miss failed for %s", task_id, exc_info=True)
        _release_intent_claim(q, task_id, error="finalize-on-miss failed", intent=intent)
        return CANCEL_FAILED
