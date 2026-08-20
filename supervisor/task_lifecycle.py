"""Queue-owned acceptance, cancellation, and replay-safe resume transitions.

This module is a code boundary only: ``supervisor.queue`` remains the single
state authority and every mutation still runs under its existing process lock.
Imports of the queue are intentionally lazy so the public queue API can re-export
these helpers without creating an import cycle.
"""

from __future__ import annotations

import itertools
import logging
import pathlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.utils import utc_now_iso

# Cancellation settlement PUBLICATION (typed outcome vocabulary, cancelled
# result fields, owed-before-settle registration, stored-truth publication,
# miss-lane delivery) lives in supervisor.cancel_publication — a module-size
# code boundary exactly like supervisor.queue_transitions. Imported here so
# this module keeps ONE public surface for callers, tests, and the
# supervisor.queue re-exports.
from supervisor.cancel_publication import (  # noqa: F401 -- intentional public re-exports
    CANCEL_ALREADY_SETTLED,
    CANCEL_CANCELLED,
    CANCEL_FAILED,
    CANCEL_NOT_FOUND,
    _CANCEL_TERMINALIZED,
    _cancel_result_fields,
    _cascade_delivery_row_locked,
    _deliver_on_miss,
    _is_workspace_task_record,
    _load_result_row,
    _publish_cancelled_task,
    _reconcile_delegated_runs_on_kill,
    _reconstructed_cost_fields,
    _register_owed_terminal_delivery,
    _salvage_cancelled_output,
    _settle_or_reopen_intent,
)


# Cancellation CUSTODY (claim, capture, confirmed death, settled write, owed
# delivery) lives in supervisor.cancel_custody — the same module-size code
# boundary as supervisor.cancel_publication and supervisor.queue_transitions.
# Imported here so this module keeps ONE public surface for callers, tests and
# the supervisor.queue re-exports; the cascade protocol below stays here.
from supervisor.cancel_custody import (  # noqa: F401 -- supervisor/task_lifecycle.py facade re-exports
    SETTLED_ALREADY,
    _active_intent,
    _claim_intent,
    _durable_settled_status,
    _finalize_cancel_intent_on_miss,
    _finish_captured_pending,
    _finish_captured_running,
    _intent_outcome_fields,
    _queue_module,
    _reaping_owner_abandoned,
    _recover_stranded_reaping_slot,
    _release_intent_claim,
    _restore_custody,
    _settle_intent,
    _worker_possibly_alive,
    cancel_task_custody,
)

log = logging.getLogger(__name__)


BUDGET_ROOT_FENCES: Dict[str, Dict[str, Any]] = {}


# Tasks whose subtree cancellation has begun (v6.82.0): descendant admission is
# FENCED under the queue lock, because a schedule event already draining would be
# admitted after any number of cascade sweeps. Bounded in memory (newest kept) —
# a cancelled tree is terminal, so an evicted entry names a tree that settled long
# ago; the registry is per-process by design (a restart has no live descendants to
# admit, and terminal task results are the durable truth).
CANCELLED_ROOT_FENCES: Dict[str, str] = {}
_CANCELLED_ROOT_FENCE_CAP = 4096
_CANCELLED_ROOT_FENCE_GRACE_SEC = 300.0
# Ids fenced by cascades that are STILL RUNNING. Pruning protects the union, not
# just the caller's own set: a second large cascade could otherwise push the cap
# over and evict an older in-flight cascade's fences, re-opening admission into a
# tree that is still being torn down. Add-only within a cascade; the whole entry
# is dropped when that cascade returns.
_ACTIVE_CASCADE_FENCES: Dict[str, set[str]] = {}
_CASCADE_TOKEN_SEQ = itertools.count()


def _next_cascade_token(task_id: str) -> str:
    """A unique key for one cascade's protected-fence set (concurrent cascades on
    the SAME target must not share, or the first to finish unprotects the other)."""
    return f"{task_id}:{next(_CASCADE_TOKEN_SEQ)}"


def _prune_cancellation_fences(*, protected: set[str]) -> None:
    """Bound the fence registry without evicting ANY in-flight cascade's ids.

    A single cascade may capture up to the per-root child ceiling (500) and several
    can overlap, so an oldest-first trim must skip the union of every RUNNING
    cascade's ids — protecting only the caller would let a second cascade evict an
    older one's fences and re-open admission into a tree still being torn down.
    Recently completed cascades receive a grace window too. Without it, the next
    cap-bound cascade could evict the just-completed root immediately after its
    active ownership was released, admitting a delayed schedule event from the
    supervisor queue. The cap is therefore soft while active/recent fences exist;
    a later prune removes them after the grace interval.
    """
    if len(CANCELLED_ROOT_FENCES) <= _CANCELLED_ROOT_FENCE_CAP:
        return
    live: set[str] = set(protected)
    for owned in _ACTIVE_CASCADE_FENCES.values():
        live |= owned
    for stale in list(CANCELLED_ROOT_FENCES):
        if len(CANCELLED_ROOT_FENCES) <= _CANCELLED_ROOT_FENCE_CAP:
            return
        if stale in live:
            continue
        try:
            stamp = str(CANCELLED_ROOT_FENCES.get(stale) or "").replace("Z", "+00:00")
            age = datetime.now(timezone.utc).timestamp() - datetime.fromisoformat(stamp).timestamp()
            if age < _CANCELLED_ROOT_FENCE_GRACE_SEC:
                continue
        except (TypeError, ValueError):
            # Unknown provenance fails safe: keep the fence instead of reopening
            # admission into a tree whose cancellation age cannot be proven.
            continue
        CANCELLED_ROOT_FENCES.pop(stale, None)


def root_cancellation_fenced(task: Dict[str, Any], root_task_id: str = "") -> bool:
    """Whether this task descends from ANY task whose cancellation has begun.

    The full ANCESTRY is walked (not just root/immediate parent): `/api/tasks/{id}/
    cancel` accepts any live id, so a mid-tree cascade must also refuse a grandchild
    that a still-live deeper descendant schedules afterwards — its root is the
    original root and its parent is not the cancelled node. Callers hold the queue
    lock, so the live maps are a consistent view; the walk is depth-bounded exactly
    like the cascade snapshot's.
    """
    if not CANCELLED_ROOT_FENCES:
        return False
    for candidate in (
        str(root_task_id or "").strip(),
        str(task.get("root_task_id") or "").strip(),
        str(task.get("parent_task_id") or "").strip(),
    ):
        if candidate and candidate in CANCELLED_ROOT_FENCES:
            return True
    q = _queue_module()
    live: Dict[str, Dict[str, Any]] = {
        str(item.get("id") or ""): item for item in q.PENDING if isinstance(item, dict)
    }
    live.update({
        str(rid): meta["task"] for rid, meta in q.RUNNING.items()
        if isinstance(meta, dict) and isinstance(meta.get("task"), dict)
    })
    parent = str(task.get("parent_task_id") or "").strip()
    seen: set[str] = set()
    while parent and parent not in seen and len(seen) < 100:
        if parent in CANCELLED_ROOT_FENCES:
            return True
        seen.add(parent)
        ancestor = live.get(parent)
        parent = str(ancestor.get("parent_task_id") or "").strip() if isinstance(ancestor, dict) else ""
    return False


def apply_budget_root_admission_fence(task: Dict[str, Any], root_task_id: str) -> bool:
    """Reject new work while a root is explicitly budget-paused OR being cancelled.

    The monetary authority remains the physical-attempt ledger.  This marker is
    only an admission latch, preventing a budget increase from silently resuming
    a root after one of its dispatches was refused.

    It is also THE root-level admission latch for subtree cancellation (v6.82.0):
    a cascade cannot rely on re-sweeping, because a `schedule_subagent` event
    already draining in the supervisor would be admitted after any number of
    sweeps and leave a live child under a Cancelled root. Both are answers to the
    same question — "may this root accept new work?" — so they share one latch
    and one call site instead of growing a second admission check.
    """
    if root_cancellation_fenced(task, root_task_id):
        task["_admission_blocked"] = "root_cancelled"
        return True
    fence = BUDGET_ROOT_FENCES.get(str(root_task_id or ""))
    if not isinstance(fence, dict) or str(fence.get("status") or "") not in {
        "active", "paused",
    }:
        return False
    task["_admission_blocked"] = "root_budget_fence"
    task["_budget_root_task_id"] = root_task_id
    task["_budget_fence_id"] = str(fence.get("fence_id") or "")
    return True


def restore_queue_fences(
    raw_acceptance: Any, raw_budget: Any,
) -> tuple[set[str], bool, bool]:
    """Validate snapshot fences and restore the small root-budget admission map."""
    malformed_acceptance = not isinstance(raw_acceptance, list)
    fenced_roots: set[str] = set()
    if not malformed_acceptance:
        for fence in raw_acceptance:
            if not isinstance(fence, dict):
                malformed_acceptance = True
                break
            status = str(fence.get("status") or "")
            root_id = str(fence.get("root_task_id") or "")
            if status in {"active", "sealed"}:
                if not root_id:
                    malformed_acceptance = True
                    break
                fenced_roots.add(root_id)
    malformed_budget = not isinstance(raw_budget, list)
    restored: Dict[str, Dict[str, Any]] = {}
    if not malformed_budget:
        for fence in raw_budget:
            if not isinstance(fence, dict):
                malformed_budget = True
                break
            root_id = str(fence.get("root_task_id") or "").strip()
            fence_id = str(fence.get("fence_id") or "").strip()
            status = str(fence.get("status") or "")
            if status in {"active", "paused"}:
                if not root_id or not fence_id:
                    malformed_budget = True
                    break
                # Read old v6.64 candidates, but deliberately discard their
                # synchronized subtree lists and replay classification.  One
                # durable marker is the complete admission state.
                restored[root_id] = {
                    "status": "paused",
                    "scope": "root",
                    "root_task_id": root_id,
                    "fence_id": fence_id,
                    "auto_resume": False,
                    "paused_at": str(fence.get("paused_at") or utc_now_iso()),
                }
    if not malformed_budget:
        BUDGET_ROOT_FENCES.clear()
        BUDGET_ROOT_FENCES.update(restored)
    return fenced_roots, malformed_acceptance, malformed_budget


def record_scheduled_admission(
    task: Dict[str, Any], admitted: Any, record: Dict[str, Any],
) -> None:
    """Project a cron dispatch refusal into terminal task/schedule state."""
    q = _queue_module()
    block = (
        str(admitted.get("_admission_blocked") or "")
        if isinstance(admitted, dict)
        else ""
    )
    if not block:
        record["failure_count"] = int(record.get("failure_count") or 0)
        record["last_error"] = ""
        return
    detail = f"Scheduled task was not queued: {block}."
    try:
        from ouroboros.task_results import STATUS_FAILED, write_task_result

        write_task_result(
            q.DRIVE_ROOT,
            str(task["id"]),
            STATUS_FAILED,
            result=detail,
            reason_code=block,
            cost_usd=0.0,
        )
    except Exception:
        q.log.warning(
            "Failed to terminalize admission-blocked scheduled task %s",
            task.get("id"),
            exc_info=True,
        )
    record["failure_count"] = int(record.get("failure_count") or 0) + 1
    record["last_error"] = detail


def task_subtree_is_live(task_id: str, *, ignore_intents: bool = False) -> bool:
    """Cheap liveness pre-check for the HTTP cascade-cancel path (v6.82).

    True when the task itself is queued/running, when it still has live
    descendants in the queue, or when it holds an ACTIVE durable cancel intent
    (or the legacy ``cancel_requested`` status latch of pre-redesign files) —
    the intent's settle still has honest work to do. Everything else is
    inactive and must keep today's 404 contract.

    ``ignore_intents=True`` is the PHYSICAL variant for the cascade
    postcondition (GR2-1e): the root's own cascade intent now survives until
    that postcondition passes, so the postcondition must judge queue/durable
    liveness only — counting the coordination intent itself would make the
    check circular and the cascade unable to ever report success.
    """
    q = _queue_module()
    task_id = str(task_id or "").strip()
    if not task_id:
        return False
    with q._queue_lock:
        self_running = task_id in q.RUNNING
        self_pending = any(
            isinstance(task, dict) and str(task.get("id") or "") == task_id
            for task in q.PENDING
        )
        descendants = [
            str(row.get("task_id") or "")
            for row in _live_descendants_locked(q, task_id, exclude_task_id=task_id)
        ]
    if self_pending:
        return True
    # A row whose DURABLE result already settled is a worker winding down, not
    # live work: its own finalizer owns the removal, and counting it as live would
    # make a cascade fail its postcondition and answer 503 for the documented
    # natural-completion race. The durable reads happen OUTSIDE the queue lock.
    if self_running and not _durable_settled_status(q, task_id):
        return True
    if any(tid and not _durable_settled_status(q, tid) for tid in descendants):
        return True
    if ignore_intents:
        return False
    try:
        from ouroboros.cancel_intents import has_active_intent
        from ouroboros.task_results import STATUS_CANCEL_REQUESTED, load_task_result

        if has_active_intent(q.DRIVE_ROOT, task_id):
            return True
        existing = load_task_result(q.DRIVE_ROOT, task_id) or {}
        return str(existing.get("status") or "") == STATUS_CANCEL_REQUESTED
    except Exception:
        return False


def cancel_task_by_id(task_id: str, *, cascade: bool = False) -> bool:
    """Cancel a task and, when requested, its atomically captured live subtree.

    The cascade is SYNCHRONOUS end to end (v6.82.0): the caller is answered only
    once the tree is actually torn down. That single decision is what removes the
    whole family of split-transaction hazards an ack-before-teardown design needs
    machinery for — no durable pre-acknowledgement latch, no partial-latch
    taxonomy, no fence ownership handed between a begin and a background teardown,
    no rollback that could withdraw a concurrent cascade's fences. What remains is
    the ADMISSION FENCE plus bounded re-sweeps.

    A cascade RE-SWEEPS: the snapshot is taken under the queue lock, but a
    `schedule_subagent` event already in the supervisor's drain queue can be
    admitted after it, so one snapshot alone could leave a late descendant
    running under a Cancelled root. Each sweep cancels what it saw; the loop
    stops when a fresh snapshot finds nothing new (bounded, so a pathological
    spawner cannot wedge the caller — the cancelled root's own worker is dead by
    then, so late arrivals are a draining queue, not an ongoing source).

    Every id gets a TYPED outcome from `queue.cancel_task_custody` — a boolean
    OR-aggregate would report success while a child that REFUSED to die is still
    live. Only terminalized ids are marked done; a failed id is retried by the
    next sweep. The call ends with an UNCONDITIONAL postcondition: it returns True
    only if nothing of the subtree is live any more.

    Completion always wins: custody is never taken from a task that already
    reached its own terminal result, and the durable write is refused by the
    result writer's monotonic guard if it settles mid-teardown.
    """
    q = _queue_module()
    task_id = str(task_id or "").strip()
    if not task_id:
        return False
    if not cascade:
        return q._cancel_task_by_id_single(task_id)
    # Close admission for the TARGET before the first snapshot (a schedule event
    # already draining would otherwise slip in), then each sweep fences every id it
    # captures — so a child naming a since-removed descendant as its parent is still
    # refused. The fence is the authority; the sweeps only catch children admitted
    # before it existed.
    cascade_token = _next_cascade_token(task_id)
    # Scope on the ROOT intent: a watchdog replay must re-run the cascade, not a
    # single cancel that would settle the root and leave descendants live.
    _record_cascade_scope(q, task_id)
    with q._queue_lock:
        CANCELLED_ROOT_FENCES[task_id] = utc_now_iso()
        _ACTIVE_CASCADE_FENCES[cascade_token] = {task_id}
        _prune_cancellation_fences(protected={task_id})
        # Capture the root's task row NOW (for the one cascade summary message):
        # after the sweeps it has left both live maps.
        root_task_row: Dict[str, Any] = {}
        running_meta = q.RUNNING.get(task_id)
        if isinstance(running_meta, dict) and isinstance(running_meta.get("task"), dict):
            root_task_row = dict(running_meta["task"])
        else:
            root_task_row = next(
                (dict(item) for item in q.PENDING
                 if isinstance(item, dict) and str(item.get("id") or "") == task_id),
                {},
            )
        if not root_task_row:
            # THE POLTERGEIST SHAPE: the root is already settled (budget hard
            # stop) while its children keep running, so it is in neither live map
            # and its task-result row carries no chat_id. Without a routing row
            # the lineage resolves to chat 0 and the whole tree's teardown is
            # delivered NOWHERE — the incident's silent ending. Any live
            # descendant carries the lineage chat, so borrow it.
            root_task_row = _cascade_delivery_row_locked(q, task_id)
    cancelled = False
    already: set[str] = set()
    failed: set[str] = set()
    all_outcomes: Dict[str, str] = {}
    try:
        for _sweep in range(4):
            swept, outcomes = _cancel_subtree_sweep(q, task_id, already, cascade_token)
            cancelled = swept or cancelled
            if not outcomes:
                break
            all_outcomes.update(outcomes)
            # ONLY terminalized ids are done. A `failed` id (stubborn process,
            # refused persistence) stays out of `already` so the next sweep
            # retries it.
            already.update(
                tid for tid, state in outcomes.items() if state in q._CANCEL_TERMINALIZED
            )
            failed = {tid for tid, state in outcomes.items() if state == q.CANCEL_FAILED}
        # UNCONDITIONAL postcondition. For a CASCADE the answer is a property of the
        # TREE, not a tally of who did the killing: success means nothing of the
        # subtree is live and nothing refused custody. (A concurrent cascade over
        # an overlapping subtree may legitimately have done the work — reporting
        # that as a failure would turn a settled tree into a 503. The endpoint's
        # own liveness pre-check still answers 404 for a tree that was never live.)
        # PHYSICAL liveness only (GR2-1e): the root's own cascade intent stays
        # open until this very check passes, so counting it would be circular.
        still_live = task_subtree_is_live(task_id, ignore_intents=True)
        # GR3-8: `failed` holds the LAST sweep's refusals, but a CONCURRENT
        # cascade/custody may have settled those very ids after that sweep ran
        # (its claim refused ours, then it finished the teardown). Re-judge each
        # against the CURRENT durable status — only a genuinely-unsettled
        # refusal vetoes a converged tree; a stale one must not turn a settled
        # subtree into a skipped summary and a 503.
        failed = {tid for tid in failed if tid and not _durable_settled_status(q, tid)}
        if failed or still_live:
            log.error(
                "Subtree cancellation for %s did not settle (failed=%s, still_live=%s)",
                task_id, sorted(failed), still_live,
            )
            return False
        if not cancelled:
            log.info("Subtree %s was already down when this cascade ran", task_id)
        # GR3-1c: the tree's ONE summary is ALWAYS registered as owed BEFORE the
        # settle — INCLUDING the replay/already-down path (`not cancelled`), the
        # exact branch a crash mid-cascade replays through; skipping the message
        # there replays the incident to silence. Idempotent by the per-intent
        # delivery_id (GR4-2); a chat-less tree records a typed handoff row.
        # A2: sweeps suppress per-task delivery; this is the tree's one message.
        summary_owed = True
        try:
            from supervisor.owner_stop import graceful_summary_suppressed
            from supervisor.terminal_delivery import deliver_cascade_summary

            # Q4=A: a graceful stop whose root COMPLETED (owner-requested
            # finalization) is confirmed by the model's own answer + card
            # state; the cascade receipt would be a duplicate and is
            # consciously suppressed (typed forensic row inside the helper).
            if not graceful_summary_suppressed(q, task_id):
                summary_owed = deliver_cascade_summary(
                    pathlib.Path(q.DRIVE_ROOT), task_id, root_task_row, all_outcomes,
                ) is not False
        except Exception:
            log.warning("Cascade summary delivery failed for %s", task_id, exc_info=True)
            summary_owed = False
        # GR3-1: the root's cascade coordination intent settles HERE and ONLY
        # here — the postcondition is the exclusive cascade settle owner (the
        # atomic scope guard in `settle_intent` refuses every other site), and
        # the summary above was registered as owed first, so a crash in between
        # still replays the message. Fenced (GR3-1d) by the freshly-read row's
        # generation/request_id. GR4-1 (the uniform rule): a summary that could
        # NOT be durably owed leaves the cascade intent OPEN — the watchdog
        # re-runs the cascade next tick and re-attempts the registration.
        try:
            from ouroboros.cancel_intents import SCOPE_CASCADE, active_intent, settle_intent

            root_intent = active_intent(q.DRIVE_ROOT, task_id)
            if root_intent and str(root_intent.get("scope") or "") == SCOPE_CASCADE:
                if summary_owed:
                    settle_intent(
                        q.DRIVE_ROOT, task_id, outcome="cancelled",
                        detail="cascade postcondition: no live subtree remains",
                        expected_generation=root_intent.get("generation"),
                        request_id=str(root_intent.get("request_id") or ""),
                        allow_cascade_scope=True,
                    )
                else:
                    log.warning(
                        "Cascade summary for %s could not be durably owed; leaving "
                        "the cascade intent open for the watchdog", task_id,
                    )
        except Exception:
            log.debug("cascade root-intent settle failed for %s", task_id, exc_info=True)
        return True
    finally:
        # The protected set is dropped only HERE — after the postcondition — so a
        # concurrent cascade's pruning can never evict this cascade's fences while
        # any of its checks still run.
        with q._queue_lock:
            _ACTIVE_CASCADE_FENCES.pop(cascade_token, None)


def _record_cascade_scope(q: Any, task_id: str) -> None:
    """Stamp ``scope=cascade`` on the root's EXISTING intent.

    Every cancel ingress records the durable intent (owner batch-4 1=A); this
    records the SHAPE the supervisor is actually running, because a crash
    mid-cascade leaves that intent open and a watchdog replaying it as a SINGLE
    cancel would settle the root while its live descendants kept running and
    spending. Deliberately no minting: an intent this call created would be one
    nobody promised to settle.

    A failure here is LOUD (GR2-1c): the HTTP cascade ingress now mints its
    intent with the cascade scope itself, which makes this call the second
    line of defense — but for other ingresses it can still be the write that
    decides whether a crash replays the cascade or silently drops the
    descendants, so it is a warning plus a typed forensic row, never a debug
    whisper.
    """
    try:
        from ouroboros.cancel_intents import SCOPE_CASCADE, mark_intent_scope

        mark_intent_scope(q.DRIVE_ROOT, task_id, SCOPE_CASCADE)
    except Exception:
        log.warning("cascade scope record failed for %s", task_id, exc_info=True)
        try:
            q.append_jsonl(
                pathlib.Path(q.DRIVE_ROOT) / "logs" / "supervisor.jsonl",
                {"ts": utc_now_iso(), "type": "cascade_scope_record_failed",
                 "task_id": task_id},
            )
        except Exception:
            log.debug("cascade-scope forensic append failed for %s", task_id, exc_info=True)


# Watchdog cadence guards: an intent younger than this may still be riding its
# own control event; the watchdog leaves it one tick before feeding custody.
_INTENT_WATCHDOG_MIN_AGE_SEC = 10.0


def sweep_cancel_intents(*, now: Optional[float] = None) -> Dict[str, str]:
    """Feed every open (unclaimed or stale-claimed) cancel intent into custody.

    The watchdog HALF of the phase-A cancel redesign: a lost control event (the
    incident wedged four children forever) no longer strands an intent, because
    the durable projection is re-swept on the supervisor tick and each open
    intent is driven through ``cancel_task_custody`` — the ONE settle owner.
    The watchdog itself never settles or writes terminal state. Also the boot
    driver for migrated legacy latches. Returns ``{task_id: typed outcome}``.
    """
    q = _queue_module()
    try:
        from ouroboros.cancel_intents import (
            INTENT_CLAIMED, SCOPE_CASCADE, active_intents, claim_is_abandoned,
        )
        from supervisor.owner_stop import OWNER_STOP_HOLDING, sweep_owner_stop_hold
    except Exception:
        return {}
    import time as _time

    current = now if now is not None else _time.time()
    outcomes: Dict[str, str] = {}
    try:
        # GR5-6: the watchdog is THE enforcement read — a corrupt projection is
        # disclosed (typed forensic row) instead of reading as "no intents".
        intents = active_intents(q.DRIVE_ROOT, disclose_corruption=True)
    except Exception:
        log.debug("cancel-intent sweep read failed", exc_info=True)
        return {}
    for task_id, intent in intents.items():
        if intent.get("state") == INTENT_CLAIMED and not claim_is_abandoned(intent, now=current):
            continue  # custody in flight; its owner settles or releases
        if sweep_owner_stop_hold(q, task_id, intent, now=current):
            # S3 policy-aware hold (§12.2 item 9): the graceful episode owns
            # this intent until its shared deadline; custody is NOT fed.
            outcomes[task_id] = OWNER_STOP_HOLDING
            continue
        raw = str(intent.get("requested_at") or "").replace("Z", "+00:00")
        try:
            requested_ts = datetime.fromisoformat(raw).timestamp()
        except (TypeError, ValueError):
            requested_ts = 0.0
        if requested_ts and (current - requested_ts) < _INTENT_WATCHDOG_MIN_AGE_SEC:
            continue  # give the in-band control event its tick first
        try:
            if str(intent.get("scope") or "") == SCOPE_CASCADE:
                # A cascade intent replays as a CASCADE. Re-feeding it as a single
                # cancel would settle the root and leave its descendants running —
                # exactly the shape a crash mid-cascade leaves behind.
                outcomes[task_id] = (
                    CANCEL_CANCELLED if cancel_task_by_id(task_id, cascade=True)
                    else CANCEL_FAILED
                )
            else:
                outcomes[task_id] = cancel_task_custody(task_id)
        except Exception:
            log.warning("cancel-intent sweep custody failed for %s", task_id, exc_info=True)
            outcomes[task_id] = CANCEL_FAILED
    return outcomes



def _cancel_subtree_sweep(
    q: Any, task_id: str, already: set[str], cascade_token: str = "",
) -> Tuple[bool, Dict[str, str]]:
    """One snapshot pass over the live subtree of ``task_id``: fence what it sees,
    then cancel it. Returns ``(cancelled_anything, {task_id: typed_outcome})``."""
    with q._queue_lock:
        live: Dict[str, Dict[str, Any]] = {
            str(task["id"]): task
            for task in q.PENDING
            if isinstance(task, dict) and str(task.get("id") or "")
        }
        live.update({
            str(running_id): meta["task"]
            for running_id, meta in q.RUNNING.items()
            if isinstance(meta, dict) and isinstance(meta.get("task"), dict)
        })
        descendants: List[Tuple[int, str]] = []
        for live_id, task in live.items():
            if live_id == task_id:
                continue
            root_id = str(task.get("root_task_id") or "")
            current = task
            distance = 0
            seen: set[str] = set()
            reaches_target = root_id == task_id
            while isinstance(current, dict) and distance < 100:
                parent_id = str(current.get("parent_task_id") or "")
                if not parent_id or parent_id in seen:
                    break
                distance += 1
                if parent_id == task_id:
                    reaches_target = True
                    break
                seen.add(parent_id)
                current = live.get(parent_id)
            if reaches_target:
                try:
                    distance = max(distance, int(task.get("depth") or 0))
                except (TypeError, ValueError):
                    pass
                descendants.append((distance, live_id))
        cancel_order = [
            item[1] for item in sorted(descendants, reverse=True)
            if item[1] not in already
        ]
        if task_id not in already:
            cancel_order.append(task_id)
        # Fence EVERY id captured in this snapshot before the lock is released:
        # once custody removes them from PENDING/RUNNING, a
        # still-draining schedule event names a parent that no longer exists in the
        # live maps, so the ancestry walk could not reconstruct the chain. Naming
        # the whole captured subtree makes the refusal a direct parent match.
        for fenced_id in cancel_order:
            CANCELLED_ROOT_FENCES[fenced_id] = utc_now_iso()
        if cascade_token:
            # Everything this sweep fenced joins the cascade's protected set for as
            # long as the cascade runs — not just for this one prune call.
            _ACTIVE_CASCADE_FENCES.setdefault(cascade_token, set()).update(cancel_order)
        _prune_cancellation_fences(protected={task_id, *cancel_order})
    if not cancel_order:
        return False, {}
    _append_cascade_snapshot_log(q, task_id, cancel_order, already)
    # Every captured descendant gets its OWN durable intent before custody runs
    # (owner batch-4 1=A applies to the whole tree, not only the id the owner
    # named): a crash mid-cascade otherwise leaves the descendants with no
    # durable fence at all, and a restart could restore and RUN them under a
    # cancelled root. They settle inside this same call on the success path, so
    # the projection stays empty when nothing goes wrong.
    for live_id in cancel_order:
        if live_id == task_id:
            continue
        try:
            from ouroboros.cancel_intents import request_cancel

            # ``allow_settled_target=True`` (GR6-1): every id here was captured
            # from the LIVE maps under the queue lock a moment ago, so live
            # physical ownership exists even when the durable result already
            # settled (a worker burning post-task cognition). Without the flag
            # the mint no-ops and a crash mid-sweep leaves that child unfenced;
            # custody settles the intent ``already_settled`` in this same call
            # once the death is confirmed.
            request_cancel(
                q.DRIVE_ROOT, live_id,
                reason=f"subtree cancellation of {task_id}",
                source="cascade_descendant", requested_by=task_id,
                allow_settled_target=True,
            )
        except Exception:
            # AR2-1 (owner batch-4 1=A): a child whose durable intent could not
            # be written is NOT silently left unfenced. Custody still runs on it
            # this sweep (killing it now beats leaving it live), but the failure
            # is surfaced — warning + typed forensic row — and the ROOT's open
            # ``scope=cascade`` intent makes the watchdog replay the WHOLE
            # cascade after a crash, re-attempting this mint.
            log.warning(
                "cascade descendant intent write failed for %s (child of %s); "
                "custody still runs this sweep, watchdog re-feeds via the root's "
                "cascade intent", live_id, task_id, exc_info=True,
            )
            try:
                q.append_jsonl(
                    pathlib.Path(q.DRIVE_ROOT) / "logs" / "supervisor.jsonl",
                    {"ts": utc_now_iso(), "type": "cascade_descendant_intent_write_failed",
                     "root_task_id": task_id, "task_id": live_id},
                )
            except Exception:
                log.debug("descendant intent-failure forensic append failed", exc_info=True)
    outcomes: Dict[str, str] = {}
    cancelled = False
    for live_id in cancel_order:
        # Children first (cancel_order is depth-sorted), each with its own typed
        # result — the parent's success can never mask a child's refusal.
        # deliver=False: the cascade delivers ONE root summary, not N messages.
        outcomes[live_id] = q.cancel_task_custody(live_id, deliver=False)
        cancelled = outcomes[live_id] == q.CANCEL_CANCELLED or cancelled
    return cancelled, outcomes


def _append_cascade_snapshot_log(
    q: Any, task_id: str, cancel_order: List[str], already: set[str],
) -> None:
    """Record the captured snapshot — FAIL-SOFT.

    The durable latch and the fences are already committed by the time this runs,
    so letting a telemetry write raise would abort the transaction after the tree
    was fenced and latched: the caller would answer 503 and never schedule the
    teardown, leaving a cancel-requested tree still running. Losing a log line is
    the strictly smaller loss.
    """
    try:
        q.append_jsonl(
            pathlib.Path(q.DRIVE_ROOT) / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "task_cancel_subtree_snapshot",
                "root_task_id": task_id,
                "descendant_task_ids": [tid for tid in cancel_order if tid != task_id],
                "descendant_count": len([tid for tid in cancel_order if tid != task_id]),
                "resweep": bool(already),
            },
        )
    except Exception:
        log.warning("Failed to log the cascade snapshot for %s", task_id, exc_info=True)


# Acceptance-fence, budget-resume and Project-deletion transitions live in
# supervisor.queue_transitions (module-size boundary); re-exported so
# `supervisor.queue` keeps ONE public import surface and existing callers are
# unchanged. The dependency is one-way: that module imports nothing from here.
from supervisor.queue_transitions import (  # noqa: E402, F401 -- intentional public re-exports
    _live_descendants_locked,
    clear_acceptance_fence_for_root,
    resume_budget_paused_task,
    resume_project_deletions,
    run_project_deletion,
    start_project_deletion,
    task_has_live_ownership,
    transition_acceptance_fence,
)
