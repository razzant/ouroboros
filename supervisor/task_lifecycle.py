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
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.utils import truncate_review_artifact, utc_now_iso

log = logging.getLogger(__name__)

#: Character budget for the orphaned-checkout disclosure written onto a tombstoned
#: project row and sent to the owner. A bound, never a silent cut: entries are
#: dropped whole and declared by count and thread id (see
#: :func:`_orphaned_checkouts_note`), and prose goes through the
#: ``truncate_review_artifact`` SSOT, which carries the omission marker.
ORPHANED_NOTE_LIMIT = 2000


#: How long a quiesce pass waits before re-reading the live set.
#:
#: Cancellation is a REQUEST, not an instant: `cancel_task_by_id` flags a task
#: and the worker settles on its own schedule. Re-reading with no pause spun a
#: daemon thread at full speed against the queue lock the workers need to
#: finish, and — worse — reached the "did not shrink" verdict on a millisecond
#: timescale, so a deletion could be declared stuck before anything had had a
#: chance to stop. Small enough that a real deletion is not noticeably slower,
#: long enough that a settling task gets a turn (T3R-17).
_QUIESCE_SETTLE_SEC = 0.05

_PROJECT_DELETE_WORKERS_LOCK = threading.Lock()
_PROJECT_DELETE_WORKERS: set[tuple[str, str]] = set()
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


def _queue_module():
    from supervisor import queue

    return queue


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


def transition_acceptance_fence(
    *, action: str, token: str, root_task_id: str = "", task_id: str = "", outcome: str = "",
    expected_generation: Optional[int] = None,
) -> Dict[str, Any]:
    """Atomically open, inspect, release, or seal a root admission fence."""
    q = _queue_module()
    action = str(action or "").strip().lower()
    token = str(token or "").strip()
    root_task_id = str(root_task_id or task_id or "").strip()
    if not token or action not in {"begin", "inspect", "end"}:
        return {"ok": False, "status": "error", "error": "invalid acceptance fence event"}
    with q._queue_lock:
        if action == "begin":
            if not root_task_id:
                return {"ok": False, "status": "error", "error": "missing root_task_id"}
            existing = q.ACCEPTANCE_FENCES.get(root_task_id)
            if isinstance(existing, dict) and str(existing.get("token") or "") != token:
                return {
                    "ok": False,
                    "status": "error",
                    "error": f"acceptance fence already active for root {root_task_id}",
                }
            if isinstance(existing, dict):
                row = existing
            else:
                row = q.ACCEPTANCE_FENCES[root_task_id] = {
                    "token": token,
                    "root_task_id": root_task_id,
                    "task_id": str(task_id or root_task_id),
                    "status": "active",
                    "opened_at": utc_now_iso(),
                    "owner_message_generation": 0,
                }
            result = {
                "ok": True,
                "status": "active",
                "root_task_id": root_task_id,
                "token": token,
                "owner_message_generation": int(row.get("owner_message_generation") or 0),
                "queue_descendants": _live_descendants_locked(
                    q, root_task_id, exclude_task_id=str(task_id or root_task_id),
                ),
            }
        else:
            matched_root = next(
                (rid for rid, row in q.ACCEPTANCE_FENCES.items() if str(row.get("token") or "") == token),
                "",
            )
            if not matched_root:
                return {"ok": False, "status": "error", "error": "unknown acceptance fence token"}
            row = q.ACCEPTANCE_FENCES[matched_root]
            if action == "inspect":
                return {
                    "ok": True,
                    "status": str(row.get("status") or "active"),
                    "root_task_id": matched_root,
                    "token": token,
                    "owner_message_generation": int(row.get("owner_message_generation") or 0),
                    "queue_descendants": _live_descendants_locked(
                        q, matched_root, exclude_task_id=str(row.get("task_id") or matched_root),
                    ),
                }
            normalized_outcome = str(outcome or "").strip().lower()
            if normalized_outcome == "revision":
                q.ACCEPTANCE_FENCES.pop(matched_root, None)
                result = {
                    "ok": True,
                    "status": "released",
                    "root_task_id": matched_root,
                    "token": token,
                }
            elif (
                expected_generation is not None
                and int(row.get("owner_message_generation") or 0) != int(expected_generation)
            ):
                current_generation = int(row.get("owner_message_generation") or 0)
                q.ACCEPTANCE_FENCES.pop(matched_root, None)
                result = {
                    "ok": True,
                    "status": "released",
                    "root_task_id": matched_root,
                    "token": token,
                    "generation_mismatch": True,
                    "expected_generation": int(expected_generation),
                    "owner_message_generation": current_generation,
                }
            else:
                row["status"] = "sealed"
                row["outcome"] = normalized_outcome or "terminal"
                row["sealed_at"] = utc_now_iso()
                result = {
                    "ok": True,
                    "status": "sealed",
                    "root_task_id": matched_root,
                    "token": token,
                }
    q.persist_queue_snapshot(reason=f"acceptance_fence_{result['status']}")
    return result


def _live_descendants_locked(
    q: Any, root_task_id: str, *, exclude_task_id: str = "",
) -> List[Dict[str, str]]:
    """Return a compact descendant snapshot while the queue lock is held."""
    rows: List[Dict[str, str]] = []
    for task in q.PENDING:
        task_id = str(task.get("id") or "") if isinstance(task, dict) else ""
        if task_id and task_id != exclude_task_id and q._is_descendant_of(task, root_task_id):
            rows.append({"task_id": task_id, "status": "pending", "source": "supervisor_queue"})
    for task_id, meta in q.RUNNING.items():
        task = meta.get("task") if isinstance(meta, dict) else None
        if (
            task_id
            and str(task_id) != exclude_task_id
            and isinstance(task, dict)
            and q._is_descendant_of(task, root_task_id)
        ):
            rows.append({"task_id": str(task_id), "status": "running", "source": "supervisor_queue"})
    return rows


def clear_acceptance_fence_for_root(root_task_id: str) -> bool:
    """Release a terminal root's fence after its task_done is queue-visible."""
    q = _queue_module()
    root_task_id = str(root_task_id or "").strip()
    if not root_task_id:
        return False
    with q._queue_lock:
        return q.ACCEPTANCE_FENCES.pop(root_task_id, None) is not None


def task_subtree_is_live(task_id: str) -> bool:
    """Cheap liveness pre-check for the HTTP cascade-cancel path (v6.82).

    True when the task itself is queued/running, when it still has live
    descendants in the queue, or when its persisted result sits in the
    ``cancel_requested`` latch (finished mid-teardown — the per-task cancel's
    finalize-on-miss branch still has honest work to do). Everything else is
    inactive and must keep today's 404 contract.
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
    try:
        from ouroboros.task_results import STATUS_CANCEL_REQUESTED, load_task_result

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
    with q._queue_lock:
        CANCELLED_ROOT_FENCES[task_id] = utc_now_iso()
        _ACTIVE_CASCADE_FENCES[cascade_token] = {task_id}
        _prune_cancellation_fences(protected={task_id})
    cancelled = False
    already: set[str] = set()
    failed: set[str] = set()
    try:
        for _sweep in range(4):
            swept, outcomes = _cancel_subtree_sweep(q, task_id, already, cascade_token)
            cancelled = swept or cancelled
            if not outcomes:
                break
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
        still_live = task_subtree_is_live(task_id)
        if failed or still_live:
            log.error(
                "Subtree cancellation for %s did not settle (failed=%s, still_live=%s)",
                task_id, sorted(failed), still_live,
            )
            return False
        if not cancelled:
            log.info("Subtree %s was already down when this cascade ran", task_id)
        return True
    finally:
        # The protected set is dropped only HERE — after the postcondition — so a
        # concurrent cascade's pruning can never evict this cascade's fences while
        # any of its checks still run.
        with q._queue_lock:
            _ACTIVE_CASCADE_FENCES.pop(cascade_token, None)


# Typed per-id cancellation outcome (v6.82.0). A boolean cannot distinguish
# "cancelled", "it had already finished on its own" and "refused/failed" — and a
# cascade that OR-aggregates booleans reports success while a child is still live.
CANCEL_CANCELLED = "cancelled"
CANCEL_ALREADY_SETTLED = "already_settled"
CANCEL_NOT_FOUND = "not_found"
CANCEL_FAILED = "failed"
_CANCEL_TERMINALIZED = frozenset({CANCEL_CANCELLED, CANCEL_ALREADY_SETTLED, CANCEL_NOT_FOUND})


def _durable_settled_status(q: Any, task_id: str) -> str:
    """The task's own already-final outcome, or "" — read once, off the hot path."""
    try:
        from ouroboros.task_results import STATUS_CANCEL_REQUESTED, load_task_result
        from ouroboros.task_status import FINAL_STATUSES

        status = str((load_task_result(q.DRIVE_ROOT, task_id) or {}).get("status") or "")
        # cancel_requested is a latch, not an outcome the task reached itself.
        return status if status in (FINAL_STATUSES - {STATUS_CANCEL_REQUESTED}) else ""
    except Exception:
        log.debug("Could not read durable status for %s", task_id, exc_info=True)
        return ""


def cancel_task_custody(task_id: str) -> str:
    """Cancel one task and return a TYPED outcome, never a bare boolean.

    CUSTODY model, in three strictly ordered phases:

    1. UNDER THE QUEUE LOCK — capture. A pending task leaves q.PENDING; a running
       task keeps its authoritative q.RUNNING row and its worker slot is marked
       ``reaping`` so no other actor can dispatch, reap, or respawn it.
       A task that already reached its OWN terminal result is not captured at
       all: natural completion wins, keeps its result AND its own event.
    2. OUTSIDE THE LOCK — kill and JOIN the worker. Process teardown must never
       hold the global queue lock (it blocks every admission and dispatch for
       the duration), and the death must be CONFIRMED, not assumed.
    3. Only after confirmed death AND a successful durable write does the task
       become publicly cancelled: terminal result, `task_done`, worker respawn,
       drive cleanup, snapshot. If either step fails, custody is RESTORED (the
       task goes back where it came from) and the outcome is ``failed`` — the
       caller must not report a cancellation that did not happen.
    """
    q = _queue_module()
    from supervisor import workers

    task_id = str(task_id or "").strip()
    if not task_id:
        return CANCEL_NOT_FOUND

    # ---- phase 1: capture under the lock -----------------------------------
    with q._queue_lock:
        settled = _durable_settled_status(q, task_id)
        if settled:
            # Natural completion (or an earlier cancel) already decided this task.
            # A QUEUED row for a task with a terminal result is a ghost and is
            # dropped; a RUNNING row is NOT touched — that worker is finishing its
            # own cycle and its finalizer owns the removal. Taking the row without
            # killing the process would orphan it, and killing it would destroy the
            # completion this branch exists to protect. The bounded re-sweeps give
            # the finalizer time; if it is still there at the end the postcondition
            # reports honestly instead of claiming a settled tree.
            for index, item in enumerate(list(q.PENDING)):
                if str(item.get("id")) == task_id:
                    q.PENDING.pop(index)
                    break
            return CANCEL_ALREADY_SETTLED

        captured_pending = None
        for index, item in enumerate(list(q.PENDING)):
            if str(item.get("id")) == task_id:
                captured_pending = q.PENDING.pop(index)
                break
        captured_worker = None
        captured_meta = None
        if captured_pending is None:
            for worker in workers.WORKERS.values():
                if worker.busy_task_id == task_id:
                    if getattr(worker, "reaping", False):
                        # The slot is ALREADY owned — by the reaper or another
                        # in-flight custody. Exactly one owner kills, publishes
                        # and respawns; a second taker would double-kill and
                        # double-respawn the slot. `failed` is honest here: the
                        # task is not settled yet, the caller's sweep retries,
                        # and the postcondition keeps refusing success until the
                        # real owner confirms death and persists the outcome.
                        return CANCEL_FAILED
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

    if captured_pending is not None:
        return _finish_captured_pending(task_id, captured_pending)
    if captured_worker is not None:
        return _finish_captured_running(task_id, captured_worker, captured_meta or {})
    return _finalize_cancel_requested_on_miss(task_id)


def _restore_custody(task_id: str, *, pending: Any = None, worker: Any = None) -> None:
    """Release custody after a failed cancellation.

    A captured PENDING task is put back in the queue. A RUNNING task needs no
    re-insert — capture never removed its row, so there is no ghost state to
    reconstruct; releasing the slot marker is the whole restore (a stranded
    ``reaping`` slot is skipped by assign and the health check forever).
    """
    q = _queue_module()
    with q._queue_lock:
        if pending is not None and all(str(t.get("id")) != task_id for t in q.PENDING):
            q.PENDING.append(pending)
        if worker is not None:
            worker.reaping = False


def _finish_captured_pending(task_id: str, task: Dict[str, Any]) -> str:
    """A queued task has no process: persist first, publish second."""
    q = _queue_module()
    from ouroboros.task_results import STATUS_CANCELLED, load_task_result, write_task_result

    try:
        existing = load_task_result(q.DRIVE_ROOT, task_id) or {}
        stored = write_task_result(
            q.DRIVE_ROOT, task_id, STATUS_CANCELLED,
            **q._cancel_result_fields(
                task, existing=existing, result="Task cancelled by user/agent request.",
            ),
        )
    except Exception:
        log.warning("Cancel persistence failed for pending task %s", task_id, exc_info=True)
        _restore_custody(task_id, pending=task)
        return CANCEL_FAILED
    if str((stored or {}).get("status") or "") != STATUS_CANCELLED:
        # The writer's monotonic guard refused it: the task settled on its own
        # between capture and write. Its outcome and event stand.
        return CANCEL_ALREADY_SETTLED
    q._emit_cancel_task_done(task, task_id)
    q.persist_queue_snapshot(reason="cancel_pending")
    return CANCEL_CANCELLED


def _finish_captured_running(task_id: str, worker: Any, meta: Dict[str, Any]) -> str:
    """A running task: CONFIRM the process is dead, persist, then publish."""
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
        return CANCEL_FAILED
    if worker.proc.is_alive():
        # A stubborn process is NOT a cancelled task: restoring custody keeps the
        # tree honest (still live, still owned by this worker) so the caller can
        # report a refusal instead of an imaginary success.
        log.error("Worker for %s survived kill escalation; cancellation refused", task_id)
        _restore_custody(task_id, worker=worker)
        return CANCEL_FAILED

    # POST-KILL child-drive re-check, the reaper's own step: forked/workspace/
    # subagent tasks self-finalize on the CHILD drive and are copied back only on
    # task_done. A worker that died after writing its child result but before
    # copy-back would otherwise lose to our cancelled write, and
    # copy_child_task_result later refuses to overwrite it. The process is dead
    # now, so this decision is final.
    try:
        from ouroboros.headless import copy_child_task_result
        from ouroboros.task_status import FINAL_STATUSES

        child_result = copy_child_task_result(pathlib.Path(q.DRIVE_ROOT), task)
        if child_result and str(child_result.get("status") or "") in FINAL_STATUSES:
            return _publish_cancelled_task(
                q, task_id, task, worker, child_result,
                {"cost_accounting_status": "unavailable", "cost_final": False},
            )
    except Exception:
        log.debug("Child-drive terminal re-check failed for %s", task_id, exc_info=True)

    # Cost reconstruction is EVIDENCE, not custody: a ledger read that fails must
    # degrade to unknown fields rather than strand a task whose worker is already
    # dead (supervisor/events.py::_authoritative_terminal_cost treats unavailable
    # accounting the same way).
    try:
        cost_fields = q.reconstruct_task_cost(
            str(task_id), fields=True, drive_root=pathlib.Path(q.DRIVE_ROOT),
        )
    except Exception:
        log.warning("Cost reconstruction failed for cancelled %s", task_id, exc_info=True)
        cost_fields = {"cost_accounting_status": "unavailable", "cost_final": False,
                       "cost_accounting_error": "ledger_unavailable", "cost_usd": None}
    # Rescue the partial result BEFORE the durable write — symmetrically with the
    # timeout kill (task_reaper), and for a stronger reason: publication below
    # DELETES a subagent's drive, so the observability blobs this reads are the
    # only copy of the work the cancelled task had already done (BIBLE P1). An
    # owner who cancels a task should not lose strictly more than a supervisor
    # timeout would.
    try:
        from ouroboros.observability import salvaged_output_note
        salvage_note = salvaged_output_note(
            q._task_drive_for_task(task, str(task_id)), str(task_id),
            # The full copy must land on the drive that OUTLIVES the child drive
            # this publication deletes (BIBLE P1): a bounded preview alone is not
            # a rescue when it is about to become the only copy.
            preserve_root=pathlib.Path(q.DRIVE_ROOT),
        )
    except Exception:
        log.debug("Failed to salvage last LLM response for cancelled %s", task_id, exc_info=True)
        salvage_note = ""
    try:
        existing = load_task_result(q.DRIVE_ROOT, task_id) or {}
        stored = write_task_result(
            q.DRIVE_ROOT, task_id, STATUS_CANCELLED,
            **q._cancel_result_fields(
                task, existing=existing, **cost_fields,
                result="Running task cancelled and worker terminated." + salvage_note,
            ),
        )
    except Exception:
        log.warning("Cancel persistence failed for running task %s", task_id, exc_info=True)
        _restore_custody(task_id, worker=worker)
        return CANCEL_FAILED

    # ---- DURABLE BOUNDARY CROSSED -----------------------------------------
    # The task's terminal truth is on disk. Everything past this line is
    # publication and slot hygiene: it is FAIL-SOFT and idempotent, because
    # answering 503 now would report a cancellation that demonstrably happened,
    # and a raising respawn must never leave the slot stranded at `reaping`.
    return _publish_cancelled_task(q, task_id, task, worker, stored, cost_fields)


def _publish_cancelled_task(
    q: Any, task_id: str, task: Dict[str, Any], worker: Any,
    stored: Dict[str, Any], cost_fields: Dict[str, Any],
) -> str:
    """Publish the STORED terminal truth and reconcile the worker slot.

    The event carries whatever actually settled: if the worker wrote its own
    natural result before we killed it, the monotonic guard refused our
    cancellation — publishing nothing would leave that card unresolved until a
    reload, so the stored status is emitted instead.
    """
    from supervisor import workers

    from ouroboros.task_results import STATUS_CANCELLED

    settled_status = str((stored or {}).get("status") or STATUS_CANCELLED)
    # The row leaves RUNNING only NOW — death confirmed, terminal result durable.
    # A natural task_done that raced us may have consumed the row already; that
    # handler also emitted the terminal event, so whoever pops the row owns the
    # emit and the card resolves exactly once.
    with q._queue_lock:
        row_owned = q.RUNNING.pop(task_id, None) is not None
    try:
        from ouroboros.tools.services import archive_task_service_logs
        archive_task_service_logs(pathlib.Path(q.DRIVE_ROOT), str(task_id), task)
    except Exception:
        log.debug("Failed to archive service logs for cancelled task %s", task_id, exc_info=True)
    if row_owned:
        try:
            q._emit_cancel_task_done(task, task_id, cost_fields=cost_fields, status=settled_status)
        except Exception:
            log.warning("Failed to publish terminal event for %s", task_id, exc_info=True)
    # Respawn recovery is the REAPER'S canonical step 5, not a private variant.
    # The helper serializes against shutdown with the lifecycle lock and starts
    # the child outside the queue lock; on failure the marker is cleared under
    # the lock so the crash detector can recover the slot on a later tick.
    try:
        workers.respawn_worker(worker.wid)
    except Exception:
        log.warning("Respawn after cancelling %s failed; clearing reaping for recovery", task_id, exc_info=True)
        try:
            with q._queue_lock:
                slot = workers.WORKERS.get(worker.wid)
                if slot is not None:
                    slot.reaping = False
        except Exception:
            log.debug("Could not clear the slot marker for %s", task_id, exc_info=True)
    if str(task.get("delegation_role") or "") == "subagent":
        try:
            from ouroboros.headless import remove_subagent_task_drive
            remove_subagent_task_drive(q.DRIVE_ROOT, str(task_id))
        except Exception:
            log.debug("Failed to remove cancelled subagent drive for %s", task_id, exc_info=True)
    try:
        q.persist_queue_snapshot(reason="cancel_running")
    except Exception:
        log.debug("Snapshot after cancelling %s failed", task_id, exc_info=True)
    return CANCEL_ALREADY_SETTLED if settled_status != STATUS_CANCELLED else CANCEL_CANCELLED


def _finalize_cancel_requested_on_miss(task_id: str) -> str:
    """Neither queued nor running: promote a pending cancel latch to cancelled."""
    q = _queue_module()
    from ouroboros.task_results import (
        STATUS_CANCEL_REQUESTED, STATUS_CANCELLED, load_task_result, write_task_result,
    )

    try:
        existing = load_task_result(q.DRIVE_ROOT, task_id) or {}
        if str(existing.get("status") or "") != STATUS_CANCEL_REQUESTED:
            return CANCEL_NOT_FOUND
        write_task_result(
            q.DRIVE_ROOT, task_id, STATUS_CANCELLED,
            **q._cancel_result_fields(
                existing, existing=existing,
                result="Task cancelled (finished before supervisor teardown).",
            ),
        )
        q._emit_cancel_task_done(existing, task_id)
        q.persist_queue_snapshot(reason="cancel_finalize")
        return CANCEL_CANCELLED
    except Exception:
        log.debug("Cancel finalize-on-miss failed for %s", task_id, exc_info=True)
        return CANCEL_FAILED



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
    outcomes: Dict[str, str] = {}
    cancelled = False
    for live_id in cancel_order:
        # Children first (cancel_order is depth-sorted), each with its own typed
        # result — the parent's success can never mask a child's refusal.
        outcomes[live_id] = q.cancel_task_custody(live_id)
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


def resume_budget_paused_task(task_id: str) -> Dict[str, Any]:
    """Explicitly resume one zero-dispatch task and, if needed, its root latch."""
    q = _queue_module()
    task_id = str(task_id or "").strip()
    if not task_id:
        return {"ok": False, "error": "missing_task_id"}
    with q._queue_lock:
        task = next((item for item in q.PENDING if str(item.get("id") or "") == task_id), None)
        if task is None:
            return {"ok": False, "error": "task_not_pending"}
        pause = task.get("_budget_pause") if isinstance(task.get("_budget_pause"), dict) else None
        if not pause:
            # A root marker blocks every already-pending sibling without
            # copying pause state onto each task.  An explicit resume request
            # may nominate any genuinely zero-dispatch member of that root.
            candidate_root = str(task.get("root_task_id") or task_id).strip()
            candidate_fence = q.BUDGET_ROOT_FENCES.get(candidate_root)
            if not isinstance(candidate_fence, dict):
                return {"ok": False, "error": "task_not_budget_paused"}
            pause = {
                **candidate_fence,
                "status": "paused_before_dispatch",
                "physical_calls": 0,
                "replay_safe": True,
                "resume_policy": "manual_same_generation",
            }
        root_scope = str(pause.get("scope") or "") == "root"
        root_task_id = str(pause.get("root_task_id") or "").strip()
        fence = q.BUDGET_ROOT_FENCES.get(root_task_id) if root_scope and root_task_id else None
        if root_scope and not isinstance(fence, dict):
            return {"ok": False, "error": "root_budget_fence_missing", "action": "cancel_or_new_run"}
        if root_scope and str(pause.get("fence_id") or "") != str(fence.get("fence_id") or ""):
            return {"ok": False, "error": "replay_unsafe", "action": "cancel_or_new_run"}
        def _pending_member_is_replay_safe(member: Dict[str, Any]) -> tuple[bool, str]:
            member_id = str(member.get("id") or "")
            cost_fields = q.reconstruct_task_cost(
                member_id,
                fields=True,
                drive_root=pathlib.Path(member.get("budget_drive_root") or q.DRIVE_ROOT),
            )
            if cost_fields.get("cost_accounting_status") != "available":
                return False, "accounting_unavailable"
            retry_lineage = bool(
                int(member.get("_attempt") or 1) > 1
                or member.get("original_task_id") or member.get("timeout_retry_from")
            )
            return bool(
                int(cost_fields.get("total_rounds") or 0) == 0
                and not bool(cost_fields.get("ledger_integrity_degraded"))
                and not retry_lineage
            ), "replay_unsafe"

        nominated_safe, nominated_error = _pending_member_is_replay_safe(task)
        nominated_safe = bool(
            nominated_safe
            and pause.get("replay_safe")
            and pause.get("physical_calls") == 0
        )
        if not nominated_safe:
            return {
                "ok": False,
                "error": nominated_error,
                "action": "cancel_or_new_run",
            }
        if root_scope:
            # Clearing one root latch makes every pending member assignable. Check
            # those members together under the existing queue lock; completed
            # historical siblings are deliberately irrelevant.
            unsafe_members: list[str] = []
            for member in q.PENDING:
                member_id = str(member.get("id") or "")
                member_root = str(member.get("root_task_id") or member_id)
                if member_root != root_task_id or member_id == task_id:
                    continue
                member_safe, _member_error = _pending_member_is_replay_safe(member)
                if not member_safe:
                    unsafe_members.append(member_id)
            if unsafe_members:
                return {
                    "ok": False,
                    "error": "root_replay_unsafe",
                    "unsafe_task_ids": unsafe_members,
                    "action": "cancel_or_new_run",
                }

        resumed_at = utc_now_iso()
        prior_pause = dict(pause)
        task.pop("_budget_pause", None)
        task["budget_resumed_at"] = resumed_at
        if root_scope:
            q.BUDGET_ROOT_FENCES.pop(root_task_id, None)
        q.persist_queue_snapshot(
            reason="budget_root_explicit_resume" if root_scope else "budget_pause_explicit_resume",
        )
    try:
        from ouroboros.task_results import STATUS_SCHEDULED, write_task_result

        write_task_result(
            pathlib.Path(task.get("budget_drive_root") or q.DRIVE_ROOT),
            task_id,
            STATUS_SCHEDULED,
            reason_code="",
            resource_limit={
                **prior_pause,
                "status": "resumed",
                "resumed_at": resumed_at,
                "auto_resume": False,
            },
        )
    except Exception:
        q.log.debug("Failed to project explicit budget resume for %s", task_id, exc_info=True)
    q.append_jsonl(
        q.DRIVE_ROOT / "logs" / "events.jsonl",
        {
            "ts": utc_now_iso(),
            "type": "budget_task_explicitly_resumed",
            "task_id": task_id,
            "root_task_id": root_task_id if root_scope else "",
            "same_generation": True,
        },
    )
    return {"ok": True, "task_id": task_id, "same_generation": True}


def _live_project_task_ids(drive_root: object, project_id: str) -> list[str]:
    """Snapshot queued/running tasks associated with one fenced Project."""
    from ouroboros.projects_registry import project_task_bindings

    q = _queue_module()
    with q._queue_lock:
        rows = [dict(task) for task in q.PENDING if isinstance(task, dict)]
        rows.extend(
            dict(meta.get("task"))
            for meta in q.RUNNING.values()
            if isinstance(meta, dict) and isinstance(meta.get("task"), dict)
        )
    bindings = project_task_bindings(drive_root)
    associated: set[str] = set()
    by_id: dict[str, dict] = {}
    for task in rows:
        task_id = str(task.get("id") or task.get("task_id") or "").strip()
        if not task_id:
            continue
        by_id[task_id] = task
        lineage = (task_id, str(task.get("parent_task_id") or ""), str(task.get("root_task_id") or ""))
        if str(task.get("project_id") or "") == project_id or any(
            isinstance(bindings.get(candidate), dict)
            and str(bindings[candidate].get("project_id") or "") == project_id
            for candidate in lineage
            if candidate
        ):
            associated.add(task_id)
    changed = True
    while changed:
        changed = False
        for task_id, task in by_id.items():
            if task_id in associated:
                continue
            if (
                str(task.get("parent_task_id") or "") in associated
                or str(task.get("root_task_id") or "") in associated
            ):
                associated.add(task_id)
                changed = True
    return sorted(
        associated,
        key=lambda task_id: bool(str(by_id.get(task_id, {}).get("parent_task_id") or "")),
        reverse=True,
    )


def _broadcast_projects_changed(project_id: str, chat_id: Any) -> None:
    try:
        from supervisor.message_bus import get_bridge

        get_bridge().broadcast({"type": "projects_changed", "project_id": project_id, "chat_id": chat_id})
    except Exception:
        _queue_module().log.debug("projects_changed broadcast failed for %s", project_id, exc_info=True)


def _sweep_project_checkouts(drive_root: object, project_id: str) -> str:
    """Take any thread checkout that outlived the delete route's own attempt.

    Best-effort and never raising: a checkout that still cannot be removed is not
    a reason to fail the deletion — the project row's own teardown has already
    succeeded by here. But it must not VANISH either, so this returns a disclosure
    sentence ("" when nothing was left behind) that the caller writes onto the
    tombstoned row and sends to the owner. A ``log.warning`` was the whole of the
    previous disclosure and it reaches no owner surface at all.

    The sweep no longer acknowledges unmerged work on the owner's behalf (see
    ``thread_worktrees.remove_project_thread_worktrees``), so a checkout that
    became at-risk AFTER the owner's pre-fence look now survives here deliberately
    — which is precisely why the survivors have to be disclosed.
    """
    try:
        from ouroboros.thread_worktrees import remove_project_thread_worktrees

        swept = remove_project_thread_worktrees(drive_root, project_id)
        if not swept["kept"]:
            return ""
        _queue_module().log.warning(
            "Project %s tombstoned with %d thread checkout(s) still on disk: %s",
            project_id, len(swept["kept"]), swept["kept"],
        )
        return _orphaned_checkouts_note(swept["kept"])
    except Exception as exc:
        _queue_module().log.warning(
            "Project %s: thread checkout sweep failed", project_id, exc_info=True,
        )
        return truncate_review_artifact(
            "the thread checkouts could not be swept "
            f"({type(exc).__name__}: {exc}), so any that exist are still on disk.",
            limit=ORPHANED_NOTE_LIMIT,
        )


def _orphaned_checkouts_note(kept: List[Dict[str, Any]]) -> str:
    """What was left behind, and WHERE — the sentence the owner is owed.

    Names each folder and branch rather than counting them: the project is about to
    become invisible on every surface, so this text is the only thing that can
    still point the owner at work only they can reach now.

    Which is exactly why it may not be cut silently (BIBLE P1, DEVELOPMENT.md "No
    silent truncation"). A flat ``[:2000]`` over the joined lines dropped whole
    checkouts out of the only record that names them — reproduced at 30 survivors,
    where 13 vanished and the sentence ended mid-word — with no marker saying so.
    A disclosure about work the owner can no longer reach through the app is the
    last text that may lose part of itself in silence. So the bound is taken at an
    ENTRY boundary, never mid-entry, and whatever does not fit is DECLARED: the
    count, the thread ids (short, and the key to finding the folder), and where the
    unabridged list is. ``log.warning`` in the caller writes that full list, and
    says so here rather than being a record nobody knows exists.
    """
    lines = []
    for item in kept:
        path = str(item.get("path") or "").strip() or "(path unknown)"
        branch = str(item.get("branch") or "").strip()
        reason = str(item.get("reason") or "removal_failed")
        lines.append((
            item.get("thread_id"),
            f"thread {item.get('thread_id')}: {path}"
            + (f" (branch {branch})" if branch else "")
            + f" — {reason}",
        ))
    count = len(lines)
    head = (
        f"{count} thread checkout{'' if count == 1 else 's'} could not be removed and "
        f"{'is' if count == 1 else 'are'} still on disk; the project is deleted, so "
        "nothing in the app can reach them any more. "
    )
    named: List[str] = []
    omitted: List[str] = []
    used = len(head)
    for thread_id, line in lines:
        if named and used + len(line) + 2 > ORPHANED_NOTE_LIMIT:
            omitted.append(str(thread_id))
            continue
        named.append(line)
        used += len(line) + 2
    text = head + "; ".join(named)
    if omitted:
        text += (
            f". {len(omitted)} more are still on disk and NOT named above "
            f"(thread{'' if len(omitted) == 1 else 's'} {', '.join(omitted)}); "
            "their full paths and branches are in the app log, on the "
            "\"tombstoned with … thread checkout(s) still on disk\" warning."
        )
    return text


def _tell_owner_about_orphaned_checkouts(project_id: str, note: str) -> None:
    """Send the disclosure to the one surface a tombstoned project still has.

    A tombstoned row is filtered out of ``list_sidebar_projects``, so writing the
    note onto the row makes it durable and checkable but shows it to nobody. The
    owner's chat is where every other background lifecycle reports itself
    (``evolution_lifecycle.notify_owner_cycle_outcome`` is the precedent) and this
    is the same class of fact: a teardown that finished in a way the owner has to
    know about. Fully guarded — a disclosure that cannot be SENT must not take the
    deletion down with it, and the note is on the row either way.
    """
    try:
        from supervisor.message_bus import send_with_budget
        from supervisor.state import load_state

        owner_chat_id = int(load_state().get("owner_chat_id") or 0)
        if not owner_chat_id:
            return
        send_with_budget(owner_chat_id, f"⚠️ Project “{project_id}” was deleted, but {note}")
    except Exception:
        _queue_module().log.warning(
            "Project %s: could not tell the owner about the checkouts left behind: %s",
            project_id, note, exc_info=True,
        )


def run_project_deletion(
    drive_root: object,
    project_id: str,
    chat_id: Any,
    worker_key: tuple[str, str] | None = None,
) -> None:
    """Cancel a fenced Project tree and tombstone only after quiescence.

    The LAST act before the tombstone is taking any thread checkout the route
    could not take yet (I1): ``api_project_delete`` refuses on work at risk and
    removes what it can, but a checkout with a task still writing in it refuses
    ``project_busy`` there and would otherwise be left behind — a folder and a
    ``thread/*`` branch that a tombstoned project has no surface to reach. Here the
    tasks are gone, so the removal's own busy judge lets it through.

    What it does NOT do is acknowledge unmerged work for the owner. The pre-fence
    inspection is a fact about a moment that has passed by the time this runs: the
    task that made the route refuse ``project_busy`` went on to commit work and edit
    tracked files in that checkout, and the sweep's hardcoded acknowledgement
    destroyed both (P1). Each row is re-inspected by the same judge the route used,
    so a newly at-risk checkout SURVIVES — and then rides the tombstone as a
    disclosure, naming the folder and the branch, plus a chat note to the owner. The
    project is still tombstoned: keeping it alive because a folder survived was
    rejected as owner direction (§I M2), so the answer is disclosure, never silence.
    """
    from ouroboros.projects_registry import complete_project_deletion, fail_project_deletion

    q = _queue_module()

    def _finish() -> None:
        # The sweep's own answer rides the tombstone. A checkout it could not take
        # is now KEPT rather than force-removed (the sweep stopped acknowledging
        # unmerged work on the owner's behalf), so "what was left behind and where"
        # is a fact the tombstone has to carry — and the owner has to be told,
        # because a tombstoned project is on no surface that could show them.
        left_behind = _sweep_project_checkouts(drive_root, project_id)
        complete_project_deletion(drive_root, project_id, delete_error=left_behind)
        _broadcast_projects_changed(project_id, chat_id)
        if left_behind:
            _tell_owner_about_orphaned_checkouts(project_id, left_behind)

    try:
        while True:
            live_ids = _live_project_task_ids(drive_root, project_id)
            if not live_ids:
                _finish()
                return
            errors: list[str] = []
            for task_id in live_ids:
                try:
                    q.cancel_task_by_id(task_id, cascade=True)
                except Exception as exc:
                    errors.append(f"{task_id}: {type(exc).__name__}: {exc}")
            remaining = _live_project_task_ids(drive_root, project_id)
            if not remaining:
                _finish()
                return
            if set(remaining) >= set(live_ids):
                detail = "; ".join(errors) if errors else "cancel_task_by_id left tasks live"
                raise RuntimeError(f"Project deletion did not quiesce ({', '.join(remaining)}): {detail}")
            # It SHRANK, so cancellation is working — give the rest a turn to
            # settle instead of spinning against the lock they need.
            time.sleep(_QUIESCE_SETTLE_SEC)
    except Exception as exc:
        q.log.exception("Project deletion failed for %s", project_id)
        fail_project_deletion(drive_root, project_id, f"{type(exc).__name__}: {exc}")
        _broadcast_projects_changed(project_id, chat_id)
    finally:
        if worker_key is not None:
            with _PROJECT_DELETE_WORKERS_LOCK:
                _PROJECT_DELETE_WORKERS.discard(worker_key)


def start_project_deletion(drive_root: object, project_id: str, chat_id: Any) -> bool:
    """Start one cancellation worker per Project and server generation."""
    key = (str(drive_root), str(project_id))
    with _PROJECT_DELETE_WORKERS_LOCK:
        if key in _PROJECT_DELETE_WORKERS:
            return False
        _PROJECT_DELETE_WORKERS.add(key)
    threading.Thread(
        target=run_project_deletion,
        args=(drive_root, project_id, chat_id, key),
        name=f"project-delete-{project_id}",
        daemon=True,
    ).start()
    return True


def resume_project_deletions(drive_root: object) -> int:
    """Resume interrupted deletion workers from durable registry state."""
    from ouroboros.projects_registry import PROJECT_DELETING, list_sidebar_projects

    started = 0
    for project in list_sidebar_projects(drive_root):
        if str(project.get("lifecycle") or "") != PROJECT_DELETING:
            continue
        started += int(start_project_deletion(
            drive_root,
            str(project.get("id") or ""),
            project.get("chat_id"),
        ))
    return started


# --------------------------------------------------------------------------- #
# Thread deletion (X10): fence -> cancel/quiesce by EXACT thread chat id ->
# tombstone. The project pattern one level down, deliberately reusing its shape
# rather than inventing a second teardown with its own bugs.
# --------------------------------------------------------------------------- #

_THREAD_DELETE_WORKERS_LOCK = threading.Lock()
_THREAD_DELETE_WORKERS: set[tuple[str, str, int]] = set()


def _live_thread_task_ids(chat_id: int) -> list[str]:
    """Queued/running tasks belonging to ONE thread, plus their subtrees.

    Selection is by EXACT thread chat id, never by project (X10): a project can
    hold several threads, and cancelling the project's whole queue because one
    thread was deleted would kill work the owner never touched. Children are
    pulled in by lineage rather than by chat id, because a subagent inherits its
    parent's tree but not necessarily its chat.

    Children are returned FIRST so a cascade cancels from the leaves up, matching
    the project path's ordering.
    """
    q = _queue_module()
    with q._queue_lock:
        rows = [dict(task) for task in q.PENDING if isinstance(task, dict)]
        rows.extend(
            dict(meta.get("task"))
            for meta in q.RUNNING.values()
            if isinstance(meta, dict) and isinstance(meta.get("task"), dict)
        )
    by_id: dict[str, dict] = {}
    associated: set[str] = set()
    for task in rows:
        task_id = str(task.get("id") or task.get("task_id") or "").strip()
        if not task_id:
            continue
        by_id[task_id] = task
        try:
            task_chat = int(task.get("chat_id") or 0)
        except (TypeError, ValueError):
            task_chat = 0
        # A falsy chat id is "this task has no room", not "this task belongs to
        # room 0": without the guard, `_live_thread_task_ids(0)` selected every
        # chat-less task in the queue — every headless subagent — and cancelled
        # them. Unreachable today because thread #0 is the project and
        # `begin_thread_deletion` refuses it, but the selection is one caller away
        # from being wrong about the owner's whole swarm.
        if task_chat and task_chat == int(chat_id):
            associated.add(task_id)
    changed = True
    while changed:
        changed = False
        for task_id, task in by_id.items():
            if task_id in associated:
                continue
            if (
                str(task.get("parent_task_id") or "") in associated
                or str(task.get("root_task_id") or "") in associated
            ):
                associated.add(task_id)
                changed = True
    return sorted(
        associated,
        key=lambda task_id: bool(str(by_id.get(task_id, {}).get("parent_task_id") or "")),
        reverse=True,
    )


def run_thread_deletion(
    drive_root: object,
    project_id: str,
    thread_id: int,
    chat_id: int,
    worker_key: tuple | None = None,
) -> None:
    """Cancel a fenced thread's tasks and tombstone only after quiescence.

    The thread is already fenced against routing by the time this runs, so no new
    work can enter while it drains. A tree that refuses to quiesce records its
    reason on the row and stays fenced — never rolled back to active, because the
    owner asked for this thread to go and silently re-opening it would put
    messages back into a room they had written off.
    """
    from ouroboros.projects_registry import complete_thread_deletion, fail_thread_deletion

    q = _queue_module()
    try:
        while True:
            live_ids = _live_thread_task_ids(chat_id)
            if not live_ids:
                complete_thread_deletion(drive_root, project_id, thread_id)
                _broadcast_projects_changed(project_id, chat_id)
                return
            errors: list[str] = []
            for task_id in live_ids:
                try:
                    q.cancel_task_by_id(task_id, cascade=True)
                except Exception as exc:
                    errors.append(f"{task_id}: {type(exc).__name__}: {exc}")
            remaining = _live_thread_task_ids(chat_id)
            if not remaining:
                complete_thread_deletion(drive_root, project_id, thread_id)
                _broadcast_projects_changed(project_id, chat_id)
                return
            if set(remaining) >= set(live_ids):
                detail = "; ".join(errors) if errors else "cancel_task_by_id left tasks live"
                raise RuntimeError(
                    f"thread deletion did not quiesce ({', '.join(remaining)}): {detail}"
                )
            # It SHRANK, so cancellation is working — give the rest a turn to
            # settle instead of spinning against the lock they need.
            time.sleep(_QUIESCE_SETTLE_SEC)
    except Exception as exc:
        q.log.exception("Thread deletion failed for %s#%s", project_id, thread_id)
        try:
            # Recording WHY must not itself raise out of a daemon thread. The
            # project row can have moved between the fence and here (its own
            # deletion started), and losing this note is better than losing the
            # traceback that explains what actually went wrong.
            fail_thread_deletion(drive_root, project_id, thread_id, f"{type(exc).__name__}: {exc}")
        except Exception:
            q.log.debug("Could not record the thread-deletion failure", exc_info=True)
        _broadcast_projects_changed(project_id, chat_id)
    finally:
        if worker_key is not None:
            with _THREAD_DELETE_WORKERS_LOCK:
                _THREAD_DELETE_WORKERS.discard(worker_key)


def start_thread_deletion(drive_root: object, project_id: str, thread_id: int, chat_id: int) -> bool:
    """Start one cancellation worker per thread and server generation."""
    key = (str(drive_root), str(project_id), int(thread_id))
    with _THREAD_DELETE_WORKERS_LOCK:
        if key in _THREAD_DELETE_WORKERS:
            return False
        _THREAD_DELETE_WORKERS.add(key)
    threading.Thread(
        target=run_thread_deletion,
        args=(drive_root, project_id, thread_id, chat_id, key),
        name=f"thread-delete-{project_id}-{thread_id}",
        daemon=True,
    ).start()
    return True


def resume_thread_deletions(drive_root: object) -> int:
    """Resume interrupted THREAD deletions from durable registry state.

    Same reason the project version exists: a restart between the fence and the
    tombstone would otherwise leave a thread fenced forever — invisible to
    routing, still on screen, never finishing.

    Thread #0 is SKIPPED: it is the project, its synthesized lifecycle mirrors the
    project's, and `list_sidebar_projects` includes deleting projects. So a
    project mid-deletion produced a bogus thread-deletion worker for its own
    thread #0 that cancelled the project's whole tree in parallel with
    `resume_project_deletions` and then raised `thread_zero_is_the_project`,
    logging a traceback on every restart. Thread #0 belongs to the project path.
    """
    from ouroboros.projects_registry import (
        MAIN_THREAD_ID,
        THREAD_DELETING,
        list_sidebar_projects,
        project_threads,
    )

    started = 0
    for project in list_sidebar_projects(drive_root):
        for thread in project_threads(project):
            if int(thread.get("id") or 0) == MAIN_THREAD_ID:
                continue
            if str(thread.get("lifecycle") or "") != THREAD_DELETING:
                continue
            started += int(start_thread_deletion(
                drive_root,
                str(project.get("id") or ""),
                int(thread.get("id") or 0),
                int(thread.get("chat_id") or 0),
            ))
    return started
