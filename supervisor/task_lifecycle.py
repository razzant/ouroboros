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
from ouroboros.workspace_admission import PLACEMENT_FENCE_KEY, placement_fence_stale_reason
from ouroboros.workspace_ref import isolate_sealed_placement

# Cancellation settlement PUBLICATION (typed outcome vocabulary, cancelled
# result fields, owed-before-settle registration, stored-truth publication,
# miss-lane delivery) lives in supervisor.cancel_publication — a module-size
# code boundary exactly like supervisor.queue_transitions. Imported here so
# this module keeps ONE public surface for callers, tests, and the
# supervisor.queue re-exports.
from supervisor.cancel_custody_ledger import (  # noqa: F401 -- re-exported
    _queue_module,
    _active_intent,
    _reaping_owner_abandoned,
    _recover_stranded_reaping_slot,
    _claim_intent,
    _settle_intent,
    _release_intent_claim,
    _intent_outcome_fields,
    _restore_custody,
)
from supervisor.cancel_publication import (  # noqa: F401 -- intentional public re-exports
    CANCEL_ALREADY_SETTLED,
    CANCEL_CANCELLED,
    CANCEL_FAILED,
    CANCEL_NOT_FOUND,
    _CANCEL_TERMINALIZED,
    _cancel_result_fields,
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

log = logging.getLogger(__name__)


BUDGET_ROOT_FENCES: Dict[str, Dict[str, Any]] = {}


# Tasks whose subtree cancellation has begun (v6.82.0): descendant admission is
# FENCED under the queue lock, because a schedule event already draining would be
# admitted after any number of cascade sweeps. Bounded in memory (newest kept) —
# a cancelled tree is terminal, so an evicted entry names a tree that settled long
# ago; the registry is per-process by design (a restart has no live descendants to
# admit, and terminal task results are the durable truth).
CANCELLED_ROOT_FENCES: Dict[str, str] = {}


# Ids fenced by cascades that are STILL RUNNING. Pruning protects the union, not
# just the caller's own set: a second large cascade could otherwise push the cap
# over and evict an older in-flight cascade's fences, re-opening admission into a
# tree that is still being torn down. Add-only within a cascade; the whole entry
# is dropped when that cascade returns.
_ACTIVE_CASCADE_FENCES: Dict[str, set[str]] = {}


_CANCELLED_ROOT_FENCE_CAP = 4096


_CANCELLED_ROOT_FENCE_GRACE_SEC = 300.0


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


def apply_late_admission_fences(task: Dict[str, Any]) -> str:
    """The checks that must run at the LAST moment before a task becomes runnable.

    ``enqueue_task`` calls this under ``_queue_lock``, after the identity/project/budget/
    acceptance fences and immediately before the append. Two things can only be judged
    here, because both concern events that happen DURING admission:

    1. **Cancellation owns an in-flight admission.** A creation surface resolves the
       placement, writes the durable SCHEDULED record, and only then enqueues. If the
       owner cancels inside that window the durable record is cancel-latched, and a task
       that appeared in the queue after being cancelled would be the worst kind of ghost:
       one the owner believes is gone. The durable record is the lifecycle authority
       (``task_results`` keeps it monotonic and cancel-sticky), so it decides.
    2. **The placement fence** (RWSB2-05): a schedule resolves its placement at fire
       time, and the owner can delete the project, rebind it, or retire the connection
       before insertion. A stale placement is REFUSED, never re-resolved — re-resolving
       would let a rebind retroactively move work scheduled against the previous target.

    Finally the sealed placement payload is ISOLATED from the caller's dict, so a caller
    that keeps mutating the task it handed over cannot rewrite the placement of a task
    that is already queued.

    Returns the typed reason (also latched onto the task) or "".
    """
    q = _queue_module()
    task_id = str(task.get("id") or "").strip()
    if task_id:
        from ouroboros.task_results import (
            STATUS_CANCEL_REQUESTED,
            _TRULY_TERMINAL_STATUSES,
            load_task_result,
        )

        try:
            durable = str((load_task_result(q.DRIVE_ROOT, task_id) or {}).get("status") or "")
        except Exception:
            durable = ""
        if durable == STATUS_CANCEL_REQUESTED or durable in _TRULY_TERMINAL_STATUSES:
            task["_admission_blocked"] = "task_cancelled_during_admission"
            task["_durable_status"] = durable
            return "task_cancelled_during_admission"
    fence = task.get(PLACEMENT_FENCE_KEY)
    if isinstance(fence, dict):
        stale = placement_fence_stale_reason(q.DRIVE_ROOT, fence)
        if stale:
            task["_admission_blocked"] = stale
            task["_project_id"] = str(fence.get("project_id") or "")
            return stale
    isolate_sealed_placement(task)
    return ""


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


















def _release_remote_task_session(task: Dict[str, Any], *, cancelled: bool) -> None:
    """Release a cancelled task's remote binding, fail-soft.

    A CANCEL reaches the target (killing the task's process groups); an ordinary
    terminal release only drops the lease and the Home import staging. Conflating
    them is how a cancelled task keeps working abroad. Swallowing is correct for
    the same reason as `_signal_remote_admission_cancel`: a local task has no
    binding at all, and a cancel must not fail because this process has no broker.
    """
    if not isinstance(task, dict) or not task:
        return
    try:
        from ouroboros.remote_task_binding import release_remote_task_session

        release_remote_task_session(task, cancelled=cancelled)
    except Exception:
        log.debug("Remote session release failed for %s", task.get("id"), exc_info=True)


def _signal_remote_admission_cancel(task_id: str) -> bool:
    """Tell the broker to abandon an admission that is already in flight.

    Best effort ON PURPOSE, and swallowing is the correct behaviour here: a cancel
    must never fail because no broker is registered in this process (the common
    case — the task was never remote), and it must never block, because the owner
    asked for a stop. The durable latch is what GUARANTEES the task cannot appear
    in the queue afterwards; this is what makes the stop reach the target too.
    """

    try:
        from ouroboros.remote_workspace import get_remote_workspace_service

        service = get_remote_workspace_service()
    except Exception:
        return False
    cancel = getattr(service, "cancel_admission", None)
    if not callable(cancel):
        return False
    try:
        return bool(cancel(task_id))
    except Exception:
        return False




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
            from supervisor.terminal_delivery import deliver_cascade_summary

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


def _cascade_delivery_row_locked(q: Any, task_id: str) -> Dict[str, Any]:
    """A routing row for a cascade whose ROOT has already left the live maps.

    Caller holds the queue lock. Returns the first live descendant's row (they
    carry the lineage ``chat_id``), or ``{}`` when the subtree is empty too.
    """
    for task in q.PENDING:
        if isinstance(task, dict) and q._is_descendant_of(task, task_id) and task.get("chat_id"):
            return dict(task)
    for meta in q.RUNNING.values():
        task = meta.get("task") if isinstance(meta, dict) else None
        if isinstance(task, dict) and q._is_descendant_of(task, task_id) and task.get("chat_id"):
            return dict(task)
    return {}


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
    # A pending remote task may already be BOUND (assignment binds before
    # dispatch), so its session goes with it or the connection reads as busy
    # forever.
    _release_remote_task_session(task, cancelled=True)
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

    # Reach the TARGET before Home's own teardown: killing the worker only stops
    # Home, and a cancelled remote task whose process groups keep running on
    # another host is the emergency-stop invariant read too narrowly.
    _release_remote_task_session(task, cancelled=True)

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
        STATUS_CANCEL_REQUESTED, STATUS_CANCELLED, STATUS_SCHEDULED,
        load_task_result, write_task_result,
    )

    try:
        active = dict(intent or {})
        if not active:
            active = _active_intent(q, task_id)
        existing = load_task_result(q.DRIVE_ROOT, task_id) or {}
        durable = str(existing.get("status") or "")
        legacy_latch = durable == STATUS_CANCEL_REQUESTED
        if durable in {STATUS_SCHEDULED, STATUS_CANCEL_REQUESTED}:
            # For a REMOTE placement the durable latch alone is not enough: the
            # admission may already be talking to the target, so the broker's
            # in-flight admission is signalled too and the target kills whatever
            # the task already started. Otherwise the owner's cancel would stop
            # Home while remote work continued — the emergency-stop invariant
            # read narrowly.
            _signal_remote_admission_cancel(task_id)
        if durable == STATUS_SCHEDULED:
            write_task_result(
                q.DRIVE_ROOT, task_id, STATUS_CANCEL_REQUESTED,
                **q._cancel_result_fields(
                    existing, existing=existing,
                    result="Task cancelled while its admission was still in flight.",
                ),
            )
            q.persist_queue_snapshot(reason="cancel_admission_window")
            return CANCEL_CANCELLED
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
    _live_project_task_ids,
    clear_acceptance_fence_for_root,
    resume_budget_paused_task,
    resume_project_deletions,
    run_project_deletion,
    start_project_deletion,
    task_has_live_ownership,
    transition_acceptance_fence,
)
