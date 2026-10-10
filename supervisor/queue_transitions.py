"""Queue-owned lifecycle TRANSITIONS that are not cancellation custody.

Extracted from ``supervisor/task_lifecycle.py`` for the module-size gate (the
same boundary that produced ``terminal_delivery.py`` and
``delegate_containment.py``): custody grew when the Poltergeist cancel redesign
landed, and the clusters here never touch it. They share one property — each
is a queue-owned transition or read-side view of a task's ADMISSION or
QUIESCENCE state, driven entirely through the queue module:

- the acceptance FENCE (open/inspect/seal a root so no new subtask is admitted
  while its acceptance review runs),
- explicit BUDGET resume of a zero-dispatch task and its root latch (the
  exact mid-run continuation's single-use GRANT and its revocation live in
  ``supervisor/budget_resume.py`` and are re-exported here),
- fenced PROJECT deletion: cancel the project's tree, then tombstone only after
  the tree is provably quiescent,
- shared live-subtree and timeout-retry lineage views used by cancellation and
  admission without creating a second queue authority.

The dependency runs one way: this module reaches the queue lazily and imports
nothing from ``task_lifecycle``, which re-exports these names so
``supervisor.queue`` stays the single public import surface for callers.
"""

from __future__ import annotations

import contextlib
import copy
import logging
import pathlib
import threading
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.post_task_checkpoint import post_task_synthesis_in_flight
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

_PROJECT_DELETE_WORKERS_LOCK = threading.Lock()
_PROJECT_DELETE_WORKERS: set[tuple[str, str]] = set()


def _queue_module():
    from supervisor import queue

    return queue


def transition_acceptance_fence(
    *, action: str, token: str, root_task_id: str = "", task_id: str = "", outcome: str = "",
    expected_generation: Optional[int] = None,
) -> Dict[str, Any]:
    """Atomically open, inspect, release, or seal a root admission fence.

    ``begin`` is idempotent for its requester: the same token, or the SAME task with a
    new one (its answer was lost), re-adopts the ``active`` row and rebinds it, so a dead
    attempt's late events own no row. A ``sealed`` row is never re-adopted. A token that
    owns no row answers ``released`` + ``row_absent`` — idempotent after ``task_done``
    cleared the fence, and never a seal. Every answer carries the row's real status.
    """
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
            matched_root, requester = root_task_id, str(task_id or root_task_id)
            row = q.ACCEPTANCE_FENCES.get(root_task_id)
            if not isinstance(row, dict):
                row = q.ACCEPTANCE_FENCES[root_task_id] = {
                    "token": token, "root_task_id": root_task_id, "task_id": requester,
                    "status": "active", "opened_at": utc_now_iso(), "owner_message_generation": 0,
                }
            elif str(row.get("token") or "") != token:
                status = str(row.get("status") or "active")
                if status != "active" or str(row.get("task_id") or "") != requester:
                    return {"ok": False, "status": status,
                            "error": f"acceptance fence already {status} for root {root_task_id}"}
                row["token"] = token
        else:
            matched_root = next(
                (rid for rid, row in q.ACCEPTANCE_FENCES.items() if str(row.get("token") or "") == token),
                "",
            )
            if not matched_root:
                return {"ok": True, "status": "released", "token": token, "row_absent": True}
            row = q.ACCEPTANCE_FENCES[matched_root]
        current_generation = int(row.get("owner_message_generation") or 0)
        result = {"ok": True, "status": str(row.get("status") or "active"), "root_task_id": matched_root, "token": token}
        if action != "end":
            result["owner_message_generation"] = current_generation
            result["queue_descendants"] = _live_descendants_locked(
                q, matched_root, exclude_task_id=str(row.get("task_id") or matched_root),
            )
            if action == "inspect":
                return result
        elif (normalized_outcome := str(outcome or "").strip().lower()) == "revision":
            q.ACCEPTANCE_FENCES.pop(matched_root, None)
            result["status"] = "released"
        elif expected_generation is not None and current_generation != int(expected_generation):
            q.ACCEPTANCE_FENCES.pop(matched_root, None)
            result.update(
                status="released", generation_mismatch=True,
                expected_generation=int(expected_generation), owner_message_generation=current_generation,
            )
        else:
            row.update(status="sealed", outcome=normalized_outcome or "terminal", sealed_at=utc_now_iso())
            result["status"] = "sealed"
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


def clear_budget_root_fence_for_settled_tree(task: dict) -> bool:
    """Release a root budget fence once its tree has no live members left.

    The brother of ``clear_acceptance_fence_for_root``, keyed by the ROOT id
    (a fence covers the whole tree, not one task): called from the task_done
    seam, so every cancel path — pending capture, running custody, cascade —
    releases the latch as a class. A fence over a tree that still has PENDING
    or RUNNING members stays: only the last settling member clears it.
    Without this, cancelling a paused tree left the fence latched forever
    (and the snapshot restore would resurrect it after a restart). An owner
    Pause latch also stays while the answered root's late phase is paused or
    still running (D10): that remainder is the tree's last member.
    """
    q = _queue_module()
    if not isinstance(task, dict):
        return False
    root_id = str(task.get("root_task_id") or task.get("id") or "").strip()
    if not root_id:
        return False
    from ouroboros.post_task_checkpoint import late_phase_state

    late = late_phase_state(q.DRIVE_ROOT, root_id) if (q.BUDGET_ROOT_FENCES.get(root_id) or {}).get(
        "cause") == "owner_pause" else ""
    with q._queue_lock:
        if root_id not in q.BUDGET_ROOT_FENCES:
            return False
        if late and (q.BUDGET_ROOT_FENCES.get(root_id) or {}).get("cause") == "owner_pause":
            return False

        def _member(row) -> bool:
            return isinstance(row, dict) and root_id in (
                str(row.get("root_task_id") or ""), str(row.get("id") or ""),
            )

        if any(_member(row) for row in q.PENDING):
            return False
        for meta in q.RUNNING.values():
            row = meta.get("task") if isinstance(meta, dict) else None
            if _member(row):
                return False
        return q.BUDGET_ROOT_FENCES.pop(root_id, None) is not None


def sweep_orphaned_budget_fences(pending, fences, drive_root) -> list:
    """Drop restored budget fences whose trees have no live members left.

    A fence over a DEAD tree is an orphan: its members settled (a cancel
    raced the pre-crash snapshot, or the release lost the crash window
    between the in-memory pop and the snapshot persist) and no future
    task_done will ever release it. Runs at the restore seam so the latch
    cannot outlive its tree across restarts.
    """
    try:
        from ouroboros.post_task_checkpoint import late_phase_state

        live_roots = {
            str(t.get("root_task_id") or t.get("id") or "")
            for t in pending if isinstance(t, dict)
        }
        # An open late phase (D10) is the owner-paused tree's member across Restart:
        # saved, or still to be settled by recovery (the census then releases it).
        orphaned = [root for root in list(fences) if root not in live_roots and not (
            (fences.get(root) or {}).get("cause") == "owner_pause" and late_phase_state(drive_root, root))]
        for root in orphaned:
            fences.pop(root, None)
        if orphaned:
            from ouroboros.utils import append_jsonl

            append_jsonl(
                pathlib.Path(drive_root) / "logs" / "supervisor.jsonl",
                {"ts": utc_now_iso(), "type": "budget_root_fence_orphan_swept",
                 "root_task_ids": sorted(orphaned)},
            )
        return orphaned
    except Exception:
        log.warning("Orphaned budget fence sweep failed", exc_info=True)
        return []


def budget_pause_fact(task, fences=None):
    """The ONE predicate for "this queued task is budget-paused".

    A member is paused either by its own replay-safe ``_budget_pause`` row or
    by a live root budget fence over its tree (a fenced sibling carries no
    row of its own). ``fences`` defaults to the live registry; pass a snapshot
    taken under the queue lock for a consistent projection.
    """
    pause = task.get("_budget_pause") if isinstance(task, dict) else None
    if isinstance(pause, dict):
        return pause
    from supervisor.events_budget import budget_fence_selected, budget_hold_fact

    fence_map = _queue_module().BUDGET_ROOT_FENCES if fences is None else fences
    root_id = str((task or {}).get("root_task_id") or (task or {}).get("id") or "")
    fence = fence_map.get(root_id)
    if (isinstance(fence, dict) and str(fence.get("status") or "") in {"active", "paused"}
            and not budget_fence_selected(task, fence)):
        # An explicit selection recorded against THIS fence released exactly one
        # row; its siblings stay fenced (owner Q9).
        return fence
    if isinstance(task, dict) and budget_hold_fact(task) is not None:
        return dict(budget_hold_fact(task))
    return None


def queued_admitted_dispatch(task: Any) -> bool:
    """A live queued row whose own next dispatch the queue already admitted.

    A requeued retry carries recorded ``possible`` evidence; nothing of its own
    (pause, grant, owner-wait or terminalization carrier, typed or selected
    hold, owner hold, unpublished Continue) can move it past a root latch, so
    while that latch stands it is queued, not running. It never proves the
    earlier attempt's effects settled: that is the tree custody's question.
    """
    from supervisor.events_budget import BUDGET_HOLD_KEY

    return bool(isinstance(task, dict) and task.get("admitted_dispatch") == "possible"
                and not any(task.get(key) for key in (
                    "_owner_hold", "_budget_pause", "_budget_pause_resume", "_owner_wait_resume",
                    "_terminalization_retry", "_continuation_prepared", BUDGET_HOLD_KEY)))


def _resume_unstarted_owner_paused_root(q: Any, task: Dict[str, Any], fence: Dict[str, Any],
                                         observation: Dict[str, Any]) -> Dict[str, Any]:
    """The owner's Resume of a root the Pause caught while it was only queued.

    A never-started root has no members: nothing of the tree ran. A queued root
    whose own dispatch was already admitted (``queued_admitted_dispatch``, a
    requeued retry) resumes only over a fresh whole-tree observation without
    unsettled effects (``owner_pause_blockers``); it keeps its attempt and
    dispatch evidence, and its other queued members wait for explicit selection
    as after an exact root Resume (Q9). Either way the Resume reopens both the
    durable fence and the queue latch, and the same queued task becomes
    assignable again. A durable fence that cannot be reopened refuses the
    Resume: the root would only be refused at every launch. Queue lock held.
    """
    from ouroboros.owner_pause import FENCE_RELEASED, launch_lock, read_fence, set_fence_state
    from supervisor.events_budget import (
        BUDGET_HOLD_KEY, HOLD_OWNER_RESTART, HOLD_SAVED_WORK, budget_hold_fact, hold_root_resume_descendants,
    )

    task_id = str(task.get("id") or "")
    admitted = "owner_pause_blockers" in observation
    safe, error = observation["safe"], observation["unsafe_error"]
    if not safe and not admitted:
        return {"ok": False, "error": error, "action": "cancel_or_new_run"}
    hold = budget_hold_fact(task)
    # This explicit root Resume releases both its Pause and its app-stop hold.
    # All other holds, observed effects and durable write gates still bind.
    if hold and hold.get("reason") not in {HOLD_OWNER_RESTART, HOLD_SAVED_WORK}:
        return {"ok": False, "error": str(hold.get("reason") or "selection_owner_held")}
    if observation.get("blockers"):
        return {"ok": False, "error": "predecessor_writers_unsettled", "blockers": observation["blockers"]}
    if observation.get("owner_pause_blockers"):
        return {"ok": False, "error": "owner_pause_effects_unsettled", "action": "wait_for_effect_settlement",
                "blockers": observation["owner_pause_blockers"][:20]}
    root = pathlib.Path(task.get("budget_drive_root") or q.DRIVE_ROOT)
    members = [row for row in q.PENDING if admitted and row is not task
               and str(row.get("root_task_id") or "") == task_id]
    prior = [(row, copy.deepcopy(row)) for row in [task, *members]]

    def rollback() -> None:
        for row, before in prior:
            row.clear()
            row.update(before)
        q.BUDGET_ROOT_FENCES[task_id] = fence

    with launch_lock(root, task_id):
        if read_fence(root, task_id) != observation["authority"]["owner_fence"]:
            return {"ok": False, "error": "selection_authority_changed"}
        # Publish queue eligibility first, while durable launch authority is still
        # closed. A crash or failed second write cannot start the unrun root.
        q.BUDGET_ROOT_FENCES.pop(task_id, None)
        task.pop("_budget_pause", None)
        task.pop(BUDGET_HOLD_KEY, None)
        if members:
            # No root grant exists to bind: only the owner selects these members.
            hold_root_resume_descendants(q, task_id, fence, {"grant_id": "", "generation": 0})
        if not q.persist_queue_snapshot(reason="owner_pause_root_resumed"):
            rollback()
            return {"ok": False, "error": "snapshot_not_persisted"}
        try:
            set_fence_state(root, task_id, fence_id=str(fence.get("fence_id") or ""),
                            state=FENCE_RELEASED, release_reason="owner_resume_unstarted_root")
        except Exception as exc:
            rollback()
            q.persist_queue_snapshot(reason="owner_pause_root_resume_held")
            return {"ok": False, "error": "owner_pause_fence_unwritable", "detail": str(exc)[:200]}
    q.append_jsonl(q.DRIVE_ROOT / "logs" / "events.jsonl",
                   {"ts": utc_now_iso(), "type": "owner_pause_resumed", "task_id": task_id,
                    "root_task_id": task_id, "fence_id": str(fence.get("fence_id") or ""),
                    "same_generation": True, "never_started": not admitted})
    return {"ok": True, "task_id": task_id, "root_task_id": task_id, "same_generation": True,
            "owner_pause_released": True, "never_started": not admitted}


def pending_member_replay_safe(q: Any, member: Dict[str, Any]) -> Tuple[bool, str]:
    """Whether one PENDING row genuinely never dispatched (zero physical calls)."""
    from supervisor.restart_retention import never_started
    from supervisor.schedule_occurrence import restore_allowed

    if member.get("_owner_hold"):
        return False, "owner_held"
    if "_working_recovery" in member:
        # A locator alone does not authorize Resume: read its exact frozen source
        # and the target attempt's durable launch evidence, as assignment does.
        from supervisor.task_admission import _working_resume_granted

        try:
            if _working_resume_granted(member, q.DRIVE_ROOT):
                return True, ""
        except (OSError, ValueError, TypeError, KeyError):
            log.warning("Saved work Resume evidence unavailable for %s", member.get("id"), exc_info=True)
        return False, "dispatch_outcome_unknown"
    if not restore_allowed(member):
        return False, "dispatch_outcome_unknown"
    if member.get("_owner_hold"):
        return False, "owner_held"

    pause = member.get("_budget_pause") or (member.get("_budget_pause_hold") or {}).get("replaced_fence_pause") or {}
    legacy_saved_zero = (member.get("admitted_dispatch") in (None, "")
                         and pause.get("status") == "paused_before_dispatch"
                         and pause.get("replay_safe") is True and pause.get("physical_calls") == 0)
    if not never_started(member) and not legacy_saved_zero:
        return False, "dispatch_outcome_unknown"
    member_id = str(member.get("id") or "")
    cost_fields = q.reconstruct_task_cost(
        member_id, fields=True,
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


def resume_budget_paused_task(task_id: str, *, selected_by: str = "") -> Dict[str, Any]:
    """Explicitly resume one budget-paused task and, if needed, its root latch.

    A replay-safe ZERO-dispatch row is released as before. An EXACT
    continuation row (#1196) receives one single-use grant instead
    (``grant_exact_budget_resume``); it is never re-run from scratch. A row held
    behind a lifted root fence is released only by recording the selection.
    ``selected_by`` names the MODEL-issued request (owner Q9); an empty value is
    the owner's own act, which needs no root grant above itself.
    """
    q = _queue_module()
    task_id = str(task_id or "").strip()
    if not task_id:
        return {"ok": False, "error": "missing_task_id"}
    from supervisor.events_budget import (
        _held_selection_authority, budget_hold_fact, observe_held_budget_selection, select_held_budget_row,
    )

    external = observation = None
    with q._queue_lock:
        located = next((item for item in q.PENDING if str(item.get("id") or "") == task_id), None)
    if located is None:
        # A member parked WARM under the owner's Pause (its stack retained while a
        # started critic finishes) wakes on the fence; the owner's own Resume and
        # a model's selection reach it here.
        from supervisor.budget_resume import resume_warm_owner_pause_root

        warm = resume_warm_owner_pause_root(task_id, selected_by=selected_by)
        if warm is not None:
            return warm
    if located is None and not selected_by:
        # An answered root's paused remainder (D10) resumes through its own grant.
        from supervisor.budget_resume import resume_late_phase

        late = resume_late_phase(task_id)
        if late is not None:
            return late
    with q._queue_lock:
        located = next((item for item in q.PENDING if str(item.get("id") or "") == task_id), None)
        if located is not None and not located.get("_budget_pause"):
            from supervisor.restart_retention import pause_retention, park_saved_pause, RETAIN_SLEEP

            retention_root = pathlib.Path(located.get("budget_drive_root") or q.DRIVE_ROOT)
            attempt = int(located.get("_attempt") or 1)
            if pause_retention(retention_root, task_id, attempt) == RETAIN_SLEEP:
                parked = park_saved_pause(located, attempt, retention_root, pause_source="explicit_resume")
                if not parked or not parked.get("_budget_pause"):
                    return {"ok": False, "error": "sleep_retention_unwritable"}
                located.update(parked)
                if not q.persist_queue_snapshot(reason="saved_sleep_retained"):
                    return {"ok": False, "error": "snapshot_not_persisted"}
        exact = bool(located is not None and isinstance(located.get("_budget_pause"), dict)
                     and located["_budget_pause"].get("exact_continuation"))
        owner_paused = bool(exact and located["_budget_pause"].get("reason") == "owner")
        result_root = pathlib.Path((located or {}).get("budget_drive_root") or q.DRIVE_ROOT)
        located_state = copy.deepcopy(located)
        needs_selection = located is not None and budget_pause_fact(located) is not None
    if exact:
        # Fresh custody at EVERY grant, read OUTSIDE the queue lock (a harness
        # round trip is not a queue-lock tenant); the grant below re-locates the
        # row and decides on this observation. A budget pause requests stops of
        # uncovered runs; the owner's Pause only observes sent work (Batch4 5A).
        from ouroboros.budget_pause import observe_task_runs
        from ouroboros.owner_pause import fence_closed, read_fence

        root_task_id = str(located_state.get("root_task_id") or task_id)
        try:
            owner_fence = read_fence(result_root, root_task_id)
        except Exception:
            return {"ok": False, "error": "owner_pause_custody_unreadable"}
        # Owner Pause may overlay an already saved budget/sleep checkpoint.
        owner_paused = owner_paused or fence_closed(owner_fence)

        external = observe_task_runs(result_root, task_id, reason="budget_resume_uncovered_cost",
                                     request_stop=not owner_paused)
        if owner_paused:
            from supervisor.continuation_admission import action_writers

            # A saved loop is not proof that its handed tools/processes ended.
            # Observe the existing whole-tree custody owner off the queue lock.
            try:
                external["owner_pause_tree"] = {"fence": owner_fence, "blockers": action_writers(
                    q, root_task_id, drive_root=result_root,
                    owner_pause_fence_id=str(owner_fence.get("fence_id") or "") if fence_closed(owner_fence) else "")}
            except Exception as exc:
                external["owner_pause_tree"] = {"error": str(exc)}
    elif needs_selection:
        observation = observe_held_budget_selection(q, task_id)
        owner_fence = (observation.get("authority") or {}).get("owner_fence") or {}
        if (not selected_by and owner_fence.get("fence_id") and queued_admitted_dispatch(located_state)
                and str(located_state.get("root_task_id") or task_id) == task_id):
            # An owner-paused root whose own dispatch was already admitted resumes
            # only over this fresh, off-lock whole-tree observation (Pause A).
            from supervisor.continuation_admission import action_writers

            try:
                observation["owner_pause_blockers"] = action_writers(
                    q, task_id, drive_root=result_root, owner_pause_fence_id=str(owner_fence["fence_id"]))
            except Exception as exc:
                observation["owner_pause_blockers"] = [{"kind": "tree_census_unreadable", "detail": str(exc)[:200]}]
    with q._queue_lock:
        task = next((item for item in q.PENDING if str(item.get("id") or "") == task_id), None)
        if task is None:
            return {"ok": False, "error": "task_not_pending"}
        pause = task.get("_budget_pause") if isinstance(task.get("_budget_pause"), dict) else None
        if pause and pause.get("exact_continuation"):
            if task is not located or task != located_state:
                return {"ok": False, "error": "selection_authority_changed"}
            return grant_exact_budget_resume(task, pause, selected_by=selected_by, external=external)
        if observation is not None:
            if observation.get("error"):
                return {"ok": False, "error": observation["error"]}
            try:
                current = _held_selection_authority(q, task)
            except Exception:
                return {"ok": False, "error": "selection_authority_unavailable"}
            if task is not observation["candidate"] or current != observation["authority"]:
                return {"ok": False, "error": "selection_authority_changed"}
            if current["cancel_pending"] or current["stop_flags"]:
                return {"ok": False, "error": "selection_owner_held"}
        elif budget_pause_fact(task) is not None:
            return {"ok": False, "error": "selection_observation_required"}

        candidate_root = str(task.get("root_task_id") or task_id)
        candidate_fence = q.BUDGET_ROOT_FENCES.get(candidate_root)
        if (not selected_by and task_id == candidate_root
                and isinstance(candidate_fence, dict) and candidate_fence.get("cause") == "owner_pause"):
            return _resume_unstarted_owner_paused_root(q, task, candidate_fence, observation)
        hold = budget_hold_fact(task)
        if hold is not None and not pause:
            return select_held_budget_row(q, task, hold, selected_by=selected_by, observation=observation)
        if not pause:
            candidate_root = str(task.get("root_task_id") or task_id).strip()
            candidate_fence = q.BUDGET_ROOT_FENCES.get(candidate_root)
            if not isinstance(candidate_fence, dict):
                return {"ok": False, "error": "task_not_budget_paused"}
            if isinstance(task.get("_budget_pause_resume"), dict):
                return {"ok": False, "error": "resume_already_granted",
                        "grant_id": str(task["_budget_pause_resume"].get("grant_id") or "")}
            if observation is None:
                return {"ok": False, "error": "task_not_budget_paused"}
            pause = {**candidate_fence, "physical_calls": 0, "replay_safe": True}
        root_scope = str(pause.get("scope") or "") == "root"
        root_task_id = str(pause.get("root_task_id") or "").strip()
        fence = q.BUDGET_ROOT_FENCES.get(root_task_id) if root_scope and root_task_id else None
        if (isinstance(fence, dict) and fence.get("cause") == "owner_pause" and task_id == root_task_id
                and not selected_by):
            return _resume_unstarted_owner_paused_root(q, task, fence, observation)
        if root_scope and not isinstance(fence, dict):
            return {"ok": False, "error": "root_budget_fence_missing", "action": "cancel_or_new_run"}
        if root_scope and str(pause.get("fence_id") or "") != str(fence.get("fence_id") or ""):
            return {"ok": False, "error": "replay_unsafe", "action": "cancel_or_new_run"}
        if observation["authority"]["owner_fence"]:
            return {"ok": False, "error": "selection_owner_held"}
        nominated_safe, nominated_error = observation["safe"], observation["unsafe_error"]
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
            # Root Resume selects the root alone; children remain behind this
            # same fence until individually selected under its root grant (Q9).
            from supervisor.events_budget import HOLD_ROOT_FENCE_MEMBER_SELECTION, hold_budget_row

            hold = hold_budget_row(
                task, reason=HOLD_ROOT_FENCE_MEMBER_SELECTION,
                extra={"root_task_id": root_task_id, "fence_id": fence["fence_id"]})
            # Creating this hold changes authority. Observe again only AFTER
            # that mutation, and require the exact post-mutation state below.
            try:
                expected = _held_selection_authority(q, task)
            except Exception:
                return {"ok": False, "error": "selection_authority_unavailable"}
        else:
            resumed_at = utc_now_iso()
            prior_pause = dict(pause)
            task.pop("_budget_pause", None)
            task["budget_resumed_at"] = resumed_at
            q.persist_queue_snapshot(reason="budget_pause_explicit_resume")
    if root_scope:
        observation = observe_held_budget_selection(q, task_id)
        with q._queue_lock:
            if observation.get("authority") != expected:
                return {"ok": False, "error": "selection_authority_changed"}
            return select_held_budget_row(q, task, hold, selected_by=selected_by, observation=observation)
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


def _live_project_task_ids(
    drive_root: object, project_id: str, *, roots_only: bool = False,
    covering: Optional[set] = None,
) -> list[str]:
    """Snapshot queued/running tasks associated with one fenced Project.

    ``roots_only`` (GR5-5) keeps only the tasks with NO live ancestor in the
    associated set — the deletion cascades those roots and their descendants
    fall with their trees, instead of every child getting a redundant cascade
    of its own (and its own summary message beside the root's). An orphan
    child whose ancestors are all settled has no live ancestor in the set, so
    it stays and gets its own cascade.

    ``covering`` (GR7-3, re-entered cancel passes only) further keeps only the
    roots that ARE a member of that set or cover one through lineage — a
    settled root that is merely winding down must not be re-cascaded (each
    re-mint delivers a duplicate owner summary), while the root over a
    genuinely stuck/new task still is.
    """
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
    if roots_only:
        associated = {
            task_id for task_id in associated
            if not _has_live_ancestor_in_set(by_id, task_id, associated)
        }
        if covering is not None:
            associated = {
                task_id for task_id in associated
                if task_id in covering or any(
                    _has_live_ancestor_in_set(by_id, member, {task_id})
                    for member in covering
                )
            }
    return sorted(
        associated,
        key=lambda task_id: bool(str(by_id.get(task_id, {}).get("parent_task_id") or "")),
        reverse=True,
    )


def _has_live_ancestor_in_set(
    by_id: dict[str, dict], task_id: str, members: set[str],
) -> bool:
    """Whether ``task_id`` descends from another LIVE member of ``members``.

    Lineage comes from the same task rows the snapshot captured: the recorded
    ``root_task_id`` (a live root covers the whole tree even when intermediate
    parents already settled) and the parent chain walked through the live rows
    (a mid-tree live ancestor covers its branch even when the recorded root is
    already gone). Depth-bounded like the cascade snapshot's walk.
    """
    task = by_id.get(task_id, {})
    root = str(task.get("root_task_id") or "")
    if root and root != task_id and root in members:
        return True
    parent = str(task.get("parent_task_id") or "")
    seen: set[str] = set()
    while parent and parent not in seen and len(seen) < 100:
        if parent in members:
            return True
        seen.add(parent)
        parent = str(by_id.get(parent, {}).get("parent_task_id") or "")
    return False


def _broadcast_projects_changed(project_id: str, chat_id: Any) -> None:
    try:
        from supervisor.message_bus import get_bridge

        get_bridge().broadcast({"type": "projects_changed", "project_id": project_id, "chat_id": chat_id})
    except Exception:
        _queue_module().log.debug("projects_changed broadcast failed for %s", project_id, exc_info=True)


def stop_evolution_tasks(reason: str = "evolution stopped") -> Dict[str, List[str]]:
    """Stop every PENDING and RUNNING evolution task with typed outcomes (GR2-13).

    Every task — queued or mid-cycle — goes through the SAME durable ingress:
    ``request_cancel`` first (fail-closed per AR2-1: a task whose intent write
    fails is KEPT, never torn down or silently pruned), then the typed
    ``cancel_task_custody``. The old shape pruned PENDING evolution tasks in
    place — no intent, no terminal result, no ``task_done`` — and returned a
    flat "cancelled" list that counted already-settled tasks as cancellations
    and dropped intent-write failures from the caller's view entirely, so
    ``/evolve off`` declared a clean stop over live leftovers.

    Returns ``{"cancelled": [...], "already_settled": [...], "not_found": [...],
    "failed": [...], "intent_write_failed": [...]}`` — ``cancelled`` names only
    tasks THIS stop actually cancelled; ``failed`` and ``intent_write_failed``
    name tasks that are still live and need the caller to say so.
    """
    q = _queue_module()
    outcomes: Dict[str, List[str]] = {
        "cancelled": [], "already_settled": [], "not_found": [],
        "failed": [], "intent_write_failed": [],
    }
    with q._queue_lock:
        pending_ids = [
            str(task.get("id") or "")
            for task in q.PENDING
            if isinstance(task, dict) and str(task.get("type") or "") == "evolution"
        ]
        running_ids = [
            str(task_id)
            for task_id, meta in q.RUNNING.items()
            if isinstance(meta, dict)
            and isinstance(meta.get("task"), dict)
            and str(meta["task"].get("type") or "") == "evolution"
        ]
    for task_id in dict.fromkeys([*pending_ids, *running_ids]):
        if not task_id:
            continue
        # Durable intent FIRST (owner batch-4 1=A): every cancel ingress goes
        # through the projection, so a crash mid-teardown leaves the
        # supervisor watchdog an owner instead of a half-killed evolution
        # cycle with nothing to finish it. FAIL-CLOSED (AR2-1): a task whose
        # intent could not be recorded is NOT torn down this pass — an
        # unfenced kill would recreate the unreplayable cancel; the caller
        # surfaces the incomplete stop and the owner retries.
        try:
            from ouroboros.cancel_intents import request_cancel

            # GR6-1 live-ownership check at the ingress: a durably-settled
            # evolution task whose worker is still winding down (post-task
            # cognition) must still be fenced and killed — ``already_settled``
            # is only terminal when no live ownership remains.
            intent = request_cancel(
                q.DRIVE_ROOT, task_id, reason=reason, source="evolution_stop",
                allow_settled_target=task_has_live_ownership(task_id),
            )
        except Exception:
            q.log.warning(
                "evolution-stop cancel intent failed for %s; task kept "
                "(owner 1=A: no cancel without a durable intent)",
                task_id, exc_info=True,
            )
            outcomes["intent_write_failed"].append(task_id)
            continue
        try:
            outcome = q.drive_cancel_intent_scope(
                str(intent.get("task_id") or task_id),
            )
        except Exception:
            q.log.warning(
                "Failed to cancel evolution task %s (%s)", task_id, reason, exc_info=True
            )
            outcomes["failed"].append(task_id)
            continue
        bucket = {
            q.CANCEL_CANCELLED: "cancelled",
            q.CANCEL_ALREADY_SETTLED: "already_settled",
            q.CANCEL_NOT_FOUND: "not_found",
        }.get(outcome, "failed")
        outcomes[bucket].append(task_id)
    return outcomes


def evolution_stop_report(outcomes: Dict[str, List[str]]) -> tuple[List[str], bool]:
    """Compose honest ``/evolve off`` message lines from typed stop outcomes.

    ONE composer for both stop ingresses (owner chat and agent tool), so the
    two surfaces cannot drift: "Cancelled" names only real cancellations,
    already-settled tasks are reported as what they are, and any task that is
    STILL LIVE (custody failure or a refused intent write) makes the stop
    INCOMPLETE — returned as ``(lines, incomplete)``.
    """
    lines: List[str] = []
    if outcomes.get("cancelled"):
        lines.append("🛑 Cancelled evolution task(s): " + ", ".join(outcomes["cancelled"]))
    settled = [*outcomes.get("already_settled", []), *outcomes.get("not_found", [])]
    if settled:
        lines.append("ℹ️ Already settled (nothing left to cancel): " + ", ".join(settled))
    still_live = [*outcomes.get("failed", []), *outcomes.get("intent_write_failed", [])]
    if still_live:
        lines.append(
            "⚠️ Evolution stop INCOMPLETE — still live: " + ", ".join(still_live)
            + ". These tasks were kept (cancellation could not be made durable or "
            "the teardown failed); retry the stop or cancel them individually."
        )
    return lines, bool(still_live)


# GR6-1c wind-down bounds: a durably-settled task still in the live maps is a
# worker/finalizer winding down, not a stuck deletion — the quiescence check
# defers briefly and re-checks instead of failing instantly. Bounded so a
# genuinely wedged finalizer still fails VISIBLY rather than spinning forever.
_WIND_DOWN_MAX_ROUNDS = 20
_WIND_DOWN_SLEEP_SEC = 0.5


def _settled_status(drive_root: object, task_id: str) -> str:
    """The task's own already-settled durable status, or "" — fail-soft."""
    try:
        from ouroboros.task_results import load_task_result
        from ouroboros.task_status import SETTLED_STATUSES

        status = str((load_task_result(drive_root, task_id) or {}).get("status") or "")
        return status if status in SETTLED_STATUSES else ""
    except Exception:
        log.debug("settled-status read failed for %s", task_id, exc_info=True)
        return ""


def task_subtree_is_live(task_id: str, *, ignore_intents: bool = False) -> bool:
    """Cheap liveness pre-check for the HTTP cascade-cancel path (v6.82).

    True when the task itself is queued/running, when it still has live
    descendants in the queue, when THIS process still runs its paid post-task
    synthesis (the direct-chat turn's in-flight key), or when it holds an
    ACTIVE durable cancel intent (or the legacy ``cancel_requested`` status
    latch of pre-redesign files) — the intent's settle still has honest work
    to do. Everything else is inactive and must keep today's 404 contract.

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
    if self_running and not _settled_status(q.DRIVE_ROOT, task_id):
        return True
    if any(tid and not _settled_status(q.DRIVE_ROOT, tid) for tid in descendants):
        return True
    # The direct-chat turn's paid post-task synthesis is PHYSICAL liveness with
    # no queue row at all (GR6-1): the row settled and the turn's loop returned
    # while the in-process thread still bills. Custody refuses it as "still
    # live"; the cascade postcondition must see the same fact, or it re-judges
    # that refusal against the stored ``completed`` alone, settles the root's
    # cascade intent and the per-stage gate loses the Stop it was waiting on.
    if post_task_synthesis_in_flight(q.DRIVE_ROOT, task_id):
        return True
    from ouroboros.review_operation import task_has_live_review_operation

    if task_has_live_review_operation(q.DRIVE_ROOT, task_id):
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


def _live_retry_target_locked(q: Any, task_id: str, *, results) -> Tuple[str, str]:
    """Resolve an old root id to its one validated live retry leaf.

    A live row is not lineage authority by itself: ingress fields can be stale
    or malformed, and choosing one of several candidates would let a cancel
    intent settle against the wrong physical task. Follow the reciprocal
    host-written result chain (old points to new; new points back to old), then
    require the live task row to satisfy the existing root-retry contract.
    Corrupt, overlong, cyclic, or ambiguous chains raise so custody leaves the
    intent open and fails closed.
    """
    requested = str(task_id or "").strip()
    if not requested:
        return requested, ""
    live_rows: list[Dict[str, Any]] = [
        row for row in q.PENDING if isinstance(row, dict)
    ]
    live_rows.extend(
        meta["task"]
        for meta in q.RUNNING.values()
        if isinstance(meta, dict) and isinstance(meta.get("task"), dict)
    )
    live_by_id = {
        str(row.get("id") or ""): row
        for row in live_rows
        if str(row.get("id") or "")
    }

    from ouroboros.task_results import resolve_task_lineage

    requested_result = results.load(requested)
    requested_shape = (
        requested_result
        if requested_result
        else live_by_id.get(requested, {})
    )
    requested_lineage = resolve_task_lineage(
        requested,
        metadata=requested_shape.get("metadata"),
        root_task_id=requested_shape.get("root_task_id"),
        parent_task_id=requested_shape.get("parent_task_id"),
        delegation_role=requested_shape.get("delegation_role"),
        original_task_id=requested_shape.get("original_task_id"),
        timeout_retry_from=requested_shape.get("timeout_retry_from"),
    )
    if not requested_lineage.get("is_root_task"):
        # Retry aliases exist only for top-level root attempts. A subagent or
        # other descendant is already a physical id; scanning root-shaped rows
        # from its wider tree would misclassify an unrelated root retry as this
        # child's successor and block cascade custody.
        return requested, ""

    chain_predecessor: Dict[str, str] = {}
    current_id = requested
    seen = {requested}
    try:
        max_edges = max(0, int(getattr(q, "QUEUE_MAX_RETRIES", 0) or 0))
    except (TypeError, ValueError, OverflowError):
        max_edges = 0
    logical_root = requested
    while True:
        current = results.load(current_id) or {}
        if current_id == requested:
            logical_root = str(current.get("root_task_id") or requested)
        superseded_by = str(current.get("superseded_by") or "").strip()
        retry_task_id = str(current.get("retry_task_id") or "").strip()
        # Subagent/evolution timeout retries intentionally reuse the exact id.
        # Their result carries retry_task_id=self as attempt metadata, not as a
        # physical lineage edge.
        if not superseded_by and retry_task_id == current_id:
            break
        if not superseded_by and not retry_task_id:
            break
        if not superseded_by or superseded_by != retry_task_id:
            raise RuntimeError(
                f"timeout retry lineage from {current_id} is not reciprocal"
            )
        if len(chain_predecessor) >= max_edges:
            raise RuntimeError(
                f"timeout retry lineage from {requested} exceeds retry authority"
            )
        successor_id = superseded_by
        if successor_id in seen:
            raise RuntimeError(
                f"timeout retry lineage from {requested} contains a cycle"
            )
        successor = results.load(successor_id) or {}
        if (
            str(successor.get("supersedes_task_id") or "") != current_id
            or str(successor.get("original_task_id") or "") != current_id
            or str(successor.get("timeout_retry_from") or "") != current_id
        ):
            raise RuntimeError(
                f"timeout retry lineage {current_id} -> {successor_id} is incomplete"
            )
        chain_predecessor[successor_id] = current_id
        seen.add(successor_id)
        current_id = successor_id

    relevant_live: list[str] = []
    if requested in live_by_id:
        relevant_live.append(requested)
    for candidate_id, predecessor_id in chain_predecessor.items():
        row = live_by_id.get(candidate_id)
        if row is None:
            continue
        lineage = resolve_task_lineage(
            candidate_id,
            metadata=row.get("metadata"),
            root_task_id=row.get("root_task_id"),
            parent_task_id=row.get("parent_task_id"),
            delegation_role=row.get("delegation_role"),
            original_task_id=row.get("original_task_id"),
            timeout_retry_from=row.get("timeout_retry_from"),
        )
        if (
            not lineage.get("is_retry_root_attempt")
            or str(lineage.get("root_task_id") or "") != logical_root
            or str(lineage.get("original_task_id") or "") != predecessor_id
            or str(lineage.get("timeout_retry_from") or "") != predecessor_id
        ):
            raise RuntimeError(
                f"live timeout retry {candidate_id} fails root-lineage validation"
            )
        relevant_live.append(candidate_id)

    # A root-shaped live retry under the same logical root but outside the
    # reciprocal chain is authority corruption, not an ignorable bystander.
    for candidate_id, row in live_by_id.items():
        if candidate_id == requested or candidate_id in chain_predecessor:
            continue
        lineage = resolve_task_lineage(
            candidate_id,
            metadata=row.get("metadata"),
            root_task_id=row.get("root_task_id"),
            parent_task_id=row.get("parent_task_id"),
            delegation_role=row.get("delegation_role"),
            original_task_id=row.get("original_task_id"),
            timeout_retry_from=row.get("timeout_retry_from"),
        )
        if (
            lineage.get("is_retry_root_attempt")
            and str(lineage.get("root_task_id") or "") == logical_root
        ):
            raise RuntimeError(
                f"live timeout retry {candidate_id} is outside the durable chain"
            )

    if len(relevant_live) > 1:
        raise RuntimeError(
            f"multiple live timeout attempts resolve from {requested}: "
            f"{', '.join(relevant_live)}"
        )
    if relevant_live:
        return relevant_live[0], ""
    if chain_predecessor:
        from ouroboros.task_status import SETTLED_STATUSES

        leaf_status = str(
            (results.load(current_id) or {}).get("status")
            or ""
        )
        if leaf_status in SETTLED_STATUSES:
            return current_id, leaf_status
    return requested, ""


def task_has_live_ownership(task_id: str, *, ignore_review_operation: str = '', ownership=None) -> bool:
    """Whether executable or paused custody remains for this task: a RUNNING row,
    busy worker, direct turn, still-billing synthesis or saved late remainder.

    The pipeline persists the durable terminal result BEFORE post-task
    cognition ends, so "the status is settled" and "the worker is dead" are
    two different facts — ``already_settled`` is a terminal answer ONLY when
    this predicate is False. Cancel INGRESSES consult it and pass
    ``allow_settled_target=True`` to ``request_cancel`` while ownership is
    live, so custody kills the still-spending worker instead of no-oping;
    completion-wins keeps the stored result either way. (The worker-process
    twin — the agent tool cannot see the live maps — is
    ``ouroboros.task_status.task_has_live_queue_ownership`` over the queue
    snapshot.)
    """
    q = _queue_module()
    from supervisor import workers

    task_id = str(task_id or "").strip()
    if not task_id:
        return False
    from supervisor.task_ownership import TaskOwnershipRead, prepare_retry_chain
    from ouroboros.review_operation import task_has_live_review_operation, paused_acceptance_preparations
    from ouroboros.owner_pause import fence_closed, FENCE_REQUESTED, FENCE_PAUSED, FENCE_RELEASED

    ownership = ownership or TaskOwnershipRead(q.DRIVE_ROOT)
    accessed = set()
    def read(tid):
        accessed.add(tid)
        return ownership.load(tid)

    try:
        current = read(task_id)
        if task_has_live_review_operation(q.DRIVE_ROOT, task_id, exclude_owner_id=ignore_review_operation,
                                          result_loader=read):
            return True
        fence = current.get("owner_pause") or {}
        if not isinstance(fence, dict) or (fence and fence.get("state") not in
                                           {FENCE_REQUESTED, FENCE_PAUSED, FENCE_RELEASED}):
            return True
        if fence_closed(fence) and paused_acceptance_preparations(q.DRIVE_ROOT, task_id, fence['fence_id'],
                                                                 result_loader=read):
            return True
        if (current.get("root_phase_checkpoint") or {}).get("post_task_synthesis") == "paused":
            return True
        prepare_retry_chain(q, task_id, read)
    except (OSError, ValueError, TypeError, RuntimeError):
        return True  # Unknown durable ownership cannot authorize a mutation.
    with q._queue_lock:
        if not ownership.unchanged(accessed):
            return True
        if task_id in q.RUNNING:
            return True
        if any(
            worker.busy_task_id == task_id for worker in workers.WORKERS.values()
        ):
            return True
        # The in-process direct-chat turn is live PHYSICAL ownership too: it
        # spends on the owner's behalf from inside the supervisor process and
        # writes an ordinary durable running row, so it must be as
        # addressable as a pooled worker (custody stops it cooperatively).
        if workers.direct_chat_turn(task_id) is not None:
            return True
        # The turn's paid post-task synthesis OUTLIVES the turn: the loop
        # returns and ``_busy`` drops while the in-process worker thread still
        # bills, so the pipeline's in-flight key is live physical ownership
        # too — the stop ingress must not answer 404 while the spend
        # continues, and custody keeps the intent open for the stage gate.
        if post_task_synthesis_in_flight(q.DRIVE_ROOT, task_id):
            return True
        try:
            retry_target, _retry_settled_status = _live_retry_target_locked(q, task_id, results=ownership)
        except Exception:
            # An indeterminate chain cannot prove that physical ownership is
            # absent.  Fail open toward liveness so intent minting, which does
            # the authoritative locked validation, decides the request.
            return True
        return bool(
            retry_target != task_id
            and (
                retry_target in q.RUNNING
                or any(
                    worker.busy_task_id == retry_target
                    for worker in workers.WORKERS.values()
                )
            )
        )


def task_settlement_liveness(task_id: str) -> Optional[bool]:
    """The probe a destructive custody settlement asks (``task_custody.settle_child_drive``):
    True while ``task_has_live_ownership`` holds or the id waits in PENDING, None while a
    reap of this task (its slot or a queued/deferred reap job) makes absence inconclusive
    or the queue cannot be read, False only for proven absence. Callers without the
    supervisor's live maps have no such probe, so they never delete."""
    q = _queue_module()
    from supervisor import task_reaper, workers

    task_id = str(task_id or "").strip()
    if not getattr(q, "INITIALIZED", False):
        return None  # empty maps outside the supervisor process prove no absence
    try:
        from supervisor.task_ownership import settlement_reads

        locked = q._queue_lock._is_owned()
        reads = settlement_reads(q.DRIVE_ROOT, task_id, locked=locked)
        if reads is None:
            return None
        if not locked:
            task_has_live_ownership(task_id, ownership=reads)  # Prepare before the mutation lock.
        with q._queue_lock:
            if task_has_live_ownership(task_id, ownership=reads) or any(str(row.get("id") or "") == task_id for row in q.PENDING):
                return True
            with q._reap_queue.mutex:
                jobs = [*q._reap_queue.queue, *task_reaper._deferred_reap_jobs]
            if any(worker.reaping and worker.busy_task_id == task_id for worker in workers.WORKERS.values()) \
                    or any(isinstance(job, dict) and str(job.get("task_id") or "") == task_id for job in jobs):
                return None  # a reap of this task is queued or holds its slot: its process may live
        if not locked:
            settlement_reads(q.DRIVE_ROOT, task_id, locked=False, prepared=reads)
        return False
    except Exception:
        log.warning("Settlement liveness of %s is unknown", task_id, exc_info=True)
        return None


@contextlib.contextmanager
def task_settlement_interlock(stop: Any = None):
    """The ownership interlock a drive settlement moves under (``task_custody.settle_child_drive``
    ``guard``): the queue lock admission and assignment hold, so no occupant can be admitted,
    assigned or retried between the settlement's last probe and the drive's move. Yields
    whether the caller's generation is still open (``stop()`` False)."""
    q = _queue_module()
    with q._queue_lock:
        yield not (callable(stop) and stop())


def run_project_deletion(
    drive_root: object,
    project_id: str,
    chat_id: Any,
    worker_key: tuple[str, str] | None = None,
) -> None:
    """Cancel a fenced Project tree and tombstone only after quiescence.

    GR7-3 wind-down shape: after a cancel pass, the loop only RE-CHECKS
    quiescence (live set + settled statuses) — it never re-runs the cancel
    pass over a purely settled-lingering set. The previous ``continue`` shape
    re-entered the full pass every 0.5s round: the root's intent had settled
    at the cascade postcondition, so each round minted a FRESH request_id →
    fresh cascade delivery id → up to ``_WIND_DOWN_MAX_ROUNDS`` duplicate
    owner summaries. The cancel pass is re-entered ONLY when a non-settled
    (genuinely stuck or newly admitted) task appears in the remaining set,
    and then only for the roots covering those tasks.
    """
    import time as _time

    from ouroboros.projects_registry import complete_project_deletion, fail_project_deletion

    q = _queue_module()
    first_pass = True
    try:
        while True:
            live_ids = _live_project_task_ids(drive_root, project_id)
            if not live_ids:
                complete_project_deletion(drive_root, project_id)
                _broadcast_projects_changed(project_id, chat_id)
                return
            errors: list[str] = []
            nonsettled_before = {
                tid for tid in live_ids if not _settled_status(drive_root, tid)
            }
            # GR5-5: cascade only the LINEAGE ROOTS of the live set — each
            # root's cascade tears down its whole subtree and delivers the
            # tree's ONE summary; cascading every child too ran redundant
            # cascades and delivered per-child summaries beside the root's.
            # A child whose root intent write failed stays covered by the
            # next round / the quiescence fail-closed path below.
            # GR7-3: a RE-ENTERED pass targets only the roots covering the
            # non-settled members — a settled root winding down is never
            # re-cascaded (each re-mint would deliver a duplicate summary).
            for task_id in _live_project_task_ids(
                drive_root, project_id, roots_only=True,
                covering=None if first_pass else nonsettled_before,
            ):
                try:
                    # Durable intent FIRST (owner batch-4 1=A): a crash between
                    # here and the settle leaves the watchdog an owner for the
                    # teardown instead of a live task under a deleted project.
                    # ``allow_settled_target`` for a settled-but-LIVE root
                    # (GR6-1c): a durably-completed root still in RUNNING is a
                    # legitimate cascade root — without the flag no intent is
                    # minted and a crash leaves its children unfenced.
                    from ouroboros.cancel_intents import SCOPE_CASCADE, request_cancel

                    request_cancel(
                        drive_root, task_id, reason=f"project {project_id} deleted",
                        source="project_delete", scope=SCOPE_CASCADE,
                        allow_settled_target=task_has_live_ownership(task_id),
                    )
                except Exception:
                    # FAIL-CLOSED (AR2-1): no teardown without the durable intent.
                    # The task stays live this round; if the intent write keeps
                    # failing the quiescence check below raises and the deletion
                    # FAILS visibly (fail_project_deletion) instead of tearing a
                    # tree down through an unfenced, unreplayable cancel.
                    q.log.warning(
                        "project-delete cancel intent failed for %s; skipping its "
                        "teardown this round", task_id, exc_info=True,
                    )
                    errors.append(f"{task_id}: cancel_intent_write_failed")
                    continue
                try:
                    q.cancel_task_by_id(task_id, cascade=True)
                except Exception as exc:
                    errors.append(f"{task_id}: {type(exc).__name__}: {exc}")
            first_pass = False
            # Wind-down (GR6-1c + GR7-3): re-check quiescence WITHOUT re-running
            # the cancel pass. A durably-settled task still in the live maps is
            # a worker/finalizer winding down (its own finalizer owns the
            # RUNNING-row removal) — bounded, so a wedged finalizer still fails
            # VISIBLY instead of spinning forever.
            wind_down_rounds = 0
            while True:
                remaining = _live_project_task_ids(drive_root, project_id)
                if not remaining:
                    complete_project_deletion(drive_root, project_id)
                    _broadcast_projects_changed(project_id, chat_id)
                    return
                stuck = {
                    tid for tid in remaining if not _settled_status(drive_root, tid)
                }
                if stuck:
                    if nonsettled_before and stuck >= nonsettled_before:
                        # The pass killed none of the genuinely-live tasks:
                        # re-entering it would loop on the same refusal.
                        detail = "; ".join(errors) if errors else "cancel_task_by_id left tasks live"
                        raise RuntimeError(
                            f"Project deletion did not quiesce ({', '.join(sorted(remaining))}): {detail}"
                        )
                    break  # stuck/new non-settled work → re-enter the cancel pass
                if wind_down_rounds >= _WIND_DOWN_MAX_ROUNDS:
                    detail = "; ".join(errors) if errors else "cancel_task_by_id left tasks live"
                    raise RuntimeError(
                        f"Project deletion did not quiesce ({', '.join(sorted(remaining))}): "
                        f"settled task(s) never left the live maps after "
                        f"{wind_down_rounds} wind-down re-checks; {detail}"
                    )
                wind_down_rounds += 1
                _time.sleep(_WIND_DOWN_SLEEP_SEC)
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


def _close_campaign_after_owner_stop(exclude_task_id: str = "") -> None:
    """GR3-3 owner-stop backstop: close the campaign once its live task settled.

    An INCOMPLETE ``/evolve off`` / ``toggle_evolution(False)`` deliberately
    leaves the campaign OPEN over the still-live evolution task (closing it
    would declare a clean terminal that did not happen); the durable
    ``evolution_owner_stopped`` state flag blocks new cycles meanwhile. Every
    evolution terminal routes through ``_handle_evolution_task_done``, so this
    runs at exactly the moment the deferred close becomes honest — and no-ops
    whenever the owner never stopped or the campaign is already terminal.
    Never raises.

    GR4-6: the close is gated on NO OTHER evolution task being live — the
    multi-live incomplete-stop shape settles ONE task at a time, and closing
    on the first terminal would declare a clean stop over the others.
    ``exclude_task_id`` names the task whose terminal is being processed (its
    RUNNING row is popped only later, by ``_finish_task_done_dispatch``).
    """
    try:
        from supervisor.evolution_lifecycle import (
            _read_evolution_campaign,
            complete_evolution_campaign,
        )
        from supervisor.state import control_is, load_state

        campaign = _read_evolution_campaign()
        # A KNOWN owner stop or a recorded stop intent (#1307); an unknown flag closes nothing.
        if not (control_is(load_state(), "evolution_owner_stopped", True)
                or isinstance(campaign.get("stop_intent"), dict)):
            return
        if campaign.get("status") not in {"active", "paused"}:
            return
        from supervisor.queue import PENDING, RUNNING, _queue_lock

        with _queue_lock:
            live = [
                str(task.get("id") or "")
                for task in PENDING
                if isinstance(task, dict) and str(task.get("type") or "") == "evolution"
            ] + [
                str(tid)
                for tid, meta in RUNNING.items()
                if isinstance(meta, dict)
                and isinstance(meta.get("task"), dict)
                and str(meta["task"].get("type") or "") == "evolution"
            ]
        live = [tid for tid in live if tid and tid != str(exclude_task_id or "")]
        if live:
            log.info(
                "owner-stop campaign close deferred: evolution task(s) still live: %s",
                live,
            )
            return
        complete_evolution_campaign(
            "owner stop completed after the live evolution task settled",
            status="stopped",
        )
    except Exception:
        log.debug("owner-stop campaign close backstop failed", exc_info=True)


def reconcile_terminal_task_projections(drive_root, task_id: str) -> None:
    """One task-done seam for the per-task owner-control projections.

    Each domain keeps its own ``reconcile_terminal`` in its own module
    (owner_hurry, owner_quiz); this thin coordinator exists so
    ``supervisor/events.py`` — far past the module size gate — carries one
    call instead of one try/except block per domain. Every leg is fail-soft:
    a projection that cannot reconcile never blocks the terminal dispatch.
    """
    import logging

    log = logging.getLogger(__name__)
    try:
        # §19.7.2 item 5: a hurry the worker never drained loses the terminal
        # race honestly — not_applied_before_terminal.
        from ouroboros.owner_hurry import reconcile_terminal

        reconcile_terminal(drive_root, str(task_id))
    except Exception:
        log.debug("owner_hurry terminal reconcile failed for %s", task_id, exc_info=True)
    try:
        # #Q-2b structural expiry (owner decision 30=A): every still-open quiz
        # dies with its author, and the already-rendered cards learn it live.
        from ouroboros.owner_quiz import reconcile_terminal as quiz_reconcile

        expired = quiz_reconcile(drive_root, str(task_id))
        if expired:
            from supervisor.message_bus import get_bridge

            for quiz_id in expired:
                get_bridge().send_quiz_state(quiz_id, str(task_id), "expired_terminal")
    except Exception:
        log.debug("owner_quiz terminal reconcile failed for %s", task_id, exc_info=True)


# The exact-continuation grant lifecycle (#1196) lives in its own owner module;
# ``grant_exact_budget_resume`` is used above, and ``revoke_exact_budget_resume``
# is still addressed on THIS surface by ``worker_assignment`` (the same shape
# ``supervisor.queue`` uses for this file).
from supervisor.budget_resume import (  # noqa: E402, F401 -- intentional public re-export
    grant_exact_budget_resume,
    revoke_exact_budget_resume,
)
