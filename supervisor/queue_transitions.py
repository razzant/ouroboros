"""Queue-owned lifecycle TRANSITIONS that are not cancellation custody.

Extracted from ``supervisor/task_lifecycle.py`` for the module-size gate: the
acceptance FENCE, explicit BUDGET resume, fenced PROJECT deletion, and shared
live-subtree / timeout-retry lineage views. One-way dependency: the queue is
reached lazily; nothing is imported from ``task_lifecycle``.
"""

from __future__ import annotations

import logging
import pathlib
import threading
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

_PROJECT_DELETE_WORKERS_LOCK = threading.Lock()
_PROJECT_DELETE_WORKERS: set[tuple[str, str]] = set()


def _queue_module():
    from supervisor import queue

    return queue


def _acceptance_fence_owner_dead_locked(q: Any, fence: Dict[str, Any]) -> bool:
    """True when the fence's recorded owner task is provably not live: not
    RUNNING, not PENDING (a same-id retry waiting behind the fence), not in the
    reaper's not-yet-provably-dead registry (a wedged kill holds the task out of
    RUNNING while the worker may still live), and not the in-process direct-chat
    lane currently running it (never in RUNNING/PENDING).  Caller holds ``q._queue_lock``."""
    owner = str(fence.get("task_id") or "").strip()
    if not owner:
        return False  # no owner recorded: not provably dead
    if owner in q.RUNNING:
        return False
    for task in q.PENDING:
        if isinstance(task, dict) and str(task.get("id") or "") == owner:
            return False
    from supervisor import workers
    from supervisor.task_reaper import task_reaping_in_progress

    if task_reaping_in_progress(owner):
        return False
    busy, live_task_id, _activity = workers.chat_turn_liveness()
    return not (busy and str(live_task_id or "") == owner)


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
    reconcile_event = None  # emitted AFTER lock release: file IO under the lock stalls every queue reader on a slow FS
    with q._queue_lock:
        if action == "begin":
            if not root_task_id:
                return {"ok": False, "status": "error", "error": "missing root_task_id"}
            existing = q.ACCEPTANCE_FENCES.get(root_task_id)
            re_adopted = reconciled_dead_owner = False
            if isinstance(existing, dict) and str(existing.get("token") or "") != token:
                if str(existing.get("status") or "") != "active":
                    # A sealed fence terminalizes with its task on task_done.
                    return {"ok": False, "status": "error",
                            "error": f"acceptance fence already sealed for root {root_task_id}"}
                if _acceptance_fence_owner_dead_locked(q, existing):
                    # Dead-owner reconcile: the new owner's fresh row replaces the
                    # leaked fence; the audit event is emitted after lock release.
                    reconcile_event = {
                        "ts": utc_now_iso(), "type": "acceptance_fence_dead_owner_reconciled", "root_task_id": root_task_id,
                        "dead_task_id": str(existing.get("task_id") or ""), "new_task_id": str(task_id or root_task_id),
                    }
                    existing = None
                    reconciled_dead_owner = True
                elif str(existing.get("task_id") or "") != str(task_id or root_task_id):
                    return {
                        "ok": False, "status": "error",
                        "error": f"acceptance fence has a different live owner for root {root_task_id}",
                    }
                else:
                    # Idempotent re-adoption: a worker that lost its token (ack timeout
                    # on a slow drive) adopts the EXISTING fence instead of a paid retry spin.
                    re_adopted = True
            if isinstance(existing, dict):
                row = existing
            else:
                row = q.ACCEPTANCE_FENCES[root_task_id] = {
                    "token": token, "root_task_id": root_task_id, "task_id": str(task_id or root_task_id), "status": "active",
                    "opened_at": utc_now_iso(), "owner_message_generation": 0,
                }
            result = {
                "ok": True, "status": "active", "root_task_id": root_task_id, "token": str(row.get("token") or token),
                "owner_message_generation": int(row.get("owner_message_generation") or 0),
                "queue_descendants": _live_descendants_locked(q, root_task_id, exclude_task_id=str(task_id or root_task_id)),
            }
            if re_adopted:
                result["re_adopted"] = True
            if reconciled_dead_owner:
                result["reconciled_dead_owner"] = True
        else:
            matched_root = next(
                (rid for rid, row in q.ACCEPTANCE_FENCES.items() if str(row.get("token") or "") == token), "")
            if not matched_root:
                return {"ok": False, "status": "error", "error": "unknown acceptance fence token"}
            row = q.ACCEPTANCE_FENCES[matched_root]
            if action == "inspect":
                return {
                    "ok": True, "status": str(row.get("status") or "active"), "root_task_id": matched_root, "token": token,
                    "owner_message_generation": int(row.get("owner_message_generation") or 0),
                    "queue_descendants": _live_descendants_locked(
                        q, matched_root, exclude_task_id=str(row.get("task_id") or matched_root)),
                }
            normalized_outcome = str(outcome or "").strip().lower()
            if normalized_outcome == "revision":
                q.ACCEPTANCE_FENCES.pop(matched_root, None)
                result = {"ok": True, "status": "released", "root_task_id": matched_root, "token": token}
            elif (expected_generation is not None
                    and int(row.get("owner_message_generation") or 0) != int(expected_generation)):
                q.ACCEPTANCE_FENCES.pop(matched_root, None)
                result = {
                    "ok": True, "status": "released", "root_task_id": matched_root, "token": token, "generation_mismatch": True,
                    "expected_generation": int(expected_generation), "owner_message_generation": int(row.get("owner_message_generation") or 0),
                }
            else:
                row["status"] = "sealed"
                row["outcome"] = normalized_outcome or "terminal"
                row["sealed_at"] = utc_now_iso()
                result = {"ok": True, "status": "sealed", "root_task_id": matched_root, "token": token}
    if reconcile_event is not None:
        try:
            q.append_jsonl(q.DRIVE_ROOT / "logs" / "supervisor.jsonl", reconcile_event)
        except Exception:
            log.warning("Failed to persist %s", reconcile_event["type"], exc_info=True)
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
    (and the snapshot restore would resurrect it after a restart).
    """
    q = _queue_module()
    if not isinstance(task, dict):
        return False
    root_id = str(task.get("root_task_id") or task.get("id") or "").strip()
    if not root_id:
        return False
    with q._queue_lock:
        if root_id not in q.BUDGET_ROOT_FENCES:
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
        live_roots = {
            str(t.get("root_task_id") or t.get("id") or "")
            for t in pending if isinstance(t, dict)
        }
        orphaned = [root for root in list(fences) if root not in live_roots]
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
    fence_map = _queue_module().BUDGET_ROOT_FENCES if fences is None else fences
    root_id = str((task or {}).get("root_task_id") or (task or {}).get("id") or "")
    fence = fence_map.get(root_id)
    if isinstance(fence, dict) and str(fence.get("status") or "") in {"active", "paused"}:
        return fence
    return None


def _drop_acceptance_fences(event_type: str, matches: Any, **extra: Any) -> List[str]:
    """Drop every fence row where ``matches(q, row)`` holds; audit + persist.
    Every teardown seam (reaper, watchdog sweep) funnels here; persist is best-effort."""
    q = _queue_module()
    with q._queue_lock:
        roots = [root for root, row in q.ACCEPTANCE_FENCES.items() if isinstance(row, dict) and matches(q, row)]
        for root in roots:
            q.ACCEPTANCE_FENCES.pop(root, None)
    if roots:
        try:
            q.append_jsonl(
                q.DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {"ts": utc_now_iso(), "type": event_type, "root_task_ids": roots, **extra},
            )
            q.persist_queue_snapshot(reason=event_type)
        except Exception:
            log.warning("Failed to persist %s", event_type, exc_info=True)
    return roots


def release_acceptance_fence_for_dead_owner(task_id: str) -> bool:
    """Release fences OWNED by one confirmed-dead task (reaper seam): matched on the
    fence's recorded owner id, so a reaped child never drops its still-reviewing
    root's fence.  Called after confirmed death, BEFORE retry admission."""
    task_id = str(task_id or "").strip()
    if not task_id:
        return False
    return bool(_drop_acceptance_fences(
        "acceptance_fence_owner_reaped",
        lambda _q, row: str(row.get("task_id") or "") == task_id,
        task_id=task_id,
    ))


def gc_acceptance_fences_for_dead_owners() -> List[str]:
    """Watchdog sweep dropping fences whose owner is provably dead (task_done
    clears a fence; a worker killed mid-review never sends one)."""
    return _drop_acceptance_fences(
        "acceptance_fence_dead_owner_gc", _acceptance_fence_owner_dead_locked,
    )


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
    if self_running and not _settled_status(q.DRIVE_ROOT, task_id):
        return True
    if any(tid and not _settled_status(q.DRIVE_ROOT, tid) for tid in descendants):
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


def _live_retry_target_locked(q: Any, task_id: str) -> Tuple[str, str]:
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

    from ouroboros.task_results import load_task_result, resolve_task_lineage

    requested_result = load_task_result(q.DRIVE_ROOT, requested, strict=True)
    requested_shape = (
        requested_result
        if isinstance(requested_result, dict)
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
        current = load_task_result(q.DRIVE_ROOT, current_id, strict=True) or {}
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
        successor = load_task_result(q.DRIVE_ROOT, successor_id, strict=True) or {}
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
            (load_task_result(q.DRIVE_ROOT, current_id, strict=True) or {}).get("status")
            or ""
        )
        if leaf_status in SETTLED_STATUSES:
            return current_id, leaf_status
    return requested, ""


def task_has_live_ownership(task_id: str) -> bool:
    """Whether live PHYSICAL ownership remains for this task: a RUNNING row or
    a busy worker slot (GR6-1, the one predicate behind the class rule).

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
    with q._queue_lock:
        if task_id in q.RUNNING:
            return True
        if any(
            worker.busy_task_id == task_id for worker in workers.WORKERS.values()
        ):
            return True
        try:
            retry_target, _retry_settled_status = _live_retry_target_locked(q, task_id)
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
