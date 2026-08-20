"""Queue-owned lifecycle TRANSITIONS that are not cancellation custody.

Extracted from ``supervisor/task_lifecycle.py`` for the module-size gate (the
same boundary that produced ``terminal_delivery.py`` and
``delegate_containment.py``): custody grew when the Poltergeist cancel redesign
landed, and the three clusters here never touch it. They share one property —
each is a queue-owned transition of a task's ADMISSION or QUIESCENCE state,
driven entirely through the queue module:

- the acceptance FENCE (open/inspect/seal a root so no new subtask is admitted
  while its acceptance review runs),
- explicit BUDGET resume of a zero-dispatch task and its root latch,
- fenced PROJECT deletion: cancel the project's tree, then tombstone only after
  the tree is provably quiescent.

The dependency runs one way: this module reaches the queue lazily and imports
nothing from ``task_lifecycle``, which re-exports these names so
``supervisor.queue`` stays the single public import surface for callers.
"""

from __future__ import annotations

import logging
import pathlib
import threading
from typing import Any, Dict, List, Optional

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
            request_cancel(
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
            outcome = q.cancel_task_custody(task_id)
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
        return any(
            worker.busy_task_id == task_id for worker in workers.WORKERS.values()
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
        from supervisor.state import load_state

        if not bool(load_state().get("evolution_owner_stopped")):
            return
        if _read_evolution_campaign().get("status") not in {"active", "paused"}:
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
