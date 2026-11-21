"""The assignment pass checks saved sleeps and technical pauses outside Q.
Sleep readiness is recorded once. A technical wait re-reads its original source
at entry and on this pass. Both use the exact grant and the same control vetoes.
The snapshot keeps its proposed handoff blocked until the pause row CAS lands.
"""
from __future__ import annotations

import logging
import pathlib
import time
from types import SimpleNamespace
from typing import Any, Dict, List

from ouroboros.store import acquisition_budget

log = logging.getLogger(__name__)

SLEEP_WAKE = "sleep_wake"


def _sleepers(q: Any, only: str) -> List[Dict[str, Any]]:
    with q._queue_lock:
        return [dict(task) for task in q.PENDING if isinstance(task, dict)
                and isinstance(task.get("_budget_pause"), dict)
                and (not only or task.get("id") == only)
                and task["_budget_pause"].get("reason") in {"authority", "sleep"}]


@acquisition_budget(0)
def authority_readable(task, row, root):
    """Re-read the named fault. A row from an older writer may have no typed cause."""
    from ouroboros import owner_pause, queue_observation, usage_admission

    cause = str(row.get("cause") or "").partition(":")[0]
    task_id = str(task.get("id") or "")
    tree = str(task.get("root_task_id") or task_id)
    try:
        if cause in {"dispatch_fence_unreadable", "dispatch_fence_invalid"}:
            document = queue_observation.dispatch_probe(root)
            return document is not None and queue_observation.valid_dispatch_facts(document)
        if cause in {"read_failed", "malformed_binding"}:
            return bool(usage_admission.task_billing_fields(task, tree, None, root))
        if cause == "start_custody_unreadable":
            return bool(usage_admission.settlement_billing_fields(root, task_id, tree))
        if cause == "owner_launch_authority_unavailable":
            with owner_pause.launch_lock(root, tree, wait=False):
                return True
        if cause == "owner_pause_authority_unreadable":
            return not owner_pause.member_fence(SimpleNamespace(task_id=task_id, root_task_id=tree,
                                                                budget_drive_root=root, drive_root=task.get("drive_root") or root,
                                                                task_metadata=task.get("metadata"), task_lifecycle_bound=True))
    except Exception:
        log.debug("Technical wait of %s still lacks %s", task_id, cause, exc_info=True)
    return False


def _readiness(q: Any, task: Dict[str, Any], row: Dict[str, Any]) -> str:
    from ouroboros.model_sleep import wake_reason

    root = pathlib.Path(task.get("budget_drive_root") or q.DRIVE_ROOT)
    ctx = SimpleNamespace(
        task_id=str(task.get("id") or ""), root_task_id=str(task.get("root_task_id") or task.get("id") or ""),
        drive_root=str(task.get("drive_root") or task.get("child_drive_root") or root),
        budget_drive_root=str(root), task_attempt=int(task.get("_attempt") or 1),
        _loop_mailbox_seen_ids=set(row.get("sleep_seen") or ()), task_metadata=task.get("metadata") or {})
    with acquisition_budget(0):  # the next assignment tick retries a contended read; this one never stalls
        return wake_reason(ctx, row["sleep"])


def _vetoed(q: Any, task: Dict[str, Any]) -> str:
    """Why a ready sleeper may not wake now (``""`` = it may)."""
    from ouroboros.owner_pause import member_fence
    from ouroboros.work_stop import continuation_refusal, current_stop, writer_wait

    observed = dict(task)
    if (observed.get("_budget_pause") or {}).get("reason") == "authority":
        observed.pop("_budget_pause")
    stop = current_stop(observed, {"budget": q.BUDGET_ROOT_FENCES})
    if stop or writer_wait(task) or continuation_refusal(task):
        return str(stop.get("reason") or stop.get("cause") or "continuation_unavailable")
    fence = member_fence(SimpleNamespace(
        task_id=str(task.get("id") or ""), root_task_id=str(task.get("root_task_id") or task.get("id") or ""),
        budget_drive_root=str(task.get("budget_drive_root") or q.DRIVE_ROOT)))
    return "owner_pause" if fence else ""


def wake_ready_sleepers(q: Any = None, *, only: str = "") -> List[Dict[str, Any]]:
    """One pass: record readiness, then grant the ready and unvetoed (typed outcomes)."""
    from ouroboros.budget_pause import BudgetPauseSuperseded, budget_pause_row, set_budget_pause
    from ouroboros.store_hot_facts import read_ahead, serving
    from supervisor.budget_resume import grant_exact_sleep_resume

    if q is None:
        from supervisor import queue as q
    from supervisor.restart_retention import publish_pending_parks
    if not only:
        publish_pending_parks(q, "budget_pause_exact_continuation")
    outcomes: List[Dict[str, Any]] = []
    batch = []
    sleepers = _sleepers(q, only)
    roots = {id(task): pathlib.Path(task.get("budget_drive_root") or q.DRIVE_ROOT) for task in sleepers}
    facts = read_ahead([(roots[id(task)], str(task.get(key) or task.get("id") or ""))
                        for task in sleepers for key in ("id", "root_task_id")]) if sleepers else {}
    try:
        for task in sleepers:
            task_id = str(task.get("id") or "")
            root = roots[id(task)]
            try:
                with serving(facts):
                    row = budget_pause_row(root, task_id)
                if row.get("state") != "paused":
                    continue
                if row.get("reason") == "authority":
                    if not authority_readable(task, row, root):
                        continue
                    ready = {"reason": row["cause"]}
                elif row.get("reason") == "sleep" and isinstance(row.get("sleep"), dict):
                    ready = row.get("sleep_ready") or {}
                else:
                    continue
                if not ready:
                    reason = _readiness(q, task, row)
                    if not reason:
                        continue
                    ready = {"reason": reason, "observed_at": time.time()}
                    # Readiness was read off-lock: a Resume, revocation or newer
                    # pause that landed meanwhile wins; the next pass re-observes.
                    set_budget_pause(root, task_id, {**row, "sleep_ready": ready},
                                     expected_pause_id=str(row.get("pause_id") or ""),
                                     expected_state=str(row.get("state") or ""),
                                     expected_grant_id=str((row.get("grant") or {}).get("grant_id") or ""))
            except BudgetPauseSuperseded:
                log.debug("Sleep readiness of %s superseded by a newer pause row", task_id, exc_info=True)
                continue
            except Exception:
                log.warning("Sleep readiness of %s unreadable; it keeps sleeping", task_id, exc_info=True)
                continue
            from ouroboros.budget_pause import observe_task_runs

            external = observe_task_runs(root, task_id, reason="sleep_wake_check", request_stop=False)
            with q._queue_lock:
                live = next((item for item in q.PENDING if str(item.get("id") or "") == task_id), None)
                if live is None or not isinstance(live.get("_budget_pause"), dict):
                    continue
                with serving(facts, strict=True):  # the tree's fence as the pass read it, never a store under Q
                    veto = _vetoed(q, live)
                if veto:
                    outcomes.append({"task_id": task_id, "ok": False, "error": "sleep_wake_vetoed", "veto": veto,
                                     "ready": ready})
                    continue
                pause = live["_budget_pause"]
            granted = grant_exact_sleep_resume(live, pause, external=external, batch=batch)
            granted.update(task_id=task_id, ready=ready)
            outcomes.append(granted)
    finally:
        if batch:
            _publish_batch(q, batch)
    return outcomes


def _publish_batch(q, batch):
    try:
        saved = q.publish_queue_snapshot(reason="budget_exact_resume_granted")
    except Exception:
        log.warning("Could not publish the proposed wake grants", exc_info=True)
        saved = False
    for result, finish in batch:
        result.update(finish(saved))


def publish_wake_grant(q, task, pause, row, handoff, record_grant, granted, batch):
    """The snapshot carries the proposed handoff beside its blocking pause. Only the row CAS opens it."""
    from supervisor.budget_resume import _sleep_policy_allows
    from supervisor.queue_publication import mark_dirty_locked

    def current():
        return any(item is task for item in q.PENDING) and task.get("_budget_pause") is pause

    with q._queue_lock:
        if not current() or task.get("_budget_pause_resume"):
            return {"ok": False, "error": "sleep_wake_vetoed"}
        task["_budget_pause_resume"] = handoff
        mark_dirty_locked("budget_exact_resume_proposed")

    def finish(saved):
        try:
            if not saved:
                return {"ok": False, "error": "snapshot_not_persisted"}
            root = pathlib.Path(task.get("budget_drive_root") or q.DRIVE_ROOT)
            if not _sleep_policy_allows(task, row, root):
                return {"ok": False, "error": "sleep_wake_vetoed"}
            if refusal := record_grant():
                return refusal
            with q._queue_lock:
                confirmed = current() and task.get("_budget_pause_resume") is handoff
                if confirmed:
                    task.pop("_budget_pause")
                    task["budget_resumed_at"] = handoff["granted_at"]
                    mark_dirty_locked("budget_exact_resume_granted")
            if confirmed:
                return granted()
            from ouroboros.budget_pause import set_budget_pause
            set_budget_pause(root, task["id"], row, expected_pause_id=handoff["pause_id"],
                             expected_state="resume_granted", expected_grant_id=handoff["grant_id"])
            return {"ok": False, "error": "selection_authority_changed"}
        except Exception as exc:
            return {"ok": False, "error": "grant_not_recorded", "detail": str(exc)[:200]}
        finally:
            with q._queue_lock:
                if task.get("_budget_pause") is pause and task.get("_budget_pause_resume") is handoff:
                    task.pop("_budget_pause_resume", None)
                    mark_dirty_locked("budget_exact_resume_withdrawn")

    result = {}
    if batch is None:
        _publish_batch(q, [(result, finish)])
    else:
        batch.append((result, finish))
    return result
