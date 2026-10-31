"""The sleep-wake policy for COLD model sleeps.

A cold sleep is parked like an exact pause (``_budget_pause`` marker, reason
``sleep``). This pass, run
from the ordinary assignment tick (no second scheduler), reads readiness from
the canonical records (``model_sleep.wake_reason`` over the task's mailbox,
the selected tasks' results, its custody rows and the chosen time) OUTSIDE the
queue lock, records it once on the same durable pause row
(``sleep_ready``), and — under the queue lock — mints the single-use grant
through the distinct policy entry (``grant_exact_sleep_resume``) with the typed
``sleep_wake`` authority. That authority is the sleeper's own readiness, not
the owner's Resume and not its root's model selection; it is VETOED, keeping
the readiness recorded, by any hold on the row (an owner Restart or Panic
hold), by a closed owner Pause fence over its tree and by a root that is
itself paused — and every money/deadline/lifetime/custody refusal of the
grant owner still applies. Readiness is never permission. The off-lock record
compares the pause id, state and grant it read, so a Resume, revocation or
newer pause landing meanwhile wins and the next pass re-observes.
"""

from __future__ import annotations

import logging
import pathlib
import time
from types import SimpleNamespace
from typing import Any, Dict, List

log = logging.getLogger(__name__)

SLEEP_WAKE = "sleep_wake"


def _sleepers(q: Any) -> List[Dict[str, Any]]:
    with q._queue_lock:
        return [dict(task) for task in q.PENDING if isinstance(task, dict)
                and isinstance(task.get("_budget_pause"), dict)
                and task["_budget_pause"].get("reason") == "sleep"
                and isinstance(task["_budget_pause"].get("sleep"), dict)]


def _readiness(q: Any, task: Dict[str, Any], row: Dict[str, Any]) -> str:
    from ouroboros.model_sleep import wake_reason
    from ouroboros.store import acquisition_budget

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

    stop = current_stop(task, {"budget": q.BUDGET_ROOT_FENCES})
    if stop or writer_wait(task) or continuation_refusal(task):
        return str(stop.get("reason") or stop.get("cause") or "continuation_unavailable")
    fence = member_fence(SimpleNamespace(
        task_id=str(task.get("id") or ""), root_task_id=str(task.get("root_task_id") or task.get("id") or ""),
        budget_drive_root=str(task.get("budget_drive_root") or q.DRIVE_ROOT)))
    return "owner_pause" if fence else ""


def wake_ready_sleepers(q: Any = None) -> List[Dict[str, Any]]:
    """One pass: record readiness, then grant the ready and unvetoed (typed outcomes)."""
    from ouroboros.budget_pause import BudgetPauseSuperseded, budget_pause_row, set_budget_pause
    from supervisor.budget_resume import grant_exact_sleep_resume

    if q is None:
        from supervisor import queue as q
    outcomes: List[Dict[str, Any]] = []
    for task in _sleepers(q):
        task_id = str(task.get("id") or "")
        root = pathlib.Path(task.get("budget_drive_root") or q.DRIVE_ROOT)
        try:
            row = budget_pause_row(root, task_id)
            if not isinstance(row.get("sleep"), dict):
                continue
            ready = row.get("sleep_ready") if isinstance(row.get("sleep_ready"), dict) else {}
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
            veto = _vetoed(q, live)
            if veto:
                outcomes.append({"task_id": task_id, "ok": False, "error": "sleep_wake_vetoed", "veto": veto,
                                 "ready": ready})
                continue
            granted = grant_exact_sleep_resume(live, live["_budget_pause"], external=external)
        outcomes.append({"task_id": task_id, "ready": ready, **granted})
    return outcomes
