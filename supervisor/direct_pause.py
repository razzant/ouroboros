"""Finish an inline exact park before releasing the direct activity registration."""

from __future__ import annotations

import logging
from types import SimpleNamespace

log = logging.getLogger(__name__)


def park_direct_pause(task: dict, event: dict) -> bool:
    from ouroboros.budget_pause import end_dispatch_fence
    from ouroboros.utils import append_jsonl, utc_now_iso
    from supervisor import queue as q, workers as pool
    from supervisor.events_budget import _handle_budget_pause, install_exact_budget_pause
    from supervisor.message_bus import get_bridge

    task_id = str(task["id"])
    checkpoint = event["resource_limit"].get("checkpoint") or {}
    try:
        shim = SimpleNamespace(
            RUNNING=pool.RUNNING, PENDING=pool.PENDING, WORKERS=pool.WORKERS, DRIVE_ROOT=pool.DRIVE_ROOT,
            sort_pending=q.sort_pending, persist_queue_snapshot=q.persist_queue_snapshot, bridge=get_bridge())
        if event["resource_limit"].get("before_task_start"):
            _handle_budget_pause(event, shim)
        else:
            install_exact_budget_pause(shim, task_id, checkpoint, evt=event, source="direct_turn_inline_park")
        append_jsonl(pool.DRIVE_ROOT / "logs" / "supervisor.jsonl", {
            "ts": utc_now_iso(), "type": "direct_turn_budget_pause_parked_inline",
            "task_id": task_id, "chat_id": event.get("chat_id"), "pause_id": checkpoint.get("pause_id")})
        return True
    except Exception:
        log.error("Direct turn %s could not be parked inline; its event takes the ordinary path",
                  task_id, exc_info=True)
        return False
    finally:
        end_dispatch_fence(task_id)


def recover_direct_pause(task: dict, error: BaseException | None = None) -> bool:
    """A raise after pausing cannot turn the same saved attempt into a terminal.

    A fresh input whose body never ran (``error.before_task_start``: its preparation failed, or a Stop
    or a bound ended its wait for storage) has no pause of its own and nothing to hold; its ordinary
    failure concludes it (False). A start that resumes saved work (``governing_continuation``) keeps
    that work whether or not its body ran, so the pause record decides, as it does after any body;
    one that cannot be read keeps the turn held for an explicit decision with no replay grant, as the
    unreadable authority it is (``work_stop`` ``authority``), never as work saved when the app stopped."""
    from ouroboros.budget_pause import budget_pause_row, death_returns_to_pause, pause_event, end_dispatch_fence
    from ouroboros.owner_wait import governing_continuation
    from supervisor import workers as pool

    try:
        if getattr(error, "before_task_start", False) and not governing_continuation(task):
            return False
        row = budget_pause_row(task.get("budget_drive_root") or pool.DRIVE_ROOT, str(task["id"]))
        if not death_returns_to_pause(row, int(task.get("_attempt") or 1)):
            return False
        if not row.get("source_ref"):
            from ouroboros.working_checkpoint import complete_pause_from_working
            row = complete_pause_from_working(task.get("budget_drive_root") or pool.DRIVE_ROOT,
                str(task["id"]), row, int(task.get("_attempt") or 1)) or row
        if row.get("source_ref"):
            event = pause_event(task, row)
            if not park_direct_pause(task, event):
                pool.get_event_q().put(event)
        else:
            retain_direct_hold(task, unsaved=True)
        return True
    except Exception:
        log.error("Direct turn %s: its pause evidence is unreadable; held for an explicit decision",
                  task["id"], exc_info=True)
        retain_direct_hold(task, unsaved=False)
        return True
    finally:
        end_dispatch_fence(str(task["id"]))


def retain_direct_hold(task: dict, *, unsaved: bool) -> None:
    """Keep a failed inline park visible on the existing queue, with no replay grant. ``unsaved``: a pause
    of this attempt is recorded without a saved source; otherwise its pause evidence was unreadable."""
    import pathlib
    import time
    from ouroboros.budget_pause import parkable_direct_task
    from supervisor import queue as q, workers as pool
    from supervisor.events_budget import HOLD_UNSAVED_PAUSE, hold_budget_row

    with q._queue_lock:
        if str(task["id"]) in pool.RUNNING or any(row.get("id") == task["id"] for row in pool.PENDING):
            return  # an already-published owner wins
        held = parkable_direct_task(task)
        held.setdefault("priority", 0)
        held.setdefault("queued_at", time.time())
        hold_budget_row(held, reason=HOLD_UNSAVED_PAUSE if unsaved else "authority",
                        detail=("Inline actor ended while pausing; saved continuation unavailable. Explicit control required."
                                if unsaved else "Its pause evidence was unreadable; whether it saved work is "
                                "unknown. Explicit control required."),
                        result_root=pathlib.Path(task.get("budget_drive_root") or pool.DRIVE_ROOT),
                        existing_result_only=True)
        pool.PENDING.append(held)
        q.sort_pending()
        try:
            if not q.persist_queue_snapshot(reason="direct_pause_recovery_held"):
                log.error("Direct pause hold of %s could not be snapshotted; live queue retains it", task["id"])
        except Exception:
            log.error("Direct pause hold snapshot failed for %s; live queue retains it", task["id"], exc_info=True)
