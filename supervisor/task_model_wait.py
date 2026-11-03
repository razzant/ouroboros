"""Supervisor projection of live model waits; the queue still owns task lifetime."""

from __future__ import annotations

import copy
from typing import Any
from ouroboros.model_wait import quota_waited_seconds  # noqa: F401 -- historical supervisor import surface


def model_waiting(meta: dict) -> bool:
    """A typed wait spares idle detection; auth still spends execution time."""
    task = meta.get("task") or {}
    attempt = int(meta.get("attempt") or task.get("_attempt") or 1)
    return any(isinstance(row, dict) and row.get("state") == "waiting"
               and row.get("task_attempt") == attempt
               for row in (task.get("model_waits") or {}).values())


def handle_task_model_wait(event: dict, ctx: Any) -> None:
    """Apply a current-attempt, monotonic worker fact without minting a task."""
    from supervisor.queue import _queue_lock
    from supervisor.log_addressing import address_ctx_event

    task_id, wait_id = str(event.get("task_id") or ""), str(event.get("wait_id") or "")
    revision, attempt = event.get("revision"), event.get("task_attempt")
    if (not task_id or not wait_id or not isinstance(revision, int) or isinstance(revision, bool)
            or revision <= 0 or not isinstance(attempt, int) or isinstance(attempt, bool)
            or event.get("state") not in {"waiting", "resolved"}):
        return
    payload = copy.deepcopy(event)
    row = {key: value for key, value in payload.items()
           if key not in {"type", "ts", "task_id", "quota_clock", "is_progress"}}
    operation_event = False
    with _queue_lock:
        meta = ctx.RUNNING.get(task_id)
        if isinstance(meta, dict):
            task = meta.get("task") or {}
            expected = int(meta.get("attempt") or task.get("_attempt") or 1)
            if attempt != expected:
                return
            waits = dict(task.get("model_waits") or {})
            previous = waits.get(wait_id) or {}
            if int(previous.get("revision") or 0) >= revision:
                return
            waits[wait_id] = row
            task["model_waits"] = waits
            # The row stays visible for any owner (an independent review operation
            # included); only the author's own owner moves the author's execution clock.
            # A reviewer's quota or result wait is never an author exclusion (#939).
            author_clock = payload.get("model_wait_owner_id", "") == str(task.get("model_wait_owner_id") or "")
            previous_clock = meta.get("model_wait_quota_clock") or {}
            clock = payload.get("quota_clock") or {}
            if author_clock and int(clock.get("revision") or 0) > int(previous_clock.get("revision") or 0):
                meta["model_wait_quota_clock"] = clock
        else:
            # A finished managed task cannot be resurrected by a late waiter.
            # Direct turns have no queue row and retain their own live owner.
            from supervisor.workers import direct_chat_turn

            if direct_chat_turn(task_id) is None:
                from ouroboros.post_task_checkpoint import post_task_model_wait

                owner = post_task_model_wait(ctx.DRIVE_ROOT, task_id)
                if owner is not None and payload.get("model_wait_owner_id", "") == owner.owner_id:
                    with owner.lock:
                        current = owner.waits.get(wait_id) or {}
                        if owner.closed or attempt != owner.attempt or current.get("revision") != revision:
                            return
                    payload["chat_id"] = owner.task.get("chat_id")
                elif isinstance(payload.get("review_operation"), dict):
                    operation_event = True
                else:
                    return
    if operation_event:
        # The author ended; its paid review operation may still be live. Proven
        # off the queue lock: it reads the canonical row and the controller process.
        from ouroboros.review_operation import review_operation_event_admitted

        if not review_operation_event_admitted(ctx.DRIVE_ROOT, payload):
            return
        payload["chat_id"] = payload["review_operation"].get("chat_id")
    address_ctx_event(ctx, payload)
    from ouroboros import dialogue_store

    dialogue_store.append_quietly(ctx.DRIVE_ROOT, "progress", payload)
    ctx.bridge.push_log(payload)


EVENT_HANDLERS = {"task_model_wait": handle_task_model_wait}
