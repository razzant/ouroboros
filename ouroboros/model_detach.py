"""Explicit same-id handoff over the existing cold-sleep continuation owner."""

from __future__ import annotations

import time
from typing import Any


LOST_HANDLES = "Process-local browser and REPL sessions are lost; reopen them and re-read needed state."


def is_detach(sleep: Any) -> bool:
    return isinstance(sleep, dict) and sleep.get("intent") == "continue_in_background"


def detached_fields(task: dict, pause: dict) -> dict:
    """Record the fact only with a saved continuation, preserving its first date."""
    sleep = pause.get("sleep") or {}
    if not is_detach(sleep) or not pause.get("source_ref"):
        return {}
    metadata = task.get("metadata") or {}
    at = task.get("detached_at") or metadata.get("detached_at") or sleep["detached_at"]
    by = task.get("detached_by") or metadata.get("detached_by") or sleep["detached_by"]
    return {"detached_at": at, "detached_by": by,
            "metadata": {**metadata, "detached_at": at, "detached_by": by}}


def pool_available() -> dict:
    """Called only by an in-process head, against its actual supervisor pool."""
    from supervisor.workers import worker_pool_admission_state

    return worker_pool_admission_state()


def owner_question_pending(ctx: Any) -> bool:
    requested = str(getattr(ctx, "_owner_wait_requested", "") or "")
    sleep = getattr(ctx, "_model_sleep", None) or {}
    return bool(requested and requested != "sleep:" + str(sleep.get("sleep_id") or ""))


def preserve_owner_question(ctx: Any, messages: list) -> None:
    """A question later in the same batch keeps its warm owner-wait owner."""
    if is_detach(getattr(ctx, "_model_sleep", None)) and owner_question_pending(ctx):
        ctx._model_sleep = None
        messages.append({"role": "user", "content": "[SYSTEM NOTICE]\nThe background handoff was withdrawn: "
                         "your owner question waits in this process."})


def arm_detach(ctx: Any, reason: str) -> dict:
    from ouroboros.model_sleep import MODE_COLD, ColdSleepBlocked, request_sleep
    from ouroboros.utils import in_worker_process, utc_now_iso

    owner = (getattr(ctx, "task_metadata", None) or {}).get("_execution_owner") or {}
    if in_worker_process() or owner.get("kind") == "pooled":
        return {"ok": True, "status": "already_in_background", "task_id": ctx.task_id,
                "note": "This work already runs on a pooled executor; nothing changed."}
    if owner_question_pending(ctx):
        return {"ok": False, "error": "owner_question_pending", "note": "Owner-question waits stay warm."}
    if not callable(getattr(ctx, "owner_wait_callback", None)):
        return {"ok": False, "error": "no_continuation_owner"}
    pool = pool_available()
    if not pool.get("available"):
        return {"ok": False, "error": "worker_pool_unavailable", "pool": pool,
                "note": "The dialogue stays in this process."}
    chosen = {"intent": "continue_in_background", "reason": str(reason or ""),
              "detached_at": time.time(), "detached_by": "mind", "wake_at": utc_now_iso(),
              "senders": [], "tasks": [], "runs": [], "any_mail": False}
    try:
        result = request_sleep(ctx, chosen, MODE_COLD)
    except ColdSleepBlocked as exc:
        return {"ok": False, "error": "detach_blocked", "blockers": exc.blockers}
    if result.get("reason") != "sleep_armed":
        return {**result, "ok": False, "error": "detach_control_pending"}
    return {**result, "ok": True, "status": "detach_armed", "task_id": ctx.task_id,
            "note": "The same work and id continue in a pooled worker after this tool batch. "
                    "It becomes eligible immediately, subject to capacity, leases and holds. " + LOST_HANDLES}
