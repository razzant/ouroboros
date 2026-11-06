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
    """The project pool's admission as the server, its only owner, sees it.

    In the server (an inline executor) that is the live pool table. A conversation carrier owns no pool
    table (its own ``supervisor.workers`` is empty), so it reads the server's published facts: the queue
    document's pool members and disablement, and the durable update marker (the server's in-memory writer
    gate is not visible from here; its transaction marker is). An unreadable document is no availability.
    """
    from ouroboros.utils import in_worker_process

    if not in_worker_process():
        from supervisor.workers import worker_pool_admission_state

        return worker_pool_admission_state()
    from ouroboros.config import DATA_DIR
    from ouroboros.queue_observation import load_queue_document, pool_observation_status

    document = load_queue_document(DATA_DIR)
    # An unmeasured pool is no availability and no absence: its reason stays ``unknown``, its count null.
    unknown = pool_observation_status(document) == "unknown"
    started = int(document.get("worker_total") or 0) + int(document.get("lane_carrier_total") or 0) > 0
    disabled = ("queue_document_unreadable" if "_snapshot_invalid" in document or "_snapshot_missing" in document
                else "unknown" if unknown else str(document.get("worker_pool_disabled_reason") or ""))
    try:
        from supervisor.update_merge import active_update_tx

        tx = active_update_tx()
        update = f"managed_update_tx:{tx.get('phase') or 'unknown'}" if tx else ""
    except Exception:
        update = "managed_update_tx:unreadable"
    reason = disabled or update or ("" if started else "no_workers")
    return {"available": not reason, "reason_code": "" if not reason else "worker_pool_unavailable",
            "disabled_reason": reason, "worker_count": None if unknown else int(document.get("worker_total") or 0),
            "source": "published"}


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
    """Arm the handoff: the same work and id continue on the project pool after this tool batch.

    The fact is the carrier's lane, never the process: a turn carried by the conversation
    lane (a worker or an inline executor) holds the owner's foreground; a pooled attempt
    already runs in the background. Detaching sets ``detached_at/by`` and releases the
    foreground reply through the cold-sleep continuation owner (``model_sleep``).
    """
    from ouroboros.model_sleep import MODE_COLD, ColdSleepBlocked, request_sleep
    from ouroboros.task_carrier import LANE_PROJECT
    from ouroboros.utils import utc_now_iso

    owner = (getattr(ctx, "task_metadata", None) or {}).get("_execution_owner") or {}
    if owner.get("kind") == "pooled" or owner.get("lane") == LANE_PROJECT:
        return {"ok": True, "status": "already_in_background", "task_id": ctx.task_id,
                "note": "This work already runs on a pooled executor; nothing changed."}
    if owner_question_pending(ctx):
        return {"ok": False, "error": "owner_question_pending", "note": "Owner-question waits stay warm."}
    from ouroboros.native_task_wait import armed_conflict

    if conflict := armed_conflict(ctx):  # an owed native task wait is never overwritten
        return {"ok": False, "error": "task_wait_owed", "note": conflict}
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
