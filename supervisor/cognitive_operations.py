"""Typed operation and wait facts shared by idle enforcement and reply preparation.

In-flight LLM, review, VLM and tool work is leased as typed ``cognitive_operation``
facts for the idle rail; display-only in_slot_wait marks classify what a call awaits,
not progress or permission. Leases are bounded by deadline and execution ceiling;
deadline, budget, cancellation and ceiling enforcement never depend on these facts.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger(__name__)


def _active_operation_progressing(meta: Dict[str, Any], now: float) -> bool:
    """Return whether a typed cognitive operation is physically in flight."""
    active = meta.get("active_operation_leases") if isinstance(meta, dict) else None
    if not isinstance(active, dict):
        return False
    live = False
    for operation_id, row in list(active.items()):
        try:
            until = float(row.get("until_ts") if isinstance(row, dict) else row)
        except (TypeError, ValueError):
            until = 0.0
        if until > now:
            if not isinstance(row, dict) or row.get("kind") != "in_slot_wait":
                live = True
        else:
            active.pop(operation_id, None)
    if not active:
        meta.pop("active_operation_leases", None)
    return live


def _handle_cognitive_operation(evt: Dict[str, Any], ctx: Any) -> None:
    """Track bounded operation/wait facts; only work leases spare the idle rail."""
    task_id = str(evt.get("task_id") or "")
    operation_id = str(evt.get("operation_id") or "")
    phase = str(evt.get("phase") or "").strip().lower()
    if not task_id or not operation_id or phase not in {"started", "finished", "failed"}:
        return
    running = getattr(ctx, "RUNNING", None)
    meta = running.get(task_id) if isinstance(running, dict) else None
    if not isinstance(meta, dict):
        return
    expected_attempt = meta.get("attempt")
    if expected_attempt is None:
        task = meta.get("task") if isinstance(meta.get("task"), dict) else {}
        expected_attempt = task.get("_attempt")
    supplied_attempt = evt.get("task_attempt")
    if supplied_attempt not in (None, ""):
        try:
            if int(supplied_attempt) != int(expected_attempt or 1):
                return
        except (TypeError, ValueError):
            return
    active = meta.get("active_operation_leases")
    if not isinstance(active, dict):
        active = {}
        meta["active_operation_leases"] = active
    if phase in {"finished", "failed"}:
        row = active.get(operation_id)
        if isinstance(row, dict):
            stored_attempt = row.get("task_attempt")
            if supplied_attempt not in (None, "") and stored_attempt not in (None, ""):
                try:
                    if int(stored_attempt) != int(supplied_attempt):
                        return
                except (TypeError, ValueError):
                    return
            for key in ("execution_id", "round_id", "slot_id"):
                supplied = str(evt.get(key) or "")
                stored = str(row.get(key) or "")
                if stored and not supplied:
                    return
                if supplied and stored and supplied != stored:
                    return
        if phase == "finished" and isinstance(row, dict) and row.get("kind") == "tool":
            # A tool call that physically completed is this task's own work, like a
            # completed model round (events_budget) or a narration line
            # (events_chat_delivery): the lease that spared the idle rail while the
            # call ran closes INTO a fresh progress stamp, so the round that follows
            # starts inside a full idle window instead of inheriting the time spent
            # in earlier tools. Stamped before the pop so no tick reads the task as
            # both lease-less and stale. Deadline, absolute ceiling, budget and
            # cancellation never consult this stamp.
            meta["last_progress_at"] = time.time()
        active.pop(operation_id, None)
        if not active:
            meta.pop("active_operation_leases", None)
        return
    now = time.time()
    try:
        requested_until = float(evt.get("lease_until") or 0.0)
    except (TypeError, ValueError):
        requested_until = 0.0
    from ouroboros.config import OPERATION_WINDOW_FALLBACK_SEC, resolve_task_execution_timeout_sec
    from ouroboros.deadline_utils import parse_deadline_ts
    from ouroboros.model_wait import execution_elapsed_seconds

    task = meta.get("task") if isinstance(meta.get("task"), dict) else {}
    ceiling = resolve_task_execution_timeout_sec(task)
    # A finite execution bound ends the lease where the task's remaining execution ends
    # (the timeout rail's own clock); without one the operation's own finite window bounds
    # it from now, so a lost terminal never spares the idle rail forever.
    hard_until = (now + max(0.0, float(ceiling) - execution_elapsed_seconds(meta, now)) if ceiling is not None
                  else now + float(OPERATION_WINDOW_FALLBACK_SEC))
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    deadline = parse_deadline_ts(task.get("deadline_at") or metadata.get("deadline_at"))
    if deadline is not None:
        hard_until = min(hard_until, deadline.timestamp())
    until = hard_until if requested_until <= 0 else min(requested_until, hard_until)
    if until > now:
        active[operation_id] = {
            "kind": str(evt.get("kind") or "cognitive"),
            "until_ts": until,
            "task_attempt": expected_attempt,
            "execution_id": str(evt.get("execution_id") or ""),
            "round_id": str(evt.get("round_id") or ""),
            "slot_id": str(evt.get("slot_id") or ""),
            "owner_id": str(evt.get("model_wait_owner_id") or ""),
            "wait_for_tasks": list(evt.get("wait_for_tasks") or ()) if evt.get("kind") == "in_slot_wait" else [],
        }


def _handle_review_late_result(evt: Dict[str, Any], ctx: Any) -> None:
    """Persist a late review settlement without changing the aggregate row."""
    payload = {"ts": evt.get("ts", utc_now_iso()), **{
        key: value for key, value in evt.items() if key != "ts"
    }}
    try:
        from supervisor.log_addressing import address_task_event

        address_task_event(getattr(ctx, "RUNNING", None), ctx.DRIVE_ROOT, payload)
    except Exception:
        log.debug("late review result addressing failed", exc_info=True)
    try:
        append_jsonl(ctx.DRIVE_ROOT / "logs" / "events.jsonl", payload)
    except Exception:
        log.debug("late review result persistence failed", exc_info=True)
    try:
        from ouroboros.tools.plan_ask import attach_late_reply

        attach_late_reply(ctx.DRIVE_ROOT, payload)
    except Exception:
        log.warning("historical plan review remains unresolved", exc_info=True)
    try:
        ctx.bridge.push_log(payload)
    except Exception:
        log.debug("late review result live projection failed", exc_info=True)


EVENT_HANDLERS = {
    "cognitive_operation": _handle_cognitive_operation,
    "review_late_result": _handle_review_late_result,
}
