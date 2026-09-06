"""Emitting one control event, and waiting for its durable handler outcome.

A control tool states an intent; the supervisor decides. This module owns that
boundary: the serialization pre-check, the live queue with its deferred
``pending_events`` fallback, and the confirmation reads that turn an emission
into a receipt the tool may report — the task-result admission record for a
promotion, the exact chat annotation for a manual target or a steer. A tool
never reports work as scheduled on the strength of having emitted an event.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
from pathlib import Path
from typing import Any, Dict

from ouroboros.tool_policy import swarm_router_turn
from ouroboros.tools.registry import ToolContext
from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger(__name__)


# Guards parent-side shared ctx state mutated during (possibly parallel)
# schedule_subagent emission within one tool-call round. Process-local: a parent
# ctx is never shared across processes, so a threading.Lock is sufficient.


_SCHEDULE_EMIT_LOCK = threading.Lock()


_PROMOTE_CONFIRM_TIMEOUT_SEC = 15.0


_PROMOTE_CONFIRM_POLL_SEC = 0.05


def _emit_control_event(ctx: ToolContext, evt: Dict[str, Any]) -> str:
    """Emit a control event live when possible, preserving legacy fallback."""
    def _mark_typed_routing_action() -> None:
        event_type = str(evt.get("type") or "")
        if event_type not in {"promote_chat_to_task", "routing_manual_target", "steer_task"}:
            return
        # Keep a turn-local fact on the existing ToolContext so finalization can
        # expose the typed action on task_done. The supervisor receipt remains the
        # routing authority, while any non-empty final model prose is a separate
        # conversational answer and must stay durable across every transport.
        action = (
            "route_to_project"
            if event_type == "promote_chat_to_task" and bool(evt.get("routed_from_main"))
            else event_type
        )
        setattr(ctx, "_typed_routing_action_emitted", action)

    try:
        from multiprocessing.reduction import ForkingPickler

        ForkingPickler.dumps(dict(evt))
    except Exception as exc:
        log.warning("Control event is not multiprocessing-serializable", exc_info=True)
        try:
            root = Path(str(getattr(ctx, "budget_drive_root", "") or ctx.drive_root))
            append_jsonl(
                root / "logs" / "supervisor.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "control_event_serialization_failed",
                    "event_type": str(evt.get("type") or ""),
                    "task_id": str(evt.get("task_id") or ""),
                    "routing_token": str(evt.get("routing_token") or ""),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        except Exception:
            log.debug("Failed to record control-event serialization failure", exc_info=True)
        return "serialization_failed"

    def _record_emitted(mode: str) -> None:
        if str(evt.get("type") or "") != "promote_chat_to_task":
            return
        try:
            root = Path(str(getattr(ctx, "budget_drive_root", "") or ctx.drive_root))
            append_jsonl(
                root / "logs" / "supervisor.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "promote_chat_to_task_emitted",
                    "task_id": str(evt.get("task_id") or ""),
                    "routing_token": str(evt.get("routing_token") or ""),
                    "transport_mode": mode,
                    "sender_pid": os.getpid(),
                },
            )
        except Exception:
            log.debug("Failed to record promote emission", exc_info=True)

    event_queue = getattr(ctx, "event_queue", None)
    if event_queue is not None:
        try:
            event_queue.put_nowait(dict(evt))
            _mark_typed_routing_action()
            _record_emitted("live")
            return "live"
        except (AttributeError, queue.Full):
            pass
        except Exception:
            log.warning("Live control event emission failed; falling back to pending_events", exc_info=True)
    with _SCHEDULE_EMIT_LOCK:
        ctx.pending_events.append(evt)
    _mark_typed_routing_action()
    _record_emitted("deferred")
    return "deferred"


def _promotion_pool_disabled_from_snapshot(ctx: ToolContext) -> str:
    """Cheap early refusal for the known crash-storm state.

    The supervisor handler remains authoritative.  This projection only keeps
    source/project side effects from starting when the latest durable snapshot
    already says the executor pool was deliberately disabled.
    """
    try:
        root = Path(str(getattr(ctx, "budget_drive_root", "") or ctx.drive_root))
        snapshot = json.loads(
            (root / "state" / "queue_snapshot.json").read_text(encoding="utf-8")
        )
        reason = str(snapshot.get("worker_pool_disabled_reason") or "")
        if reason not in {"", "unknown"}:
            return reason
        if int(snapshot.get("worker_total") or 0) <= 0:
            return "no_workers"
        return ""
    except Exception:
        return ""


def _routing_status_root(ctx: ToolContext) -> Path:
    return Path(str(getattr(ctx, "budget_drive_root", "") or ctx.drive_root))


def _wait_for_promotion_admission(
    ctx: ToolContext,
    task_id: str,
    routing_token: str,
    *,
    client_message_id: str = "",
    timeout_sec: float = _PROMOTE_CONFIRM_TIMEOUT_SEC,
) -> Dict[str, Any]:
    """Wait for matching-token admission (SSOT moved to routing_wait, #198)."""
    from ouroboros.routing_wait import wait_for_promotion_admission

    return wait_for_promotion_admission(
        _routing_status_root(ctx), task_id, routing_token,
        client_message_id=client_message_id, timeout_sec=timeout_sec,
        poll_sec=_PROMOTE_CONFIRM_POLL_SEC,
    )


def _wait_for_routing_annotation(
    ctx: ToolContext,
    client_message_id: str,
    routing_token: str,
    *,
    timeout_sec: float = _PROMOTE_CONFIRM_TIMEOUT_SEC,
) -> Dict[str, Any]:
    """Wait for an exact chat-annotation receipt (SSOT in routing_wait, #198)."""
    from ouroboros.routing_wait import wait_for_routing_annotation

    return wait_for_routing_annotation(
        _routing_status_root(ctx), client_message_id, routing_token,
        timeout_sec=timeout_sec, poll_sec=_PROMOTE_CONFIRM_POLL_SEC,
    )


def _emit_and_wait_for_routing(
    ctx: ToolContext,
    evt: Dict[str, Any],
) -> tuple[str, Dict[str, Any]]:
    """Emit one routing event and return only its durable handler outcome."""
    mode = _emit_control_event(ctx, evt)
    if mode == "serialization_failed":
        return mode, {
            "status": "rejected",
            "reason": "event_serialization_failed",
            "detail": "The routing event was not emitted.",
        }
    timeout = _PROMOTE_CONFIRM_TIMEOUT_SEC if mode == "live" else 0.0
    if str(evt.get("type") or "") == "promote_chat_to_task":
        try:
            return mode, _wait_for_promotion_admission(
                ctx,
                str(evt.get("task_id") or ""),
                str(evt.get("routing_token") or ""),
                client_message_id=str(evt.get("client_message_id") or ""),
                timeout_sec=timeout,
            )
        except Exception as exc:
            if not swarm_router_turn(ctx):
                raise
            log.warning("Routing admission receipt failed after event emission", exc_info=True)
            return mode, {
                "status": "unconfirmed",
                "reason": "admission_confirmation_failed",
                "detail": type(exc).__name__,
            }
    return mode, _wait_for_routing_annotation(
        ctx,
        str(evt.get("client_message_id") or ""),
        str(evt.get("routing_token") or ""),
        timeout_sec=timeout,
    )
