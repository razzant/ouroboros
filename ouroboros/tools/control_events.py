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

from ouroboros.runtime_limits import get_promote_confirm_wait_sec
from ouroboros.tool_capabilities import ROUTING_VERBS
from ouroboros.tools.registry import ToolContext
from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger(__name__)


# Guards parent-side shared ctx state mutated during (possibly parallel)
# schedule_subagent emission within one tool-call round. Restart's send-once
# section also enters the deferred append below, so this lock must be reentrant.


_SCHEDULE_EMIT_LOCK = threading.RLock()


_PROMOTE_CONFIRM_TIMEOUT_SEC = get_promote_confirm_wait_sec()


_PROMOTE_CONFIRM_POLL_SEC = 0.05


def request_restart_intent(ctx: ToolContext, reason: str, *, evolution: bool = False) -> dict:
    """Send once per requester attempt; an uncertain ACK never duplicates the request."""
    import time
    import uuid

    from ouroboros.queue_observation import load_queue_document, queue_stamp
    from ouroboros.tool_access_paths import canonical_data_root

    task_id, attempt = str(ctx.task_id or ""), int(getattr(ctx, "task_attempt", None) or 1)
    with _SCHEDULE_EMIT_LOCK:  # the existing lock for parallel calls sharing one ToolContext
        request = getattr(ctx, "pending_restart_request", None)
        if not request or (request["task_id"], request["task_attempt"]) != (task_id, attempt):
            request = {"type": "restart_request", "phase": "intent", "request_id": uuid.uuid4().hex,
                       "task_id": task_id, "task_attempt": attempt, "reason": reason,
                       "evolution_restart": evolution, "ts": utc_now_iso()}
            ctx.pending_restart_request = request
            request["transport"] = _emit_control_event(ctx, dict(request))
    deadline = time.monotonic() + (_PROMOTE_CONFIRM_TIMEOUT_SEC if request["transport"] == "live" else 0)
    while True:
        document = load_queue_document(canonical_data_root(ctx))
        if queue_stamp(document):
            for key in ("restart_reply", "restart_intent"):
                receipt = document.get(key) or {}
                if receipt.get("request_id") == request["request_id"] and receipt.get("status") != "awaiting":
                    return receipt
        if time.monotonic() >= deadline:
            return {"status": "unconfirmed", "request_id": request["request_id"]}
        time.sleep(_PROMOTE_CONFIRM_POLL_SEC)


def _emit_control_event(ctx: ToolContext, evt: Dict[str, Any]) -> str:
    """Emit a control event live when possible, preserving legacy fallback."""
    def _mark_typed_routing_action() -> None:
        event_type = str(evt.get("type") or "")
        if not any(event_type in events for events in ROUTING_VERBS.values()):
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
                    # The owner message this root came from: without it the durable
                    # ingress row cannot be joined to the message that caused it.
                    "client_message_id": str(evt.get("client_message_id") or ""),
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
    from ouroboros.tool_custody import record_control_handoff, forget_unemitted_control

    if str(evt.get("type") or "") == "promote_chat_to_task":
        from ouroboros.promotion_intent import prepare_promotion_intent

        try:
            prepare_promotion_intent(_routing_status_root(ctx), evt)
        except Exception as exc:
            return "intent_persist_failed", {"status": "rejected", "reason": "intent_persist_failed",
                                             "detail": f"New work was not emitted: {type(exc).__name__}: {exc}"}
    record_control_handoff(evt)
    mode = _emit_control_event(ctx, evt)
    if mode == "serialization_failed":
        forget_unemitted_control(evt)
        return mode, {
            "status": "rejected",
            "reason": "event_serialization_failed",
            "detail": "The routing event was not emitted.",
        }
    timeout = _PROMOTE_CONFIRM_TIMEOUT_SEC if mode == "live" else 0.0
    kind = str(evt.get("type") or "")
    if kind == "promote_chat_to_task":
        receipt = _wait_for_promotion_admission(
            ctx,
            str(evt.get("task_id") or ""),
            str(evt.get("routing_token") or ""),
            client_message_id=str(evt.get("client_message_id") or ""),
            timeout_sec=timeout,
        )
    else:
        receipt = _wait_for_routing_annotation(
            ctx,
            str(evt.get("client_message_id") or ""),
            str(evt.get("routing_token") or ""),
            timeout_sec=timeout,
        )
    # Emitted, but its outcome was not seen in time: it may still land, or already
    # has. The turn keeps the exact identity, so a later acting round reports the
    # recorded outcome (``loop_model_call._append_routing_receipts``), even for an
    # act no owner message carries (an agent-steer id, a wake's promote).
    if str(receipt.get("status") or "unconfirmed") == "unconfirmed" and kind in {"promote_chat_to_task", "steer_task"}:
        acts = getattr(ctx, "unconfirmed_routing_acts", None)
        if isinstance(acts, list):
            acts.append({
                "action": kind if kind == "steer_task" or not evt.get("routed_from_main") else "route_to_project",
                "client_message_id": str(evt.get("client_message_id") or ""),
                "routing_token": str(evt.get("routing_token") or ""),
                "target": str(evt.get("task_id") or evt.get("target_task_id") or ""),
            })
    return mode, receipt
