"""``POST /api/tasks/{task_id}/hurry`` — the text-free owner hurry ingress (HQ1).

Split out of ``gateway/tasks.py`` by the module-size boundary (the same shape as
``gateway/task_events.py``); ``gateway.tasks`` re-exports ``api_task_hurry`` so
route wiring and tests keep one namespace.

The request body carries ONLY a client-generated stable ``request_id`` — there
is no text field and no chat side effect anywhere on this path (§19.7.2 items
1-4): no ``steer_task``, no ``owner_text``, no ``send_message``, no outbox row,
no routing receipt, no WS chat frame. The acknowledgement is the HTTP response;
durable facts are the mailbox control, the ``owner_hurry`` task-result
projection, and one non-chat ``owner_hurry`` event.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.gateway._helpers import json_error, json_exception, request_drive_root, request_json_or
from ouroboros.task_results import resolve_task_lineage, validate_task_id
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

_REQUEST_ID_MAX = 128


def _admit_hurry_locked(task_id: str) -> Tuple[Optional[Dict[str, Any]], str, int]:
    """The queue-owned liveness/admission transaction (§19.7.2 item 2).

    Returns ``(task_row, refusal_reason, attempt)``. Runs UNDER the queue lock:
    live membership (RUNNING or PENDING), root-only eligibility (managed
    children refused), sealed-acceptance-fence refusal, and the cancel-pending
    re-check all happen inside one transaction so a racing cancel ingress and
    this admission cannot both win.
    """
    from supervisor import queue as q

    with q._queue_lock:
        attempt = 1
        task: Optional[Dict[str, Any]] = None
        meta = q.RUNNING.get(task_id) if isinstance(q.RUNNING, dict) else None
        if isinstance(meta, dict) and isinstance(meta.get("task"), dict):
            task = dict(meta["task"])
            raw_attempt = meta.get("attempt")
            if raw_attempt is None:
                raw_attempt = task.get("_attempt")
            try:
                attempt = max(1, int(raw_attempt)) if raw_attempt is not None else 1
            except (TypeError, ValueError):
                attempt = 1
        if task is None:
            task = next(
                (
                    dict(row) for row in list(q.PENDING or [])
                    if isinstance(row, dict) and str(row.get("id") or "") == task_id
                ),
                None,
            )
            if task is not None:
                try:
                    attempt = max(1, int(task.get("_attempt") or 1))
                except (TypeError, ValueError):
                    attempt = 1
        if task is None:
            # The in-process direct-chat turn drains the same owner mailbox a
            # pooled task does, so a typed hurry reaches it the same way.
            from supervisor.workers import direct_chat_turn

            task = direct_chat_turn(task_id)
        if task is None:
            return None, "task_not_live", attempt
        lineage = resolve_task_lineage(
            task_id,
            metadata=task.get("metadata"),
            root_task_id=task.get("root_task_id"),
            parent_task_id=task.get("parent_task_id"),
            delegation_role=task.get("delegation_role"),
            original_task_id=task.get("original_task_id"),
            timeout_retry_from=task.get("timeout_retry_from"),
        )
        if not bool(lineage["is_root_task"]):
            return None, "not_a_root_task", attempt
        fence = q.ACCEPTANCE_FENCES.get(str(task.get("root_task_id") or task_id))
        if isinstance(fence, dict) and str(fence.get("status") or "") == "sealed":
            return None, "acceptance_fence_sealed", attempt
        try:
            from ouroboros.cancel_intents import cancel_pending

            if cancel_pending(q.DRIVE_ROOT, task_id):
                # A pending/active stop WINS and owns the terminal reason
                # (§19.7.2 item 2); hurry is refused, never queued behind it.
                return None, "cancel_pending", attempt
        except Exception:
            log.debug("hurry cancel-pending check failed for %s", task_id, exc_info=True)
        return task, "", attempt


async def api_task_hurry(request: Request) -> JSONResponse:
    """POST /api/tasks/{task_id}/hurry — typed task-local acceleration control."""
    try:
        task_id = validate_task_id(request.path_params.get("task_id"))
    except ValueError as exc:
        return json_error(str(exc), 400)
    body = await request_json_or(request, {})
    if not isinstance(body, dict):
        return json_error("request body must be a JSON object", 400, task_id=task_id)
    request_id = str(body.get("request_id") or "").strip()
    if not request_id or len(request_id) > _REQUEST_ID_MAX:
        return json_error(
            "request_id is required (a stable client-generated id, reused on retry)",
            400, task_id=task_id, reason_code="request_id_required",
        )
    if any(key not in {"request_id"} for key in body):
        # Text-free by contract (HQ1 owner decision: no visible chat message) — a body
        # smuggling text/fields is refused rather than silently dropped.
        return json_error(
            "hurry accepts only {\"request_id\"} — it carries no text",
            400, task_id=task_id, reason_code="unexpected_fields",
        )
    # Executable gateway ABI (ABI-3, Q7=A): schema gate after the bespoke
    # required/closed-world checks (their typed reason codes stay first) — it
    # adds the declared-type enforcement (a non-string request_id is refused,
    # never coerced).
    from ouroboros.gateway.contracts import TaskHurryRequest
    from ouroboros.gateway.schema import validate_ingress

    schema_errors = validate_ingress(body, TaskHurryRequest)
    if schema_errors:
        return json_error(
            f"invalid request body: {schema_errors[0]}", 400,
            task_id=task_id, reason_code="invalid_request_body",
            schema_errors=schema_errors[:8],
        )
    drive_root = request_drive_root(request)
    try:
        task, refusal, attempt = _admit_hurry_locked(task_id)
    except Exception as exc:
        return json_exception(exc, 503)
    if refusal == "task_not_live":
        return json_error(
            "task is not live (terminal, unknown, or not a queue task)",
            404, task_id=task_id, reason_code="task_not_live",
        )
    if refusal:
        return json_error(
            f"hurry refused: {refusal}", 409, task_id=task_id, reason_code=refusal,
        )
    assert task is not None
    try:
        from supervisor.queue import _task_drive_for_task

        from ouroboros.owner_hurry import emit_event, record_requested
        from ouroboros.owner_mailbox import KIND_HURRY, write_owner_message

        requested_at = utc_now_iso()
        # Request-id idempotency (§19.7.2 item 3): the same request_id on the
        # live attempt returns the existing acknowledgement; DIFFERENT ids
        # collapse onto the one executable effect latch (no second control).
        outcome = record_requested(
            drive_root, task_id,
            request_id=request_id, attempt=attempt, requested_at=requested_at,
        )
        block = outcome.get("block") if isinstance(outcome.get("block"), dict) else {}
        # Physical mailbox root EXACTLY as steer_task: direct task ->
        # DRIVE_ROOT; forked task -> its child drive. The non-empty text is
        # the parser-required internal reason, never dialogue. EVERY request
        # (fresh, same-id retry, or a duplicate with a DIFFERENT id, e.g.
        # after a mailbox write failure + page reload minted a new id)
        # appends idempotently under the effect LATCH's request_id: the drain
        # dedupes by msg_id, so a duplicate line is invisible while a lost
        # control is healed instead of a false "already accepted" acknowledgement.
        latch_request_id = str(block.get("request_id") or "") or request_id
        drive = _task_drive_for_task(task, task_id)
        if not write_owner_message(
            drive, "owner_hurry", task_id,
            msg_id=f"hurry:{latch_request_id}", kind=KIND_HURRY,
        ):
            return json_error(
                "the durable hurry control could not be written — retry with "
                "the same request_id",
                503, task_id=task_id, reason_code="mailbox_write_failed",
            )
        if not outcome.get("duplicate"):
            emit_event(
                None, drive_root, task_id,
                phase="requested", request_id=request_id,
                requested_at=requested_at, attempt=attempt,
            )
    except Exception as exc:
        return json_exception(exc, 503)
    return JSONResponse({
        "ok": True,
        "task_id": task_id,
        "request_id": request_id,
        "state": str(outcome.get("state") or "requested"),
        "attempt_key": int(block.get("attempt_key") or attempt),
        "duplicate": bool(outcome.get("duplicate")),
    })


__all__ = ["api_task_hurry"]
