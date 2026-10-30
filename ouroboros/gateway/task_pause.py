"""``POST /api/tasks/{task_id}/pause`` — the owner's Pause of a whole task tree.

Owner Batch4 5A. The body carries a client-generated stable ``request_id``
(reused on retry: the acknowledgement is idempotent) and may declare who
pressed: ``actor`` ``owner`` (the owner's own client) or ``ouroboros``, with
an optional ``reason``, one sentence the owner reads under the work's name. An
actor the core does not know declares nobody; any other field is refused. The
declaration names the Pause's author (``control_actor.pause_author``) and
grants no power; there is no chat side effect. The response is sent only
after the root's durable fence landed (``supervisor/owner_pause_control.py``); ``state`` is the
accepted action state — ``requested`` while members settle, ``paused`` once
saved, ``released`` on a retry after Resume. A new Pause needs a fresh ID.
``202`` with ``latch_pending`` answers a fence that is durable (every launch
gate refuses) while its queue latch waited on the root's busy launch lock: the
Pause is accepted, not refused, and the same ``request_id`` completes it.
Resume is the existing ``/resume`` endpoint.

The accept step takes the queue lock and writes durable records, so it runs
off the event loop (``run_sync_to_completion``): a held lock or a slow disk
never stalls health, Stop or Panic, and a disconnected request still settles
the one acceptance it started (a retry with the same ``request_id`` rejoins it).
"""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.gateway._helpers import (
    json_error, json_exception, migrating_error, request_drive_root, request_json_or, run_sync_to_completion,
)
from ouroboros.control_actor import PAUSE_AUTHORS
from ouroboros.task_results import validate_task_id
from ouroboros.utils import utc_now_iso

_REQUEST_ID_MAX = 128
_FIELDS = frozenset({"request_id", "actor", "reason"})
_CONFLICTS = frozenset({"not_a_root_task", "cancel_pending", "task_terminal", "root_result_missing",
                        "realtime_pause_unsupported"})


async def api_task_pause(request: Request) -> JSONResponse:
    try:
        task_id = validate_task_id(request.path_params.get("task_id"))
    except ValueError as exc:
        return json_error(str(exc), 400)
    body = await request_json_or(request, {})
    if not isinstance(body, dict) or any(key not in _FIELDS for key in body):
        return json_error("pause accepts only {\"request_id\"}, an optional \"actor\" and \"reason\"", 400,
                          task_id=task_id, reason_code="unexpected_fields")
    request_id = body.get("request_id")
    if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > _REQUEST_ID_MAX:
        return json_error("request_id is required (a stable client-generated id, reused on retry)",
                          400, task_id=task_id, reason_code="request_id_required")
    reason = body.get("reason", "")
    if not isinstance(reason, str):
        return json_error("reason is the declared author's sentence (a string)", 400,
                          task_id=task_id, reason_code="pause_reason_invalid")
    actor = body.get("actor")
    by = {"actor": actor, "text": reason.strip(), "at": utc_now_iso()} if actor in PAUSE_AUTHORS else None
    if migrating := await run_sync_to_completion(migrating_error, request_drive_root(request)):
        return migrating  # P1: the Pause fence is a store row; 503 while the boot import owns the store
    try:
        from supervisor.owner_pause_control import request_owner_pause

        result = await run_sync_to_completion(request_owner_pause, task_id, request_id=request_id.strip(), by=by)
    except Exception as exc:
        return json_exception(exc, 503)
    if result.get("ok"):
        return JSONResponse(result, status_code=202 if result.get("latch_pending") else 200)
    error = str(result.get("error") or "pause_refused")
    status = 404 if error == "task_not_live" else 409 if error in _CONFLICTS else 503
    return json_error(f"pause refused: {error}", status, task_id=task_id, reason_code=error,
                      **({"root_task_id": result["root_task_id"]} if result.get("root_task_id") else {}))


def owner_tree_control_routes() -> list:
    """The owner's whole-tree Pause and the Continue of an interrupted root (Batch4)."""
    from starlette.routing import Route

    from ouroboros.gateway.task_continue import api_task_continue

    return [Route("/api/tasks/{task_id}/pause", endpoint=api_task_pause, methods=["POST"]),
            Route("/api/tasks/{task_id}/continue", endpoint=api_task_continue, methods=["POST"])]
