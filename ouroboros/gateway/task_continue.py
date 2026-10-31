"""``POST /api/tasks/{task_id}/continue`` — the owner's Continue of an interrupted root.

Owner Batch4 (1A). The body carries ONLY ``{"action_nonce"}``: a random id the
client keeps across retry and reload, so the same press always answers the
same admission (``supervisor/continuation_admission.py``); a new press is a
new nonce. A timeout on the client is never "not admitted" — it retries with
the same nonce. ``held`` names a Continue admitted but waiting on the
interrupted task's own unsettled writers.

The admission takes the queue lock and writes durable records, so it runs off
the event loop (``run_sync_to_completion``): a disconnected request still
settles the one admission it started, and the same nonce rejoins it.
"""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.gateway._helpers import json_error, json_exception, request_json_or, run_sync_to_completion
from ouroboros.task_results import validate_task_id

_NOT_FOUND = frozenset({"predecessor_missing"})
_UNAVAILABLE = frozenset({"predecessor_record_unreadable", "successor_record_unreadable",
                          "continuation_claim_unwritable", "queue_snapshot_persist_failed",
                          "continuation_unconfirmed", "task_id_lookup_failed", "queue_not_restored"})


async def api_task_continue(request: Request) -> JSONResponse:
    try:
        task_id = validate_task_id(request.path_params.get("task_id"))
    except ValueError as exc:
        return json_error(str(exc), 400)
    body = await request_json_or(request, {})
    if not isinstance(body, dict) or any(key != "action_nonce" for key in body):
        return json_error("continue accepts only {\"action_nonce\"}", 400, task_id=task_id,
                          reason_code="unexpected_fields")
    try:
        from supervisor.continuation_admission import admit_continuation

        result = await run_sync_to_completion(admit_continuation, task_id,
                                              action_nonce=str(body.get("action_nonce") or ""))
    except ValueError as exc:
        return json_error(str(exc), 400, task_id=task_id, reason_code="invalid_action_nonce")
    except Exception as exc:
        return json_exception(exc, 503)
    if result.get("ok"):
        return JSONResponse(result)
    error = str(result.get("error") or "continue_refused")
    status = 404 if error in _NOT_FOUND else 503 if error in _UNAVAILABLE else 409
    return json_error(f"continue refused: {error}", status, task_id=task_id, reason_code=error,
                      **{key: result[key] for key in ("successor_task_id", "cause", "gaps", "state", "action_nonce", "unconfirmed") if result.get(key)})
