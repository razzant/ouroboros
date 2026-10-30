"""``POST /api/tasks/{task_id}/owner-keep``: keep or release the materials of exactly this task.

The body says what to do; ``source`` is stamped here from the request the gateway actually
authenticated (``{basis: "authenticated_gateway_request", route}``): it names that request and claims no
chat message, and no caller can declare who approved it. The answer is what ``owner_keep`` found
under the task's custody lock, never a promise about material already accepted for deletion.
"""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.gateway._helpers import (
    json_error, json_exception, migrating_error, request_drive_root, request_json_or, run_sync_to_completion,
)
from ouroboros.owner_keep import set_owner_keep
from ouroboros.task_results import validate_task_id

_FIELDS = frozenset({"keep", "note"})
_ROUTE = "POST /api/tasks/{task_id}/owner-keep"


async def api_task_owner_keep(request: Request) -> JSONResponse:
    try:
        task_id = validate_task_id(request.path_params.get("task_id"))
    except ValueError as exc:
        return json_error(str(exc), 400)
    body = await request_json_or(request, {})
    if (not isinstance(body, dict) or any(key not in _FIELDS for key in body)
            or not isinstance(body.get("keep"), bool) or not isinstance(body.get("note", ""), str)):
        return json_error("owner-keep accepts {\"keep\": true|false} and an optional \"note\"", 400,
                          task_id=task_id, reason_code="unexpected_fields")
    root = request_drive_root(request)
    if migrating := await run_sync_to_completion(migrating_error, root):
        return migrating
    source = {"basis": "authenticated_gateway_request", "route": _ROUTE}
    try:
        result = await run_sync_to_completion(set_owner_keep, root, task_id, keep=body["keep"], source=source,
                                              note=body.get("note", "").strip())
    except Exception as exc:
        return json_exception(exc, 503)
    if result["status"] == "refused":
        return json_error(f"owner-keep refused: {result['reason']}", 404 if result["reason"] == "task_not_found" else 503,
                          task_id=task_id, reason_code=result["reason"])
    return JSONResponse(result)
