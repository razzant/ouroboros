"""Activity › Background: the services tasks kept running, and the owner's Stop of one launch.

Both routes are thin: ``launch_custody.kept_services`` is the one inventory and ``launch_custody.stop_kept_service``
the one stop, the same handler ``stop_service(service_id=...)`` uses for a root task. Nothing here starts, restarts
or signals anything else, and no process is named by a pid or a port.
"""

from __future__ import annotations

from typing import Any, Dict, List

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from ouroboros.gateway._helpers import json_error, request_drive_root, request_json_or

_NO_STORE = {"Cache-Control": "no-store"}


def _titled(root: Any) -> Dict[str, Any]:
    """The inventory with each originating task's own name, when its record says one."""
    from ouroboros.launch_custody import kept_services
    from ouroboros.task_results import load_task_result

    view = kept_services(root)
    services: List[Dict[str, Any]] = []
    for item in view["services"]:
        try:
            task = load_task_result(root, item["task_id"]) if item.get("task_id") else None
        except Exception:
            task = None
        title = str((task or {}).get("suggested_name") or (task or {}).get("title") or "").strip()
        services.append({**item, "task_title": title} if title else dict(item))
    return {**view, "services": services}


async def api_kept_services(request: Request) -> JSONResponse:
    """What tasks left running now, as custody reads it (one census); never a stop."""
    return JSONResponse(await run_in_threadpool(_titled, request_drive_root(request)), headers=_NO_STORE)


async def api_kept_service_stop(request: Request) -> JSONResponse:
    """Stop ONE kept launch by its id, with the owner's reason; the receipt says what a fresh read found after."""
    from ouroboros.launch_custody import stop_kept_service

    body = await request_json_or(request, {})
    if not isinstance(body, dict):
        return json_error("request body must be a JSON object", 400)
    receipt = await run_in_threadpool(
        stop_kept_service, request_drive_root(request), str(request.path_params.get("launch_id") or ""),
        reason=str(body.get("reason") or ""), actor="owner:gateway")
    return JSONResponse(receipt, status_code=400 if receipt.get("outcome") == "refused" else 200, headers=_NO_STORE)


def kept_service_routes() -> List[Route]:
    return [
        Route("/api/services/kept", endpoint=api_kept_services, methods=["GET"]),
        Route("/api/services/kept/{launch_id}/stop", endpoint=api_kept_service_stop, methods=["POST"]),
    ]
