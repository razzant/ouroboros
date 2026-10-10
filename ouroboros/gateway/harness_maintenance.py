"""Thin owner HTTP projection of the shared engine-owned CLI maintenance service."""
from __future__ import annotations

import asyncio

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from ouroboros import harness_maintenance as service
from ouroboros.gateway._helpers import run_sync_to_completion


def harness_maintenance_routes() -> list[Route]:
    """The maintenance family shares handlers and operation identity across reads/cancel."""
    return [
        Route("/api/claudexor/maintenance/harnesses", api_harness_maintenance_inventory, methods=["GET"]),
        Route("/api/claudexor/maintenance/operations", api_harness_maintenance_create, methods=["POST"]),
        Route("/api/claudexor/maintenance/operations/{operation_id}", api_harness_maintenance_operation, methods=["GET"]),
        Route("/api/claudexor/maintenance/operations/{operation_id}/cancel", api_harness_maintenance_operation, methods=["POST"]),
    ]


def _boolean(request: Request, name: str) -> bool:
    values = request.query_params.getlist(name)
    if len(values) > 1 or (values and values[0] not in {"true", "false"}):
        raise ValueError(f"{name} must be true or false")
    return bool(values and values[0] == "true")


async def api_harness_maintenance_inventory(request: Request) -> JSONResponse:
    try:
        if set(request.query_params) - {"harness", "fresh", "checkLatest"}:
            raise ValueError("Unsupported maintenance query")
        return JSONResponse(await asyncio.to_thread(
            service.inspect_harnesses, request.query_params.getlist("harness"),
            fresh=_boolean(request, "fresh"), check_latest=_boolean(request, "checkLatest")))
    except Exception as exc:
        status, body = service.maintenance_problem(exc)
        return JSONResponse(body, status_code=status)


async def api_harness_maintenance_create(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        result = await run_sync_to_completion(service.start_maintenance, body,
                                              request.headers.get("Idempotency-Key", ""))
        return JSONResponse(result, status_code=202)
    except Exception as exc:
        status, body = service.maintenance_problem(exc)
        return JSONResponse(body, status_code=status)


async def api_harness_maintenance_operation(request: Request) -> JSONResponse:
    try:
        operation_id = request.path_params["operation_id"]
        if request.method == "POST":
            result = await run_sync_to_completion(service.cancel_maintenance, operation_id)
        else:
            result = await asyncio.to_thread(service.inspect_operation, operation_id)
        return JSONResponse(result)
    except Exception as exc:
        status, body = service.maintenance_problem(exc)
        return JSONResponse(body, status_code=status)
