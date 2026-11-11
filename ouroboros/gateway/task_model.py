"""POST /api/tasks/{task_id}/model: one owner act on a durable root's primary."""
from __future__ import annotations

import asyncio

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.gateway._helpers import json_error, json_exception, request_json_or
from ouroboros.gateway.schema import validate_ingress
from ouroboros.gateway.work_contracts import TaskModelRequest
from ouroboros.owner_model import ModelActRefused, request as request_model


async def api_task_model(request: Request) -> JSONResponse:
    from ouroboros.composer_cognition import normalize_model_choice
    from ouroboros.task_results import validate_task_id

    try:
        task_id = validate_task_id(request.path_params.get("task_id"))
    except ValueError as exc:
        return json_error(str(exc), 400, reason_code="task_model_invalid")
    body = await request_json_or(request, {})
    errors = validate_ingress(body, TaskModelRequest)
    if isinstance(body, dict) and set(body) - {"request_id", "action", "model", "use_local"}:
        errors.append("Unexpected fields in a model act")
    if errors:
        return json_error(errors[0], 400, task_id=task_id, reason_code="task_model_invalid")
    if not body["request_id"].strip() or len(body["request_id"]) > 128:
        return json_error("A stable request_id of 1–128 characters is required", 400,
                          task_id=task_id, reason_code="request_id_required")
    if body["action"] != "set" and any(key in body for key in ("model", "use_local")):
        return json_error("Only set accepts model and use_local", 400, task_id=task_id, reason_code="task_model_invalid")
    try:
        if body["action"] == "set":
            chosen = normalize_model_choice({**body, "kind": "model"})
            body = {**body, "model": chosen["model"], "use_local": chosen["use_local"]}
    except ValueError as exc:
        return json_error(str(exc), 400, task_id=task_id, reason_code="task_model_invalid")
    try:
        result = await asyncio.to_thread(request_model, task_id, body)
    except ModelActRefused as exc:
        return json_error(str(exc), exc.status, task_id=task_id, reason_code=exc.code)
    except Exception as exc:
        return json_exception(exc, 503)
    return JSONResponse(result)
