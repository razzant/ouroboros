"""Scheduled task HTTP surface."""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.gateway._helpers import json_error, json_exception, request_drive_root, request_json_or
from ouroboros.schedule_contract import RESERVED_TEMPLATE_FIELDS, cron_error, schedule_id_error, timezone_error


def _enabled_value(payload: dict) -> bool | str:
    if "enabled" not in payload:
        return True
    value = payload.get("enabled")
    if isinstance(value, bool):
        return value
    return "enabled must be a JSON boolean"


async def api_schedules_list(_request: Request) -> JSONResponse:
    try:
        from supervisor.queue import list_scheduled_tasks

        return JSONResponse(list_scheduled_tasks(request_drive_root(_request)))
    except Exception as exc:
        return json_exception(exc)


async def api_schedules_upsert(request: Request) -> JSONResponse:
    try:
        body = await request_json_or(request, {})
        if not isinstance(body, dict):
            return json_error("request body must be a JSON object", 400)
        if err := schedule_id_error(str(body.get("id") or "")):
            return json_error(err, 400)
        trigger = body.get("trigger") if isinstance(body.get("trigger"), dict) else {}
        if str(trigger.get("type") or "cron") == "cron":
            expr = str(trigger.get("expr") or body.get("cron") or "").strip()
            if err := cron_error(expr):
                return json_error(err, 400)
            trigger = {"type": "cron", "expr": expr}
        else:
            return json_error("trigger.type must be cron", 400)
        task = body.get("task") if isinstance(body.get("task"), dict) else {}
        if RESERVED_TEMPLATE_FIELDS & set(task):
            return json_error("scheduled task templates cannot include workspace/drive fields; use /api/tasks for workspace preflight", 400)
        if "metadata" in task and not isinstance(task.get("metadata"), dict):
            return json_error("scheduled task template metadata must be an object", 400)
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        if RESERVED_TEMPLATE_FIELDS & set(metadata):
            return json_error("scheduled task template metadata cannot include reserved lineage/workspace fields", 400)
        if str(task.get("type") or "task") != "task":
            return json_error("scheduled task templates must use type='task'", 400)
        if "project_id" in task:
            from ouroboros.project_facts import explicit_project_id_ok

            # (RWS v2 §7) The template's ONLY placement input. It is validated for SHAPE
            # here and for EXISTENCE at fire time — a project can be created, deleted or
            # rebound between writing the row and running it, so existence now would be a
            # promise this surface cannot keep.
            if not explicit_project_id_ok(str(task.get("project_id") or "")):
                return json_error(
                    "scheduled task project_id must be filesystem-safe (alphanumeric/_/-/., no spaces or slashes)",
                    400,
                )
        if "priority" in task:
            try:
                int(task.get("priority"))
            except (TypeError, ValueError):
                return json_error("scheduled task priority must be an integer", 400)
        if err := timezone_error(str(body.get("timezone") or "")):
            return json_error(err, 400)
        if not task:
            task = {
                "type": "task",
                "text": str(body.get("description") or body.get("name") or "Scheduled task"),
            }
        enabled = _enabled_value(body)
        if isinstance(enabled, str):
            return json_error(enabled, 400)
        record = {
            "id": str(body.get("id") or "").strip(),
            "name": str(body.get("name") or body.get("id") or "scheduled-task").strip(),
            "description": str(body.get("description") or "").strip(),
            "enabled": enabled,
            "timezone": str(body.get("timezone") or "").strip(),
            "trigger": trigger,
            "task": task,
        }
        from supervisor.queue import upsert_scheduled_task

        return JSONResponse({"ok": True, "schedule": upsert_scheduled_task(record, drive_root=request_drive_root(request))})
    except Exception as exc:
        return json_exception(exc)


async def api_schedules_delete(request: Request) -> JSONResponse:
    try:
        schedule_id = str(request.path_params.get("schedule_id") or "").strip()
        if err := schedule_id_error(schedule_id):
            return json_error(err, 400)
        from supervisor.queue import remove_scheduled_task

        return JSONResponse({"ok": remove_scheduled_task(schedule_id, drive_root=request_drive_root(request))})
    except Exception as exc:
        return json_exception(exc)
