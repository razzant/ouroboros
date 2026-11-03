"""Scheduled task HTTP surface."""

from __future__ import annotations

import asyncio

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
    return await asyncio.to_thread(_schedules_list, _request)


def _schedules_list(_request: Request) -> JSONResponse:
    try:
        from supervisor.queue import ScheduleStoreUnreadable, load_schedule_store, schedule_activity_projection

        try:
            store = load_schedule_store(request_drive_root(_request))
        except ScheduleStoreUnreadable as exc:
            # "No schedules" is a claim; an unparseable table means the state is
            # UNKNOWN, and the Activity section says so instead of showing empty.
            return json_error(str(exc), 503)
        from supervisor.followup_policy import observed_store
        return JSONResponse(schedule_activity_projection(observed_store(request_drive_root(_request), store)))
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
        trigger_type = str(trigger.get("type") or "cron")
        if trigger_type == "cron":
            expr = str(trigger.get("expr") or body.get("cron") or "").strip()
            if err := cron_error(expr):
                return json_error(err, 400)
            trigger = {"type": "cron", "expr": expr}
        elif trigger_type == "once":
            # One-shot definitions use the same timestamp parser as task-authored
            # follow-ups; lifecycle controls use the dedicated action endpoint.
            from ouroboros.deadline_utils import parse_deadline_ts

            instant = parse_deadline_ts(str(trigger.get("run_at") or "").strip())
            if instant is None:
                return json_error("trigger.run_at must be a parseable ISO 8601 instant", 400)
            trigger = {"type": "once", "run_at": instant.isoformat()}
        else:
            return json_error("trigger.type must be cron or once", 400)
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
        # This owner door creates Main self-work or the named room's default.
        # Existing followups retain their recorded explicit resource/none intent.
        task = {**task, "metadata": dict(task.get("metadata") or {})}
        project_id = str(task.get("project_id") or "").strip()
        new_intent = {"kind": "room_default", "project_id": project_id} if project_id else {"kind": "system_repo"}
        enabled = _enabled_value(body)
        if isinstance(enabled, str):
            return json_error(enabled, 400)
        # Only the fields the owner authors here. Provenance, run history and the
        # consumed receipt belong to the runtime and are merged from the CURRENT
        # row inside the write lock — reading them here would merge a snapshot
        # that is already stale.
        record = {
            "id": str(body.get("id") or "").strip(),
            "name": str(body.get("name") or body.get("id") or "scheduled-task").strip(),
            "description": str(body.get("description") or "").strip(),
            "enabled": enabled,
            "timezone": str(body.get("timezone") or "").strip(),
            "trigger": trigger,
            "task": task,
        }
        from supervisor.queue import ScheduleRefused, ScheduleStoreUnreadable, upsert_scheduled_task

        try:
            stored = upsert_scheduled_task(
                record, drive_root=request_drive_root(request), actor="owner:gateway", writer_kind="owner",
                reason=str(body.get("reason") or "").strip(), new_resource_intent=new_intent)
        except ScheduleRefused as refusal:
            # 409 when the write could not be made SAFELY (its audit is down);
            # 400 when the request itself asks for something this door cannot do.
            return json_error(refusal.message, 409 if refusal.status == "audit_unavailable" else 400)
        except ScheduleStoreUnreadable as exc:
            return json_error(str(exc), 409)
        return JSONResponse({"ok": stored.get("audit") == "recorded", "schedule": stored})
    except Exception as exc:
        return json_exception(exc)


async def api_schedules_action(request: Request) -> JSONResponse:
    """Apply one named lifecycle action to an existing schedule.

    The owner surface names the ACTION it wants; nothing infers a command from a
    free-text reason. Both the Activity buttons and the agent tool land on the
    same audited seam in ``supervisor.queue_schedules``, so they cannot drift
    apart in what a disable, a delete or a restore actually means.
    """
    try:
        schedule_id = str(request.path_params.get("schedule_id") or "").strip()
        if err := schedule_id_error(schedule_id):
            return json_error(err, 400)
        body = await request_json_or(request, {})
        if not isinstance(body, dict):
            return json_error("request body must be a JSON object", 400)
        from supervisor.queue import SCHEDULE_ACTIONS, mutate_scheduled_task

        action = str(body.get("action") or "").strip().lower()
        if action not in SCHEDULE_ACTIONS:
            return json_error(f"action must be one of {sorted(SCHEDULE_ACTIONS)}", 400)
        reason = str(body.get("reason") or "").strip()
        if not reason:
            return json_error("reason is required for a schedule lifecycle action", 400)
        return JSONResponse(mutate_scheduled_task(
            action, schedule_id, reason=reason, actor="owner:gateway", writer_kind="owner",
            drive_root=request_drive_root(request), expected_hold_id=str(body.get("expected_hold_id") or ""),
            relation=str(body.get("relation") or ""),
        ))
    except Exception as exc:
        return json_exception(exc)


async def api_schedules_delete(request: Request) -> JSONResponse:
    try:
        schedule_id = str(request.path_params.get("schedule_id") or "").strip()
        if err := schedule_id_error(schedule_id):
            return json_error(err, 400)
        from supervisor.queue import mutate_scheduled_task

        return JSONResponse(mutate_scheduled_task(
            "delete", schedule_id, reason="owner deleted the schedule", actor="owner:gateway", writer_kind="owner",
            drive_root=request_drive_root(request),
        ))
    except Exception as exc:
        return json_exception(exc)
