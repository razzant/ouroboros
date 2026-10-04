"""Desktop startup & background endpoints: the host's sign-in registration and keep-running choice.

Transport only; the entry, its states and its target live in
``ouroboros.desktop_autostart``. OS registration is not settings.json, so the
autostart routes stay outside the owner settings write seam; a change is still
audited. The keep-running choice IS settings.json: its POST holds the document
lock, maps a contended lock to a typed refusal and answers ``saved=false``
before the commit, like the other single-decision owner endpoints.
"""

from __future__ import annotations

import logging

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros import desktop_autostart
from ouroboros.gateway._helpers import json_error, request_json_or, run_sync_to_completion
from ouroboros.gateway.owner_settings import owner_audit, owner_write_guard, settings_document_mutation, unsaved_error

log = logging.getLogger(__name__)


async def api_desktop_autostart_get(request: Request) -> JSONResponse:
    try:
        return JSONResponse(await run_sync_to_completion(desktop_autostart.autostart_status))
    except OSError as exc:
        log.warning("Host sign-in entry could not be read: %s", exc)
        return json_error("Host startup entry could not be read", 500)


async def api_desktop_autostart_post(request: Request) -> JSONResponse:
    body = await request_json_or(request, None)
    if not isinstance(body, dict) or set(body) != {"enabled"} or not isinstance(body["enabled"], bool):
        return json_error('request body must be {"enabled": true} or {"enabled": false}', 400)
    try:
        status = await run_sync_to_completion(desktop_autostart.autostart_status, body["enabled"])
    except OSError as exc:
        log.warning("Host sign-in entry could not be changed: %s", exc)
        return json_error("Host startup entry could not be changed", 500)
    if status["state"] == "unavailable":
        return json_error(status["reason"], 409)
    owner_audit(request, "desktop_autostart", {"enabled": body["enabled"], "state": status["state"]})
    return JSONResponse(status)


async def api_desktop_background_get(request: Request) -> JSONResponse:
    return JSONResponse(await run_sync_to_completion(desktop_autostart.background_status))


@owner_write_guard
async def api_desktop_background_post(request: Request) -> JSONResponse:
    body = await request_json_or(request, None)
    if not isinstance(body, dict) or set(body) != {"enabled"} or not isinstance(body["enabled"], bool):
        return unsaved_error('request body must be {"enabled": true} or {"enabled": false}', 400)
    status = await run_sync_to_completion(_record_background_choice, body["enabled"])
    if status["state"] == "unavailable":
        return unsaved_error(status["reason"], 409)
    owner_audit(request, "desktop_background", {"enabled": body["enabled"], "state": status["state"]})
    return JSONResponse(status)


def _record_background_choice(enabled: bool) -> dict:
    with settings_document_mutation():  # a generic save mid-merge cannot revert this key
        return desktop_autostart.background_status(enabled)
