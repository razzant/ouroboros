"""Save the composer's effort, live Subagents choice and chat model for its room."""

from __future__ import annotations

import os
from typing import Any, Dict

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from ouroboros.composer_cognition import (
    main_cognition, normalize_model_choice,
    request_project_id, update_project_cognition, update_project_model_choice,
)
from ouroboros.gateway._helpers import request_drive_root, request_json_or
from ouroboros.gateway.owner_settings import (
    _owner_audit,
    _owner_update_settings,
    owner_write_guard,
    record_settings_change,
    settings_document_digest,
    settings_document_mutation,
    unsaved_error,
)
from ouroboros.settings_scales import EFFORT_FORMAT_KEY, EFFORT_RANGE_KEYS, EFFORT_SCALE, effort_range, effort_rank

# The body's fields, in the order of ``EFFORT_RANGE_KEYS`` (min, recommended = TASK, max).
RANGE_FIELDS = ("min", "recommended", "max")


def owner_composer_routes() -> list[Route]:
    """The four composer capsules' owner writes, for Main or the named room.

    Their paths and handlers stay together here; the gateway mounts the family.
    Context's writer retains its settings/capability authority in ``settings``.
    """
    from ouroboros.gateway.settings import api_owner_context_mode

    return [
        Route("/api/owner/context-mode", endpoint=api_owner_context_mode, methods=["POST"]),
        Route("/api/owner/effort-range", endpoint=api_owner_effort_range, methods=["POST"]),
        Route("/api/owner/subagents-mode", endpoint=api_owner_subagents_mode, methods=["POST"]),
        Route("/api/owner/chat-model", endpoint=api_owner_chat_model, methods=["POST"]),
    ]


def effort_range_refusal(body: Any) -> str:
    """A complete triple of tiers with min ≤ max; primary recommended is independent."""
    if not isinstance(body, dict):
        return "JSON body must be an object with min, recommended and max."
    tiers: Dict[str, str] = {}
    for field in RANGE_FIELDS:
        value = str(body.get(field) or "").strip().lower()
        if value not in EFFORT_SCALE:
            return f"'{field}' must be one of: {', '.join(EFFORT_SCALE)}."
        tiers[field] = value
    if effort_rank(tiers["min"]) > effort_rank(tiers["max"]):
        return "The helper range must be ordered min ≤ max."
    return ""


@owner_write_guard
async def api_owner_effort_range(request: Request) -> JSONResponse:
    """Persist the owner's effort range; every participant started afterwards reads it."""
    from ouroboros.gateway.settings import _run_settings_writer

    body = await request_json_or(request, {})
    # Off the event loop, under the document lock (held inside), like every settings writer.
    return await _run_settings_writer(_api_owner_effort_range_sync, request, body)


def _api_owner_effort_range_sync(request: Request, body: Any) -> JSONResponse:
    refusal = effort_range_refusal(body)
    if refusal:
        return unsaved_error(refusal, 400, code="effort_range_invalid")
    try:
        project_id = request_project_id(body)
    except ValueError as exc:
        return unsaved_error(str(exc), 400, code="effort_range_invalid")
    values = {key: str(body[field]).strip().lower() for key, field in zip(EFFORT_RANGE_KEYS, RANGE_FIELDS)}
    current = effort_range(values)

    def _set_range(current: Dict[str, Any]) -> Dict[str, Any]:
        current.update(values)
        current[EFFORT_FORMAT_KEY] = 1
        return current

    with settings_document_mutation():
        if project_id:
            try:
                old, cognition = update_project_cognition(request_drive_root(request), project_id, effort_range=current)
            except ValueError as exc:
                return unsaved_error(str(exc), 404, code="project_unavailable")
            previous = old["effort_range"]
        else:
            previous = main_cognition()["effort_range"]
            digest = settings_document_digest()
            _owner_update_settings(_set_range, digest, authored_keys=EFFORT_RANGE_KEYS)
            for key, value in values.items():
                os.environ[key] = value
            cognition = main_cognition()
    _owner_audit(request, "effort_range", {"project_id": project_id, "effort_range": current, "previous_effort_range": previous})
    return JSONResponse({"ok": True, "project_id": project_id, "effort_range": current, "cognition": cognition})


@owner_write_guard
async def api_owner_subagents_mode(request: Request) -> JSONResponse:
    from ouroboros.gateway.settings import _run_settings_writer

    return await _run_settings_writer(_api_owner_subagents_mode_sync, request, await request_json_or(request, {}))


def _api_owner_subagents_mode_sync(request: Request, body: Any) -> JSONResponse:
    from ouroboros.config import VALID_SUBAGENTS_MODES

    mode = str(body.get("mode") or "").strip().lower() if isinstance(body, dict) else ""
    if mode not in VALID_SUBAGENTS_MODES:
        return unsaved_error("Mode must be off, auto or swarm.", 400, code="subagents_mode_invalid")
    try:
        project_id = request_project_id(body)
    except ValueError as exc:
        return unsaved_error(str(exc), 400, code="subagents_mode_invalid")
    with settings_document_mutation():
        if project_id:
            try:
                old, cognition = update_project_cognition(request_drive_root(request), project_id, subagents_mode=mode)
            except ValueError as exc:
                return unsaved_error(str(exc), 404, code="project_unavailable")
            previous = old["subagents_mode"]
        else:
            previous = main_cognition()["subagents_mode"]
            _owner_update_settings(lambda current: {**current, "OUROBOROS_SUBAGENTS_MODE": mode},
                                   settings_document_digest(), authored_keys=("OUROBOROS_SUBAGENTS_MODE",))
            os.environ["OUROBOROS_SUBAGENTS_MODE"] = mode
            cognition = main_cognition()
    _owner_audit(request, "subagents_mode", {"project_id": project_id, "subagents_mode": mode, "previous_subagents_mode": previous})
    return JSONResponse({"ok": True, "project_id": project_id, "subagents_mode": mode, "cognition": cognition})


@owner_write_guard
async def api_owner_chat_model(request: Request) -> JSONResponse:
    """Save a room's choice, or change Main itself from the Main composer."""
    from ouroboros.gateway.settings import _run_settings_writer

    return await _run_settings_writer(_api_owner_chat_model_sync, request, await request_json_or(request, {}))


def _api_owner_chat_model_sync(request: Request, body: Any) -> JSONResponse:
    if not isinstance(body, dict):
        return unsaved_error("JSON body must be an object with a kind.", 400, code="chat_model_invalid")
    try:
        choice = normalize_model_choice({key: value for key, value in body.items() if key != "project_id"})
        project_id = request_project_id(body)
    except ValueError as exc:
        return unsaved_error(str(exc), 400, code="chat_model_invalid")
    if not project_id and choice["kind"] != "model":
        return unsaved_error("Main chat chooses Main's model; only a Project can follow Main.",
                             400, code="chat_model_invalid")
    with settings_document_mutation():
        if project_id:
            try:
                previous, choice = update_project_model_choice(request_drive_root(request), project_id, choice)
            except ValueError as exc:
                return unsaved_error(str(exc), 404, code="project_unavailable")
            result = {"ok": True, "project_id": project_id, "model_choice": choice, "previous_model_choice": previous}
        else:
            result = _write_main_route(request, choice)
        if choice["kind"] == "model":
            from ouroboros.gateway.ui_preferences import record_model_recent
            from ouroboros.gateway.owner_settings import CommitBoundary, post_commit_failure_response

            try:
                record_model_recent(request_drive_root(request), choice["model"], choice["use_local"])
            except Exception as exc:
                boundary = CommitBoundary()
                boundary.commit()
                boundary.at("model recents")
                return post_commit_failure_response(exc, boundary)
    _owner_audit(request, "chat_model", result)
    return JSONResponse(result)


def _write_main_route(request: Request | None, choice: dict, *, origin: dict | None = None) -> dict:
    """Main's role writer owns the save; only same-source subscription pins survive."""
    from ouroboros.gateway.settings import _active_main_route
    from ouroboros.model_slots import (
        MODEL_ACCOUNTS_KEY, MODEL_CONTEXT_WINDOWS_KEY, apply_model_role_override,
        model_role_option, normalize_model_role_options,
    )
    from ouroboros.provider_models import parse_claudexor_model, provider_for_model

    result = {"ok": True, "project_id": ""}

    def transform(current):
        old = _active_main_route(current)
        model, local = choice["model"], choice["use_local"]
        subscription = not local and provider_for_model(model) == "claudexor"
        pin = str(model_role_option(MODEL_ACCOUNTS_KEY, "main", settings=current) or "")
        keep = (subscription and not old["use_local"] and old["provider"] == "claudexor"
                and parse_claudexor_model(model)[0] == parse_claudexor_model(old["model"])[0])
        updated = apply_model_role_override(current, role="main", model=model,
                                            credential_profile_id=pin if keep else "", use_local=local)
        reset = False
        if model != old["model"]:
            windows, _ = normalize_model_role_options(MODEL_CONTEXT_WINDOWS_KEY, current.get(MODEL_CONTEXT_WINDOWS_KEY))
            if "main" in windows:
                reset = bool(windows.pop("main"))
                updated[MODEL_CONTEXT_WINDOWS_KEY] = normalize_model_role_options(MODEL_CONTEXT_WINDOWS_KEY, windows)[1]
        result.update(main_route={"model": model, "use_local": local},
                      previous_main_route={key: old[key] for key in ("model", "use_local")},
                      account="kept" if pin and keep else "auto" if subscription else "none", window_reset=reset)
        return updated

    record_settings_change(request, _owner_update_settings(
        transform, settings_document_digest(), authored_keys=("OUROBOROS_MODEL", "USE_LOCAL_MAIN")), origin=origin)
    return result


__all__ = ["RANGE_FIELDS", "api_owner_chat_model", "api_owner_effort_range", "api_owner_subagents_mode",
           "effort_range_refusal"]
