"""Save the composer's effort, live Subagents choice, chat model and processing tier for its room."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from ouroboros.composer_cognition import (
    main_cognition, normalize_model_choice,
    request_project_id, update_project_cognition, update_project_model_choice, update_project_processing_choice,
)
from ouroboros.gateway._helpers import request_drive_root, request_json_or
from ouroboros.gateway.owner_settings import (
    BindingRefused,
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
    """The four composer capsules' owner writes (Model's tier its own), for Main or the named room.

    Their paths and handlers stay together here; the gateway mounts the family.
    Context's writer retains its settings/capability authority in ``settings``.
    """
    from ouroboros.gateway.settings import api_owner_context_mode

    return [
        Route("/api/owner/context-mode", endpoint=api_owner_context_mode, methods=["POST"]),
        Route("/api/owner/effort-range", endpoint=api_owner_effort_range, methods=["POST"]),
        Route("/api/owner/subagents-mode", endpoint=api_owner_subagents_mode, methods=["POST"]),
        Route("/api/owner/chat-model", endpoint=api_owner_chat_model, methods=["POST"]),
        Route("/api/owner/chat-processing", endpoint=api_owner_chat_processing, methods=["POST"]),
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
        return "The subagent range must be ordered min ≤ max."
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


def binding_condition_refusal(condition: Any) -> str:
    """Main's four shown settings, optional API connection and confirmed Ultrafast destination."""
    from ouroboros.gateway.owner_settings import CONNECTION_CONDITION, connection_condition_valid

    if (not isinstance(condition, dict) or not all(isinstance(token, str) and token for token in condition.values())
            or not MAIN_BINDING_KEYS <= set(condition) <= MAIN_BINDING_KEYS | {CONNECTION_CONDITION, "ultrafast"}
            or (CONNECTION_CONDITION in condition and not connection_condition_valid(condition[CONNECTION_CONDITION]))):
        return "Reload Main's model and account to get their current settings and connection."
    return ""


def named_connection(condition: Any) -> str:
    from ouroboros.gateway.owner_settings import CONNECTION_CONDITION

    return str((condition or {}).get(CONNECTION_CONDITION, "")).partition("#")[0]


def destination_refusal(condition: Any, connection: str, carries_ultrafast: bool) -> str:
    """For a read the lock found current: the revision must name the API ``connection`` the write's route sends to
    (none where it has none), and is required where that route carries Ultrafast; "" when it does."""
    named = named_connection(condition)
    if named and named != connection:
        return "Main's binding names another connection than its route's."
    if not named and connection and carries_ultrafast:
        return "Ultrafast on an API connection needs that connection's revision as shown."
    return ""


MAIN_BINDING_KEYS = {"OUROBOROS_MODEL", "USE_LOCAL_MAIN", "OUROBOROS_MODEL_ACCOUNTS", "OUROBOROS_MODEL_PROCESSING_PREFERENCES"}
MAIN_CONFIRMATION = ("Confirm Ultrafast for this model and account in Main or Settings → Models, "
                     "or choose Standard or Default for Main first.")


def _effective_main_route(settings: dict) -> dict:
    from ouroboros.gateway.settings import _active_main_route
    from ouroboros.server_runtime import apply_runtime_provider_defaults

    return _active_main_route(apply_runtime_provider_defaults(settings)[0])


def check_main_ultrafast(stored: dict, proposed: dict, condition: dict | None) -> None:
    """Compare the confirmed destination with what this locked write would actually enable or transfer."""
    from ouroboros.gateway.owner_settings import (
        CONNECTION_CONDITION, SettingsChangedElsewhere, connection_revision, settings_in_place_of,
    )
    from ouroboros.model_slots import MODEL_ACCOUNTS_KEY, model_role_option

    after = settings_in_place_of(proposed)
    if _main_tier(after) != "ultrafast":
        return

    def destination(settings):
        route = _effective_main_route(settings)
        pin = model_role_option(MODEL_ACCOUNTS_KEY, "main", settings=settings) if route["provider"] == "claudexor" else ""
        return [route["model"], route["use_local"], pin or ""]

    before = settings_in_place_of(stored)
    target = destination(after)
    model, local, pin = target
    connection = _api_connection(model, local)
    if connection:
        # A supplied revision already matched the locked read. An ordinary credentials save carries none.
        shown = (condition[CONNECTION_CONDITION] if named_connection(condition) == connection
                 else connection_revision(stored, connection, effective=before))
        if connection_revision(proposed, connection, effective=after) != shown:
            raise BindingRefused("Save the connection with Main on Standard or Default, then confirm Ultrafast "
                                 "for the updated connection in Main or Settings → Models.")
    if _main_tier(before) == "ultrafast" and target == destination(before):
        return
    if not condition or not MAIN_BINDING_KEYS <= condition.keys():
        raise BindingRefused(MAIN_CONFIRMATION)
    try:
        chosen = json.loads(condition.get("ultrafast", ""))
    except (TypeError, ValueError):
        chosen = None
    if not isinstance(chosen, list) or len(chosen) != 5 or chosen[:3] != target or type(chosen[1]) is not bool:
        raise BindingRefused(MAIN_CONFIRMATION)
    if refusal := destination_refusal(condition, connection, True):
        raise BindingRefused(refusal)
    store, fingerprint = chosen[3:]
    if pin:
        from ouroboros.gateway.models import observed_profile
        from ouroboros.provider_models import parse_claudexor_model

        if not all(isinstance(value, str) and value for value in (store, fingerprint)):
            raise BindingRefused("The catalog could not identify this pinned account. Refresh it, choose Auto, or choose Standard.")
        if observed_profile(parse_claudexor_model(model)[0], pin) != (store, fingerprint):
            raise SettingsChangedElsewhere("The pinned account changed or could not be verified. Reopen Models and choose again.")
    elif store != "" or fingerprint != "":
        raise BindingRefused("Auto and API routes do not bind an individual subscription account.")


def _api_connection(model: str, use_local: bool) -> str:
    """The picker id of the API connection ``model`` sends to (``kind`` or ``kind@name``); "" for the subscription
    engine, the local model or an invalid route, which have no such connection."""
    from ouroboros.provider_models import PROVIDER_KINDS, connection_id, parse_model_route

    parts = parse_model_route(model)
    return connection_id(parts.kind, parts.connection) if not use_local and parts.valid and not parts.local \
        and parts.kind in PROVIDER_KINDS else ""


def _main_tier(current: Dict[str, Any]) -> str:
    """Main's own tier in the document the lock read (an unreadable map holds none; its token binds the write)."""
    from ouroboros.model_slots import MODEL_PROCESSING_PREFERENCES_KEY, model_role_option

    try:
        return str(model_role_option(MODEL_PROCESSING_PREFERENCES_KEY, "main", settings=current) or "")
    except ValueError:
        return ""


def _api_owner_chat_model_sync(request: Request, body: Any) -> JSONResponse:
    if not isinstance(body, dict):
        return unsaved_error("JSON body must be an object with a kind.", 400, code="chat_model_invalid")
    try:
        choice = normalize_model_choice({key: value for key, value in body.items() if key not in ("project_id", "if_unchanged")})
        project_id = request_project_id(body)
    except ValueError as exc:
        return unsaved_error(str(exc), 400, code="chat_model_invalid")
    if not project_id and choice["kind"] != "model":
        return unsaved_error("Main chat chooses Main's model; only a Project can follow Main.",
                             400, code="chat_model_invalid")
    # Main's model on the binding the owner was shown (its tier included): a move elsewhere answers 409, never a tier
    # that came meanwhile carried to the new model unasked.
    # The form and the body's own destination are checked here; the stored state only under the lock.
    condition = body.get("if_unchanged")
    if condition is not None and (project_id or choice["kind"] != "model" or binding_condition_refusal(condition)
                                  or destination_refusal(condition, _api_connection(choice["model"], choice["use_local"]), False)):
        return unsaved_error("Only Main's model accepts its binding read tokens, exactly.", 400, code="chat_model_invalid")
    with settings_document_mutation():
        if project_id:
            try:
                previous, choice = update_project_model_choice(request_drive_root(request), project_id, choice)
            except ValueError as exc:
                return unsaved_error(str(exc), 404, code="project_unavailable")
            result = {"ok": True, "project_id": project_id, "model_choice": choice, "previous_model_choice": previous}
        else:
            try:
                result = _write_main_route(request, choice, if_unchanged=condition)
            except BindingRefused as exc:
                return unsaved_error(str(exc), 400, code="chat_model_invalid")
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


@owner_write_guard
async def api_owner_chat_processing(request: Request) -> JSONResponse:
    """Save a room's processing tier, or Main's own from the Main composer; ``""`` is Default."""
    from ouroboros.gateway.settings import _run_settings_writer

    return await _run_settings_writer(_api_owner_chat_processing_sync, request, await request_json_or(request, {}))


def _api_owner_chat_processing_sync(request: Request, body: Any) -> JSONResponse:
    from ouroboros.model_slots import (
        MODEL_PROCESSING_PREFERENCES_KEY, PROCESSING_PREFERENCE_KEY, normalize_call_tier, normalize_model_role_options,
        normalize_processing_preference,
    )

    from ouroboros.gateway.client_contracts import OwnerChatProcessingRequest
    from ouroboros.gateway.schema import validate_ingress

    if errors := validate_ingress(body, OwnerChatProcessingRequest):  # the card's tier writer's grammar too
        return unsaved_error(errors[0], 400, code="chat_processing_invalid")
    try:
        project_id = request_project_id(body)
        tier = normalize_call_tier(body["processing"])
    except ValueError as exc:
        return unsaved_error(str(exc), 400, code="chat_processing_invalid")
    condition = body.get("if_unchanged")
    if condition is not None:
        if project_id or tier != "ultrafast":
            return unsaved_error("Only Main Ultrafast accepts settings conditions.", 400, code="chat_processing_invalid")
        if refusal := binding_condition_refusal(condition):
            return unsaved_error(refusal, 400, code="chat_processing_invalid")
    with settings_document_mutation():
        if project_id:
            try:
                previous, tier = update_project_processing_choice(request_drive_root(request), project_id, tier)
            except ValueError as exc:
                return unsaved_error(str(exc), 404, code="project_unavailable")
            result = {"ok": True, "project_id": project_id, "processing_choice": tier,
                      "previous_processing_choice": previous}
        else:
            # Main's own tier is its role option; Default removes it, so Main takes the default tier again.
            # Nothing else of Main moves: its model, account and window stay as they are.
            result = {"ok": True, "project_id": ""}

            def transform(current):
                roles, _ = normalize_model_role_options(MODEL_PROCESSING_PREFERENCES_KEY,
                                                        current.get(MODEL_PROCESSING_PREFERENCES_KEY))
                previous = str(roles.get("main") or "")
                result.update(main_processing=tier, previous_main_processing=previous,
                              processing_default=normalize_processing_preference(current.get(PROCESSING_PREFERENCE_KEY)))
                if previous == tier:
                    return None
                if tier:
                    roles["main"] = tier
                else:
                    roles.pop("main", None)
                return {**current, MODEL_PROCESSING_PREFERENCES_KEY: normalize_model_role_options(
                    MODEL_PROCESSING_PREFERENCES_KEY, roles)[1]}

            def check_unchanged(document):
                from ouroboros.gateway.owner_settings import settings_in_place_of

                main = _effective_main_route(settings_in_place_of(document))
                if refusal := destination_refusal(condition, _api_connection(main["model"], main["use_local"]), True):
                    raise BindingRefused(refusal)

            digest = settings_document_digest() if condition is None else ""
            try:
                record_settings_change(request, _owner_update_settings(
                    transform, digest, authored_keys=(MODEL_PROCESSING_PREFERENCES_KEY,), if_unchanged=condition,
                    check_unchanged=check_unchanged if condition is not None else None))
            except BindingRefused as exc:
                return unsaved_error(str(exc), 400, code="chat_processing_invalid")
    _owner_audit(request, "chat_processing", result)
    return JSONResponse(result)


def _write_main_route(request: Request | None, choice: dict, *, origin: dict | None = None,
                      if_unchanged: dict | None = None) -> dict:
    """Main's role writer owns the save; only same-source subscription pins survive. ``if_unchanged``: the binding
    read the owner chose on, compared under the lock; a current one that carries Ultrafast onto an API connection
    without naming it raises ``BindingRefused``."""
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
        if if_unchanged is not None and (refusal := destination_refusal(
                if_unchanged, _api_connection(model, local), _main_tier(current) == "ultrafast")):
            raise BindingRefused(refusal)
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
        transform, settings_document_digest() if if_unchanged is None else "", authored_keys=("OUROBOROS_MODEL", "USE_LOCAL_MAIN"),
        if_unchanged=if_unchanged), origin=origin)
    return result


__all__ = ["RANGE_FIELDS", "api_owner_chat_model", "BindingRefused", "binding_condition_refusal", "destination_refusal", "api_owner_chat_processing", "api_owner_effort_range",
           "api_owner_subagents_mode", "effort_range_refusal"]
