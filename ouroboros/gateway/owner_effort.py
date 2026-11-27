"""Save the composer's effort, live Subagents choice, chat model and processing tier for its room."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Mapping

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from ouroboros.composer_cognition import (
    MAIN_CHAT_MODEL_KEY, MAIN_CHAT_PROCESSING_KEY, chat_consent_word, main_chat_choice, main_chat_saved,
    main_chat_route_setting, main_cognition, model_choice_token, normalize_model_choice,
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
from ouroboros.provider_connections import api_connection_for_model as _api_connection
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
    """Save the model new work starts on in the named room, or in the Main chat; no role's model moves."""
    from ouroboros.gateway.settings import _run_settings_writer

    return await _run_settings_writer(_api_owner_chat_model_sync, request, await request_json_or(request, {}))


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
MAIN_CONFIRMATION = ("Confirm Ultrafast for this model and account in Settings → Models, "
                     "or choose Standard or Default for Main first.")


def _effective_main_route(settings: dict) -> dict:
    from ouroboros.gateway.settings import _active_main_route
    from ouroboros.server_runtime import apply_runtime_provider_defaults

    return _active_main_route(apply_runtime_provider_defaults(settings)[0])


def _main_destination(settings: dict) -> list:
    """``[model, use_local, pin]`` Main's own work runs on in ``settings``, as its Ultrafast rule names a destination:
    the active route by the resolver every route consumer shares, with its pinned subscription account (``""``:
    Auto, an API route or the local lane)."""
    from ouroboros.model_slots import MODEL_ACCOUNTS_KEY, model_role_option

    route = _effective_main_route(settings)
    pin = model_role_option(MODEL_ACCOUNTS_KEY, "main", settings=settings) if route["provider"] == "claudexor" else ""
    return [route["model"], route["use_local"], pin or ""]


def check_main_ultrafast(stored: dict, proposed: dict, condition: dict | None) -> None:
    """Compare the confirmed destination with what this locked write would actually enable or transfer."""
    from ouroboros.gateway.owner_settings import (
        CONNECTION_CONDITION, SettingsChangedElsewhere, connection_revision, settings_in_place_of,
    )

    after = settings_in_place_of(proposed)
    if _main_tier(after) != "ultrafast":
        return
    before = settings_in_place_of(stored)
    target = _main_destination(after)
    model, local, pin = target
    connection = _api_connection(model, local)
    if connection:
        # A supplied revision already matched the locked read. An ordinary credentials save carries none.
        shown = (condition[CONNECTION_CONDITION] if named_connection(condition) == connection
                 else connection_revision(stored, connection, effective=before))
        if connection_revision(proposed, connection, effective=after) != shown:
            raise BindingRefused("Save the connection with Main on Standard or Default, then confirm Ultrafast "
                                 "for the updated connection in Settings → Models.")
    if _main_tier(before) == "ultrafast" and target == _main_destination(before):
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


def _main_tier(current: Dict[str, Any]) -> str:
    """Main's own tier in the document the lock read (an unreadable map holds none; its token binds the write)."""
    from ouroboros.model_slots import MODEL_PROCESSING_PREFERENCES_KEY, model_role_option

    try:
        return str(model_role_option(MODEL_PROCESSING_PREFERENCES_KEY, "main", settings=current) or "")
    except ValueError:
        return ""


# A page loaded before the Main chat kept its own model and tier sends these two writes with no scope of their own:
# there a body without ``project_id`` meant Main the role (its model with Main's binding tokens, its tier bare).
STALE_COMPOSER = "Reload the page: this composer saves its chat's own model and tier now, never Main's."
MAIN_CHAT_SCOPE = "main_chat"


def _scope_refusal(body: dict, project_id: str) -> str:
    """Why a composer write names no single scope, or ``""``. A room is named by its ``project_id`` alone, as it
    always was; the Main chat by ``scope: "main_chat"``. A body with neither is the older page's write of Main the
    role: it is refused with the way out, never answered as saved over another setting."""
    if project_id:
        return "A room's write names its project_id alone." if "scope" in body else ""
    return "" if body.get("scope") == MAIN_CHAT_SCOPE else STALE_COMPOSER


def _api_owner_chat_model_sync(request: Request, body: Any) -> JSONResponse:
    if not isinstance(body, dict):
        return unsaved_error("JSON body must be an object with a kind.", 400, code="chat_model_invalid")
    if "if_unchanged" in body:
        return unsaved_error(STALE_COMPOSER, 400, code="chat_model_invalid")
    try:
        choice = normalize_model_choice({key: value for key, value in body.items() if key != "project_id"})
        project_id = request_project_id(body)
    except ValueError as exc:
        return unsaved_error(str(exc), 400, code="chat_model_invalid")
    if refusal := _scope_refusal(body, project_id):
        return unsaved_error(refusal, 400, code="chat_model_invalid")
    with settings_document_mutation():
        if project_id:
            try:
                previous, choice = update_project_model_choice(request_drive_root(request), project_id, choice)
            except ValueError as exc:
                return unsaved_error(str(exc), 404, code="project_unavailable")
        else:
            previous, choice = write_main_chat_model(request, choice)
        result = {"ok": True, "project_id": project_id, "model_choice": choice, "previous_model_choice": previous,
                  "choice_token": model_choice_token(project_id, choice)}
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
    """Save the processing tier of new work in the named room, or in the Main chat; ``""`` is Default."""
    from ouroboros.gateway.settings import _run_settings_writer

    return await _run_settings_writer(_api_owner_chat_processing_sync, request, await request_json_or(request, {}))


def _api_owner_chat_processing_sync(request: Request, body: Any) -> JSONResponse:
    from ouroboros.model_slots import normalize_call_tier

    from ouroboros.gateway.client_contracts import OwnerChatProcessingRequest
    from ouroboros.gateway.schema import validate_ingress

    if isinstance(body, dict) and "if_unchanged" in body:
        return unsaved_error(STALE_COMPOSER, 400, code="chat_processing_invalid")
    if errors := validate_ingress(body, OwnerChatProcessingRequest):  # the card's tier writer's grammar too
        return unsaved_error(errors[0], 400, code="chat_processing_invalid")
    try:
        project_id = request_project_id(body)
        tier = normalize_call_tier(body["processing"])
    except ValueError as exc:
        return unsaved_error(str(exc), 400, code="chat_processing_invalid")
    if refusal := _scope_refusal(body, project_id):
        return unsaved_error(refusal, 400, code="chat_processing_invalid")
    try:
        asked = _chat_confirmation(body.get("confirmed")) if tier == "ultrafast" and not project_id else None
        with settings_document_mutation():
            if project_id:
                try:
                    previous, tier = update_project_processing_choice(request_drive_root(request), project_id, tier)
                except ValueError as exc:
                    return unsaved_error(str(exc), 404, code="project_unavailable")
            else:
                previous, tier = write_main_chat_processing(request, tier, confirmed=asked)
    except ChatRouteUnconfirmed as exc:
        return unsaved_error(str(exc), 409, code="settings_changed_elsewhere")
    result = {"ok": True, "project_id": project_id, "processing_choice": tier, "previous_processing_choice": previous}
    _owner_audit(request, "chat_processing", result)
    return JSONResponse(result)


def _write_main_chat(request: Request | None, key: str, value: str, origin: dict | None, *,
                     confirmed: list | None = None) -> dict:
    """One of the Main chat's own two settings, written under the document's lock; a value already saved writes
    nothing. Neither is a role's key, so Main's model, account, window and tier stay as they are. ``confirmed``: what
    an Ultrafast was asked for, which that locked read judges before anything lands (``keep_chat_ultrafast``), a
    tier already saved included. Returns the chat's choice as that read held it."""
    held: dict = {}

    def transform(current):
        held.update(main_chat_choice(current))
        return None if confirmed is None and str(current.get(key) or "") == value else {**current, key: value}

    transform.chat_ultrafast = confirmed  # server-made: no request names it
    record_settings_change(request, _owner_update_settings(transform, settings_document_digest(), authored_keys=(key,)),
                           origin=origin)
    return held


def write_main_chat_model(request: Request | None, choice: dict, *, origin: dict | None = None) -> tuple[dict, dict]:
    """Save the model new work in the Main chat starts on, which a new room copies; Main's reference follows Main.
    The composer's capsule and a cited ``switch_model(scope="new")`` share this writer. ``(previous, saved)``."""
    current = normalize_model_choice(choice)
    held = _write_main_chat(request, MAIN_CHAT_MODEL_KEY, main_chat_route_setting(current), origin)
    return held["model_choice"], current


class ChatRouteUnconfirmed(ValueError):
    """The Main chat's Ultrafast was not asked for the destination its new work runs on now: nothing is saved."""


ROUTE_NOT_ASKED = ("The model or profile new work runs on here is not the one Ultrafast was asked for, so it was not "
                   "set. Choose Ultrafast again.")
CONNECTION_NOT_SHOWN = ("Ultrafast on an API connection needs that connection's revision as shown. "
                        "Reopen the model sheet and choose Ultrafast again.")
CONNECTION_MOVED = ("The connection this model is sent through changed after Ultrafast was asked for, so it was "
                    "not set. Choose Ultrafast again.")


def _chat_destination(settings: Mapping, choice: Mapping | None = None) -> list:
    """``[model, use_local, pin]`` new work in the Main chat runs on in ``settings``: its own model on Auto, else
    Main's destination (``_main_destination``). ``choice``: the chat's model choice where the caller holds it."""
    choice = main_chat_saved(settings)[0] if choice is None else choice
    return ([choice["model"], bool(choice["use_local"]), ""] if choice["kind"] == "model"
            else _main_destination(dict(settings)))


def chat_ultrafast_route(settings: Mapping, choice: Mapping, consent: str) -> tuple[list, str] | None:
    """The confirmed route and, for API, the random revision that still identifies its connection. A read only;
    a missing or changed connection cannot come back on the old word. Named profiles need no engine witness."""
    from ouroboros.provider_connections import live_connection_account

    try:
        destination = _chat_destination(settings, choice)
        account = ""
        if _api_connection(*destination[:2]):
            account = live_connection_account(destination[0])
            if not account:
                return None
        return (destination, account) if consent and consent == chat_consent_word(destination, account) else None
    except Exception:  # no destination is named: nothing is confirmed
        import logging

        logging.getLogger(__name__).warning("The Main chat's destination could not be read; its Ultrafast is withheld",
                                            exc_info=True)
        return None


def _chat_confirmation(confirmed: Any) -> list:
    """What a Main chat's Ultrafast was asked for, ``[model, use_local, account, connection revision]``, from the
    request's ``confirmed`` (``work_contracts.ConfirmedChatUltrafast``): the destination by name and, on an API
    route, its connection as shown. Names only: no account behind a profile is identified, so nothing is asked of
    the engine. The locked write compares destination and revision with its own read (``keep_chat_ultrafast``).
    ``ChatRouteUnconfirmed`` otherwise."""
    from ouroboros.gateway.owner_settings import CONNECTION_CONDITION, connection_condition_valid

    confirmed = confirmed if isinstance(confirmed, dict) else {}
    model, local, account = (confirmed.get(key) for key in ("model", "use_local", "account"))
    if not (isinstance(model, str) and model and type(local) is bool and isinstance(account, str)):
        raise ChatRouteUnconfirmed(ROUTE_NOT_ASKED)
    revision = confirmed.get(CONNECTION_CONDITION, "")
    if not isinstance(revision, str) or (revision and not connection_condition_valid(revision)):
        raise ChatRouteUnconfirmed(CONNECTION_NOT_SHOWN)
    return [model, local, account, revision]


def write_main_chat_processing(request: Request | None, tier: Any, *, confirmed: list | None = None,
                               origin: dict | None = None) -> tuple[str, str]:
    """Save the Main chat's own processing tier for new work there (``""``: Default). ``(previous, saved)``.

    Ultrafast is its owner's consent to one destination's price: ``confirmed`` is what they were asked for
    (``_chat_confirmation``), and the write's one locked read judges it (``keep_chat_ultrafast``).
    ``ChatRouteUnconfirmed`` or the settings writer's own refusal otherwise."""
    from ouroboros.model_slots import normalize_call_tier

    current = normalize_call_tier(tier)
    if current == "ultrafast" and confirmed is None:
        raise ChatRouteUnconfirmed(ROUTE_NOT_ASKED)
    # The locked writer stamps the destination's word using its own current connection revision.
    held = _write_main_chat(request, MAIN_CHAT_PROCESSING_KEY, current, origin,
                            confirmed=confirmed if current == "ultrafast" else None)
    return held["processing_choice"], current


def keep_chat_ultrafast(stored: dict, proposed: dict, revisions: dict, confirmed: list | None = None) -> None:
    """The Main chat's Ultrafast, judged on the write's one locked read, ``stored``, and the document as it would
    land, ``proposed`` (changed in place).

    It stands on one destination: the model, locality and profile name its new work runs on (``_chat_destination``,
    the resolver Main's own rule uses) and what that route's API connection sends to, compared by the connection's
    revision in this process and never stored. Only the chat's own write sets it, with the destination and revision
    its owner was asked for (``confirmed``, ``_chat_confirmation``): ``ChatRouteUnconfirmed`` where the read holds
    another. The saved word also names the planned random API revision, never the confirmation receipt (which is
    process-local). Other writes only preserve the word when route and API connection are unchanged. Admitted work
    never reads the chip again; its API calls compare their own connection (``route_choice.task_processing``)."""
    from ouroboros.gateway.owner_settings import connection_revision, settings_in_place_of

    saved = proposed.get(MAIN_CHAT_PROCESSING_KEY)
    if not isinstance(saved, str) or saved.partition("@")[0] != "ultrafast":
        return
    try:
        before, after = settings_in_place_of(stored), settings_in_place_of(proposed)
        destination = _chat_destination(after)
        connection = _api_connection(destination[0], destination[1])
        shown = connection_revision(stored, connection, effective=before) if connection else ""
        same = (_chat_destination(before) == destination
                and (connection_revision(proposed, connection, effective=after) if connection else "") == shown)
    except Exception as exc:  # an account map or connection that cannot be read names no destination
        if confirmed is not None:
            raise ChatRouteUnconfirmed("Main's account could not be read, so Ultrafast was not set.") from exc
        import logging

        logging.getLogger(__name__).warning("The Main chat's destination could not be read; its Ultrafast ends",
                                            exc_info=True)
        proposed[MAIN_CHAT_PROCESSING_KEY] = ""
        return
    if confirmed is not None:
        if not same or confirmed[:3] != destination:
            raise ChatRouteUnconfirmed(ROUTE_NOT_ASKED)
        if confirmed[3] != shown:
            raise ChatRouteUnconfirmed(CONNECTION_MOVED if connection and confirmed[3].partition("#")[0] == connection
                                       else CONNECTION_NOT_SHOWN)
        revision = revisions["after"].get(connection, "")
        if connection and not revision:
            raise ChatRouteUnconfirmed("This API connection has no saved revision, so Ultrafast was not set.")
        account = f"api:{connection}#{revision}" if connection else ""
        proposed[MAIN_CHAT_PROCESSING_KEY] = f"ultrafast@{chat_consent_word(destination, account)}"
    elif not (same and stored.get(MAIN_CHAT_PROCESSING_KEY) == saved and (
            not connection or revisions["before"].get(connection) == revisions["after"].get(connection))):
        proposed[MAIN_CHAT_PROCESSING_KEY] = ""


def _write_main_route(request: Request | None, choice: dict, *, origin: dict | None = None) -> dict:
    """Main's role writer owns the save (a cited ``switch_model(scope="main")``; Settings › Models has its own):
    only same-source subscription pins survive, and a manual window goes with a changed model."""
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


__all__ = ["RANGE_FIELDS", "api_owner_chat_model", "BindingRefused", "destination_refusal", "api_owner_chat_processing", "api_owner_effort_range",
           "api_owner_subagents_mode", "effort_range_refusal", "write_main_chat_model", "write_main_chat_processing"]
