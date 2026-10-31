"""Owner decisions about safety: two levels, rules and the resolved checkers."""

from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.gateway.owner_settings import (
    _owner_audit, _owner_read_settings_raw, _owner_update_settings, owner_write_guard,
    settings_document_digest, settings_document_mutation, unsaved_error,
)

# Models inherits Light then Main. The same owner transaction can edit that
# inherited route or the catalog mark without exposing a generic-save bypass.
SAFETY_MODEL_KEYS = frozenset({
    "OUROBOROS_MODEL_SAFETY", "OUROBOROS_MODEL_LIGHT", "OUROBOROS_MODEL",
    "USE_LOCAL_LIGHT", "USE_LOCAL_MAIN", "OUROBOROS_MODEL_ACCOUNTS",
    "OUROBOROS_MODEL_CONTEXT_WINDOWS", "OUROBOROS_MODEL_PROCESSING_PREFERENCES",
    "OUROBOROS_SUBAGENTS",
})


def audit_safety_change(request: Request, before: dict, after: dict) -> None:
    """Record resolved checker changes, including unauthored provider proposals."""
    from ouroboros.config import safety_configuration, get_runtime_mode

    previous, current = safety_configuration(before), safety_configuration(after)
    if previous != current:
        keys = SAFETY_MODEL_KEYS | {"OUROBOROS_SAFETY_MODE", "OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "OUROBOROS_SAFETY_OWNER_RULES"}
        _owner_audit(request, "safety_settings", {"runtime_mode": get_runtime_mode(),
                    "changed_settings": sorted(k for k in keys if before.get(k) != after.get(k)),
                    "previous_checker": previous[0], "checker": current[0]})


async def _write(request: Request, writer: Any) -> JSONResponse:
    from ouroboros.gateway.settings import _json_body_or_empty, _run_settings_writer

    return await _run_settings_writer(writer, request, await _json_body_or_empty(request))


@owner_write_guard
async def api_owner_safety_mode(request: Request) -> JSONResponse:
    return await _write(request, _api_owner_safety_mode_sync)


def _api_owner_safety_mode_sync(request: Request, body: Any) -> JSONResponse:
    from ouroboros import config

    if not isinstance(body, dict):
        return unsaved_error("Expected a JSON object", 400)
    kind = body.get("kind", "your_work")
    if kind not in {"your_work", "consciousness"}:
        return unsaved_error("'kind' must be your_work or consciousness", 400)
    mode = str(body.get("mode") or "").strip().lower()
    if mode not in config.VALID_SAFETY_MODES:
        return unsaved_error("'mode' must be one of: full, light, off", 400)
    key = "OUROBOROS_CONSCIOUSNESS_SAFETY_MODE" if kind == "consciousness" else "OUROBOROS_SAFETY_MODE"
    _persist(request, {key: mode}, "safety_mode", {"kind": kind, "safety_mode": mode})
    return JSONResponse({"ok": True, "safety_mode": mode})


@owner_write_guard
async def api_owner_safety_rules(request: Request) -> JSONResponse:
    return await _write(request, _api_owner_safety_rules_sync)


def _api_owner_safety_rules_sync(request: Request, body: Any) -> JSONResponse:
    rules = body.get("rules") if isinstance(body, dict) else None
    if not isinstance(rules, str) or len(rules.encode("utf-8")) > 64_000:
        return unsaved_error("'rules' must be text of at most 64,000 UTF-8 bytes", 400)
    _persist(request, {"OUROBOROS_SAFETY_OWNER_RULES": rules}, "safety_rules", {"rules_bytes": len(rules.encode("utf-8"))})
    return JSONResponse({"ok": True, "rules": rules})


@owner_write_guard
async def api_owner_safety_model(request: Request) -> JSONResponse:
    return await _write(request, _api_owner_safety_model_sync)


def _api_owner_safety_model_sync(request: Request, body: Any) -> JSONResponse:
    settings = body.get("settings") if isinstance(body, dict) else None
    if not isinstance(settings, dict) or not settings or set(settings) - SAFETY_MODEL_KEYS:
        return unsaved_error("'settings' must contain Safety Supervisor model or catalog settings only", 400)
    from ouroboros.configured_subagents import normalize_configured_subagents, roster_save_error
    from ouroboros.config import load_settings, safety_configuration
    from ouroboros.gateway.settings import ALLOW_EMPTY_REVIEW_POOL, review_pool_save_judgement

    settings = dict(settings)
    try:
        for key, value in settings.items():
            if key in {"OUROBOROS_MODEL", "OUROBOROS_MODEL_LIGHT", "OUROBOROS_MODEL_SAFETY"} and not isinstance(value, str):
                raise ValueError(f"{key} must be a model string")
            if key.startswith("USE_LOCAL_") and not isinstance(value, bool):
                raise ValueError(f"{key} must be a boolean")
        if "OUROBOROS_SUBAGENTS" in settings:
            canonical = normalize_configured_subagents(settings["OUROBOROS_SUBAGENTS"])[1]
            stored = load_settings()
            roster_error = roster_save_error(canonical, stored, settings)
            if roster_error:
                return unsaved_error(roster_error, 400)
            pool_error = review_pool_save_judgement(canonical, stored, allow_empty=body.get(ALLOW_EMPTY_REVIEW_POOL) is True)
            if pool_error:
                return unsaved_error(pool_error, 400, code="empty_review_pool")
            settings["OUROBOROS_SUBAGENTS"] = canonical
        safety_configuration(settings)
        _persist(request, settings, "safety_model", {"changed_settings": sorted(settings)})
    except (TypeError, ValueError) as exc:
        return unsaved_error(str(exc), 400)
    return JSONResponse({"ok": True})


def _persist(request: Request, changes: dict, action: str, audit: dict) -> dict:
    from ouroboros.config import apply_settings_to_env

    def merge(current):
        proposed = {**current, **changes}
        if action == "safety_model":
            from ouroboros.config import load_settings_lock_held
            from ouroboros.gateway.settings import REVIEW_LANES_KEY
            from ouroboros.server_runtime import has_startup_ready_provider

            # Provider access can come from the operator environment.
            effective = {**load_settings_lock_held(), **changes}
            if str(effective.get("LOCAL_MODEL_SOURCE", "") or "").strip() and not has_startup_ready_provider(effective):
                raise ValueError("Local-only setups must route at least one model to the local runtime.")
            if changes.get("OUROBOROS_SUBAGENTS") not in (None, ""):
                proposed.pop(REVIEW_LANES_KEY, None)
        return proposed

    with settings_document_mutation():
        digest = settings_document_digest()
        previous = _owner_read_settings_raw()
        _owner_update_settings(merge, digest,
                               authored_keys=tuple(changes), allow_safety_lowering=True)
        # Projection only; the owner writer has already released its file lock.
        from ouroboros.config import load_settings
        apply_settings_to_env(load_settings())
    _owner_audit(request, action, {**audit, "previous": {key: previous.get(key) for key in changes}})
    return previous
