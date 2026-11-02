"""Built-in web_search Source/Model resolution, shared by dispatch and projections.

This setting selects only the built-in transport. Skills, MCP and browser tools
remain independent choices. Resolution reads settings; it performs no provider I/O.
"""
from __future__ import annotations

import importlib.util
import sys

from ouroboros.provider_models import legacy_openai_base

# The model Ouroboros recommends for each built-in source, the one home of that choice: curated, never
# a ranking of model names or catalogs. A blank saved Model follows it; a saved model stays as saved.
# Checked 2026-11-01; re-check the source's own guide before changing its row:
# OpenAI recommends gpt-5.5 with Responses web_search (developers.openai.com/api/docs/guides/tools-web-search);
# OpenRouter lists openrouter:web_search on the openai/gpt-5.5 endpoints
# (openrouter.ai/docs/guides/features/server-tools/web-search; GET /api/v1/models/openai/gpt-5.5/endpoints);
# Anthropic lists claude-sonnet-5-5 (platform.claude.com/docs/en/about-claude/models/overview), and its
# web_search_20250305 tool (llm_anthropic.py) serves every model. If stale: a retired id fails its own
# leg loudly (Auto then tries the next source); a superseded one keeps answering until a release moves it.
SEARCH_MODELS = {"openai": "gpt-5.5", "openrouter": "openai/gpt-5.5", "anthropic": "claude-sonnet-5-5", "ddgs": ""}
SEARCH_BACKENDS = {"openai": "openai_responses", "openrouter": "openrouter_server_tool", "anthropic": "anthropic_server_tool", "ddgs": "ddgs"}


def resolve_web_search_route(*, backend=None, model=None, settings=None, per_call=False) -> dict:
    """Resolve from an explicit settings document, else from what this process's dispatch reads.

    ``model`` stands in for the saved selection (a Settings draft keeps the saved meaning). With
    ``per_call`` a nonblank ``model`` is the one a ``web_search`` call named, and it is mandatory:
    only legs that send that model are candidates, so a model no built-in transport serves is an
    error, never Auto's other models or ddgs."""
    from ouroboros.config import runtime_setting

    setting = settings.get if settings is not None else runtime_setting
    source = str(setting("OUROBOROS_WEBSEARCH_BACKEND") or "auto") if backend is None else str(backend or "auto")
    source = source.strip().lower()
    selected = str(setting("OUROBOROS_WEBSEARCH_MODEL") or "") if model is None else str(model or "")
    selected = selected.strip()
    strict = per_call and bool(selected)
    result = {"scope": "builtin_web_search", "source": source, "model": selected, "legs": [], "unavailable": [],
              "note": "Applies only to built-in web_search; skills, MCP and browser tools remain independent choices."
                      + (" This call named its model: only these legs send it (a routed model or named connection has one),"
                         " and no other model or ddgs substitutes." if strict else "")}
    if source not in {"auto", *SEARCH_MODELS}:
        return {**result, "error": "Unknown web search source"}
    from ouroboros.provider_models import parse_model_route

    sep, parsed = "::" in selected, parse_model_route(selected)
    # A known route head (``kind[@name]::``) names its connection; an unknown one stays its own text.
    prefix, native = ((parsed.kind, parsed.model) if sep and parsed.model != selected
                      else selected.partition("::")[::2])
    connection = parsed.connection if parsed.valid else ""
    legacy_anthropic = selected.partition("/")[2] if not sep and selected.startswith("anthropic/") else ""
    unsupported = ("This model route has no built-in search transport"
                   if sep and (prefix not in {"openai", "anthropic", "openrouter"} or not native or not parsed.valid) else
                   "Direct search sources require a native model id"
                   if sep and prefix in {"openai", "anthropic"} and "/" in native else "")
    if source == "ddgs":
        if strict:  # a named model is never silently dropped
            return {**result, "error": "The ddgs source takes no model; omit model to search with ddgs"}
        candidates = [("ddgs", "")]  # ddgs takes no model: a saved model of any route is irrelevant here
    elif unsupported and (strict or source != "auto"):
        return {**result, "error": unsupported}  # a strict source or a named model never substitutes another
    elif source == "auto":
        if unsupported:
            # Auto keeps the legs that need no model and says the selection is not applied.
            result["unapplied_model"] = selected
            result["note"] += f" {unsupported}, so Auto does not apply the saved model and uses the recommended models."
            candidates = [("anthropic", SEARCH_MODELS["anthropic"]), ("ddgs", "")]
        elif not selected:
            candidates = list(SEARCH_MODELS.items())
        else:
            # The selection's own legs send its model; Auto's recommendation and ddgs follow only a saved one.
            if sep:
                if prefix == "openrouter" and "/" not in native:
                    native = f"openai/{native}"
                own = ([("openai", native), ("openrouter", f"openai/{native}")] if prefix == "openai" and not strict
                       else [(prefix, native)])  # a named route or connection is the call's whole choice
                rest = [("ddgs", "")]
            elif "/" in selected:
                # A legacy anthropic/<model> keeps the owner's model on the direct Anthropic leg too.
                own = [("openrouter", selected)] + ([("anthropic", legacy_anthropic)] if legacy_anthropic else [])
                rest = ([] if legacy_anthropic else [("anthropic", SEARCH_MODELS["anthropic"])]) + [("ddgs", "")]
            else:
                own = [("openai", selected), ("openrouter", f"openai/{selected}")]
                rest = [("anthropic", SEARCH_MODELS["anthropic"]), ("ddgs", "")]
            candidates = own if strict else own + rest
    else:
        if sep and prefix != source:
            return {**result, "error": "The selected model belongs to a different search source"}
        chosen = native if sep else selected
        if source == "openai" and "/" in chosen:
            return {**result, "error": "OpenAI search requires its native model id"}
        if source == "openrouter" and chosen and "/" not in chosen:
            chosen = f"openai/{chosen}"
        if source == "anthropic" and not sep and chosen:
            if chosen.startswith("anthropic/"):
                chosen = chosen.partition("/")[2]
            elif model is None:
                # Only persisted bare values carry the old ignored-model meaning.
                # A per-call native id (even the same spelling) is an explicit choice.
                result["ignored_legacy_model"] = chosen
                chosen = ""
                result["note"] += " The saved bare model was not applied by Anthropic; select anthropic::model to change it."
            if "/" in chosen:
                return {**result, "error": "Anthropic search requires its native model id"}
        candidates = [(source, chosen or SEARCH_MODELS[source])]
    for name, wire_model in candidates:
        reason, leg_connection = "", connection if name == prefix else ""
        if name != "ddgs" and not _leg_configured(name, leg_connection, settings):
            reason = "no_credentials"
        elif name == "openai" and not leg_connection and legacy_openai_base(settings):
            reason = "legacy_openai_base_url"  # Legacy compatible credentials are not official Responses access.
        elif name in _SDK_PACKAGES and not _package_discoverable(_SDK_PACKAGES[name]):
            reason = "package_missing"  # discoverable is not proof of a working import; dispatch still reports that
        explicit_model = (bool(selected) and name != "ddgs" and not result.get("ignored_legacy_model")
                          and not result.get("unapplied_model"))
        if source == "auto" and name == "anthropic" and prefix != "anthropic" and not legacy_anthropic:
            explicit_model = False
        leg = {"source": name, "backend": SEARCH_BACKENDS[name], "model": wire_model,
               "model_source": "selection" if explicit_model else "provider_default",
               **({"connection": leg_connection} if leg_connection else {})}
        result["unavailable" if reason else "legs"].append({**leg, "unavailable_reason": reason} if reason else leg)
    return result


def named_connection_routes() -> list[dict]:
    """The resolver's own verdict on a call naming each configured named connection under the configured
    Source: the legs it may use and those it cannot, with their cause. For availability and diagnostics only:
    an omitted call never tries a named connection, and only a selection or a call that names one sends on it."""
    from ouroboros.provider_connections import configured_connections
    from ouroboros.provider_models import connection_id

    return [resolve_web_search_route(model=f"{connection_id(kind, name)}::{SEARCH_MODELS[kind]}", per_call=True)
            for kind, name in configured_connections() if name and kind in SEARCH_MODELS]


def _leg_configured(source: str, connection: str, settings) -> bool:
    """Whether the search leg's connection (built-in unless the selection named one) can serve."""
    from ouroboros.provider_models import connection_id, model_has_credentials, model_has_credentials_in_settings

    route = f"{connection_id(source, connection)}::search"
    return model_has_credentials(route) if settings is None else model_has_credentials_in_settings(route, settings)


# The optional SDKs a leg imports itself; openai and openrouter use installed dependencies.
_SDK_PACKAGES = {"anthropic": "anthropic", "ddgs": "ddgs"}


def _package_discoverable(package: str) -> bool:
    """Local discovery only: no import, network or metadata read on a settings or context read."""
    try:
        return bool(sys.modules.get(package)) if package in sys.modules else importlib.util.find_spec(package) is not None
    except (ImportError, ValueError):
        return False
