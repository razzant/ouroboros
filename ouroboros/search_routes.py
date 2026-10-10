"""Built-in web_search Source/Model resolution, shared by dispatch and projections.

This setting selects only the built-in transport. Skills, MCP and browser tools
remain independent choices. Resolution reads settings; it performs no provider I/O.
"""
from __future__ import annotations

import importlib.util
import sys

SEARCH_MODELS = {"openai": "gpt-5.2", "openrouter": "openai/gpt-5.2", "anthropic": "claude-sonnet-4-6", "ddgs": ""}
SEARCH_BACKENDS = {"openai": "openai_responses", "openrouter": "openrouter_server_tool", "anthropic": "anthropic_server_tool", "ddgs": "ddgs"}


def resolve_web_search_route(*, backend=None, model=None, settings=None) -> dict:
    """Resolve from an explicit settings document, else from what this process's dispatch reads."""
    from ouroboros.config import runtime_setting

    setting = settings.get if settings is not None else runtime_setting
    source = str(setting("OUROBOROS_WEBSEARCH_BACKEND") or "auto") if backend is None else str(backend or "auto")
    source = source.strip().lower()
    selected = str(setting("OUROBOROS_WEBSEARCH_MODEL") or "") if model is None else str(model or "")
    selected = selected.strip()
    result = {"scope": "builtin_web_search", "source": source, "model": selected, "legs": [], "unavailable": [],
              "note": "Applies only to built-in web_search; skills, MCP and browser tools remain independent choices."}
    if source not in {"auto", *SEARCH_MODELS}:
        return {**result, "error": "Unknown web search source"}
    prefix, sep, native = selected.partition("::")
    legacy_anthropic = selected.partition("/")[2] if not sep and selected.startswith("anthropic/") else ""
    unsupported = ("This model route has no built-in search transport"
                   if sep and (prefix not in {"openai", "anthropic", "openrouter"} or not native) else
                   "Direct search sources require a native model id"
                   if sep and prefix in {"openai", "anthropic"} and "/" in native else "")
    if source == "ddgs":
        candidates = [("ddgs", "")]  # ddgs takes no model: a saved model of any route is irrelevant here
    elif unsupported and source != "auto":
        return {**result, "error": unsupported}  # a strict source never substitutes another model
    elif source == "auto":
        if unsupported:
            # Auto keeps the legs that need no model and says the selection is not applied.
            result["unapplied_model"] = selected
            result["note"] += f" {unsupported}, so Auto does not apply the saved model and uses provider defaults."
            candidates = [("anthropic", SEARCH_MODELS["anthropic"]), ("ddgs", "")]
        elif not selected:
            candidates = list(SEARCH_MODELS.items())
        elif sep:
            if prefix == "openrouter" and "/" not in native:
                native = f"openai/{native}"
            candidates = ([("openai", native), ("openrouter", f"openai/{native}")] if prefix == "openai"
                          else [(prefix, native)]) + [("ddgs", "")]
        elif "/" in selected:
            # A saved legacy anthropic/<model> keeps the owner's model on the direct Anthropic leg too.
            candidates = [("openrouter", selected), ("anthropic", legacy_anthropic or SEARCH_MODELS["anthropic"]),
                          ("ddgs", "")]
        else:
            candidates = [("openai", selected), ("openrouter", f"openai/{selected}"),
                          ("anthropic", SEARCH_MODELS["anthropic"]), ("ddgs", "")]
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
        if name == "ddgs":
            try:
                ready = bool(sys.modules.get("ddgs")) if "ddgs" in sys.modules else importlib.util.find_spec("ddgs") is not None
            except (ImportError, ValueError):
                ready = False
        else:
            ready = bool(setting({"openai": "OPENAI_API_KEY", "openrouter": "OPENROUTER_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}[name]))
            if name == "openai" and setting("OPENAI_BASE_URL"):
                ready = False  # Legacy compatible credentials are not official Responses access.
        explicit_model = (bool(selected) and name != "ddgs" and not result.get("ignored_legacy_model")
                          and not result.get("unapplied_model"))
        if source == "auto" and name == "anthropic" and prefix != "anthropic" and not legacy_anthropic:
            explicit_model = False
        leg = {"source": name, "backend": SEARCH_BACKENDS[name], "model": wire_model,
               "model_source": "selection" if explicit_model else "provider_default"}
        result["legs" if ready else "unavailable"].append(leg)
    return result
