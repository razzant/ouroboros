"""Live Conversation: beta configuration, consumer and approved Talk A surface."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ouroboros.contracts.live_conversation import LiveConfig
from ouroboros.contracts.realtime import RealtimeSetupError
from ouroboros.live_conversation import call_offer, end_feature_calls

# Calls are off until the owner turns them on; Gemini is the default engine.
DEFAULTS = {"enabled": False, "provider": "gemini_live:gemini", "model": "", "voice": "", "instructions": "",
            "pause": "normal", "interruptions": True, "thinking": "", "search": True}


def validate(value):
    """The stored configuration as the typed LiveConfig the host reads; unknown keys are refused."""
    if not isinstance(value, dict) or set(value) - DEFAULTS.keys():
        raise RealtimeSetupError("configuration", "unknown call configuration")
    return LiveConfig(**{**DEFAULTS, **value})


def registered_catalogs():
    """Loaded catalog metadata for Settings; the host decides call availability."""
    from ouroboros import extension_registry_state as registry

    with registry._lock:
        return [(key, spec["catalog"])
                for key, spec in registry._realtime_providers.items()
                if spec["skill"] in registry._extensions and spec["skill"] not in registry._unloading]


def model_json(model):
    value = asdict(model)
    value["honours"] = sorted(model.honours)
    return value


def configuration_view(config, catalogs):
    selected = next((catalog for key, catalog in catalogs if key == config.provider), None)
    try:
        model = selected.model(config.model) if selected else None
    except RealtimeSetupError:
        model = None
    offered = call_offer(config)["offered"]
    return {**asdict(config), "offered": offered, "selected_model": model_json(model) if model else None,
            "catalogs": [{"id": key, "label": catalog.label, "privacy": catalog.privacy,
                          "models": [model_json(item) for item in catalog.models]} for key, catalog in catalogs]}


def register(api):
    from .driver import call_tools, run
    from ouroboros.utils import atomic_write_json

    state_dir = Path(api.get_state_dir())
    path = state_dir / "conversation.json"

    async def handle(host):
        await run(host, state_dir=state_dir)

    def configure():
        try:
            return validate(json.loads(path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            return validate({})

    async def settings(request):
        from starlette.responses import JSONResponse

        catalogs = registered_catalogs()
        # Ordinary extension settings route; never derives beta enablement from keys.
        if request.method == "POST":
            try:
                try:
                    stored = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    stored = {}  # a missing, unreadable or malformed file never blocks the owner's own switch
                config = validate(await request.json())
                if not config.provider:
                    raise RealtimeSetupError("configuration", "Choose an engine.")
                if config.enabled:
                    catalog = next((catalog for key, catalog in catalogs if key == config.provider), None)
                    if catalog is None:
                        raise RealtimeSetupError("provider_unavailable", "The selected engine is not loaded.")
                    model = catalog.model(config.model)
                    if config.voice and config.voice not in {voice.id for voice in model.voices}:
                        raise RealtimeSetupError("voice_unavailable", "Choose a voice from the selected model's catalog.")
                    if "thinking" in model.honours and config.thinking and config.thinking not in model.thinking_levels:
                        raise RealtimeSetupError("configuration", "Choose a thinking level from the selected model's catalog.")
                record = asdict(config)
                # The beta default belongs to the product, not to this install: a switch the owner
                # never flipped is not written, so a later change of the default still reaches them.
                if "enabled" not in (stored if isinstance(stored, dict) else {}) and config.enabled == DEFAULTS["enabled"]:
                    record.pop("enabled")
                atomic_write_json(path, record)
                if not config.enabled: end_feature_calls("live_conversation:call")
            except (ValueError, RealtimeSetupError) as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
            # The value just written is the view; a file the owner cannot read back never hides a saved change.
            return JSONResponse(configuration_view(config, catalogs))
        return JSONResponse(configuration_view(configure(), catalogs))

    api.register_live_conversation("call", handle, configure=configure, call_tools=call_tools(state_dir))
    api.register_route("config", settings, methods=["GET", "POST"])
    api.register_settings_section("conversation", title="Calls (beta)", schema={"components": [
        {"type": "markdown", "text": "The call skills are on by default. To talk, turn on Calls (beta), choose an engine "
         "and add its API key in Settings → Secrets, under the keys the skills request. Gemini and OpenAI do not "
         "serve Russia or Belarus."},
        {"type": "form", "route": "config", "method": "POST", "fields": [
            {"name": "enabled", "type": "checkbox", "label": "Calls (beta)", "default": False},
            {"name": "provider", "type": "select", "label": "Engine", "default": DEFAULTS["provider"], "options": []},
            {"name": "model", "type": "select", "label": "Model", "options": []},
            {"name": "voice", "type": "select", "label": "Voice", "options": []},
            {"name": "instructions", "type": "textarea", "label": "Instructions for calls"},
            {"name": "pause", "type": "select", "label": "Pause before answering", "options": []},
            {"name": "interruptions", "type": "checkbox", "label": "Speaking interrupts Ouroboros"},
            {"name": "thinking", "type": "select", "label": "Thinking", "options": []},
            {"name": "search", "type": "checkbox", "label": "Built-in search"},
        ]},
    ]})
