"""Shared LLM entry for live routes, separate from text-provider defaults.

Calls use Google Gemini Live. It reads the provider credential registry with the
task-scoped settings reader; the key alone never invents a text route or changes
Main's selection.
"""
from __future__ import annotations

from dataclasses import replace

from ouroboros.gateways.realtime_engine import (
    GOOGLE_INPUT_RATE, GOOGLE_OUTPUT_RATE, EngineRoute, LiveEngine, SessionConfig,
)
from ouroboros.gateways.realtime_transport import LiveEngineRefusal
from ouroboros.provider_models import PROVIDER_ENV_KEYS
from ouroboros.settings_integrity import runtime_setting


def live_key_present(settings) -> bool:
    """Whether the owner configured the Google AI key calls need (``settings`` maps names to values)."""
    return bool(str(settings.get(PROVIDER_ENV_KEYS["google"]) or "").strip())


def make_live_engine(client, route: EngineRoute) -> LiveEngine:
    from ouroboros.gateways.realtime_gemini import GeminiLiveEngine

    if route.engine != "google":
        raise LiveEngineRefusal("engine", "Calls use Google Gemini Live.")
    if not route.model.strip() or "::" in route.model:
        raise LiveEngineRefusal("model", "A live route requires the provider's unqualified model ID.")
    key = str(runtime_setting(PROVIDER_ENV_KEYS["google"], "") or "").strip()
    if not key:
        raise LiveEngineRefusal("missing_credential", f"Configure {PROVIDER_ENV_KEYS['google']} for calls.")
    engine = GeminiLiveEngine(api_key=key)
    engine.route = replace(route, input_rate=GOOGLE_INPUT_RATE, output_rate=GOOGLE_OUTPUT_RATE)
    return engine


async def open_configured_engine(engine: LiveEngine, route: EngineRoute, config: SessionConfig) -> LiveEngine:
    if (route.engine, route.model, route.voice) != (config.route.engine, config.route.model, config.route.voice):
        raise LiveEngineRefusal("route_mismatch", "Session configuration must use the selected live route.")
    await engine.open(config)
    return engine
