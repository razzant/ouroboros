"""Provider-specific model ID helpers and direct-provider defaults."""

from __future__ import annotations

OPENAI_DIRECT_DEFAULTS = {
    "main": "openai::gpt-5.5",
    "code": "openai::gpt-5.5",
    "light": "openai::gpt-5.5-mini",
    "fallback": "openai::gpt-5.5-mini",
}

CLOUDRU_DIRECT_DEFAULTS = {
    "main": "cloudru::zai-org/GLM-4.7",
    "code": "cloudru::zai-org/GLM-4.7",
    "light": "cloudru::zai-org/GLM-4.7",
    "fallback": "cloudru::zai-org/GLM-4.7",
}

GIGACHAT_DIRECT_DEFAULTS = {
    "main": "gigachat::GigaChat-3-Ultra",
    "code": "gigachat::GigaChat-3-Ultra",
    "light": "gigachat::GigaChat-3-Ultra",
    "fallback": "gigachat::GigaChat-3-Ultra",
}

ANTHROPIC_DIRECT_DEFAULTS = {
    "main": "anthropic::claude-opus-4-8",
    "code": "anthropic::claude-opus-4-8",
    "light": "anthropic::claude-sonnet-4-6",
    "fallback": "anthropic::claude-sonnet-4-6",
}

_DIRECT_PROVIDER_DEFAULTS = {
    "openai": OPENAI_DIRECT_DEFAULTS,
    "anthropic": ANTHROPIC_DIRECT_DEFAULTS,
    "cloudru": CLOUDRU_DIRECT_DEFAULTS,
    "gigachat": GIGACHAT_DIRECT_DEFAULTS,
}

_ANTHROPIC_MODEL_ALIASES = {
    "claude-opus-4.6": "claude-opus-4-6",
    "claude-opus-4.7": "claude-opus-4-7",
    "claude-opus-4.8": "claude-opus-4-8",
    "claude-sonnet-4.6": "claude-sonnet-4-6",
}


def normalize_anthropic_model_id(model_id: str) -> str:
    text = str(model_id or "").strip()
    return _ANTHROPIC_MODEL_ALIASES.get(text, text)


def migrate_model_value(provider: str, value: str) -> str:
    text = str(value or "").strip()
    if provider == "openai":
        if text.startswith("openai/"):
            return f"openai::{text[len('openai/'):]}"
        return text
    if provider == "anthropic":
        if text.startswith("anthropic::"):
            return f"anthropic::{normalize_anthropic_model_id(text[len('anthropic::'):])}"
        if text.startswith("anthropic/"):
            return f"anthropic::{normalize_anthropic_model_id(text[len('anthropic/'):])}"
        return text
    if provider == "cloudru":
        if text.startswith("cloudru::"):
            return text
        if text.startswith("cloudru/"):
            return f"cloudru::{text[len('cloudru/'):]}"
        return text
    return text


def compute_direct_review_models_fallback(
    provider: str,
    main_model: str,
    light_model: str = "",
    *,
    review_runs: int = 3,
) -> list[str]:
    """Return direct-provider review fallback preserving commit-triad shape.

    The quorum-safe shape is ``[main, light, light]`` when main/light are
    distinct provider-prefixed lanes; otherwise it degrades to ``[main] * N``.
    """
    if provider not in _DIRECT_PROVIDER_DEFAULTS:
        return []
    provider_prefix = f"{provider}::"
    main = migrate_model_value(provider, main_model)
    if not main.startswith(provider_prefix):
        return []
    light = migrate_model_value(provider, light_model) if light_model else ""
    default_light = migrate_model_value(provider, _DIRECT_PROVIDER_DEFAULTS[provider].get("light", ""))
    light_slot = light if light.startswith(provider_prefix) else default_light
    if light_slot and light_slot != main:
        return [main, light_slot, light_slot]
    return [main] * int(review_runs or 3)


def normalize_model_identity(model: str) -> str:
    text = str(model or "").strip()
    if text.endswith(" (local)"):
        text = text[:-8]
    if text.startswith("openai::"):
        return f"openai/{text[len('openai::'):]}"
    if text.startswith("openai-compatible::"):
        return f"openai-compatible/{text[len('openai-compatible::'):]}"
    if text.startswith("cloudru::"):
        return f"cloudru/{text[len('cloudru::'):]}"
    if text.startswith("gigachat::"):
        return f"gigachat/{text[len('gigachat::'):]}"
    if text.startswith("anthropic::"):
        return f"anthropic/{normalize_anthropic_model_id(text[len('anthropic::'):])}"
    if text.startswith("anthropic/"):
        return f"anthropic/{normalize_anthropic_model_id(text[len('anthropic/'):])}"
    return text
