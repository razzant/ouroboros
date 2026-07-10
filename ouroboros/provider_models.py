"""Provider-specific model ID helpers, direct-provider defaults, and the
provider registry (SSOT for prefix→provider→credentials knowledge that was
previously duplicated across llm.py, pricing.py, agent_task_pipeline.py and
deep_self_review.py)."""

from __future__ import annotations

import os

# Direct-provider prefix → canonical provider name. Un-prefixed models route
# through OpenRouter. Order matters only for readability; prefixes are disjoint.
PROVIDER_PREFIXES: tuple[tuple[str, str], ...] = (
    ("openai::", "openai"),
    ("anthropic::", "anthropic"),
    ("cloudru::", "cloudru"),
    ("gigachat::", "gigachat"),
    ("openai-compatible::", "openai-compatible"),
    ("openrouter::", "openrouter"),
)

# Primary credential env var per provider (single-key providers).
PROVIDER_ENV_KEYS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "cloudru": "CLOUDRU_FOUNDATION_MODELS_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def provider_for_model(model: str) -> str:
    """Return the execution provider for a model id (``local`` for local lanes)."""
    name = str(model or "").strip()
    if name.endswith(" (local)"):
        return "local"
    for prefix, provider in PROVIDER_PREFIXES:
        if name.startswith(prefix):
            return provider
    return "openrouter"


def provider_has_credentials(provider: str) -> bool:
    """Return True when the environment carries usable credentials for a provider."""
    if provider == "local":
        return True
    if provider == "openai-compatible":
        compat = str(os.environ.get("OPENAI_COMPATIBLE_API_KEY", "") or "").strip()
        legacy_key = str(os.environ.get("OPENAI_API_KEY", "") or "").strip()
        legacy_base = str(os.environ.get("OPENAI_BASE_URL", "") or "").strip()
        return bool(compat or (legacy_key and legacy_base))
    if provider == "gigachat":
        creds = str(os.environ.get("GIGACHAT_CREDENTIALS", "") or "").strip()
        user = str(os.environ.get("GIGACHAT_USER", "") or "").strip()
        password = str(os.environ.get("GIGACHAT_PASSWORD", "") or "").strip()
        return bool(creds or (user and password))
    env_key = PROVIDER_ENV_KEYS.get(provider, "OPENROUTER_API_KEY")
    return bool(str(os.environ.get(env_key, "") or "").strip())


def model_has_credentials(model: str) -> bool:
    """Return True when the model's provider has usable credentials configured."""
    return provider_has_credentials(provider_for_model(model))


def resolve_credentialed_model(default_model: str) -> str:
    """Return ``default_model`` if its provider is credentialed, else the first
    configured model slot whose provider has credentials (light → fallback →
    main → heavy). Falls back to ``default_model`` when nothing is credentialed
    so callers surface the original provider error rather than a silent swap."""
    if model_has_credentials(default_model):
        return default_model
    # LIGHT/MAIN/HEAVY are single-model slots; FALLBACKS is a comma chain expanded via the
    # shared SSOT parser (which also honors the legacy singular OUROBOROS_MODEL_FALLBACK)
    # instead of testing the whole comma-string as one broken model id. Empty Heavy/Light
    # (default -> Main) simply contribute nothing here. Lazy import: config imports this
    # module, so importing config at module load would be circular.
    from ouroboros.config import parse_fallback_chain
    candidates: list[str] = []
    light = str(os.environ.get("OUROBOROS_MODEL_LIGHT", "") or "").strip()
    if light:
        candidates.append(light)
    candidates.extend(parse_fallback_chain())
    for env_name in ("OUROBOROS_MODEL", "OUROBOROS_MODEL_HEAVY"):
        raw = str(os.environ.get(env_name, "") or "").strip()
        if raw:
            candidates.append(raw)
    for candidate in candidates:
        if model_has_credentials(candidate):
            return candidate
    return default_model


OPENAI_DIRECT_DEFAULTS = {
    "main": "openai::gpt-5.5",
    "heavy": "openai::gpt-5.5",
    "light": "openai::gpt-5.4-mini",
    "fallback": "openai::gpt-5.4-mini",
}

CLOUDRU_DIRECT_DEFAULTS = {
    "main": "cloudru::zai-org/GLM-4.7",
    "heavy": "cloudru::zai-org/GLM-4.7",
    "light": "cloudru::zai-org/GLM-4.7",
    "fallback": "cloudru::zai-org/GLM-4.7",
}

GIGACHAT_DIRECT_DEFAULTS = {
    "main": "gigachat::GigaChat-3-Ultra",
    "heavy": "gigachat::GigaChat-3-Ultra",
    "light": "gigachat::GigaChat-3-Ultra",
    "fallback": "gigachat::GigaChat-3-Ultra",
}

ANTHROPIC_DIRECT_DEFAULTS = {
    "main": "anthropic::claude-opus-4-8",
    "heavy": "anthropic::claude-opus-4-8",
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


# Conservative static vision map by normalized id/prefix. The OpenRouter
# /models overlay (llm.py) refines this at runtime; static knowledge only
# covers families whose vision support is long-established.
_VISION_MODEL_PREFIXES: tuple[str, ...] = (
    "openai/gpt-5", "openai/gpt-4o", "openai/gpt-4.1", "openai/o3", "openai/o4",
    "google/gemini-", "anthropic/claude-",
    "x-ai/grok-4", "x-ai/grok-3",
    "qwen/qwen-vl", "qwen/qwen2.5-vl", "qwen/qwen3-vl",
    "mistralai/pixtral", "meta-llama/llama-4", "meta-llama/llama-3.2-90b-vision",
    "openai/gpt-5.5",
)

# Runtime overlay: model_id → bool, fed from OpenRouter /models
# architecture.input_modalities by llm.py (same lifecycle as its
# supported-parameters cache).
_VISION_OVERLAY: dict = {}


def update_vision_overlay(model_id: str, supports: bool) -> None:
    normalized = normalize_model_identity(model_id)
    if normalized:
        _VISION_OVERLAY[normalized] = bool(supports)


def supports_vision(model_id: str) -> bool:
    """True when the model accepts native image input blocks."""
    # Local lanes have no vision regardless of family name; check the RAW id —
    # normalize_model_identity strips the " (local)" suffix.
    if str(model_id or "").strip().endswith(" (local)"):
        return False
    normalized = normalize_model_identity(model_id)
    if not normalized:
        return False
    if normalized in _VISION_OVERLAY:
        return _VISION_OVERLAY[normalized]
    return normalized.startswith(_VISION_MODEL_PREFIXES)


# NOTE (v6.33.0): the static per-model context-window table was REMOVED. It
# perpetually went stale (1M-beta models hard-coded to 200K, [1m] ignored). The
# agent's OWN operating window is the owner low/max context MODE (the SSOT — see
# context_budget.py / loop.py), and external-model windows are resolved by
# Capability Evidence (ouroboros.capability_evidence: confirmed provider metadata
# / local health, or route-fingerprinted owner-ack), fail-closed when unknown.


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
