"""Provider-specific model ID helpers, direct-provider defaults, and the
provider registry (SSOT for prefix→provider→credentials knowledge that was
previously duplicated across llm.py, pricing.py, agent_task_pipeline.py and
deep_self_review.py)."""

from __future__ import annotations

import os

from ouroboros.model_slots import parse_fallback_chain
from ouroboros.settings_defaults import OPENROUTER_DEFAULTS, OPENROUTER_REVIEW_DEFAULTS, SETTINGS_DEFAULTS  # noqa: F401

# MiniMax exposes the same OpenAI-compatible API on two regional hosts. Keep the
# mapping centralized so transport, capability evidence, and settings diagnostics
# fingerprint the exact endpoint selected by the owner.
MINIMAX_REGION_ENDPOINTS: dict[str, str] = {
    "global_en": "https://api.minimax.io/v1",
    "cn_zh": "https://api.minimaxi.com/v1",
}
MINIMAX_DEFAULT_REGION = "global_en"


def resolve_minimax_base_url(region: str = "") -> str:
    """Return the configured MiniMax OpenAI-compatible endpoint."""
    selected = str(region or "").strip().lower() or MINIMAX_DEFAULT_REGION
    return MINIMAX_REGION_ENDPOINTS.get(selected, MINIMAX_REGION_ENDPOINTS[MINIMAX_DEFAULT_REGION])


# Direct-provider prefix → canonical provider name. Un-prefixed models route
# through OpenRouter. Order matters only for readability; prefixes are disjoint.
PROVIDER_PREFIXES: tuple[tuple[str, str], ...] = (
    ("openai::", "openai"),
    ("anthropic::", "anthropic"),
    ("minimax::", "minimax"),
    ("cloudru::", "cloudru"),
    ("gigachat::", "gigachat"),
    ("openai-compatible::", "openai-compatible"),
    ("openrouter::", "openrouter"),
)

# Primary credential env var per provider (single-key providers).
PROVIDER_ENV_KEYS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "minimax": "MINIMAX_API_KEY",
    "cloudru": "CLOUDRU_FOUNDATION_MODELS_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

# Settings whose content-bound grant lets an external skill bypass the core LLM
# transport and incur model spend.  This belongs with the provider/credential
# registry, not in the frozen PluginAPI contract.
MODEL_PROVIDER_CREDENTIAL_KEYS: frozenset[str] = frozenset({
    *PROVIDER_ENV_KEYS.values(),
    "OPENAI_COMPATIBLE_API_KEY",
    "GIGACHAT_CREDENTIALS",
    "GIGACHAT_PASSWORD",
})

# EVERY env/settings key ``llm.LLM._resolve_remote_target`` reads for a provider, GROUPED so
# a credential and the fields it is useless without travel together or not at all (GigaChat
# needs CREDENTIALS *or* USER+PASSWORD plus its endpoint/scope; Cloud.ru's key is meaningless
# against the wrong base_url; the openai-compatible lane legitimately falls back to the legacy
# OPENAI_* pair).  Deriving a per-run credential set from anything but this table is guessing:
# `anthropic/claude-sonnet-4.6` is an OPENROUTER model id, only `anthropic::…` is direct.
PROVIDER_CREDENTIAL_GROUPS: dict[str, tuple[str, ...]] = {
    "openrouter": ("OPENROUTER_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "minimax": ("MINIMAX_API_KEY", "MINIMAX_REGION"),
    "cloudru": ("CLOUDRU_FOUNDATION_MODELS_API_KEY", "CLOUDRU_FOUNDATION_MODELS_BASE_URL"),
    "gigachat": (
        "GIGACHAT_CREDENTIALS", "GIGACHAT_PASSWORD", "GIGACHAT_USER",
        "GIGACHAT_BASE_URL", "GIGACHAT_SCOPE", "GIGACHAT_VERIFY_SSL_CERTS",
    ),
    "openai-compatible": (
        "OPENAI_COMPATIBLE_API_KEY", "OPENAI_COMPATIBLE_BASE_URL",
        "OPENAI_API_KEY", "OPENAI_BASE_URL",
    ),
    "local": (),
}

# Settings keys that hold a ROUTED model identity (prefix -> provider via provider_for_model).
# Superset of the live slots; a key absent from settings still declares whatever
# ``config.SETTINGS_DEFAULTS`` will hand the runtime, which is why declared_model_settings()
# fills the defaults in rather than treating "unset" as "unused".
MODEL_SETTING_KEYS: tuple[str, ...] = (
    "OUROBOROS_MODEL", "OUROBOROS_MODEL_HEAVY", "OUROBOROS_MODEL_LIGHT",
    "OUROBOROS_MODEL_VISION", "OUROBOROS_MODEL_CONSCIOUSNESS",
    "OUROBOROS_MODEL_FALLBACKS", "OUROBOROS_MODEL_FALLBACK",
    "OUROBOROS_MODEL_DEEP_SELF_REVIEW", "OUROBOROS_WEBSEARCH_MODEL",
    "OUROBOROS_REVIEW_MODELS", "OUROBOROS_SCOPE_REVIEW_MODELS",
    "OUROBOROS_SCOPE_REVIEW_MODEL",
)

# Settings keys whose value is a Claude Agent SDK / Claude Code model NAME (``opus[1m]``),
# NOT a routed model identity: they carry no provider prefix, so provider_for_model would
# mis-route them to OpenRouter.  Their transport is the Anthropic SDK subprocess, which
# authenticates with ANTHROPIC_API_KEY (the Claude runtime gateways:
# tools/claude_advisory_review.py), so a non-empty value DECLARES the anthropic provider.
CLAUDE_SDK_MODEL_SETTING_KEYS: tuple[str, ...] = ("CLAUDE_CODE_MODEL", "CLAUDE_AGENT_SDK_MODEL")


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


def local_only_review_route_env() -> bool:
    """Whether review slots must inherit the configured local Main route."""
    local_main = str(os.environ.get("USE_LOCAL_MAIN", "") or "").strip().lower()
    if local_main not in {"1", "true", "yes", "on"}:
        return False
    return not any(
        provider_has_credentials(provider)
        for provider in (
            "openrouter", "openai", "anthropic", "minimax", "cloudru", "gigachat",
            "openai-compatible",
        )
    )


def review_model_uses_local(model: str) -> bool:
    """Return the transport route for a resolved review slot."""
    return local_only_review_route_env() or provider_for_model(str(model or "").strip()) == "local"


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
    # (default -> Main) simply contribute nothing here.
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


def declared_model_settings(settings: dict) -> dict[str, str]:
    """Return the model slots a settings mapping DECLARES, with runtime defaults filled in.

    An absent or empty slot is not "unused": the server falls back to
    ``SETTINGS_DEFAULTS`` for it, so the default's provider is genuinely reachable and
    must be declared."""
    declared: dict[str, str] = {}
    for key in (*MODEL_SETTING_KEYS, *CLAUDE_SDK_MODEL_SETTING_KEYS):
        value = str((settings or {}).get(key) or "").strip()
        if not value:
            value = str(SETTINGS_DEFAULTS.get(key) or "").strip()
        if value:
            declared[key] = value
    return declared


def providers_for_declared_models(declared: dict) -> dict[str, list[str]]:
    """Map ``{settings key: model string}`` to ``{provider: sorted model strings}``.

    Comma chains (fallbacks, review triads) are expanded; the Claude-SDK slots resolve to
    ``anthropic`` by transport rather than by prefix."""
    found: dict[str, set] = {}
    for key, raw in (declared or {}).items():
        text = str(raw or "").strip()
        if not text:
            continue
        if str(key) in CLAUDE_SDK_MODEL_SETTING_KEYS:
            found.setdefault("anthropic", set()).add(text)
            continue
        for part in text.split(","):
            model = part.strip()
            if model:
                found.setdefault(provider_for_model(model), set()).add(model)
    return {provider: sorted(models) for provider, models in sorted(found.items())}


def credential_keys_for_providers(providers) -> tuple[str, ...]:
    """Return the ordered, de-duplicated credential keys a provider set needs."""
    keys: list[str] = []
    for provider in providers:
        group = PROVIDER_CREDENTIAL_GROUPS.get(str(provider))
        if group is None:
            # Unknown provider: fail OPEN with its primary key rather than silently
            # handing a run no credential at all.
            group = tuple(filter(None, (PROVIDER_ENV_KEYS.get(str(provider), ""),)))
        for key in group:
            if key not in keys:
                keys.append(key)
    return tuple(keys)


ALL_PROVIDER_CREDENTIAL_KEYS: frozenset[str] = frozenset(
    key for group in PROVIDER_CREDENTIAL_GROUPS.values() for key in group
)


def provider_credential_plan(settings: dict) -> dict:
    """Derive WHICH provider credentials a settings mapping's declared models actually need.

    Returns ``{declared_model_slots, providers, planned_keys, fail_open}``.  ``fail_open`` is
    the disclosed escape hatch: when nothing resolves (a settings mapping with no model slot
    at all) the plan is the FULL credential universe, because a benchmark that dies on a
    missing key at hour six is worse than one that carries a spare."""
    declared = declared_model_settings(settings)
    providers = providers_for_declared_models(declared)
    planned = credential_keys_for_providers(providers)
    fail_open = not planned
    if fail_open:
        planned = tuple(sorted(ALL_PROVIDER_CREDENTIAL_KEYS))
    return {
        "declared_model_slots": declared,
        "providers": providers,
        "planned_keys": sorted(planned),
        "fail_open": fail_open,
    }


OPENAI_DIRECT_DEFAULTS = {
    "main": "openai::gpt-5.6-terra",
    "heavy": "",
    "light": "openai::gpt-5.6-luna",
    "fallback": "openai::gpt-5.6-sol",
    # Deep self-review is a real slot with a SHIPPED default; without a
    # per-provider value a direct-only install keeps an unreachable
    # OpenRouter-form id it has no credential for (v6.82.0). Only providers whose
    # model genuinely carries the >=1M window this review sizes against get one —
    # Cloud.ru and GigaChat are documented BELOW that floor, so filling their slot
    # would advertise a deep review that is doomed to overflow its real route.
    #
    # DELIBERATELY plain Sol, NOT the OpenRouter default's `-pro`: that suffix is an
    # OpenRouter slug, not an OpenAI model id. Live-probed 2026-07-29 against
    # api.openai.com: `gpt-5.6-sol-pro` on /v1/chat/completions -> 404; the pro
    # reasoning mode exists only on /v1/responses as `reasoning.mode="pro"` (200),
    # and passing `reasoning` to /v1/chat/completions -> 400 "Unknown parameter".
    # Every LLM call in llm.py is a chat.completions call, so a direct-OpenAI
    # install runs deep review on plain Sol — an owner-accepted capability
    # difference from the OpenRouter default, disclosed in README/ARCHITECTURE
    # rather than papered over with a slug that does not exist.
    "deep_self_review": "openai::gpt-5.6-sol",
}

CLOUDRU_DIRECT_DEFAULTS = {
    "main": "cloudru::zai-org/GLM-4.7",
    "heavy": "cloudru::zai-org/GLM-4.7",
    "light": "cloudru::zai-org/GLM-4.7",
    "fallback": "cloudru::zai-org/GLM-4.7",
}

GIGACHAT_DIRECT_DEFAULTS = {
    # Ultra is available only to individuals in Freemium. GigaChat 2 Max is
    # available to personal and legal-entity paid plans as well, so it is the
    # strongest current default that does not strand B2B/CORP installs.
    "main": "gigachat::GigaChat-2-Max",
    "heavy": "gigachat::GigaChat-2-Max",
    "light": "gigachat::GigaChat-2-Max",
    "fallback": "gigachat::GigaChat-2-Max",
}

MINIMAX_DIRECT_DEFAULTS = {
    "main": "minimax::MiniMax-M3",
    "heavy": "minimax::MiniMax-M3",
    "light": "minimax::MiniMax-M2.7",
    "fallback": "minimax::MiniMax-M2.7",
    # NO deep_self_review default: MiniMax documents M3 as "up to 1M tokens" with a
    # GUARANTEED minimum of 512K (platform.minimax.io, 2026-08). Deep review sizes
    # against a firm 1M window, so the guaranteed floor decides — the slot follows
    # the Cloud.ru/GigaChat clear-instead-of-fill path; owners can opt in manually.
}

ANTHROPIC_DIRECT_DEFAULTS = {
    "main": "anthropic::claude-opus-5",
    "heavy": "",
    "light": "anthropic::claude-sonnet-5",
    "fallback": "anthropic::claude-sonnet-5",
    # Deep self-review is a real slot with a SHIPPED default; without a
    # per-provider value a direct-only install keeps an unreachable
    # OpenRouter-form id it has no credential for (v6.82.0).
    "deep_self_review": "anthropic::claude-opus-5",
}

DIRECT_PROVIDER_DEFAULTS = {
    "openai": OPENAI_DIRECT_DEFAULTS,
    "anthropic": ANTHROPIC_DIRECT_DEFAULTS,
    "cloudru": CLOUDRU_DIRECT_DEFAULTS,
    "gigachat": GIGACHAT_DIRECT_DEFAULTS,
    "minimax": MINIMAX_DIRECT_DEFAULTS,
}

# Review panels are declared as provider ROLE sequences, then compiled against
# the owner's current provider-prefixed Main/Light values. This keeps explicit
# custom models useful while making the shipped single-provider policy obvious:
# OpenAI and Anthropic run their strongest Main model three independent times.
# The older MiniMax mixed panel is preserved because that profile was not part of
# this policy change.
DIRECT_PROVIDER_REVIEW_ROLES = {
    "openai": ("main", "main", "main"),
    "anthropic": ("main", "main", "main"),
    "cloudru": ("main", "main", "main"),
    "gigachat": ("main", "main", "main"),
    "minimax": ("main", "light", "light"),
}

DIRECT_PROVIDER_SCOPE_DEFAULTS = {
    provider: defaults["main"]
    for provider, defaults in DIRECT_PROVIDER_DEFAULTS.items()
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
    if provider == "minimax":
        if text.startswith("minimax::"):
            return text
        if text.startswith("minimax/"):
            return f"minimax::{text[len('minimax/'):]}"
        return text
    return text


def compute_direct_review_models_fallback(
    provider: str,
    main_model: str,
    light_model: str = "",
    *,
    review_runs: int = 3,
) -> list[str]:
    """Compile a direct-provider review panel from declarative role names."""
    if provider not in DIRECT_PROVIDER_DEFAULTS:
        return []
    provider_prefix = f"{provider}::"
    main = migrate_model_value(provider, main_model)
    if not main.startswith(provider_prefix):
        return []
    light = migrate_model_value(provider, light_model) if light_model else ""
    default_light = migrate_model_value(provider, DIRECT_PROVIDER_DEFAULTS[provider].get("light", ""))
    light_slot = light if light.startswith(provider_prefix) else default_light
    role_models = {"main": main, "light": light_slot or main}
    roles = DIRECT_PROVIDER_REVIEW_ROLES.get(provider, ("main", "light", "light"))
    compiled = [role_models.get(role, main) for role in roles]
    count = max(1, int(review_runs or 3))
    return [compiled[index % len(compiled)] for index in range(count)]


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
    if text.startswith("minimax::"):
        return f"minimax/{text[len('minimax::'):]}"
    if text.startswith("anthropic::"):
        return f"anthropic/{normalize_anthropic_model_id(text[len('anthropic::'):])}"
    if text.startswith("anthropic/"):
        return f"anthropic/{normalize_anthropic_model_id(text[len('anthropic/'):])}"
    return text
