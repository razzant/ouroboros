"""Helpers shared by server startup, onboarding, and WebSocket liveness."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from ouroboros.provider_models import (
    ANTHROPIC_DIRECT_DEFAULTS,
    CLOUDRU_DIRECT_DEFAULTS,
    GIGACHAT_DIRECT_DEFAULTS,
    MINIMAX_DIRECT_DEFAULTS,
    OPENAI_DIRECT_DEFAULTS,
    compute_direct_review_models_fallback,
    migrate_model_value,
)
from ouroboros.config import SETTINGS_DEFAULTS, _DIRECT_PROVIDER_REVIEW_RUNS
from ouroboros.settings_vocabulary import parse_model_list
from ouroboros.utils import utc_now_iso


_DIRECT_PROVIDER_AUTO_DEFAULTS = {
    "openai": {
        "OUROBOROS_MODEL": OPENAI_DIRECT_DEFAULTS["main"],
        "OUROBOROS_MODEL_HEAVY": OPENAI_DIRECT_DEFAULTS["heavy"],
        "OUROBOROS_MODEL_LIGHT": OPENAI_DIRECT_DEFAULTS["light"],
        "OUROBOROS_MODEL_FALLBACKS": OPENAI_DIRECT_DEFAULTS["fallback"],
        "OUROBOROS_MODEL_DEEP_SELF_REVIEW": OPENAI_DIRECT_DEFAULTS["deep_self_review"],
    },
    "anthropic": {
        "OUROBOROS_MODEL": ANTHROPIC_DIRECT_DEFAULTS["main"],
        "OUROBOROS_MODEL_HEAVY": ANTHROPIC_DIRECT_DEFAULTS["heavy"],
        "OUROBOROS_MODEL_LIGHT": ANTHROPIC_DIRECT_DEFAULTS["light"],
        "OUROBOROS_MODEL_FALLBACKS": ANTHROPIC_DIRECT_DEFAULTS["fallback"],
        "OUROBOROS_MODEL_DEEP_SELF_REVIEW": ANTHROPIC_DIRECT_DEFAULTS["deep_self_review"],
    },
    "cloudru": {
        "OUROBOROS_MODEL": CLOUDRU_DIRECT_DEFAULTS["main"],
        "OUROBOROS_MODEL_HEAVY": CLOUDRU_DIRECT_DEFAULTS["heavy"],
        "OUROBOROS_MODEL_LIGHT": CLOUDRU_DIRECT_DEFAULTS["light"],
        "OUROBOROS_MODEL_FALLBACKS": CLOUDRU_DIRECT_DEFAULTS["fallback"],
    },
    "gigachat": {
        "OUROBOROS_MODEL": GIGACHAT_DIRECT_DEFAULTS["main"],
        "OUROBOROS_MODEL_HEAVY": GIGACHAT_DIRECT_DEFAULTS["heavy"],
        "OUROBOROS_MODEL_LIGHT": GIGACHAT_DIRECT_DEFAULTS["light"],
        "OUROBOROS_MODEL_FALLBACKS": GIGACHAT_DIRECT_DEFAULTS["fallback"],
    },
    # NB: no OUROBOROS_MODEL_DEEP_SELF_REVIEW — MiniMax guarantees only a 512K
    # window floor ("up to 1M"), below the 1M deep-review sizing target, so the
    # slot takes the same clear-instead-of-fill branch as Cloud.ru/GigaChat.
    "minimax": {
        "OUROBOROS_MODEL": MINIMAX_DIRECT_DEFAULTS["main"],
        "OUROBOROS_MODEL_HEAVY": MINIMAX_DIRECT_DEFAULTS["heavy"],
        "OUROBOROS_MODEL_LIGHT": MINIMAX_DIRECT_DEFAULTS["light"],
        "OUROBOROS_MODEL_FALLBACKS": MINIMAX_DIRECT_DEFAULTS["fallback"],
    },
}
# Legacy values that should be auto-replaced with a provider's direct defaults.
# Cloud.ru, GigaChat, and MiniMax intentionally have NO entry: such a provider-only user's
# main/code/light slots match the shipped SETTINGS_DEFAULTS or the shared
# _PRIOR_SHIPPED_SLOT_DEFAULTS below (google/gemini era) and migrate via the
# `current in {"", default, *legacy}` check, and the review/scope slots are
# rebuilt from the provider-prefixed main by _normalize_direct_review_models
# — so no per-provider legacy set is needed (verified by
# test_apply_runtime_provider_defaults_cloudru_*).
_DIRECT_PROVIDER_LEGACY_DEFAULTS = {
    "openai": {
        # v6.82.0: the outgoing v6.81 OpenAI slot defaults (gpt-5.5 main/heavy,
        # gpt-5.4-mini light/fallback, both spellings) migrate to the new gpt-5.6
        # slot defaults; all four models stay LIVE (no retirement remap).
        "OUROBOROS_MODEL": {
            "anthropic/claude-opus-4.6", "openai/gpt-5.5", "openai::gpt-5.5",
        },
        "OUROBOROS_MODEL_HEAVY": {
            "anthropic/claude-opus-4.6", "openai/gpt-5.5", "openai::gpt-5.5",
        },
        "OUROBOROS_MODEL_LIGHT": {
            "anthropic/claude-sonnet-4.6", "openai/gpt-5.4-mini", "openai::gpt-5.4-mini",
        },
        "OUROBOROS_MODEL_FALLBACKS": {
            "anthropic/claude-sonnet-4.6", "openai/gpt-5.4-mini", "openai::gpt-5.4-mini",
        },
        # The SHIPPED OpenRouter deep default and its migrated direct spelling both
        # name a router slug that does not exist on api.openai.com (404) — a direct
        # install must land on the real model, not on the -pro id.
        "OUROBOROS_MODEL_DEEP_SELF_REVIEW": {
            "openai/gpt-5.6-sol-pro", "openai::gpt-5.6-sol-pro",
            "openai/gpt-5.5-pro", "openai::gpt-5.5-pro",
        },
    },
    "anthropic": {
        # Both spellings of each previous Anthropic default (opus-4.6 -> opus-4.8 in
        # v6.8x, opus-4.8/sonnet-4.6 -> sonnet-5/opus-5 in v6.82.0) so existing
        # Anthropic-only users on any prior shipped default migrate to the current
        # slot defaults, whether stored in dot/slash or dash/colon form.
        "OUROBOROS_MODEL": {
            "anthropic/claude-opus-4.6", "anthropic::claude-opus-4-6",
            "anthropic/claude-opus-4.8", "anthropic::claude-opus-4-8",
        },
        "OUROBOROS_MODEL_HEAVY": {
            "anthropic/claude-opus-4.6", "anthropic::claude-opus-4-6",
            "anthropic/claude-opus-4.8", "anthropic::claude-opus-4-8",
        },
        "OUROBOROS_MODEL_LIGHT": {
            "anthropic/claude-sonnet-4.6", "anthropic::claude-sonnet-4-6",
        },
        "OUROBOROS_MODEL_FALLBACKS": {
            "anthropic/claude-sonnet-4.6", "anthropic::claude-sonnet-4-6",
        },
    },
}
_DIRECT_PROVIDER_LEGACY_DEFAULTS["openai"]["OUROBOROS_MODEL_LIGHT"].add("openai::gpt-4.1")
_DIRECT_PROVIDER_LEGACY_DEFAULTS["openai"]["OUROBOROS_MODEL_FALLBACKS"].add("openai::gpt-4.1")
_LEGACY_GEMINI_31_FLASH_LITE = "google/gemini-" + "3.1-flash-lite"
_LEGACY_GEMINI_31_PRO_PREVIEW = "google/gemini-" + "3.1-pro-preview"
_LEGACY_GEMINI_3_FLASH_PREVIEW = "google/gemini-" + "3-flash-preview"
for _legacy_defaults in _DIRECT_PROVIDER_LEGACY_DEFAULTS.values():
    for _slot in ("OUROBOROS_MODEL", "OUROBOROS_MODEL_HEAVY", "OUROBOROS_MODEL_LIGHT"):
        _legacy_defaults[_slot].add(_LEGACY_GEMINI_31_FLASH_LITE)
# Outgoing SHIPPED OpenRouter defaults (v6.81 and earlier), applied for EVERY
# exclusive-direct provider (incl. cloudru/gigachat/minimax, which have no per-provider
# legacy table): before v6.82.0 a stored copy of the shipped default matched the
# `current in {"", default}` check because SETTINGS_DEFAULTS still carried it;
# after the defaults refresh these stored copies are still "the old DEFAULT, not
# an explicit choice" and must keep migrating to the provider slots. All models
# here stay LIVE (no retirement remap).
_PRIOR_SHIPPED_SLOT_DEFAULTS = {
    "OUROBOROS_MODEL": {"google/gemini-3.5-flash"},
    "OUROBOROS_MODEL_HEAVY": {"google/gemini-3.5-flash"},
    "OUROBOROS_MODEL_LIGHT": {"google/gemini-3.5-flash"},
    "OUROBOROS_MODEL_FALLBACKS": {"anthropic/claude-sonnet-4.6"},
    # v6.81's shipped deep-review value: an upgraded direct-provider install still
    # carries it, and it is just as unreachable without an OpenRouter credential.
    "OUROBOROS_MODEL_DEEP_SELF_REVIEW": {"openai/gpt-5.5-pro", "openai::gpt-5.5-pro"},
}
_ALL_MODEL_SLOT_KEYS = tuple(_DIRECT_PROVIDER_AUTO_DEFAULTS["openai"].keys())
_SCOPE_REVIEW_LEGACY_DEFAULTS = frozenset({
    "",
    "anthropic/claude-opus-4.6",
    "anthropic::claude-opus-4-6",
    "openai/gpt-5.5",
    "openai::gpt-5.5",
    "openai/gpt-5.5-pro",
    "openai::gpt-5.5-pro",
    "openai/gpt-" + "5.4",
    "openai::gpt-" + "5.4",
    "openai/gpt-" + "5.4-pro",
    "openai::gpt-" + "5.4-pro",
    "openai/gpt-" + "5.4-mini",
    "openai::gpt-" + "5.4-mini",
    # The v6.55.0-v6.81 shipped scope-review default (both spellings).
    "anthropic/claude-fable-5",
    "anthropic::claude-fable-5",
})
# The prior shipped scope-review defaults (gpt-5.5 pre-v6.55.0, claude-fable-5
# v6.55.0-v6.81; both spellings each). On the aggregator/general path (no exclusive
# direct provider) a saved value equal to one of them was an old shipped DEFAULT, not
# an explicit user choice — remap it to the current shipped default (gpt-5.6-terra as
# of v6.82.0) so an upgraded install picks up the current scope reviewer instead of
# silently keeping an off-default slot. Scope-review keys ONLY (gpt-5.5 and fable-5
# stay first-class for main/triad slots); the direct-provider path is untouched —
# it already migrates these values via _SCOPE_REVIEW_LEGACY_DEFAULTS above.
_SCOPE_REVIEW_PRIOR_DEFAULTS = frozenset({
    "openai/gpt-5.5", "openai::gpt-5.5",
    "anthropic/claude-fable-5", "anthropic::claude-fable-5",
})
_RETIRED_MODEL_DEFAULT_REPLACEMENTS = {
    "openai/gpt-" + "5.4": "openai/gpt-5.5",
    "openai::gpt-" + "5.4": "openai::gpt-5.5",
    "openai/gpt-" + "5.4-pro": "openai/gpt-5.5-pro",
    "openai::gpt-" + "5.4-pro": "openai::gpt-5.5-pro",
    # NB: gpt-5.4-mini is intentionally absent — it is a LIVE model (the 5.5 family
    # shipped without a mini lane), so it must pass through unchanged. A prior mapping
    # here rewrote it to a non-existent "gpt-5.5-mini" and broke every call on that slot.
    _LEGACY_GEMINI_31_FLASH_LITE: "google/gemini-3.5-flash",
    _LEGACY_GEMINI_31_PRO_PREVIEW: "google/gemini-3.5-flash",
    _LEGACY_GEMINI_3_FLASH_PREVIEW: "google/gemini-3.5-flash",
}


def _truthy_setting(value) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes", "on"}


def _setting_text(settings: dict, key: str) -> str:
    return str(settings.get(key, "") or "").strip()


def _serialize_model_list(models: list[str]) -> str:
    return ",".join(model.strip() for model in models if str(model or "").strip())


def _unique_changed_keys(keys: list[str]) -> list[str]:
    return list(dict.fromkeys(keys))


def _refresh_retired_model_defaults(settings: dict) -> tuple[dict, list[str]]:
    normalized = dict(settings)
    changed: list[str] = []
    keys = [
        "OUROBOROS_MODEL",
        "OUROBOROS_MODEL_HEAVY",
        "OUROBOROS_MODEL_LIGHT",
        "OUROBOROS_MODEL_FALLBACKS",
        "CLAUDE_CODE_MODEL",
        "OUROBOROS_SCOPE_REVIEW_MODEL",
    ]
    for key in keys:
        value = _setting_text(normalized, key)
        replacement = _RETIRED_MODEL_DEFAULT_REPLACEMENTS.get(value)
        if replacement:
            normalized[key] = replacement
            changed.append(key)
    review_value = _setting_text(normalized, "OUROBOROS_REVIEW_MODELS")
    if review_value:
        models = [
            _RETIRED_MODEL_DEFAULT_REPLACEMENTS.get(item, item)
            for item in parse_model_list(review_value)
        ]
        serialized = _serialize_model_list(models)
        if serialized != review_value:
            normalized["OUROBOROS_REVIEW_MODELS"] = serialized
            changed.append("OUROBOROS_REVIEW_MODELS")
    scope_review_value = _setting_text(normalized, "OUROBOROS_SCOPE_REVIEW_MODELS")
    if scope_review_value:
        models = [
            _RETIRED_MODEL_DEFAULT_REPLACEMENTS.get(item, item)
            for item in parse_model_list(scope_review_value)
        ]
        serialized = _serialize_model_list(models)
        if serialized != scope_review_value:
            normalized["OUROBOROS_SCOPE_REVIEW_MODELS"] = serialized
            changed.append("OUROBOROS_SCOPE_REVIEW_MODELS")
    return normalized, _unique_changed_keys(changed)


def _migrate_scope_review_prior_default(settings: dict) -> tuple[dict, list[str]]:
    normalized = dict(settings)
    changed: list[str] = []
    for key in ("OUROBOROS_SCOPE_REVIEW_MODEL", "OUROBOROS_SCOPE_REVIEW_MODELS"):
        if _setting_text(normalized, key) in _SCOPE_REVIEW_PRIOR_DEFAULTS:
            normalized[key] = str(SETTINGS_DEFAULTS[key])
            changed.append(key)
    return normalized, changed


def _provider_prefix(provider: str) -> str:
    return f"{provider}::"


def _exclusive_direct_remote_provider(settings: dict) -> str:
    has_openrouter = bool(_setting_text(settings, "OPENROUTER_API_KEY"))
    has_official_openai = bool(_setting_text(settings, "OPENAI_API_KEY"))
    has_anthropic = bool(_setting_text(settings, "ANTHROPIC_API_KEY"))
    has_minimax = bool(_setting_text(settings, "MINIMAX_API_KEY"))
    has_legacy_openai_base = bool(_setting_text(settings, "OPENAI_BASE_URL"))
    has_compatible = bool(_setting_text(settings, "OPENAI_COMPATIBLE_BASE_URL"))
    has_cloudru = bool(_setting_text(settings, "CLOUDRU_FOUNDATION_MODELS_API_KEY"))
    has_gigachat = bool(_setting_text(settings, "GIGACHAT_CREDENTIALS")) or (
        bool(_setting_text(settings, "GIGACHAT_USER"))
        and bool(_setting_text(settings, "GIGACHAT_PASSWORD"))
    )
    # Mirror config._exclusive_direct_remote_provider_env: OpenRouter / legacy
    # base / compatible disqualify exclusivity; among the registered direct
    # providers return one only when exactly one is set.
    if has_openrouter or has_legacy_openai_base or has_compatible:
        return ""
    direct = [
        name for name, present in (
            ("openai", has_official_openai),
            ("anthropic", has_anthropic),
            ("minimax", has_minimax),
            ("cloudru", has_cloudru),
            ("gigachat", has_gigachat),
        ) if present
    ]
    return direct[0] if len(direct) == 1 else ""


def _normalize_direct_review_models(settings: dict, provider: str) -> str:
    main_model = migrate_model_value(provider, _setting_text(settings, "OUROBOROS_MODEL"))
    current_models = parse_model_list(_setting_text(settings, "OUROBOROS_REVIEW_MODELS"))
    migrated_models = [migrate_model_value(provider, model) for model in current_models]
    provider_prefix = _provider_prefix(provider)

    if not main_model.startswith(provider_prefix):
        return _serialize_model_list(migrated_models)

    has_foreign_models = any(not model.startswith(provider_prefix) for model in migrated_models)
    # Honor an explicit provider-matching list exactly (a single model stays a
    # single slot); expand to the stochastic fallback only when empty/foreign.
    if not migrated_models or has_foreign_models:
        user_light_raw = _setting_text(settings, "OUROBOROS_MODEL_LIGHT")
        fallback = compute_direct_review_models_fallback(
            provider,
            main_model,
            user_light_raw,
            review_runs=_DIRECT_PROVIDER_REVIEW_RUNS,
        )
        return _serialize_model_list(fallback)
    return _serialize_model_list(migrated_models)


def _normalize_direct_scope_review_model(settings: dict, provider: str) -> str:
    current_raw = _setting_text(settings, "OUROBOROS_SCOPE_REVIEW_MODEL")
    default_raw = _setting_text(SETTINGS_DEFAULTS, "OUROBOROS_SCOPE_REVIEW_MODEL")
    current = migrate_model_value(provider, current_raw) if current_raw else ""
    default = migrate_model_value(provider, default_raw) if default_raw else ""
    provider_prefix = _provider_prefix(provider)
    if provider == "openai":
        # Keep the OpenAI-only branch explicit even though the shipped scope-review
        # default is an OpenAI model again (openai/gpt-5.6-terra as of v6.82.0; it was
        # cross-provider fable-5 in v6.55.0-v6.81 and this branch kept the slot
        # callable then). Pinning the designated OpenAI scope reviewer here keeps the
        # slot on the terra reviewer (1M-window designated default) rather than on
        # whatever main model the install migrated to.
        auto_value = migrate_model_value(provider, "openai/gpt-5.6-terra")
    else:
        auto_value = migrate_model_value(
            provider,
            _DIRECT_PROVIDER_AUTO_DEFAULTS.get(provider, {}).get("OUROBOROS_MODEL", ""),
        )
    legacy_defaults = {
        migrate_model_value(provider, item) for item in _SCOPE_REVIEW_LEGACY_DEFAULTS
    }
    if current_raw in {"", default_raw, *_SCOPE_REVIEW_LEGACY_DEFAULTS} or current in {"", default, *legacy_defaults}:
        return auto_value
    if current.startswith(provider_prefix) and current_raw:
        return current
    return current or auto_value


def _normalize_direct_scope_review_models(settings: dict, provider: str) -> str:
    raw = _setting_text(settings, "OUROBOROS_SCOPE_REVIEW_MODELS")
    models = parse_model_list(raw)
    if not models:
        singular = _normalize_direct_scope_review_model(settings, provider)
        return _serialize_model_list([singular] if singular else [])
    migrated = [migrate_model_value(provider, model) for model in models]
    provider_prefix = _provider_prefix(provider)
    if raw == _setting_text(SETTINGS_DEFAULTS, "OUROBOROS_SCOPE_REVIEW_MODELS"):
        return _serialize_model_list([_normalize_direct_scope_review_model(settings, provider)])
    # A SINGLETON list still holding a prior shipped scope default migrates exactly
    # like the singular key. Without this the two keys go split-brain — the runtime
    # prefers the plural one, so an upgraded install would keep the retired
    # sub-floor reviewer and fail every Max-mode commit against the BIBLE P3 floor.
    legacy_scope_defaults = _SCOPE_REVIEW_LEGACY_DEFAULTS | _SCOPE_REVIEW_PRIOR_DEFAULTS
    if len(models) == 1 and (
        models[0] in legacy_scope_defaults or migrated[0] in legacy_scope_defaults
    ):
        return _serialize_model_list([_normalize_direct_scope_review_model(settings, provider)])
    if all(model.startswith(provider_prefix) for model in migrated):
        return _serialize_model_list(migrated)
    return _serialize_model_list([_normalize_direct_scope_review_model(settings, provider)])


def classify_runtime_provider_change(before: dict, after: dict) -> str:
    """Classify what kind of normalization ``apply_runtime_provider_defaults`` did.

    Returns one of:

    - ``"none"`` — no change, or change was purely cosmetic.
    - ``"direct_normalize"`` — OpenRouter is NOT configured, and the function
      auto-filled direct-provider defaults.  This is the only case where a
      user-facing warning is appropriate.
    - ``"reverse_migrate"`` — OpenRouter IS configured (so no exclusive-direct
      provider is active).  ``apply_runtime_provider_defaults`` returned early
      without making any changes, so this is pure housekeeping and should NOT
      produce a warning.
    """
    provider_after = _exclusive_direct_remote_provider(after)
    if provider_after:
        return "direct_normalize"
    has_openrouter_after = bool(_setting_text(after, "OPENROUTER_API_KEY"))
    if has_openrouter_after:
        return "reverse_migrate"
    return "none"


def has_remote_provider(settings: dict) -> bool:
    """Return True when any supported remote-provider credential is configured."""
    if any(
        str(settings.get(key, "") or "").strip()
        for key in (
            "OPENROUTER_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "MINIMAX_API_KEY",
            "OPENAI_COMPATIBLE_BASE_URL",
            "CLOUDRU_FOUNDATION_MODELS_API_KEY",
            "GIGACHAT_CREDENTIALS",
        )
    ):
        return True
    # GigaChat also supports user/password basic auth.
    return bool(str(settings.get("GIGACHAT_USER", "") or "").strip()) and bool(
        str(settings.get("GIGACHAT_PASSWORD", "") or "").strip()
    )


def has_local_routing(settings: dict) -> bool:
    """Return True when a task-capable model slot is routed to local."""
    return any(
        _truthy_setting(settings.get(k))
        for k in ("USE_LOCAL_MAIN", "USE_LOCAL_HEAVY", "USE_LOCAL_LIGHT", "USE_LOCAL_FALLBACK")
    )


def needs_local_model_autostart(settings: dict) -> bool:
    """Return True when any configured model lane needs the local server running."""
    return any(
        _truthy_setting(settings.get(k))
        for k in (
            "USE_LOCAL_MAIN", "USE_LOCAL_HEAVY", "USE_LOCAL_LIGHT",
            "USE_LOCAL_CONSCIOUSNESS", "USE_LOCAL_FALLBACK",
        )
    )


def has_startup_ready_provider(settings: dict) -> bool:
    """Return True when the runtime has enough provider config to serve chat.

    Used both by startup/onboarding gating and by the supervisor-start gate.
    A local model source alone is not enough unless at least one lane is
    routed to that local runtime. (The packaged launcher imports this name —
    keep it stable.)
    """
    return has_remote_provider(settings) or has_local_routing(settings)


_LOCAL_LANE_FLAG_FOR_SLOT = {
    "OUROBOROS_MODEL_LIGHT": "USE_LOCAL_LIGHT",
    "OUROBOROS_MODEL_FALLBACKS": "USE_LOCAL_FALLBACK",
}


def _clear_shipped_defaults_for_local_only(settings: dict) -> list[str]:
    """Keep a LOCAL-ONLY install self-sufficient (DEVELOPMENT: Provider Independence).

    Light and Fallback ship real remote defaults since v6.82.0. With no remote
    credential at all, an UNTOUCHED shipped default in those slots would send the
    Light lane (safety, summaries, deep subagents) to a provider the install cannot
    reach; an empty slot inherits Main, which is the local route. Only values still
    equal to the shipped default are cleared — an explicit choice is never touched.
    """
    # Only a genuine LOCAL-routing install: an empty/unconfigured settings dict (a
    # pre-onboarding first run) must keep the shipped defaults it is about to be
    # offered, and a remote install is untouched.
    if has_remote_provider(settings) or not _setting_text(settings, "LOCAL_MODEL_SOURCE"):
        return []
    changed: list[str] = []
    for key in ("OUROBOROS_MODEL_LIGHT", "OUROBOROS_MODEL_FALLBACKS"):
        # A slot the owner explicitly routed to LOCAL is reachable: the flag sends
        # that lane to the local endpoint whatever the model string says, and for
        # Fallbacks the string is the only chain entry — clearing it would not
        # inherit Main (an empty Fallbacks slot means "no cross-model fallback"),
        # it would silently delete the lane the owner asked for.
        if _truthy_setting(settings.get(_LOCAL_LANE_FLAG_FOR_SLOT[key])):
            continue
        current = _setting_text(settings, key)
        # An UPGRADED local-only install still carries the PRIOR shipped defaults, which
        # are just as unreachable — treat them exactly like the current ones.
        shipped = {_setting_text(SETTINGS_DEFAULTS, key), *_PRIOR_SHIPPED_SLOT_DEFAULTS.get(key, set())}
        if current and current in shipped:
            settings[key] = ""
            changed.append(key)
    return changed


def apply_runtime_provider_defaults(settings: dict) -> tuple[dict, bool, list[str]]:
    """Auto-fill safe runtime defaults for the agreed provider cases."""
    normalized, retired_changed = _refresh_retired_model_defaults(settings)
    provider = _exclusive_direct_remote_provider(normalized)

    if not provider:
        normalized, scope_changed = _migrate_scope_review_prior_default(normalized)
        local_changed = _clear_shipped_defaults_for_local_only(normalized)
        changed_keys = _unique_changed_keys(retired_changed + scope_changed + local_changed)
        return normalized, bool(changed_keys), changed_keys

    changed_keys: list[str] = list(retired_changed)
    provider_defaults = _DIRECT_PROVIDER_AUTO_DEFAULTS[provider]
    main_shipped_default = _setting_text(SETTINGS_DEFAULTS, "OUROBOROS_MODEL")
    for key in _ALL_MODEL_SLOT_KEYS:
        if key not in provider_defaults:
            # This provider has NO reachable value for that slot (deep review needs
            # the >=1M window Cloud.ru/GigaChat/MiniMax do not guarantee). Leaving a
            # SHIPPED default in place would keep an unreachable OpenRouter-form
            # route, so clear it — the review is then honestly unavailable. An
            # explicit owner value is never touched.
            current_shipped = _setting_text(normalized, key)
            shipped_values = {
                _setting_text(SETTINGS_DEFAULTS, key),
                *_PRIOR_SHIPPED_SLOT_DEFAULTS.get(key, set()),
            }
            if current_shipped and current_shipped in shipped_values:
                normalized[key] = ""
                changed_keys.append(key)
            continue
        raw_current = _setting_text(normalized, key)
        current = migrate_model_value(provider, raw_current)
        default = _setting_text(SETTINGS_DEFAULTS, key)
        auto_value = provider_defaults[key]
        legacy_defaults = (
            _DIRECT_PROVIDER_LEGACY_DEFAULTS.get(provider, {}).get(key, set())
            | _PRIOR_SHIPPED_SLOT_DEFAULTS.get(key, set())
        )
        # Heavy/Light default EMPTY -> Main (role-model, v6.39). Their pre-role-model
        # default was the shared Main default, so a stored value equal to it is the old
        # "follow Main" default and migrates to the provider slot exactly like "".
        extra_default = (
            main_shipped_default
            if key in ("OUROBOROS_MODEL_HEAVY", "OUROBOROS_MODEL_LIGHT")
            else ""
        )
        next_value = (
            auto_value
            if current in {"", default, extra_default, *legacy_defaults}
            else current
        )
        if next_value != raw_current:
            normalized[key] = next_value
            changed_keys.append(key)

    review_models = _normalize_direct_review_models(normalized, provider)
    if review_models != _setting_text(normalized, "OUROBOROS_REVIEW_MODELS"):
        normalized["OUROBOROS_REVIEW_MODELS"] = review_models
        changed_keys.append("OUROBOROS_REVIEW_MODELS")

    scope_review_model = _normalize_direct_scope_review_model(normalized, provider)
    if scope_review_model != _setting_text(normalized, "OUROBOROS_SCOPE_REVIEW_MODEL"):
        normalized["OUROBOROS_SCOPE_REVIEW_MODEL"] = scope_review_model
        changed_keys.append("OUROBOROS_SCOPE_REVIEW_MODEL")

    scope_review_models = _normalize_direct_scope_review_models(normalized, provider)
    if scope_review_models != _setting_text(normalized, "OUROBOROS_SCOPE_REVIEW_MODELS"):
        normalized["OUROBOROS_SCOPE_REVIEW_MODELS"] = scope_review_models
        changed_keys.append("OUROBOROS_SCOPE_REVIEW_MODELS")

    changed_keys = _unique_changed_keys(changed_keys)
    return normalized, bool(changed_keys), changed_keys


def setup_remote_if_configured(settings: dict, log) -> None:
    """Set up GitHub remote when credentials are configured."""
    slug = settings.get("GITHUB_REPO", "")
    token = settings.get("GITHUB_TOKEN", "")
    if not token:
        return
    from supervisor.git_ops import configure_personal_remote

    # configure_personal_remote ensures the official `managed` remote exists before
    # repointing `origin`, so a plain official clone never loses official updates.
    remote_ok, remote_msg, _resolved = configure_personal_remote(
        slug,
        token,
        auto_fork=not bool(str(slug or "").strip()),
    )
    if not remote_ok:
        log.warning("Remote configuration failed on startup: %s", remote_msg)


async def ws_heartbeat_loop(
    has_clients_fn: Callable[[], bool],
    broadcast_fn: Callable[[dict], Awaitable[None]],
    interval_sec: float = 15.0,
) -> None:
    """Keep embedded clients active and give watchdogs a steady liveness signal."""
    while True:
        await asyncio.sleep(interval_sec)
        if not has_clients_fn():
            continue
        await broadcast_fn({
            "type": "heartbeat",
            "ts": utc_now_iso(),
        })
