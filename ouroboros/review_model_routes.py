"""Ouroboros — the reviewer model lists the API-pinned review surfaces run.

ABI 7.0 (ABI-10, owner 5.4=A): the comma-list SETTINGS keys are retired — the
structured ``OUROBOROS_REVIEWER_SLOTS`` is the one configuration surface. The
comma ENV keys survive only as the derived runtime projection
(``reviewer_slot_config.project_reviewer_slots_into_env``) plus an operational
env-override plane (bench launchers): these getters resolve that env plane —
honouring a local-only Main route and rewriting the list when the install has
exactly one direct provider credentialed — and fall back to the shipped
``OPENROUTER_REVIEW_DEFAULTS``. Also the reviewer-quorum rule shared by every
review family.
"""

from __future__ import annotations

import dataclasses
import os

from ouroboros.model_slots import ResolvedModelTarget, _main_model, _parse_model_list
from ouroboros.provider_models import (
    compute_direct_review_models_fallback,
    local_only_review_route_env,
    migrate_model_value,
    resolve_model_target,
    review_model_uses_local,
)
from ouroboros.settings_defaults import OPENROUTER_REVIEW_DEFAULTS, SETTINGS_DEFAULTS

_DIRECT_PROVIDER_REVIEW_RUNS = 3


def _exclusive_direct_remote_provider_env() -> str:
    has_openrouter = bool(str(os.environ.get("OPENROUTER_API_KEY", "") or "").strip())
    has_openai = bool(str(os.environ.get("OPENAI_API_KEY", "") or "").strip())
    has_anthropic = bool(str(os.environ.get("ANTHROPIC_API_KEY", "") or "").strip())
    has_minimax = bool(str(os.environ.get("MINIMAX_API_KEY", "") or "").strip())
    has_legacy_base = bool(str(os.environ.get("OPENAI_BASE_URL", "") or "").strip())
    has_compatible = bool(str(os.environ.get("OPENAI_COMPATIBLE_BASE_URL", "") or "").strip())
    has_cloudru = bool(str(os.environ.get("CLOUDRU_FOUNDATION_MODELS_API_KEY", "") or "").strip())
    has_gigachat = bool(str(os.environ.get("GIGACHAT_CREDENTIALS", "") or "").strip()) or (
        bool(str(os.environ.get("GIGACHAT_USER", "") or "").strip())
        and bool(str(os.environ.get("GIGACHAT_PASSWORD", "") or "").strip())
    )
    # OpenRouter / legacy OpenAI base / OpenAI-compatible all route through the
    # OpenRouter-style stack, so their presence means "not an exclusive direct
    # provider". Among the registered direct providers, return one only when
    # exactly one is configured.
    if has_openrouter or has_legacy_base or has_compatible:
        return ""
    direct = [name for name, present in (
        ("openai", has_openai), ("anthropic", has_anthropic), ("minimax", has_minimax),
        ("cloudru", has_cloudru), ("gigachat", has_gigachat),
        ("deepseek", bool(str(os.environ.get("DEEPSEEK_API_KEY", "") or "").strip())),
    ) if present]
    return direct[0] if len(direct) == 1 else ""


def direct_provider_review_models_fallback(provider: str) -> list[str]:
    """Return the exact review-models list a direct-provider fallback emits."""
    if provider not in ("openai", "anthropic", "minimax", "cloudru", "gigachat", "deepseek"):
        return []
    main_model = str(
        os.environ.get("OUROBOROS_MODEL", SETTINGS_DEFAULTS["OUROBOROS_MODEL"]) or ""
    ).strip()
    main_model = migrate_model_value(provider, main_model)
    user_light_raw = str(os.environ.get("OUROBOROS_MODEL_LIGHT", "") or "").strip()
    return compute_direct_review_models_fallback(
        provider,
        main_model,
        user_light_raw,
        review_runs=_DIRECT_PROVIDER_REVIEW_RUNS,
    )


def adaptive_quorum(n_slots: int) -> int:
    """Reviewer-quorum SSOT for an ARBITRARY configured slot count, reused by
    triad/scope/plan/skill/acceptance review. One configured reviewer needs 1 (a loud
    single_reviewer_no_diversity degraded mode), 2 need both, 3+ keep the classic 2-of-N
    majority. DISTINCT from "configured >= quorum but fewer responded", which stays a loud
    infra quorum FAILURE at the call site."""
    return 2 if n_slots >= 3 else max(1, n_slots)


def get_review_models() -> list[str]:
    """Return the effective triad model list from the derived env plane."""
    default_str = ",".join(OPENROUTER_REVIEW_DEFAULTS["triad"])
    models_str = os.environ.get("OUROBOROS_REVIEW_MODELS", default_str) or default_str
    models = _parse_model_list(models_str)
    models = [_main_model()] * max(1, len(models)) if local_only_review_route_env() else models
    provider = _exclusive_direct_remote_provider_env()
    if not provider:
        return models

    main_model = str(os.environ.get("OUROBOROS_MODEL", SETTINGS_DEFAULTS["OUROBOROS_MODEL"]) or "").strip()
    main_model = migrate_model_value(provider, main_model)
    provider_prefix = f"{provider}::"
    if not main_model.startswith(provider_prefix):
        return models

    migrated = [migrate_model_value(provider, model) for model in models]
    if not migrated or any(not model.startswith(provider_prefix) for model in migrated):
        # Auto-expand to the [main]*N stochastic fallback ONLY when nothing usable is
        # configured (empty, or foreign models in an exclusive direct-provider setup). An
        # explicit provider-matching list is honored exactly, duplicates included.
        return direct_provider_review_models_fallback(provider)
    return migrated


def resolved_review_model_target(model: str, *, effort: str = "") -> ResolvedModelTarget:
    """Construct the ABI-4 typed target for ONE resolved reviewer model.

    The review seam's transport predicate is ``review_model_uses_local`` (a
    local-only Main route pins EVERY review slot to the local lane), so the
    typed ``provider_route`` says ``"local"`` exactly when that predicate
    does — downstream slot builders read the dataclass instead of re-asking
    the predicate per model string. Purely a typed view: the model lists
    themselves stay ``get_review_models``/``get_scope_review_models``.
    """
    target = resolve_model_target(model, effort=effort)
    if target.provider_route != "local" and review_model_uses_local(target.model_id):
        target = dataclasses.replace(target, provider_route="local")
    return target


def get_review_targets() -> tuple[ResolvedModelTarget, ...]:
    """The effective triad list as typed targets (ABI-4), same order/membership.

    TYPED VIEW FOR FUTURE CONSUMERS — no production caller yet: today's review
    lanes consume ``get_review_models`` plus ``resolved_review_model_target``
    per slot (reviewer_slot_config); wiring a whole-list consumer is review-
    surface work outside the ABI-4 sweep's byte-identical contract."""
    return tuple(resolved_review_model_target(model) for model in get_review_models())


def get_scope_review_targets() -> tuple[ResolvedModelTarget, ...]:
    """The effective scope list as typed targets (ABI-4), duplicates preserved.

    TYPED VIEW FOR FUTURE CONSUMERS — no production caller yet (see
    ``get_review_targets``)."""
    return tuple(resolved_review_model_target(model) for model in get_scope_review_models())


def get_review_enforcement() -> str:
    """Return the configured pre-commit review enforcement mode."""
    default_val = str(SETTINGS_DEFAULTS["OUROBOROS_REVIEW_ENFORCEMENT"])
    raw = (os.environ.get("OUROBOROS_REVIEW_ENFORCEMENT", default_val) or default_val).strip().lower()
    return raw if raw in {"advisory", "blocking"} else default_val


def get_scope_review_models() -> list[str]:
    """Return effective scope reviewer models, preserving duplicate model IDs."""
    default_str = ",".join(OPENROUTER_REVIEW_DEFAULTS["scope"])
    raw = os.environ.get("OUROBOROS_SCOPE_REVIEW_MODELS", "") or ""
    if not raw.strip():
        raw = os.environ.get("OUROBOROS_SCOPE_REVIEW_MODEL", default_str) or default_str
    models = _parse_model_list(raw)
    singular = str(os.environ.get("OUROBOROS_SCOPE_REVIEW_MODEL", OPENROUTER_REVIEW_DEFAULTS["scope"][0]) or "").strip()
    if not models and singular:
        models = [singular]
    if not models:
        models = _parse_model_list(default_str)
    models = [_main_model()] * max(1, len(models)) if local_only_review_route_env() else models
    provider = _exclusive_direct_remote_provider_env()
    if not provider:
        return models
    migrated = [migrate_model_value(provider, model) for model in models]
    provider_prefix = f"{provider}::"
    if migrated and all(model.startswith(provider_prefix) for model in migrated):
        return migrated
    migrated_singular = migrate_model_value(provider, singular or OPENROUTER_REVIEW_DEFAULTS["scope"][0])
    if migrated_singular.startswith(provider_prefix):
        return [migrated_singular]
    fallback = direct_provider_review_models_fallback(provider)
    return fallback[:1] if fallback else migrated
