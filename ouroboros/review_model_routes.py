"""Reviewer quorum, enforcement and typed model targets.

The review pool is the marked catalog rows read by
``reviewer_slot_config.review_pool_slots``. Factory provider panels are minted
once by ``subscription_install_presets.factory_review_rows``. This module
owns the shared quorum and enforcement readers, the per-model typed target,
and the direct-provider helpers re-exported by config.
"""

from __future__ import annotations

import dataclasses

from ouroboros.model_slots import ResolvedModelTarget
from ouroboros.provider_models import (
    resolve_model_target,
    review_model_uses_local,
)
from ouroboros.settings_defaults import SETTINGS_DEFAULTS
from ouroboros.settings_integrity import runtime_setting

_DIRECT_PROVIDER_REVIEW_RUNS = 3


def _exclusive_direct_remote_provider_env() -> str:
    has_openrouter = bool(str(runtime_setting("OPENROUTER_API_KEY", "") or "").strip())
    has_openai = bool(str(runtime_setting("OPENAI_API_KEY", "") or "").strip())
    has_anthropic = bool(str(runtime_setting("ANTHROPIC_API_KEY", "") or "").strip())
    has_minimax = bool(str(runtime_setting("MINIMAX_API_KEY", "") or "").strip())
    has_legacy_base = bool(str(runtime_setting("OPENAI_BASE_URL", "") or "").strip())
    has_compatible = bool(str(runtime_setting("OPENAI_COMPATIBLE_BASE_URL", "") or "").strip())
    has_cloudru = bool(str(runtime_setting("CLOUDRU_FOUNDATION_MODELS_API_KEY", "") or "").strip())
    has_gigachat = bool(str(runtime_setting("GIGACHAT_CREDENTIALS", "") or "").strip()) or (
        bool(str(runtime_setting("GIGACHAT_USER", "") or "").strip())
        and bool(str(runtime_setting("GIGACHAT_PASSWORD", "") or "").strip())
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
        ("deepseek", bool(str(runtime_setting("DEEPSEEK_API_KEY", "") or "").strip())),
        ("zai", bool(str(runtime_setting("ZAI_API_KEY", "") or "").strip())),
    ) if present]
    return direct[0] if len(direct) == 1 else ""


def adaptive_quorum(n_slots: int) -> int:
    """Reviewer-quorum SSOT for an ARBITRARY configured slot count, reused by
    triad/scope/plan/skill/acceptance review. One configured reviewer needs 1 (a loud
    single_reviewer_no_diversity degraded mode), 2 need both, 3+ keep the classic 2-of-N
    majority. DISTINCT from "configured >= quorum but fewer responded", which stays a loud
    infra quorum FAILURE at the call site."""
    return 2 if n_slots >= 3 else max(1, n_slots)


def resolved_review_model_target(model: str, *, effort: str = "") -> ResolvedModelTarget:
    """Construct the ABI-4 typed target for ONE resolved reviewer model.

    The review seam's transport predicate is ``review_model_uses_local`` (a
    local-only Main route pins EVERY review slot to the local lane), so the
    typed ``provider_route`` says ``"local"`` exactly when that predicate
    does — downstream slot builders read the dataclass instead of re-asking
    the predicate per model string. The pool owns model order and membership.
    """
    target = resolve_model_target(model, effort=effort)
    if target.provider_route != "local" and review_model_uses_local(target.model_id):
        target = dataclasses.replace(target, provider_route="local")
    return target


def get_review_enforcement() -> str:
    """Return the configured pre-commit review enforcement mode."""
    default_val = str(SETTINGS_DEFAULTS["OUROBOROS_REVIEW_ENFORCEMENT"])
    raw = (runtime_setting("OUROBOROS_REVIEW_ENFORCEMENT", default_val) or default_val).strip().lower()
    return raw if raw in {"advisory", "blocking"} else default_val
