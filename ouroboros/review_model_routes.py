"""Reviewer quorum and typed model targets.

The review pool is the marked catalog rows read by
``reviewer_slot_config.review_pool_slots``. Factory provider panels are minted
once by ``subscription_install_presets.factory_review_rows``. This module
owns the shared quorum reader, the per-model typed target,
and the direct-provider helpers re-exported by config.
"""

from __future__ import annotations

import dataclasses

from ouroboros.model_slots import ResolvedModelTarget
from ouroboros.provider_models import (
    resolve_model_target,
    review_model_uses_local,
)


def _exclusive_direct_remote_provider_env() -> str:
    """The one direct provider this process reaches, else '' (``provider_models.exclusive_direct_kind``)."""
    from ouroboros.provider_models import exclusive_direct_kind

    return exclusive_direct_kind()


def adaptive_quorum(n_slots: int) -> int:
    """Reviewer-quorum SSOT for an ARBITRARY configured slot count, reused by
    triad/scope/plan/skill/acceptance review. One configured reviewer needs 1 (a loud
    single_reviewer_no_diversity degraded mode), 2 need both, 3+ keep the classic 2-of-N
    quorum. DISTINCT from "configured >= quorum but fewer responded", which stays a loud
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
