"""Process-local record of each task's last observed prompt-cache split.

Extracted from ``ouroboros.usage_accounting`` (at its module size ceiling) and
re-exported from there. Nothing here is
durable and nothing is locked: a lost, evicted or stale entry only makes the
money reservation price the whole prompt as a fresh cache write again, which is
the conservative direction, so a torn read can never under-reserve.

A settled attempt stores its cached-token count per task, provider, route,
review surface and processing mode; entries expire with the cache TTL. The
count is read only to estimate the next reservation and never becomes a charge
source: settled cost comes from the provider's reported usage.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

# (task_id, provider, route identity, review surface) -> (cached tokens, monotonic stamp, horizon)
_SPLITS: Dict[Tuple[str, str, str, str, str], Tuple[int, float, float]] = {}
_SPLITS_CAP = 64


def _surface() -> str:
    """The logical prompt surface of the current usage scope: its category plus
    the review attribution. Plan, acceptance and skill reviewer sends settle
    under the task id too (ordinary reviews carry their surface only in
    ``category``), and their prefixes must never pose as the transcript's own
    split or as each other's. Only a caller-named wave (``cache_wave``) splits
    it further; a derived review round is attribution only (#1544, owner A)."""
    from ouroboros.usage_accounting import current_usage_scope

    scope = current_usage_scope()
    if scope is None:
        return ""
    category = "" if str(scope.category or "task") == "task" else str(scope.category)
    return "|".join(
        part for part in (
            category, scope.review_skill, scope.cache_wave, scope.review_slot_id,
        ) if part
    )


def _key(task_id: str, provider: str, model: str, processing_mode: str = "") -> Tuple[str, str, str, str, str]:
    """One key per (task, provider, route, surface), normalizing only model spelling.

    The two sides of this store reach it by different names for the same model:
    the fence settles under the ledger's qualified identity
    (``anthropic/claude-opus-5``) while the loop knows the configured slot
    string (``anthropic::claude-opus-5``). Keying on the raw text made the
    default direct-Anthropic install miss its own split on every read, so both
    sides normalize through the identity the fence itself uses -- one leading
    ``~`` probe marker stripped first, exactly as the reservation's own
    model-family test does. The provider and its named connection scope the cache: direct and OpenRouter caches are not shared, public router slugs keep their exact spelling, and named connections keep their own splits even when two keys use the same model.
    """
    from ouroboros.provider_models import connection_id, normalize_model_identity, parse_model_route

    model = str(model or "").strip().removeprefix("~")
    connection = connection_id(str(provider or "").strip().lower(), parse_model_route(model).connection)
    return (str(task_id or "").strip(), connection, normalize_model_identity(model), _surface(), str(processing_mode or ""))


def stash_task_cache_split(
    task_id: str, model: str, cached_tokens: int, *, provider: str = "", ttl_seconds: float,
    processing_mode: str = "",
) -> None:
    """Remember what one task+provider+model send read from the provider cache."""
    key = _key(task_id, provider, model, processing_mode)
    if not key[0] or not key[2]:
        return
    if key not in _SPLITS and len(_SPLITS) >= _SPLITS_CAP:
        _SPLITS.clear()
    _SPLITS[key] = (max(0, int(cached_tokens or 0)), time.monotonic(), float(ttl_seconds))


def last_task_cache_split(task_id: str, model: str, *, provider: str = "", processing_mode: str = "") -> Optional[int]:
    """The task's own last observed cached-token count, or None once it lapsed.

    None also covers a different provider, model, route, or review surface.
    """
    split = _SPLITS.get(_key(task_id, provider, model, processing_mode))
    if split is None or time.monotonic() - split[1] > split[2]:
        return None
    return split[0]


def invalidate_task_cache_splits(task_id: str) -> None:
    """Forget observations whose cacheable transcript prefix was rebuilt."""
    task = str(task_id or "").strip()
    for key in [key for key in _SPLITS if key[0] == task]:
        _SPLITS.pop(key, None)


def reservation_cache_ttl(request) -> str:
    """Price the applied TTL; opaque candidates fall back to the owner's setting."""
    from ouroboros.config import PROMPT_CACHE_TTL_SCALE, resolve_prompt_cache_ttl

    ttl = str(request.prompt_cache_ttl or "").strip().lower()
    if ttl in PROMPT_CACHE_TTL_SCALE:
        return ttl
    # An inspectable marker-free candidate keeps the historical conservative base-tier reservation.
    return "default" if request.candidate_measurement_kind == "canonical_json_v1" else resolve_prompt_cache_ttl()


def reset_task_cache_splits() -> None:
    """Test seam: forget every observed split (process-local, no durable state)."""
    _SPLITS.clear()
