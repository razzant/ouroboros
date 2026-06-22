"""Canonical fingerprint for evolution-campaign objectives (BUG 3 objective-repeat gate).

Leaf module by design: it imports nothing beyond the standard library, so every layer that
needs the fingerprint — ``post_task_evolution`` (suffix append), ``supervisor.evolution_lifecycle``
(per-cycle stamp + repeat counter), ``supervisor.queue`` (pause gate) and
``ouroboros.agent_startup_checks`` (absorb-clear) — can import it without re-introducing the
documented ``evolution_lifecycle`` <-> ``queue`` import cycle.

The single source of truth for the objective fingerprint. All sites that count, gate, or clear
objective repeats MUST key off :func:`canonical_objective_fingerprint`; mixing a raw hash on one
side and a suffix-stripping hash on the other would make the counter accumulate under one key
while the gate tests another (the post-task objective would then never trip the gate).
"""
from __future__ import annotations

import hashlib
import re

# The plan-review hint that ``post_task_evolution.apply_pending_request`` appends to a promoted
# objective when the source backlog item requires plan review. Kept here (and imported by
# post_task_evolution) so the append and the strip below can never drift: a single backlog item
# promoted once with requires_plan_review=True and once with False must map to ONE fingerprint.
# This literal MUST stay byte-identical to the string that is appended.
_PLAN_REVIEW_SUFFIX = (
    "\n\n(The source backlog item requires plan review: run plan_task "
    "before implementing any code.)"
)


def canonical_objective_fingerprint(objective: str) -> str:
    """Return the stable per-objective key used by the BUG 3 objective-repeat gate.

    Strips the plan-review suffix, collapses whitespace, lowercases, then sha256[:16].
    Returns ``""`` for an empty/whitespace-only objective so callers can skip bucketing
    under a meaningless key.
    """
    base = str(objective or "")
    if base.endswith(_PLAN_REVIEW_SUFFIX):
        base = base[: -len(_PLAN_REVIEW_SUFFIX)]
    norm = re.sub(r"\s+", " ", base).strip().lower()
    if not norm:
        return ""
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]
