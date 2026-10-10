"""Read-only canonical usage projection for one exact Skill Review wave."""

from __future__ import annotations

import pathlib
from typing import Any, Dict

from ouroboros._usage_rows import _skill_review_usage_bucket
from ouroboros.usage_ledger import _drive_root


def skill_review_usage_markdown(
    usage: Dict[str, Any], *, coverage_known: bool, expected: int, recorded: int,
) -> str:
    """Render the canonical per-wave attempt projection without creating totals."""
    def value(name: str) -> str:
        item = usage.get(name)
        return "unknown" if item is None else str(item)

    attempts = [item for item in (usage.get("attempts") or []) if isinstance(item, dict)]
    token_attempts = [
        item for item in attempts
        if str(item.get("state") or "") in {"dispatched", "settled", "unresolved"}
    ]
    attempt_ids = [str(item) for item in (usage.get("attempt_ids") or []) if str(item)]
    integrity_degraded = bool(usage.get("integrity_degraded"))
    coverage_complete = coverage_known and recorded >= expected
    whole_wave_final = coverage_complete and not integrity_degraded and usage.get("cost_final") is True
    cash_label = "Cash" if whole_wave_final else "Recorded-row cash"
    lines = [
        "### Review accounting",
        "",
        "- Canonical attempt IDs: " + (", ".join(attempt_ids) or "none recorded"),
        (f"- {cash_label}: settled ${{settled_usd:.6f}}; confirmed ${{confirmed_usd:.6f}}; "
         "estimated ${estimated_usd:.6f}; unresolved upper bound "
         "${unresolved_upper_bound_usd:.6f}.").format(**usage),
        (f"- Calls: API physical={int(usage.get('physical_calls') or 0)}; "
         f"subscription sessions={int(usage.get('subscription_sessions') or 0)}."),
        (f"- Reported tokens: prompt={value('prompt_tokens')}; completion={value('completion_tokens')}; "
         f"cached={value('cached_tokens')}."),
        (f"- Finality: unknown/unmetered={int(usage.get('unknown_unmetered') or 0)}; "
         f"non-final rows={int(usage.get('non_final_rows') or 0)}; "
         f"ledger integrity={'degraded' if integrity_degraded else 'verified'}; "
         f"slot attribution={'complete' if usage.get('attribution_complete') else 'incomplete'}."),
    ]
    token_gaps = [
        f"{field.removesuffix('_tokens')}={sum(item.get(field) is None for item in token_attempts)}/{len(token_attempts)} unreported"
        for field in ("prompt_tokens", "completion_tokens", "cached_tokens")
        if token_attempts and any(item.get(field) is None for item in token_attempts)
    ]
    if token_gaps:
        lines.append("- Token coverage: " + "; ".join(token_gaps) + ".")
    if integrity_degraded:
        coverage = f"unverified (ledger integrity degraded; {recorded}/{expected} visible)"
    elif coverage_complete:
        coverage = f"complete ({recorded}/{expected} recorded)"
    else:
        coverage = (
            f"incomplete ({recorded}/{expected} recorded)" if coverage_known else
            f"unknown ({recorded} physical rows / {expected} actor occurrences visible)"
        )
    finality = "" if whole_wave_final else "; whole-wave cash and finality are unavailable"
    lines.append(f"- Wave attempt coverage: {coverage}{finality}.")
    for slot_id, bucket in (usage.get("by_slot") or {}).items():
        lines.append(
            f"- Slot {slot_id}: API physical={int(bucket.get('physical_calls') or 0)}, "
            f"subscription sessions={int(bucket.get('subscription_sessions') or 0)}, "
            f"settled=${float(bucket.get('settled_usd') or 0):.6f}."
        )
    windows = usage.get("subscription_windows") or {}
    if windows:
        lines.append("- Subscription windows: " + ", ".join(
            f"{route} resets {reset_at}" for route, reset_at in sorted(windows.items())
        ) + ".")
    for attempt in usage.get("attempts") or []:
        cost = attempt.get("cost_usd")
        cost_text = "unknown" if cost is None else f"${float(cost):.6f}"
        route = attempt.get("subscription_route") or attempt.get("provider") or "unknown"
        route_label = "requested route" if attempt.get("kind") == "subscription_session" else "provider"
        lines.append(
            f"- `{attempt.get('attempt_id')}`: slot={attempt.get('review_slot_id') or 'unattributed'}, "
            f"kind={attempt.get('kind') or 'attempt'}, state={attempt.get('state') or 'unknown'}, "
            f"model={attempt.get('model') or 'unknown'}, {route_label}={route}, "
            f"profile={attempt.get('credential_profile_id') or 'automatic/undisclosed'}, "
            f"access={attempt.get('access_profile') or 'undisclosed'}, cash={cost_text}."
        )
    return "\n".join(lines)


def skill_review_attempt_coverage(
    record: Dict[str, Any], usage: Dict[str, Any],
) -> tuple[bool, int, int]:
    """Compare exact terminal actor slots with canonical physical-attempt slots."""
    expected: Dict[str, int] = {}
    actors = [item for item in (record.get("raw_actor_records") or []) if isinstance(item, dict)]
    for actor in actors:
        slot_id = str(actor.get("slot_id") or "")
        if not slot_id:
            return False, 0, 0
        expected[slot_id] = expected.get(slot_id, 0) + 1
    if not expected:
        return False, 0, 0
    observed: Dict[str, int] = {}
    for attempt in usage.get("attempts") or []:
        # The light-model extraction is a real paid attempt and stays in all
        # monetary/token totals, but it canonicalizes a session answer; it is
        # not the reviewer transport whose late settlement this coverage proves.
        if str(attempt.get("source") or "") == "review_substrate.extraction":
            continue
        if str(attempt.get("state") or "") not in {"dispatched", "settled", "unresolved"}:
            continue
        slot_id = str(attempt.get("review_slot_id") or "")
        if slot_id:
            observed[slot_id] = observed.get(slot_id, 0) + 1
    if any(count > 1 for count in expected.values()):
        # Chunked waves repeat one stable slot id. With only the owner-approved
        # wave/slot attribution, retries cannot be joined to a particular chunk.
        return False, sum(expected.values()), sum(observed.values())
    return True, sum(expected.values()), sum(
        min(count, observed.get(slot_id, 0)) for slot_id, count in expected.items()
    )


def skill_review_usage(
    drive_root: pathlib.Path | str | None = None, *, review_skill: str,
    review_wave_id: str,
) -> Dict[str, Any]:
    """Return exact current physical attempts attributed to one skill/wave
    (the ``(review_skill, review_wave_id)`` index; per-slot buckets follow)."""
    from ouroboros import usage_store

    root = _drive_root(drive_root)
    skill, wave = str(review_skill or ""), str(review_wave_id or "")
    with usage_store.read(root) as txn:
        rows = txn.attempts("review_skill = ? AND review_wave_id = ?", (skill, wave))
        degraded = usage_store.integrity_degraded(root)
    return _skill_review_usage_bucket(rows, review_skill=skill, review_wave_id=wave, integrity_degraded=degraded)


__all__ = [
    "skill_review_attempt_coverage",
    "skill_review_usage",
    "skill_review_usage_markdown",
]
