"""Pure reducers from panel actor rows to a tier and a reason line.

Owns the read-only judgement layer above a completed review run: which actors
contributed to the aggregate, the worst outcome tier among them, and the one
honest reason line naming the real blocker. Extracted from
ouroboros/review_substrate.py (v7 D06 split, re-cut on the v7next tip);
review_substrate.py re-exports every name.
"""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

from ouroboros.outcomes import OUTCOME_TIER_BEST_EFFORT, OUTCOME_TIER_BLOCKED, OUTCOME_TIER_SOLVED

if TYPE_CHECKING:  # annotation-only name; lazy under future annotations, never imported at runtime
    from ouroboros.review_records import ReviewRunResult


def _sub():
    """The parent review-substrate module, read at call time.

    The substrate members stay monkeypatch-addressable at their historical
    ``ouroboros.review_substrate`` bindings (tests rebind them there), so this
    leaf resolves every such cross-reference through the module at each call
    instead of freezing whatever object a from-import saw at import time.
    """
    from ouroboros import review_substrate

    return review_substrate


_TIER_ORDER = {OUTCOME_TIER_SOLVED: 0, OUTCOME_TIER_BEST_EFFORT: 1, OUTCOME_TIER_BLOCKED: 2}

# The reviewers' JSON keeps the identifier; the note the model READS (and may
# echo to its human) says the assessment in words — the v6.61.4 token-parroting
# class, where a ledger token becomes owner-facing prose by being quoted back.
TIER_WORDS = {
    OUTCOME_TIER_SOLVED: "a verified solution",
    OUTCOME_TIER_BEST_EFFORT: "a partial result",
    OUTCOME_TIER_BLOCKED: "blocked, with the evidence recorded",
}

# The dialogue votes historical task-acceptance actor rows carry; the review
# projection still reads them off those old rows.
DIALOGUE_STATUS_VALUES = ("continue_actionable", "unreachable_here", "stable_disagreement")


def _contributing_actors(result: ReviewRunResult) -> List[Dict[str, Any]]:
    """Actors whose verdict CONTRIBUTED to the aggregate, so a parse-degraded or
    non-responsive slot cannot inject a tier / coach / finding into a clean quorum
    result (Bible P3: one degraded slot must not poison the aggregate — the exact
    class the split-participation gate was built to avoid). For aggregate PASS only
    PASS actors speak; for FAIL only FAIL actors; for a DEGRADED/UNKNOWN aggregate
    only the cleanly-parsed PASS/FAIL actors may speak (never the degraded ones)."""
    actors = [a for a in (getattr(result, "actors", None) or []) if isinstance(a, dict)]
    agg = str(getattr(result, "aggregate_signal", "") or "").upper()
    if agg in ("PASS", "FAIL"):
        return [a for a in actors if str(a.get("signal", "")).upper() == agg]
    return [a for a in actors if str(a.get("signal", "")).upper() in ("PASS", "FAIL")]


def aggregate_outcome_tier(result: ReviewRunResult) -> str:
    """Worst-tier-wins across the actors that CONTRIBUTED to the aggregate verdict."""
    worst, worst_rank = "", -1
    for actor in _contributing_actors(result):
        parsed = actor.get("parsed") if isinstance(actor, dict) else None
        tier = str((parsed or {}).get("outcome_tier") or "").strip().lower() if isinstance(parsed, dict) else ""
        rank = _TIER_ORDER.get(tier, -1)
        if rank > worst_rank:
            worst_rank, worst = rank, tier
    return worst


def panel_reason(run: Any, *, with_tier: bool = True) -> str:
    """One honest reason line naming the REAL blocker (v6.74.0, A6): the compact
    projection's fallback when a run recorded no reason of its own.
    Accepts a ``ReviewRunResult`` or its dict/namespace record."""
    from types import SimpleNamespace

    if isinstance(run, dict):
        run = SimpleNamespace(**run)
    aggregate = str(getattr(run, "aggregate_signal", "") or "UNKNOWN").upper()
    tier = aggregate_outcome_tier(run)
    # The tier is said in words (the reviewers' JSON keeps the identifier).
    rated = f"rated {TIER_WORDS.get(tier, tier or 'unclassified')} — " if with_tier else ""
    if aggregate == "PASS":
        return f"{rated}passed"
    if aggregate == "FAIL":
        fail_slots = {
            str(actor.get("slot_id", ""))
            for actor in _contributing_actors(run)
            if str(actor.get("signal", "")).upper() == "FAIL"
        }
        named = ""
        for actor in _contributing_actors(run):
            if str(actor.get("signal", "")).upper() != "FAIL":
                continue
            parsed = actor.get("parsed") if isinstance(actor.get("parsed"), dict) else {}
            for finding in (parsed.get("findings") or []):
                if isinstance(finding, dict):
                    named = str(finding.get("item") or finding.get("recommendation") or "").strip()
                    if named:
                        break
            if not named:
                named = str(parsed.get("summary") or "").strip()
            if named:
                break
        if not named:
            # Coordinator-flattened findings (slot_id-stamped) from FAIL slots.
            for finding in (getattr(run, "parsed_findings", None) or []):
                if isinstance(finding, dict) and str(finding.get("slot_id", "")) in fail_slots:
                    named = str(finding.get("item") or finding.get("recommendation") or "").strip()
                    if named:
                        break
        if named:
            compact = _sub().truncate_review_artifact(" ".join(named.split()), limit=300)
            return f"{rated}{compact}"
        return f"{rated}reviewer FAIL without a named finding"
    reasons = [str(r) for r in (getattr(run, "degraded_reasons", None) or []) if str(r)]
    if len(reasons) > 4:
        return "; ".join(reasons[:4]) + f" ⚠️ OMISSION NOTE: +{len(reasons) - 4} more causes in the run record"
    return "; ".join(reasons) or "no valid reviewer quorum"
