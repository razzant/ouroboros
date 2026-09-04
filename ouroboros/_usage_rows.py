"""Pure row-math projections over physical-attempt ledger rows.

Extracted from ``usage_accounting.py`` at the v6.91 gate-3 fix round so the
substrate stays under the hard module gate: the accounted/summary arithmetic,
its limit/integrity decorations, and the per-row physical-call/bucket
projections. A LEAF over plain row dicts — no file I/O, no locks, and it never
imports ``ouroboros.usage_accounting``; that module re-exports every name here
so historical import and monkeypatch sites keep working unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from ouroboros.usage_ledger import _number

REVIEW_ATTRIBUTION_KEYS = ("review_skill", "review_wave_id", "review_slot_id")

class SummaryAccumulator:
    """The ``_summary`` fold as a reusable, SIGNED accumulator.

    ``add(row)`` applies one final row; ``add(row, -1)`` retracts a row that a
    later row of the same attempt has superseded. ``_summary`` itself is
    ``add`` over its rows in order, so a full replay and an incrementally
    maintained accumulator are the same arithmetic (see
    ``_usage_rows_memo._FinalsView``: the reserve path at 500K ledger rows
    otherwise re-folds every final row under the monetary lock per call).

    Only the per-route ``subscription_windows`` maximum is not retractable; a
    view that has ever seen a subscription row reports ``exact_retraction``
    False and its owner re-folds from the final rows instead.
    """

    __slots__ = (
        "settled", "confirmed", "estimated", "reserved", "unresolved", "unknown",
        "non_final_rows", "counts", "sessions", "session_windows", "exact_retraction",
    )

    def __init__(self) -> None:
        self.settled = self.confirmed = self.estimated = self.reserved = self.unresolved = 0.0
        self.unknown = 0
        # Finality is a COUNT of OPEN ROWS, not a truthiness test on dollar sums. Three of the
        # four old terms asked a STATE question of a float, so any row that is genuinely open
        # while holding $0.00 disappeared from the predicate entirely:
        #   * an ESTIMATED $0.00 — what the engine reports for a delegated subscription run
        #     whose cash it has not settled (all 8 estimated rows on a live 60-run page); and
        #   * a DISPATCHED row whose reservation is exactly 0.0, which `_reservation_cost`
        #     returns for `provider="local"`, so the projection claimed `cost_final: True`
        #     with a physical send still in flight.
        # A row is final when it is SETTLED at a known price its writer called final; anything
        # else is open, however little it costs. `_final_rows` keys by attempt_id, so a settled
        # row REPLACES its own reserved/dispatched predecessor — a row still open here is
        # really still open, and a released reservation is in neither branch.
        #
        # The count is also the DISCLOSED CAUSE, returned as `non_final_rows`: a projection
        # reporting `cost_final: false` with every dollar bucket at zero and `unknown` at zero
        # is a flag no reader can reconstruct.
        self.non_final_rows = 0
        self.counts: Dict[str, int] = {}
        # Separate "sessions and quota" axis: subscription work is already paid for, so
        # it contributes exactly $0 to money, and its real scarce resource (sessions and
        # the window that grants them) is counted here instead of being faked as cash.
        self.sessions = 0
        self.session_windows: Dict[str, str] = {}
        self.exact_retraction = True

    def add(self, row: Dict[str, Any], sign: int = 1) -> None:
        state = str(row.get("state") or "")
        if str(row.get("kind") or "") == "subscription_session":
            self.sessions += sign
            route = str(row.get("subscription_route") or "")
            reset_at = str(row.get("subscription_reset_at") or "")
            if route and reset_at:
                self.session_windows[route] = max(self.session_windows.get(route, ""), reset_at)
                self.exact_retraction = False
        if str(row.get("kind") or "") == "legacy_metadata":
            ambiguous = max(1, int(row.get("ambiguous_call_count") or 1))
            self.counts["metadata_only"] = self.counts.get("metadata_only", 0) + sign * ambiguous
            return
        self.counts[state] = self.counts.get(state, 0) + sign
        pricing_unknown = row.get("pricing_known") is False
        if state == "settled":
            cost = _number(row.get("cost_usd"))
            if cost is None:
                self.unknown += sign
                self.non_final_rows += sign
                bound = _number(row.get("reservation_upper_bound_usd"))
                if bound is not None:
                    self.unresolved += sign * bound
            else:
                self.settled += sign * cost
                if bool(row.get("cost_final")):
                    self.confirmed += sign * cost
                else:
                    self.estimated += sign * cost
                    self.non_final_rows += sign
        elif state == "reserved":
            self.non_final_rows += sign
            bound = _number(row.get("reservation_upper_bound_usd"))
            if bound is None or pricing_unknown:
                self.unknown += sign
            if bound is not None:
                self.reserved += sign * bound
        elif state in {"dispatched", "unresolved"}:
            self.non_final_rows += sign
            bound = _number(row.get("reservation_upper_bound_usd"))
            if bound is None or pricing_unknown:
                self.unknown += sign
            if bound is not None:
                self.unresolved += sign * bound

    def render(self) -> Dict[str, Any]:
        settled, confirmed, estimated, reserved, unresolved = (
            round(value, 6) for value in (
                self.settled, self.confirmed, self.estimated, self.reserved, self.unresolved,
            )
        )
        return {
            "settled_usd": settled,
            "confirmed_usd": confirmed,
            "estimated_usd": estimated,
            "reserved_usd": reserved,
            "unresolved_upper_bound_usd": unresolved,
            "accounted_usd": round(settled + reserved + unresolved, 6),
            "unknown_unmetered": self.unknown,
            # Every row that increments `unknown` is open, so the old `not unknown` term is
            # subsumed here rather than dropped.
            "non_final_rows": self.non_final_rows,
            "cost_final": not self.non_final_rows,
            "attempt_counts": {state: count for state, count in self.counts.items() if count},
            "subscription_sessions": self.sessions,
            "subscription_windows": dict(self.session_windows),
        }


def _summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    accumulator = SummaryAccumulator()
    for row in rows:
        accumulator.add(row)
    return accumulator.render()


def _with_limit(summary: Dict[str, Any], limit: Optional[float]) -> Dict[str, Any]:
    if limit is None:
        return summary
    summary["limit_usd"] = round(max(0.0, float(limit)), 6)
    summary["remaining_known_usd"] = round(max(0.0, summary["limit_usd"] - float(summary["accounted_usd"])), 6)
    return summary


def _with_integrity(summary: Dict[str, Any], degraded: bool) -> Dict[str, Any]:
    """Attach ledger integrity and prevent a torn tail from claiming final cost."""
    summary["integrity_degraded"] = bool(degraded)
    if degraded:
        summary["cost_final"] = False
    return summary


def _physical_call_count(row: Dict[str, Any]) -> int:
    kind = str(row.get("kind") or "attempt")
    if kind == "legacy_metadata":
        return 0
    if kind == "legacy_delta":
        return 0
    # A subscription session is not a core-mediated physical provider send; it is
    # counted on the sessions axis instead (see record_subscription_session).
    if kind == "subscription_session":
        return 0
    return 1 if str(row.get("state") or "") in {"dispatched", "settled", "unresolved"} else 0


def _breakdown_bucket(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    bucket = _summary(rows)

    def summed(field: str) -> Optional[int]:
        """Sum the rows that HAVE this count; None when not one of them does.

        `int(row.get(field) or 0)` collapsed an ABSENT count into a reported zero, so a
        bucket whose provider returned no token counts at all published a confident
        "0 tokens" — the render-unknown-as-zero shape this module refuses everywhere
        else (`cost = None  # legacy zero may mean unknown pricing, never "free"`). A
        zero here is now only ever a MEASURED zero, and a partially-reporting bucket
        still sums the rows that did report rather than being erased by the ones that
        did not."""
        present = [row.get(field) for row in rows if row.get(field) is not None]
        return sum(max(0, int(value)) for value in present) if present else None

    prompt = summed("prompt_tokens")
    completion = summed("completion_tokens")
    prompt_cache_ttls: Dict[str, int] = {}
    for row in rows:
        ttl = str(row.get("prompt_cache_ttl") or "").strip()
        if ttl:
            prompt_cache_ttls[ttl] = prompt_cache_ttls.get(ttl, 0) + _physical_call_count(row)
    bucket.update({
        "physical_calls": sum(_physical_call_count(row) for row in rows),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        # Unknown on BOTH halves is an unknown total; one known half is a real total
        # of what was measured, reported as the number it is.
        "total_tokens": (None if prompt is None and completion is None
                         else (prompt or 0) + (completion or 0)),
        "cached_tokens": summed("cached_tokens"),
        "cache_write_tokens": summed("cache_write_tokens"),
        "prompt_cache_ttls": prompt_cache_ttls,
    })
    return bucket


_SKILL_ATTEMPT_FIELDS = (
    "attempt_id", "review_slot_id", "kind", "state", "model", "provider", "source",
    "cost_usd", "cost_final", "reservation_upper_bound_usd", "pricing_known",
    "prompt_tokens", "completion_tokens", "cached_tokens", "subscription_route",
    "subscription_reset_at", "credential_profile_id", "access_profile",
)


def _skill_review_usage_bucket(
    rows: Sequence[Dict[str, Any]], *, review_skill: str, review_wave_id: str,
    integrity_degraded: bool,
) -> Dict[str, Any]:
    """Project one exact Skill Review wave from canonical final attempt rows."""
    selected = sorted(
        (
            row for row in rows
            if str(row.get("review_skill") or "") == review_skill
            and str(row.get("review_wave_id") or "") == review_wave_id
        ),
        key=lambda row: str(row.get("attempt_id") or ""),
    )
    grouped: Dict[str, list[Dict[str, Any]]] = {}
    for row in selected:
        grouped.setdefault(str(row.get("review_slot_id") or "(unattributed)"), []).append(row)
    by_slot = {
        slot: _with_integrity(_breakdown_bucket(grouped[slot]), integrity_degraded)
        for slot in sorted(grouped)
    }
    result = _with_integrity(_breakdown_bucket(selected), integrity_degraded)
    result.update({
        "review_skill": review_skill,
        "review_wave_id": review_wave_id,
        "attempt_ids": [str(row.get("attempt_id") or "") for row in selected],
        "attempts": [
            {key: row.get(key) for key in _SKILL_ATTEMPT_FIELDS if key in row}
            for row in selected
        ],
        "by_slot": by_slot,
        "attribution_complete": bool(selected) and all(
            str(row.get("review_slot_id") or "") for row in selected
        ),
    })
    return result
